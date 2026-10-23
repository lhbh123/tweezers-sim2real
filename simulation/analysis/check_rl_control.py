"""Independent batched IK direction and contact isolation audit, not training."""
import argparse
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--pose', default='configs/human_grasp_proximal.json')
parser.add_argument('--out', default='analysis/rl_control_audit.json')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rl_scene import BatchedTweezersScene
from rl_task import TweezerTask, TaskConfig, quat_apply

scene = BatchedTweezersScene(num_envs=8, device=args.device, pose_json=args.pose,
                            dt=1/480, spring_damping=.02,
                            tool_world_R=Rotation.from_euler('y',30,degrees=True).as_matrix() @ np.diag([1.,-1.,-1.]))
task = TweezerTask(scene, TaskConfig(reset_xy_range=0., initial_tip_clearance=.05))
device = scene.device
report = {'device':device, 'num_envs':8, 'physics_dt':scene.dt,
          'policy_dt':task.dt, 'self_collisions_enabled':scene.self_collisions,
          'notes':['Robot action is the existing task implementation, auto_reset=False.',
                   'Tools are moved away only before the IK audit to isolate the arm servo.',
                   'Contact tests intentionally overlap block and jaw only during reset.']}

# Isolate Cartesian command mapping from contacts, without altering the controller.
objects = {name: obj.data.root_state_w.clone() for name,obj in scene.objects.items()}
for state in objects.values():
    state[:,1] += .35
    state[:,7:] = 0.
scene.reset(q=task.q_target.clone(), object_states=objects)
for _ in range(32): scene.step(task.q_target)
actions = torch.tensor([[.5,0,0,0],[-.5,0,0,0],[0,.5,0,0],[0,-.5,0,0],
                        [0,0,.5,0],[0,0,-.5,0],[0,0,0,0],[.5,.5,.5,0]],device=device)
expected = actions[:,:3] * task.cfg.max_wrist_step
displacements, predicted_targets, linearization_errors = [], [], []
before_first = scene.get_state()['wrist_pos_w'].clone()
for _ in range(24):
    state = scene.get_state()
    before_pos = state['wrist_pos_w'].clone()
    before_q = state['joint_pos'][:,scene.arm_joint_indices].clone()
    jac = scene.get_wrist_jacobian().clone()
    _, _, _, _ = task.step(actions, auto_reset=False)
    after = scene.get_state()
    dq_actual = after['joint_pos'][:,scene.arm_joint_indices] - before_q
    dq_command = task.q_target[:,scene.arm_joint_indices] - before_q
    actual = after['wrist_pos_w'] - before_pos
    linear = (jac[:,:3] @ dq_actual.unsqueeze(-1)).squeeze(-1)
    predicted_targets.append((jac[:,:3] @ dq_command.unsqueeze(-1)).squeeze(-1))
    displacements.append(actual)
    linearization_errors.append((actual-linear).norm(dim=-1))
actual = torch.stack(displacements)
predicted = torch.stack(predicted_targets)
steady = actual[-12:].mean(0)
norm_expected = expected.norm(dim=-1)
cosine = (steady*expected).sum(-1) / (steady.norm(dim=-1)*norm_expected).clamp_min(1e-12)
ratio = steady.norm(dim=-1)/norm_expected.clamp_min(1e-12)
moving = norm_expected > 0
report['ik'] = {
    'actions':actions.tolist(), 'commanded_xyz_per_policy_step_m':expected.tolist(),
    'first_actual_xyz_m':actual[0].tolist(), 'steady_mean_actual_xyz_m':steady.tolist(),
    'steady_jacobian_target_prediction_m':predicted[-12:].mean(0).tolist(),
    'total_actual_xyz_m':(scene.get_state()['wrist_pos_w']-before_first).tolist(),
    'direction_cosine':cosine.tolist(), 'servo_displacement_fraction':ratio.tolist(),
    'max_actual_joint_jacobian_linearization_error_m':float(torch.stack(linearization_errors).max()),
    'zero_action_steady_displacement_m':float(steady[6].norm()),
    'pass': bool((cosine[moving]>.95).all() and (ratio[moving]>.1).all()
                 and (ratio[moving]<1.5).all() and steady[6].norm()<2e-5)}
print('IK_AUDIT',json.dumps(report['ik']),flush=True)

def contact_trial(source_env, jaw_name, cross_target_env=None):
    objects = {name: value.clone() for name,value in scene.default_object_states.items()}
    for name,state in objects.items():
        if name != 'block': state[:,1] += .25
        else:
            state[:,:3] += torch.tensor([.30,.30,.20],device=device)
        state[:,7:] = 0.
    target_env = source_env if cross_target_env is None else cross_target_env
    jaw = objects[jaw_name][target_env]
    offset = torch.tensor([[0.,0.,.07]],device=device)
    objects['block'][source_env,:3] = jaw[:3] + quat_apply(jaw[3:7].reshape(1,4),offset)[0]
    scene.reset(object_states=objects)
    immediate = scene.get_state()['block_jaw_contact_forces_w'].norm(dim=-1)
    peaks = torch.zeros((8,2),device=device)
    for _ in range(12):
        state = scene.step(scene.default_joint_pos)
        peaks = torch.maximum(peaks,state['block_jaw_contact_forces_w'].norm(dim=-1))
    others = torch.arange(8,device=device) != source_env
    return {'source_env':source_env,'target_env':target_env,'jaw':jaw_name,
            'reset_contact_max_N':float(immediate.max()),
            'peak_contact_per_env_jaw_N':peaks.tolist(),
            'source_peak_N':float(peaks[source_env].max()),
            'other_environment_peak_N':float(peaks[others].max())}

trials = [contact_trial(0,'left'), contact_trial(3,'right'),
          contact_trial(0,'left',cross_target_env=1)]
contact_pass = (all(t['source_peak_N']>.001 and t['other_environment_peak_N']<1e-5 for t in trials[:2])
                and trials[2]['source_peak_N']<1e-5
                and all(t['reset_contact_max_N']==0. for t in trials))
report['contact_isolation'] = {'trials':trials,'pass':contact_pass}
report['pass'] = report['ik']['pass'] and contact_pass
Path(args.out).write_text(json.dumps(report,indent=2))
print('CONTROL_AUDIT',json.dumps(report),flush=True)
scene.close()
app.close()
if not report['pass']: raise SystemExit(1)
