"""Isolate task-zero wrist feedback versus a constant physical servo target.

Only this independent probe intercepts robot target generation. All runs start
from the same snapshot and same Task reset bank. No task-time rigid-body writes,
fixture, additional tool force, or changes to the shared source are made.
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',required=True)
p.add_argument('--source-run',default='runs/rl_tripod_supported_zero06')
p.add_argument('--seconds',type=float,default=5.)
p.add_argument('--copies',type=int,default=8)
p.add_argument('--pose',default='configs/human_grasp_tripod_webtail_supported_v2.json')
p.add_argument('--snapshot',default='runs/tripod_webtail_v1_fine06/c-0.05_ix-0.01_mid0_snapshot.json')
p.add_argument('--modes',nargs='+',default=['task_zero_actual_joint_base','constant_saved_joint_target','anchored_cartesian_actual_base','cartesian_previous_target_base'])
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.headless=True
out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False)
source=out/'source';source.mkdir();source_run=ROOT/a.source_run
for name in ['rl_scene.py','rl_task.py','reset_reference.py','arm_clearance.py','kinematics.py','grasp_contacts.py']:
 shutil.copy2(source_run/name,source/name)
shutil.copy2(__file__,source/Path(__file__).name)
pose=source/'pose.json';shutil.copy2(ROOT/a.pose,pose)
snapshot=ROOT/a.snapshot
(out/'launch.json').write_text(json.dumps({'arguments':vars(a),'source_sha256':{str(x.relative_to(out)):hashlib.sha256(x.read_bytes()).hexdigest() for x in source.iterdir()},'snapshot':str(snapshot),'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()},indent=2))
app=AppLauncher(a).app
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(source))
from rl_scene import BatchedTweezersScene
from rl_task import TweezerTask,TaskConfig,quat_mul,quat_inv
modes=a.modes
n=len(modes)*a.copies
torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
saved=json.loads(snapshot.read_text());hq=saved['zero_velocity_variant']['rigid_body_states_13']['handle'][3:7]
R=Rotation.from_quat(np.roll(hq,-1)).as_matrix()
scene=BatchedTweezersScene(num_envs=n,device=a.device,dt=1/480,pose_json=pose,tool_world_R=R,
 spring=5.,spring_damping=.02,hand_stiffness=10.,hand_damping=.3,finger_friction=1.4,
 solver_iterations=64,solver_velocity_iterations=8,contact_offset=.0002,simplify_arm_collisions=False,
 self_collisions=True,filter_mechanical_self_pairs=True,enable_grasp_support=True)
cfg=TaskConfig(decimation=8,episode_seconds=a.seconds+1.,initial_tip_clearance=.018,reset_xy_range=.002,
 closure_bias=0.,closure_scale=.45,closure_min=-.45,closure_max=.7,seed=0,reset_snapshot=str(snapshot),
 require_web_up=True,min_web_up_component=.35,max_palm_up_component=.25,require_tripod_support=True,
 support_vertical_threshold_N=.003,independent_support_actions=True,tool_down_angle_range_deg=(20.,60.))
env=TweezerTask(scene,cfg)
# Choose the same bank pose in every environment; only environment translation differs.
q0=env.bank_q[0].repeat(n,1).clone();constant=env.bank_q_target[0].repeat(n,1).clone()
objects={k:v.clone() for k,v in scene.default_object_states.items()}
for k in ['handle','left','right']:
 objects[k]=env.bank_object_states[k][0].repeat(n,1).clone();objects[k][:,:3]+=scene.env_origins
scene.reset(q=q0,object_states=objects)
scene.robot.set_joint_position_target(constant);scene.robot.write_data_to_sim()
env.q_target[:]=constant;env.hand_target[:]=constant[:,env.hand];env.steps[:]=0
s=scene.get_state();initial_wrist_p=s['wrist_pos_w'].clone();initial_wrist_q=s['wrist_quat_w'].clone()
initial_tool_p=s['tool_root_state_w'][:,:3].clone();initial_tool_q=s['tool_root_state_w'][:,3:7].clone()
previous_command=constant.clone()
original_step=scene.step
mask={name:torch.arange(i*a.copies,(i+1)*a.copies,device=scene.device) for i,name in enumerate(modes)}
first_command=None

def intercepted_step(target,decimation=1,render=False,substeps=None):
 global previous_command,first_command
 state=scene.get_state();jac=scene.get_wrist_jacobian()
 reg=.02**2*torch.eye(6,device=scene.device).expand(n,6,6)
 qerr=quat_mul(initial_wrist_q,quat_inv(state['wrist_quat_w']))
 rot=(2*qerr[:,1:]*torch.sign(qerr[:,:1])).clamp(-.05,.05)
 def solve(position_error):
  error=torch.cat((position_error,rot),-1)
  return (jac.transpose(-1,-2)@torch.linalg.solve(jac@jac.transpose(-1,-2)+reg,error.unsqueeze(-1))).squeeze(-1).clamp(-.025,.025)
 if 'constant_saved_joint_target' in mask:
  target[mask['constant_saved_joint_target']]=constant[mask['constant_saved_joint_target']]
 if 'anchored_cartesian_actual_base' in mask:
  ids=mask['anchored_cartesian_actual_base'];dq=solve(initial_wrist_p-state['wrist_pos_w'])
  target[ids[:,None],env.arm]=state['joint_pos'][ids[:,None],env.arm]+dq[ids]
 if 'cartesian_previous_target_base' in mask:
  ids=mask['cartesian_previous_target_base'];dq=solve(torch.zeros_like(initial_wrist_p))
  target[ids[:,None],env.arm]=previous_command[ids[:,None],env.arm]+dq[ids]
 previous_command=target.clone()
 if first_command is None:first_command=target.detach().cpu().numpy().tolist()
 return original_step(target,decimation=decimation,render=render,substeps=substeps)
scene.step=intercepted_step
scene.stage.GetRootLayer().Export(str(out/'initial_scene.usda'))
(out/'task.json').write_text(json.dumps(env.metadata(),indent=2))
initial={'q_actual':q0.cpu().tolist(),'q_target':constant.cpu().tolist(),'wrist_pos_w':initial_wrist_p.cpu().tolist(),'wrist_quat_w':initial_wrist_q.cpu().tolist(),'objects':{k:v.cpu().tolist() for k,v in objects.items()}}
(out/'initial_state.json').write_text(json.dumps(initial,indent=2))
records=[];zero=torch.zeros((n,env.action_size),device=scene.device)
def cpu(v):return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
print('AB_READY',json.dumps({'modes':modes,'copies':a.copies,'arm_initial_tracking_bias':cpu((constant-q0)[0,env.arm]).tolist()}),flush=True)
steps=round(a.seconds/env.dt)
for step in range(steps):
 env.step(zero,auto_reset=False);m=env.measurements();s=m['state'];support=m['grasp_support']
 wrist_angle=torch.rad2deg(2*torch.acos((s['wrist_quat_w']*initial_wrist_q).sum(-1).abs().clamp(0,1)))
 tool_angle=torch.rad2deg(2*torch.acos((s['tool_root_state_w'][:,3:7]*initial_tool_q).sum(-1).abs().clamp(0,1)))
 record={'time':(step+1)*env.dt,'q_actual':cpu(s['joint_pos']),'q_target':cpu(env.q_target),
  'wrist_translation':cpu(s['wrist_pos_w']-initial_wrist_p),'wrist_orientation_drift_deg':cpu(wrist_angle),
  'tool_translation':cpu(s['tool_root_state_w'][:,:3]-initial_tool_p),'tool_orientation_drift_deg':cpu(tool_angle),
  'tool_angle_table_deg':cpu(m['tool_down_angle_deg']),'grip_error_m':cpu(m['grip_error']),
  'support_gate':cpu(m['grasp_support_ok']),'strength_N':cpu(support['role_contact_strength_N']),
  'middle_Fz_N':cpu(support['role_support_z_N'][:,2]),'web_Fz_N':cpu((support['web_normal_force_w']+support['web_friction_force_w'])[:,2]),
  'middle_under_count':cpu(support['middle_under_contact_count']),'web_count':cpu(support['web_contact_count']),
  'block_contact_N':cpu(m['forces'])}
 records.append(record)
 if (step+1)%60==0:
  print('AB_PROGRESS',json.dumps({'time':record['time'],'support_fraction_recent':{name:float(np.stack([r['support_gate'] for r in records[-60:]])[:,i*a.copies:(i+1)*a.copies].mean()) for i,name in enumerate(modes)}}),flush=True)
import copy
final_state=scene.get_state()
for i in range(n):
 obj={key:cpu(final_state[key+'_root_state_w'][i]).tolist() for key in ['handle','left','right']}
 snapshot_out={'schema_version':1,'source_run':str(out.relative_to(ROOT)),'source_candidate':modes[i//a.copies]+'_copy'+str(i%a.copies),'physics_time_s':a.seconds,'joint_names':scene.joint_names,
 'q_actual':cpu(final_state['joint_pos'][i]).tolist(),'q_target':cpu(env.q_target[i]).tolist(),'q_velocity':cpu(scene.robot.data.joint_vel[i]).tolist(),
 'rigid_body_states_13':obj,'world_wrist_pose_7':cpu(torch.cat((final_state['wrist_pos_w'][i],final_state['wrist_quat_w'][i]))).tolist(),
 'world_hand_pose_7':cpu(torch.cat((final_state['palm_pos_w'][i],final_state['palm_quat_w'][i]))).tolist(),
 'arm_base_world_xyz':(cpu(scene.env_origins[i])+np.array([0.,0.,scene.table_height])).tolist(),'fixture_active':False,'args':vars(a),
 'tripod_hold':{'support_gate_fraction_after1s':float(np.mean([r['support_gate'][i] for r in records if r['time']>=1.])),'evidence':'all8physics-substep mean support reconstructed by shared frozen contact reader; see records.npz'}}
 zero_state=copy.deepcopy(snapshot_out)
 zero_state['q_velocity']=[0.]*len(scene.joint_names)
 for body in zero_state['rigid_body_states_13'].values():body[7:]=[0.]*6
 snapshot_out['zero_velocity_variant']=zero_state
 (out/(snapshot_out['source_candidate']+'_snapshot.json')).write_text(json.dumps(snapshot_out,indent=2)+'\n')
arrays={k:np.stack([r[k] for r in records]) for k in records[0]};np.savez_compressed(out/'records.npz',**arrays)
active=arrays['time']>=1.
report={'source_run':a.source_run,'modes':{},'first_joint_target':first_command,'all_object_writes_reset_only':True,'same_snapshot_hand_targets_and_bank_pose':True}
for i,name in enumerate(modes):
 sl=slice(i*a.copies,(i+1)*a.copies)
 report['modes'][name]={
  'support_gate_fraction_per_env_after1s':arrays['support_gate'][active,sl].mean(0).tolist(),
  'middle_Fz_N_mean_min_max':[float(arrays['middle_Fz_N'][active,sl].mean()),float(arrays['middle_Fz_N'][active,sl].min()),float(arrays['middle_Fz_N'][active,sl].max())],
  'web_Fz_N_mean_min_max':[float(arrays['web_Fz_N'][active,sl].mean()),float(arrays['web_Fz_N'][active,sl].min()),float(arrays['web_Fz_N'][active,sl].max())],
  'wrist_translation_final_m':arrays['wrist_translation'][-1,sl].tolist(),
  'wrist_orientation_final_deg':arrays['wrist_orientation_drift_deg'][-1,sl].tolist(),
  'tool_orientation_final_deg':arrays['tool_orientation_drift_deg'][-1,sl].tolist(),
  'max_grip_error_m':float(arrays['grip_error_m'][:,sl].max()),'max_block_contact_N':float(arrays['block_contact_N'][:,sl].max()),
  'final_arm_target_shift_from_reset_rad':(arrays['q_target'][-1,sl][:,cpu(env.arm).astype(int)]-cpu(constant)[sl][:,cpu(env.arm).astype(int)]).tolist()}
(out/'report.json').write_text(json.dumps(report,indent=2))
print('AB_RESULT',json.dumps(report['modes']),flush=True)
app.close()
