"""Finite, labeled scripted actuation experiment, NOT a learned-policy result.

Uses exactly the task action interface; robot targets only. No runtime object
state/force writes. Tests whether cube contact changes the empty-pinch balance.
"""
import argparse, hashlib, itertools, json, shutil, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',required=True)
p.add_argument('--pose',default='configs/human_grasp_tripod_horizontal_settled_v1.json')
p.add_argument('--snapshot',default='runs/tripod_horizontal_hold12/closure0_copy0_snapshot.json')
p.add_argument('--coordinated',action='store_true',help='Collect varied demonstrations through the final coordinated action space')
p.add_argument('--copies',type=int,default=32)
p.add_argument('--repeat-successful',action='store_true',help='Use measured successful coordinated closure1.05 and middle residual−.005')
p.add_argument('--reset-seed',type=int,default=17)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.headless=True
out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
paths=[ROOT/'src'/n for n in ['arm_command.py','arm_clearance.py','grasp_contacts.py','kinematics.py','reset_reference.py','rl_scene.py','rl_task.py']]+[ROOT/a.pose,ROOT/a.snapshot,Path(__file__)]
hashes={}
for path in paths:
    hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();shutil.copy2(path,source/path.name)
(out/'launch.json').write_text(json.dumps(dict(arguments=vars(a),source_sha256=hashes,algorithm='scripted_mechanical_feasibility_NOT_RL'),indent=2))
app=AppLauncher(a).app
sys.path.insert(0,str(source))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from rl_scene import BatchedTweezersScene
from rl_task import TaskConfig,TweezerTask
torch.set_num_threads(4)
variants=[dict(closure=c,index_extra=ix,middle_PIP=mid) for c,ix,mid in itertools.product([0.,.5,1.],[.025,.035,.045],[-.035,-.02,0.])]
if a.coordinated:
    variants=[dict(closure=[.95,1.,1.05][i%3],index_extra=0.,middle_PIP=[-.005,0.,.005][(i//3)%3]) for i in range(a.copies)]
    if a.repeat_successful:variants=[dict(closure=1.05,index_extra=0.,middle_PIP=-.005) for _ in range(a.copies)]
snapshot=json.loads((source/Path(a.snapshot).name).read_text())
R=Rotation.from_quat(np.roll(snapshot['rigid_body_states_13']['handle'][3:7],-1)).as_matrix()
scene=BatchedTweezersScene(num_envs=len(variants),device=a.device,dt=1/480,pose_json=source/Path(a.pose).name,
    tool_world_R=R,tool_position=(.48,0,.95),spring=5.,spring_damping=.02,solver_iterations=64,
    solver_velocity_iterations=8,contact_offset=.0002,simplify_arm_collisions=False,enable_grasp_support=True)
cfg=TaskConfig(decimation=8,episode_seconds=5.,initial_tip_clearance=.018,reset_xy_range=0,closure_bias=0.,
    closure_scale=2.,closure_min=-.5,closure_max=2.,reset_snapshot=str(source/Path(a.snapshot).name),
    require_web_up=True,max_palm_up_component=.25,require_tripod_support=True,independent_support_actions=True,
    index_support_action_rad=.08,middle_support_action_rad=.12,command_space_arm=True,tool_down_angle_range_deg=(20.,60.))
if a.coordinated:
    cfg.closure_scale=1.2;cfg.closure_min=-.1;cfg.closure_max=1.2
    cfg.index_support_action_rad=.008;cfg.middle_support_action_rad=.025
    cfg.reset_xy_range=.002;cfg.seed=a.reset_seed;cfg.max_wrist_descent=.019;cfg.support_reward_weight=1.5
task=TweezerTask(scene,cfg)
# A common centered reset isolates pinch parameters from initial XY randomization.
if not a.coordinated:
    task.bank_q=task.bank_q[:1];task.bank_q_target=task.bank_q_target[:1];task.bank_delta=task.bank_delta[:1]
    task.bank_object_states={k:v[:1] for k,v in task.bank_object_states.items()}
task.reset()
(out/'task.json').write_text(json.dumps(task.metadata(),indent=2));scene.stage.Export(str(out/'initial_scene.usda'))
dev=scene.device
parameters=torch.tensor([[v['closure']/cfg.closure_scale,v['index_extra']/cfg.index_support_action_rad,v['middle_PIP']/cfg.middle_support_action_rad] for v in variants],device=dev)
rows=[];arrays={k:[] for k in ['q','tool','left','right','block','action','observations','lift','both','support','support_z','strength','web_z','tip','gap']}
start=task.measurements()['tip'].clone();block=scene.get_state()['block_root_state_w'][:,:3].clone()
target=block.clone();target[:,2]+=.0015
print('SCRIPTED_PICK_START',len(variants),flush=True)
for step in range(task.max_steps):
    t=step*task.dt;measurement=task.measurements();goal=start.clone()
    if t>=.5:
        alpha=min(1.,(t-.5)/.75);goal=start+alpha*(target-start)
    if t>=2.:goal[:,2]+=min(.06,(t-2.)*.04)
    act=torch.zeros((len(variants),6),device=dev)
    act[:,:3]=((goal-measurement['tip'])/cfg.max_wrist_step).clamp(-1/3,1/3)
    act[:,3:]=parameters*max(0.,min(1.,(t-1.25)/.65))
    arrays['observations'].append(task.observe(measurement).detach().cpu().numpy().copy())
    _,reward,done,info=task.step(act,auto_reset=False)
    m=info['measurements'];s=m['state'];g=m['grasp_support']
    def add(k,v):arrays[k].append(v.detach().cpu().numpy().copy())
    add('q',s['joint_pos']);add('tool',s['tool_root_state_w']);add('left',s['left_root_state_w']);add('right',s['right_root_state_w']);add('block',s['block_root_state_w']);add('action',act)
    for k,v in [('lift',m['lift']),('both',m['both']),('support',m['grasp_support_ok']),('support_z',g['role_support_z_N']),('strength',g['role_contact_strength_N']),('web_z',g['web_support_z_N']),('tip',m['tip']),('gap',m['gap'])]:add(k,v)
    if step%60==59:print('SCRIPTED_PICK_PROGRESS',t+task.dt,'max_lift',float(m['lift'].max()),'both',int(m['both'].sum()),'support',int(m['grasp_support_ok'].sum()),flush=True)
values={k:np.stack(v) for k,v in arrays.items()};np.savez_compressed(out/'trajectory_arrays.npz',**values)
if a.coordinated:
    np.savez_compressed(out/'demonstrations.npz',observations=values['observations'],actions=values['action'],
        success=task.episode_success.cpu().numpy(),episode_id=np.arange(len(variants)))
for i,v in enumerate(variants):
    final=slice(-60,None)
    rows.append({**v,'environment':i,'max_lift_m':float(values['lift'][:,i].max()),'final_lift_m':float(values['lift'][-1,i]),
        'bilateral_fraction':float(values['both'][:,i].mean()),'last_second_support_fraction':float(values['support'][final,i].mean()),
        'last_second_bilateral_fraction':float(values['both'][final,i].mean()),'task_success':bool(task.episode_success[i]),
        'last_second_mean_middle_up_N':float(values['support_z'][final,i,2].mean()),'last_second_mean_web_up_N':float(values['web_z'][final,i].mean())})
(out/'report.json').write_text(json.dumps(dict(algorithm='scripted_actuation_NOT_RL',object_runtime_writes=False,variants=rows),indent=2))
scene.stage.Export(str(out/'final_scene.usda'));print('SCRIPTED_PICK_DONE',json.dumps(rows),flush=True)
scene.close();app.close()
