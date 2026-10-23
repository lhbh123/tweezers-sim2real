"""Physical actuation feasibility only: hold, 50mm lift, and ±10mm X excursion.

This scripted diagnostic is not a policy, PPO training schedule, or demonstration
of learned task success. Shared source is frozen, never edited. Only robot joint
commands are written after reset; tools remain dynamic and unsupported externally.
"""
import argparse,copy,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--copies',type=int,default=4);p.add_argument('--seconds',type=float,default=12.)
p.add_argument('--pose',default='configs/human_grasp_tripod_horizontal_settled_v1.json');p.add_argument('--low-stow-hold',action='store_true');p.add_argument('--snapshot',default='runs/tripod_horizontal_hold12/closure0_copy0_snapshot.json')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True
out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
for n in ['rl_scene.py','rl_task.py','arm_command.py','kinematics.py','arm_clearance.py','reset_reference.py','grasp_contacts.py']:shutil.copy2(ROOT/'src'/n,source/n)
shutil.copy2(__file__,source/Path(__file__).name);shutil.copy2(ROOT/a.pose,source/'pose.json')
(out/'launch.json').write_text(json.dumps({'arguments':vars(a),'source_sha256':{str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()},'diagnostic_schedule_not_used_by_PPO':True},indent=2))
app=AppLauncher(a).app
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaacsim.core.prims import RigidPrim
sys.path.insert(0,str(source));from rl_scene import BatchedTweezersScene
from rl_task import TweezerTask,TaskConfig
speeds=([0.,0.] if a.low_stow_hold else [0.,.01,.02,.04]);mode_names=(['tip_z_0.7458','tip_z_0.7276'] if a.low_stow_hold else list(map(str,speeds)));n=len(speeds)*a.copies;torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
snapshot=json.loads((ROOT/a.snapshot).read_text());qh=snapshot['zero_velocity_variant']['rigid_body_states_13']['handle'][3:7];R=Rotation.from_quat(np.roll(qh,-1)).as_matrix()
scene=BatchedTweezersScene(num_envs=n,device=a.device,dt=1/480,pose_json=source/'pose.json',tool_world_R=R,
 spring=5.,spring_damping=.02,hand_stiffness=10.,hand_damping=.3,finger_friction=1.4,
 solver_iterations=64,solver_velocity_iterations=8,contact_offset=.0002,simplify_arm_collisions=False,
 self_collisions=True,filter_mechanical_self_pairs=True,enable_grasp_support=True)
cfg=TaskConfig(decimation=8,episode_seconds=a.seconds+1,initial_tip_clearance=.018,reset_xy_range=.002,
 closure_bias=0.,closure_scale=.45,closure_min=-.45,closure_max=.7,seed=0,reset_snapshot=str(ROOT/a.snapshot),
 require_web_up=True,min_web_up_component=.35,max_palm_up_component=.25,require_tripod_support=True,
 support_vertical_threshold_N=.003,independent_support_actions=True,command_space_arm=True,tool_down_angle_range_deg=(20.,60.))
env=TweezerTask(scene,cfg);q0=env.bank_q[0].repeat(n,1).clone();target=env.bank_q_target[0].repeat(n,1).clone();objects={k:v.clone() for k,v in scene.default_object_states.items()}
for k in ['handle','left','right']:
 objects[k]=env.bank_object_states[k][0].repeat(n,1).clone();objects[k][:,:3]+=scene.env_origins
if a.low_stow_hold:
 points=np.array([[.48,0.,z] for z in [.7458,.7276] for _ in range(a.copies)])
 built=env.reset_reference.build(points,env_origins=scene.env_origins,position_frame='local')
 q0=built['q_actual'];target=built['q_target']
 for k in ['handle','left','right']:objects[k]=built['object_states'][k]
 objects['block'][:,:3]=scene.env_origins+torch.tensor([-.1,.3,.7261],device=scene.device)
scene.reset(q=q0,object_states=objects);env.q_target[:]=target;env.hand_target[:]=target[:,env.hand]
env.arm_command_reference[:]=env.arm_command.fk_jacobian(target[:,env.arm])[0]
scene.robot.set_joint_position_target(target);scene.robot.write_data_to_sim();s0=scene.get_state()
def cpu(v):return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
initial_wrist=cpu(s0['wrist_pos_w']).copy();initial_hand_q=cpu(s0['joint_pos'][:,env.hand]).copy();origins=cpu(scene.env_origins)
# Read-only table contacts for every robot and tool rigid body at every physics step.
paths=[x.replace('/env_0/','/env_.*/') for x in scene.grasp_contact_reader.paths]
paths += ['/World/envs/env_.*/Tweezers/'+x for x in ['Handle','Left','Right']]
table_view=RigidPrim(paths,name='motion_allbody_table_contacts',track_contact_forces=True,reset_xform_properties=False,
 contact_filter_prim_paths_expr=[['/World/envs/env_.*/Table'] for _ in paths],max_contact_count=max(32768,n*256));table_view.initialize()
row_env=torch.tensor([int(x.split('/env_')[1].split('/')[0]) for x in table_view.prim_paths],device=scene.device)
max_table=torch.zeros(n,device=scene.device);table_pair_count=0;original_update=scene.update

def observed_update(dt=None):
 global max_table,table_pair_count
 original_update(dt)
 if dt == 0:return
 matrix=table_view.get_contact_force_matrix(dt=scene.dt);force=matrix.norm(dim=-1).amax(-1)
 maximum=torch.zeros(n,device=scene.device);maximum.scatter_reduce_(0,row_env,force,reduce='amax',include_self=True)
 max_table=torch.maximum(max_table,maximum);table_pair_count+=1
scene.update=observed_update
(out/'task.json').write_text(json.dumps(env.metadata(),indent=2));scene.stage.GetRootLayer().Export(str(out/'initial_scene.usda'))
(out/'initial_state.json').write_text(json.dumps({'joint_names':scene.joint_names,'q_actual':cpu(q0).tolist(),'q_target':cpu(target).tolist(),'env_origins':origins.tolist(),'wrist_pos_w':initial_wrist.tolist(),'source_snapshot':a.snapshot},indent=2))

def offset(t,speed):
 if speed==0:return np.zeros(3)
 t=max(0.,t-1.);distance=speed*t
 z=min(distance,.05);x=0.
 if distance>.05:
  d=distance-.05
  if d<=.01:x=d
  elif d<=.03:x=.02-d
  elif d<=.04:x=d-.04
 return np.array([x,0.,z])

records=[];action=torch.zeros((n,env.action_size),device=scene.device)
print('MOTION_READY',json.dumps({'speeds_mps':speeds,'copies':a.copies,'table_contact_sources':len(table_view.prim_paths)}),flush=True)
for step in range(round(a.seconds/env.dt)):
 t=step*env.dt
 for i,speed in enumerate(speeds):
  d=(offset(t+env.dt,speed)-offset(t,speed))/cfg.max_wrist_step
  action[i*a.copies:(i+1)*a.copies,:3]=torch.tensor(d,device=scene.device,dtype=torch.float32)
 env.step(action,auto_reset=False);m=env.measurements();s=m['state'];support=m['grasp_support']
 records.append({'time':t+env.dt,'q_actual':cpu(s['joint_pos']).copy(),'q_target':cpu(env.q_target).copy(),
  'wrist_pos_w':cpu(s['wrist_pos_w']).copy(),'wrist_quat_w':cpu(s['wrist_quat_w']).copy(),
  'palm_pos_w':cpu(s['palm_pos_w']).copy(),'palm_quat_w':cpu(s['palm_quat_w']).copy(),
  'objects':np.stack([cpu(s[k+'_root_state_w']).copy() for k in ['handle','left','right']],axis=1),
  'support_gate':cpu(m['grasp_support_ok']).copy(),'orientation_gate':cpu(m['orientation_ok']).copy(),
  'grip_error_m':cpu(m['grip_error']).copy(),'tool_angle_table_deg':cpu(m['tool_down_angle_deg']).copy(),
  'tool_to_tip_angle_table_deg':cpu(m['root_to_tip_down_angle_deg']).copy(),
  'middle_Fz_N':cpu(support['role_support_z_N'][:,2]).copy(),'web_Fz_N':cpu((support['web_normal_force_w']+support['web_friction_force_w'])[:,2]).copy(),
  'strength_N':cpu(support['role_contact_strength_N']).copy(),'middle_under_count':cpu(support['middle_under_contact_count']).copy(),
  'web_count':cpu(support['web_contact_count']).copy(),'block_contact_N':cpu(m['forces']).copy()})
 if (step+1)%60==0:
  print('MOTION_PROGRESS',json.dumps({'time':t+env.dt,'recent_support_fraction':{mode_names[i]:float(np.array([r['support_gate'] for r in records[-60:]])[:,i*a.copies:(i+1)*a.copies].mean()) for i,v in enumerate(speeds)},'max_table_force_N':float(max_table.max())}),flush=True)
arrays={k:np.stack([r[k] for r in records]) for k in records[0]};np.savez_compressed(out/'records.npz',**arrays)
active=arrays['time']>=1.;report={'diagnostic_schedule_not_used_by_PPO':True,'support_threshold_N':.003,'table_contact_all_physics_steps_read':table_pair_count,'modes':{}}
for i,speed in enumerate(speeds):
 sl=slice(i*a.copies,(i+1)*a.copies);w=arrays['wrist_pos_w'][:,sl]-initial_wrist[sl]
 report['modes'][mode_names[i]]={'support_fraction_per_env_after1s':arrays['support_gate'][active,sl].mean(0).tolist(),
 'orientation_fraction_per_env_after1s':arrays['orientation_gate'][active,sl].mean(0).tolist(),
 'middle_Fz_N_min_mean_max':[float(arrays['middle_Fz_N'][active,sl].min()),float(arrays['middle_Fz_N'][active,sl].mean()),float(arrays['middle_Fz_N'][active,sl].max())],
 'web_Fz_N_min_mean_max':[float(arrays['web_Fz_N'][active,sl].min()),float(arrays['web_Fz_N'][active,sl].mean()),float(arrays['web_Fz_N'][active,sl].max())],
 'actual_wrist_delta_final_m':w[-1].tolist(),'actual_wrist_x_min_max_m':[float(w[:,:,0].min()),float(w[:,:,0].max())],
 'max_grip_error_m':float(arrays['grip_error_m'][:,sl].max()),'max_block_contact_N':float(arrays['block_contact_N'][:,sl].max()),'max_robot_tool_table_contact_N':cpu(max_table)[sl].tolist()}
(out/'report.json').write_text(json.dumps(report,indent=2));print('MOTION_RESULT',json.dumps(report),flush=True);app.close()
