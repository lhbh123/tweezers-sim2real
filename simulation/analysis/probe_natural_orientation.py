"""Cold-reset, fixture-free orientation sweep of the observed natural grasp.

Only robot joint targets are written after the single reset. Candidates rotate
the wrist and all three observed tool bodies together, then face real gravity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--snapshot',default='runs/natural_hold_distal_open060/stable_snapshot.json')
parser.add_argument('--pose',default='configs/human_grasp_distal_suspend30.json')
parser.add_argument('--scene-source',default='runs/rl_distal_open250_render/rl_scene.py')
parser.add_argument('--out',default='runs/natural_orientation_sweep01')
parser.add_argument('--seconds',type=float,default=10.)
parser.add_argument('--angles',type=float,nargs='+',default=[30.,40.,50.,60.])
parser.add_argument('--closures',type=float,nargs='+',default=[0.,.2,.4])
parser.add_argument('--signs',type=float,nargs='+',default=[-1.,1.])
parser.add_argument('--source-only',action='store_true')
parser.add_argument('--copies',type=int,default=8)
parser.add_argument('--all-robot-contacts',action='store_true')
parser.add_argument('--spring',type=float,default=1.5)
parser.add_argument('--spring-damping',type=float,default=.02)
parser.add_argument('--rest-open-angle',type=float,default=None)
parser.add_argument('--tip-position',type=float,nargs=3,default=[.48,0.,.88])
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.headless=True
out=Path(args.out)
out.mkdir(parents=True,exist_ok=False)
app=AppLauncher(args).app

import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
source_path=Path(args.scene_source).resolve()
scene_code=source_path.read_bytes()
(out/'rl_scene_source.py').write_bytes(scene_code)
spec=importlib.util.spec_from_file_location('orientation_scene',source_path)
scene_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(scene_module)
BatchedTweezersScene=scene_module.BatchedTweezersScene
from reset_reference import StableResetReference
from isaaclab.utils.math import quat_apply,quat_mul,quat_inv
from isaacsim.core.prims import RigidPrim
from kinematics import ASSET,stl_triangles

raw=json.loads(Path(args.snapshot).read_text())
pose=json.loads(Path(args.pose).read_text())
if args.rest_open_angle is not None:pose['tool_geometry']['open_angle_rad']=args.rest_open_angle
runtime_pose=out/'runtime_pose.json';runtime_pose.write_text(json.dumps(pose,indent=2))
source_R=Rotation.from_quat(np.array(raw['rigid_body_states_13']['handle'])[[4,5,6,3]]).as_matrix()
source_hand_R=Rotation.from_quat(np.array(raw['world_hand_pose_7'])[[4,5,6,3]]).as_matrix()
candidates=[{'name':'source_orientation'+(str(i) if args.source_only else ''),
             'extra_closure':0.,'goal_R':source_R} for i in range(args.copies if args.source_only else 1)]
for angle in ([] if args.source_only else args.angles):
    z=np.array([np.cos(np.deg2rad(angle)),0.,-np.sin(np.deg2rad(angle))])
    for sign in args.signs:
        x=np.array([0.,sign,0.]);R=np.column_stack([x,np.cross(z,x),z])
        for closure in args.closures:
            candidates.append({'name':f'table{angle:g}_x{sign:+g}_close{closure:g}',
                               'extra_closure':closure,'goal_R':R})
n=len(candidates)
scene=BatchedTweezersScene(num_envs=n,device=args.device,dt=1/480,pose_json=runtime_pose,
    tool_world_R=source_R,spring=args.spring,spring_damping=args.spring_damping,hand_stiffness=10.,hand_damping=.3,
    solver_iterations=64,solver_velocity_iterations=8,tool_width=.009,bridge_width=.012,
    contact_offset=.0002,simplify_arm_collisions=False,self_collisions=True,
    filter_mechanical_self_pairs=True)
dev=scene.device
reference=StableResetReference(args.snapshot,scene)
q=scene.default_joint_pos.clone();targets=q.clone()
objects={name:state.clone() for name,state in scene.default_object_states.items()}

# Visual hand mesh geometry only supplies a conservative table reach diagnostic.
fk=scene.kin.fk(dict(zip(scene.joint_names,reference.q_actual)),root='right_hand_link')
hand_points=[]
for link in scene.kin.root.findall('link'):
    name=link.get('name');mesh=link.find('visual/geometry/mesh')
    if name not in fk or mesh is None: continue
    vertices=stl_triangles(ASSET/mesh.get('filename')).reshape(-1,3)
    hand_points.append(vertices@fk[name][:3,:3].T+fk[name][:3,3])
hand_points=np.concatenate(hand_points)
hand_source=np.array(raw['world_hand_pose_7'][:3])
source_world_hand_vertices=hand_points@source_hand_R.T+hand_source
for i,c in enumerate(candidates):
    try:
        batch=reference.build(args.tip_position,env_origins=scene.env_origins[i:i+1],
            position_frame='local',desired_tip_orientation=c['goal_R'])
        q[i]=batch['q_actual'][0];targets[i]=batch['q_target'][0]
        for name,delta in pose['closure_delta'].items():
            targets[i,scene.joint_names.index(name)]+=c['extra_closure']*delta
        for name,state in batch['object_states'].items():objects[name][i]=state[0]
        c['ik_error']=float(batch['ik_error'][0]);c['reachable']=True
    except ValueError as error:
        c['reachable']=False;c['ik_error_message']=str(error)
    dR=c['goal_R']@source_R.T
    c['goal_palm_normal_w']=(dR@source_hand_R[:,1]).tolist()
    c['goal_tool_axis_w']=c['goal_R'][:,2].tolist()
    c['goal_tool_angle_down_from_table_deg']=float(np.rad2deg(np.arcsin(-c['goal_R'][2,2])))
    c['hand_visual_min_z_relative_to_tip_m']=float(((source_world_hand_vertices-reference.tip_position)@dR.T)[:,2].min())
    c['goal_handle_quaternion_wxyz']=np.roll(Rotation.from_matrix(c['goal_R']).as_quat(),1).tolist()
    c['goal_R']=c['goal_R'].tolist()
objects['block'][:,0]=scene.env_origins[:,0]-.1
objects['block'][:,1]=scene.env_origins[:,1]+.3

# Tools already carry PhysxContactReportAPI. Source paths are explicitly mapped
# back to env/side; wildcard filters remain separated by PhysX environment IDs.
grips=[]
for side in ['Left','Right']:
    view=RigidPrim('/World/envs/env_.*/Tweezers/'+side,name='orientation_grip_'+side,
        track_contact_forces=True,reset_xform_properties=False,
        contact_filter_prim_paths_expr=[
            '/World/envs/env_.*/Robot/root_joint/right_hand_thumb_rota_link2',
            '/World/envs/env_.*/Robot/root_joint/right_hand_index_rota_link2'],max_contact_count=4096)
    view.initialize()
    assert len(view.prim_paths)==n,view.prim_paths
    grips.append(view)
all_views=[];all_robot_paths=[];all_forces=[]
if args.all_robot_contacts:
    from pxr import Usd,UsdPhysics
    robot_path='/World/envs/env_0/Robot'
    all_robot_paths=[str(p.GetPath()) for p in Usd.PrimRange(scene.stage.GetPrimAtPath(robot_path))
                     if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for name in ['Handle','Left','Right']:
        view=RigidPrim('/World/envs/env_.*/Tweezers/'+name,name='all_contacts_'+name,
            track_contact_forces=True,reset_xform_properties=False,
            contact_filter_prim_paths_expr=[p.replace('/env_0/','/env_.*/') for p in all_robot_paths],
            max_contact_count=8192)
        view.initialize();all_views.append(view)
scene.reset(q=q,object_states=objects)
s=scene.get_state()
initial_tool=s['tool_root_state_w'].clone()
relative_pos=quat_apply(quat_inv(s['wrist_quat_w']),initial_tool[:,:3]-s['wrist_pos_w'])
relative_quat=quat_mul(quat_inv(s['wrist_quat_w']),initial_tool[:,3:7])
f0=scene.kin.fk(dict(zip(scene.joint_names,reference.q_actual)))
wrist_to_hand_R=f0['wrist_3_link'][:3,:3].T@f0['right_hand_link'][:3,:3]
hand_normal_in_wrist=torch.tensor(wrist_to_hand_R[:,1],device=dev,dtype=torch.float32).expand(n,3)

def corners(size,centre):
    return torch.tensor(np.array([np.array(centre)+np.array(size)*[x,y,z]
        for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]]),device=dev,dtype=torch.float32)
points={'handle':corners((.012,.009,.006),(0,0,-.003))}
for name in ['left','right']:
    points[name]=torch.cat([corners((.001,.009,.125),(0,0,.0625)),corners((.001,.005,.015),(0,0,.1325))])
minimum_clearance=torch.full((n,),float('inf'),device=dev)
max_cube_force=torch.zeros(n,device=dev)
max_error=torch.zeros(n,device=dev)
first_loss=torch.full((n,),-1.,device=dev)
trace=[];forces=[];early_states=[];start=time.perf_counter();steps=round(args.seconds/scene.dt)
print('ORIENTATION_START',json.dumps({'num_envs':n,'candidates':candidates}),flush=True)
for step in range(steps):
    s=scene.step(targets)
    tool=s['tool_root_state_w']
    expected=s['wrist_pos_w']+quat_apply(s['wrist_quat_w'],relative_pos)
    expected_q=quat_mul(s['wrist_quat_w'],relative_quat)
    error=(tool[:,:3]-expected).norm(dim=-1)
    orient=1-(tool[:,3:7]*expected_q).sum(-1).square().clamp(0,1)
    clear=torch.full((n,),float('inf'),device=dev)
    for name,p in points.items():
        state=s[('tool' if name=='handle' else name)+'_root_state_w']
        pp=p[None].expand(n,-1,-1);qq=state[:,None,3:7].expand(-1,len(p),-1)
        world=state[:,None,:3]+quat_apply(qq.reshape(-1,4),pp.reshape(-1,3)).reshape(n,-1,3)
        clear=torch.minimum(clear,(world[:,:,2]-scene.env_origins[:,None,2]-scene.table_height).amin(-1))
    minimum_clearance=torch.minimum(minimum_clearance,clear)
    max_error=torch.maximum(max_error,error)
    max_cube_force=torch.maximum(max_cube_force,s['block_jaw_contact_forces_w'].norm(dim=-1).amax(-1))
    lost=(error>.035)|(orient>.3)|(clear<.001)
    first_loss=torch.where((first_loss<0)&lost,torch.full_like(first_loss,(step+1)*scene.dt),first_loss)
    if step+1 in [48,96,240]:
        early_states.append({'time_s':(step+1)*scene.dt,'joint_pos':s['joint_pos'].cpu().tolist(),
            'joint_targets':targets.cpu().tolist(),'body_names':scene.robot.body_names,
            'body_pos_local':(s['body_pos_w']-scene.env_origins[:,None,:]).cpu().tolist(),
            'body_quat_w':s['body_quat_w'].cpu().tolist(),
            'tool_body_states':{name:s[('tool' if name=='handle' else name)+'_root_state_w'].cpu().tolist()
                                for name in ['handle','left','right']}})
    if step%8==7:
        mapped=np.zeros((n,2,2,3))
        for side,view in enumerate(grips):
            contact=view.get_contact_force_matrix(dt=scene.dt)
            if torch.is_tensor(contact):contact=contact.cpu().numpy()
            for j,path in enumerate(view.prim_paths):
                env=int(path.split('/env_')[1].split('/')[0]);mapped[env,side]=contact[j]
        forces.append(mapped)
        if all_views:
            all_mapped=np.zeros((n,3,len(all_robot_paths),3))
            for side,view in enumerate(all_views):
                contact=view.get_contact_force_matrix(dt=scene.dt)
                if torch.is_tensor(contact):contact=contact.cpu().numpy()
                for j,path in enumerate(view.prim_paths):
                    env=int(path.split('/env_')[1].split('/')[0]);all_mapped[env,side]=contact[j]
            all_forces.append(all_mapped)
        trace.append(np.concatenate([np.full((n,1),(step+1)*scene.dt),tool.cpu().numpy(),
            error[:,None].cpu().numpy(),orient[:,None].cpu().numpy(),clear[:,None].cpu().numpy()],axis=1))
    if (step+1)%960==0:
        print('ORIENTATION_PROGRESS',json.dumps({'time_s':(step+1)*scene.dt,'retained':int((first_loss<0).sum()),
            'first_loss_s':first_loss.tolist()}),flush=True)
trace=np.stack(trace);forces=np.stack(forces)
if all_forces:all_forces=np.stack(all_forces)
tail=trace[:,0,0]>=args.seconds-1
tail_xyz_range=np.ptp(trace[tail,:,1:4],axis=0)
tail_contact=np.linalg.norm(forces[tail].sum(axis=2),axis=-1)
contact_fraction=(tail_contact>.003).all(-1).mean(0)
palm=quat_apply(s['wrist_quat_w'],hand_normal_in_wrist).cpu().numpy()
axis=quat_apply(s['tool_root_state_w'][:,3:7],torch.tensor([0.,0.,1.],device=dev).expand(n,3)).cpu().numpy()
tip_offset=torch.tensor([0.,0.,.1325],device=dev).expand(n,3)
tip_points={name:s[name+'_root_state_w'][:,:3]+quat_apply(s[name+'_root_state_w'][:,3:7],tip_offset)
            for name in ['left','right']}
tip_separation=tip_points['right']-tip_points['left']
tip_gap=tip_separation.norm(dim=-1);spread_axis=tip_separation/tip_gap[:,None].clamp_min(1e-9)
tip_half=torch.tensor([.0005,.0025,.0075],device=dev)
projected_half=torch.zeros(n,device=dev)
for name in ['left','right']:
    local_axis=quat_apply(quat_inv(s[name+'_root_state_w'][:,3:7]),spread_axis)
    projected_half+=(local_axis.abs()*tip_half).sum(-1)
stable_count=0
for i,c in enumerate(candidates):
    range_max=float(tail_xyz_range[i].max())
    final_q=trace[-1,i,4:8];dq=np.abs(trace[tail,i,4:8]@final_q)
    angular_range=float(np.rad2deg(2*np.arccos(np.clip(dq,0,1))).max())
    retained=bool(c['reachable'] and first_loss[i]<0 and minimum_clearance[i]>.005 and max_cube_force[i]<1e-6)
    stable=bool(retained and range_max<.001 and angular_range<1. and contact_fraction[i]>=.95)
    c.update({'retained':retained,'settled_stable':stable,'first_loss_s':float(first_loss[i]),
        'max_grip_translation_m':float(max_error[i]),'min_table_clearance_m':float(minimum_clearance[i]),
        'max_block_contact_N':float(max_cube_force[i]),'last_1s_xyz_range_m':tail_xyz_range[i].tolist(),
        'last_1s_orientation_range_deg':angular_range,'last_1s_bilateral_fraction_gt3mN':float(contact_fraction[i]),
        'last_1s_jaw_contact_mean_N':tail_contact[:,i].mean(0).tolist(),
        'final_palm_normal_w':palm[i].tolist(),'final_tool_axis_w':axis[i].tolist(),
        'final_tip_center_gap_m':float(tip_gap[i]),
        'final_tip_projected_inner_gap_m':float(tip_gap[i]-projected_half[i]),
        'final_tip_spread_axis_w':spread_axis[i].cpu().tolist(),
        'axis_aligned_12mm_cube_projected_width_m':float(.012*spread_axis[i].abs().sum()),
        'final_tool_angle_down_from_table_deg':float(np.rad2deg(np.arcsin(np.clip(-axis[i,2],-1,1)))),
        'last_1s_reported_speed_mean_mps':float(np.linalg.norm(trace[tail,i,8:11],axis=-1).mean()),
        'last_1s_position_difference_speed_mean_mps':float(np.linalg.norm(np.diff(trace[tail,i,1:4],axis=0)/(8*scene.dt),axis=-1).mean())})
    if all_views:
        average=np.linalg.norm(all_forces[tail,i],axis=-1).mean(0)
        peak=np.linalg.norm(all_forces[:,i],axis=-1).max(0)
        c['all_robot_contacts_mean_gt1mN']=[{'tool':['handle','left','right'][a],
            'robot':all_robot_paths[b].rsplit('/',1)[-1],'mean_force_N':float(average[a,b]),
            'peak_force_N':float(peak[a,b])} for a,b in zip(*np.nonzero(average>.001))]
        allowed=[p.rsplit('/',1)[-1] in ['right_hand_thumb_rota_link2','right_hand_index_rota_link2'] for p in all_robot_paths]
        c['max_non_distal_robot_contact_N']=float(peak[:,~np.array(allowed)].max())
    if stable:
        stable_count+=1
        current=scene.kin.fk(dict(zip(scene.joint_names,s['joint_pos'][i].cpu().tolist())))
        worldposes={}
        for link in ['wrist_3_link','right_hand_link']:
            T=current[link].copy();T[:3,3]+=[0.,0.,scene.table_height]
            worldposes[link]=np.r_[T[:3,3],np.roll(Rotation.from_matrix(T[:3,:3]).as_quat(),1)].tolist()
        bodies={}
        for name in ['handle','left','right']:
            state=s[('tool' if name=='handle' else name)+'_root_state_w'][i].cpu().numpy().copy()
            state[:3]-=scene.env_origins[i].cpu().numpy();bodies[name]=state.tolist()
        snap={'schema_version':1,'source_run':str(out.resolve()),'source_candidate':c['name'],
            'physics_time_s':args.seconds,'joint_names':scene.joint_names,
            'q_actual':s['joint_pos'][i].cpu().tolist(),'q_target':targets[i].cpu().tolist(),
            'q_velocity':s['joint_vel'][i].cpu().tolist(),'rigid_body_states_13':bodies,
            'world_wrist_pose_7':worldposes['wrist_3_link'],'world_hand_pose_7':worldposes['right_hand_link'],
            'arm_base_world_xyz':[0.,0.,scene.table_height],'fixture_active':False,
            'args':vars(args),'natural_hold':c}
        snap['zero_velocity_variant']={k:snap[k] for k in ['q_actual','q_target']}
        snap['zero_velocity_variant']['q_velocity']=[0.]*18
        snap['zero_velocity_variant']['rigid_body_states_13']={k:v[:7]+[0.]*6 for k,v in bodies.items()}
        (out/(c['name']+'_snapshot.json')).write_text(json.dumps(snap,indent=2))
report={'source_snapshot':args.snapshot,'source_snapshot_sha256':hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest(),
    'source_pose':args.pose,'source_scene_sha256':hashlib.sha256(scene_code).hexdigest(),
    'palm_normal_definition':'right_hand_link local +Y, flexor side verified with MCP flexion and STL surface normal',
    'seconds':args.seconds,'num_envs':n,'stable_count':stable_count,'candidates':candidates,
    'device':dev,'physics_dt':scene.dt,'tool_dynamic_gravity_enabled':True,'fixture_present':False,
    'spring_stiffness_Nm_per_rad':args.spring,'spring_damping_Nms_per_rad':args.spring_damping,
    'spring_rest_open_angle_rad':pose['tool_geometry']['open_angle_rad'],
    'task_time_state_writes':False,'constant_joint_targets':True,
    'self_collisions':True,'mechanical_filter_pairs':scene.filtered_mechanical_pairs,
    'contact_source_rows':[v.prim_paths for v in grips],'wall_s':time.perf_counter()-start,
    'criteria':'No loss (35 mm / quaternion metric .3), >5 mm table clearance, no block contact, tail1s position range<1mm, orientation range<1deg, bilateral >3mN in95% samples.',
    'reach_note':'hand_visual_min_z_relative_to_tip is source mesh geometry at reset, not a collision-free arm trajectory guarantee.'}
(out/'report.json').write_text(json.dumps(report,indent=2))
(out/'early_states.json').write_text(json.dumps({'env_origins':scene.env_origins.cpu().tolist(),
    'joint_names':scene.joint_names,'samples':early_states}))
np.savez_compressed(out/'trace.npz',trace=trace,grip_forces=forces,all_robot_contact_forces=all_forces)
print('ORIENTATION_RESULT',json.dumps({'out':str(out),'stable_count':stable_count,'candidates':candidates}),flush=True)
scene.close();app.close()
