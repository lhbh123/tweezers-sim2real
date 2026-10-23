"""Independent reconstruction of placement from saved simulator states."""
import argparse,json,ast,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--geometry',action='store_true');a=p.parse_args();run=Path(a.run)
archive=np.load(run/'trajectory.npz');d={k:archive[k] for k in archive.files};archive.close();r=json.loads((run/'evaluation.json').read_text());cfg=r['task']['place_config'];dt=float(d['dt']);orig=d['origins'];b=d['block'];tip=d['tip'];goal=d['goal']+orig
xy=np.linalg.norm(b[:,:,:2]-goal[None,:,:2],axis=-1)
lift=b[:,:,2]-orig[None,:,2]-(.72+.0061)
valid_lift=(lift>.045)&d['both']&(d['retained']<.035)&d['face']
physical_place=(xy<cfg['goal_tolerance_m'])&(abs(lift)<.0015)&(np.linalg.norm(b[:,:,7:10],axis=-1)<.01)&(np.linalg.norm(b[:,:,10:13],axis=-1)<.5)&(d['forces'].max(-1)<.003)&(tip[:,:,2]-b[:,:,2]>.035)&(d['retained']<.035)&d['face']
lo,hi=r['task']['tool_down_angle_range_deg']
root_to_tip=d['tip']-d['tool'][:,:,:3]
root_angle=np.rad2deg(np.arctan2(-root_to_tip[:,:,2],np.linalg.norm(root_to_tip[:,:,:2],axis=-1)))
angle_ok=(d['normal_angle']>=lo)&(d['normal_angle']<=hi)&(root_angle>=lo)&(root_angle<=hi)
valid_lift&=angle_ok
physical_place&=angle_ok
if 'orientation_ok' in d:
 valid_lift&=d['orientation_ok']&(d['orientation_error']<.3)
 physical_place&=d['orientation_ok']&(d['orientation_error']<.3)
rows=[]
for i,claimed in enumerate(r['success_per_env']):
 count=0;longest=0;placed=False;ever_lift=False;lift_count=0;longest_lift=0
 for k in range(len(b)):
  if not d['active'][k,i]:continue
  lift_count=lift_count+1 if valid_lift[k,i] else 0;longest_lift=max(longest_lift,lift_count);ever_lift|=lift_count>=max(1,round(r['task']['success_definition'].get('continuous_bilateral_lift_s',0)/dt))
  count=count+1 if physical_place[k,i] and ever_lift else 0;longest=max(longest,count)
  placed|=count>=round(cfg['stable_seconds']/dt)
 rows.append(dict(environment=i,claimed=claimed,reconstructed=placed,consistent=claimed==placed,
  longest_stable_place_s=longest*dt,longest_bilateral_lift_s=longest_lift*dt,
  start_xy=d['start'][i,:2].tolist(),goal_xy=d['goal'][i,:2].tolist(),transfer_distance_m=float(np.linalg.norm(d['start'][i,:2]-d['goal'][i,:2]))))
module=ast.parse((run/'source/pick_place_task.py').read_text());bad=[]
for node in ast.walk(module):
 if isinstance(node,ast.FunctionDef) and node.name not in ['reset','__init__']:
  for c in ast.walk(node):
   if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and ('write_root' in c.func.attr or 'set_external' in c.func.attr or c.func.attr=='reset'):bad.append((node.name,c.func.attr))
# step may call reset only at terminal episode boundaries; inspect separately, not treat as runtime grab.
bad=[x for x in bad if x!=('step','reset')]
report=dict(run=str(run),episodes=len(rows),reconstructed_successes=sum(x['reconstructed'] for x in rows),all_claims_consistent=all(x['consistent'] for x in rows),no_runtime_object_control_calls=not bad,invalid_calls=bad,
 start_xy_min=d['start'][:,:2].min(0).tolist(),start_xy_max=d['start'][:,:2].max(0).tolist(),goal_xy_min=d['goal'][:,:2].min(0).tolist(),goal_xy_max=d['goal'][:,:2].max(0).tolist(),
 minimum_transfer_distance_m=min(x['transfer_distance_m'] for x in rows),rows=rows)
if a.geometry:
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
 from kinematics import Kinematics
 from arm_clearance import ArmTableClearance
 robot_meta=r['task'];kin=Kinematics(Path(robot_meta['robot_asset_dir'])/'robot.urdf',palm_link=robot_meta.get('palm_link','right_hand_link'),hand_root_link=robot_meta.get('hand_root_link'),arm_joint_names=robot_meta.get('arm_joint_names')) if robot_meta.get('robot_asset_dir') else Kinematics();clear=ArmTableClearance(kin);names=r['task']['joint_names'];worst=1.;violations=0;limit_bad=0
 for k in range(len(b)):
  for i in range(len(rows)):
   if not d['active'][k,i]:continue
   q=dict(zip(names,d['q'][k,i]));fk=kin.fk(q)
   for name,verts in {**clear.vertices,**clear.hand_vertices}.items():
    T=fk[name];z=(verts@T[2,:3]+T[2,3]).min()
    worst=min(worst,float(z));violations+=int(z<-.0001)
   limit_bad+=sum(q[n]<kin.by_name[n]['lower']-1e-4 or q[n]>kin.by_name[n]['upper']+1e-4 for n in names)
 report['source_mesh_table_audit']=dict(samples=int(d['active'].sum()),min_moving_robot_table_clearance_m=worst,below_table_link_samples=violations,joint_limit_violations=int(limit_bad),scope='every active CONTROL sample, arm+hand source vertices versus infinite tabletop plane; not an all-substep self-collision audit')
from scipy.spatial.transform import Rotation
geo=r['task']['tool_geometry'];tool_min=float('inf');tool_bad=0
for name in ['tool','left','right']:
 state=d[name];live_state=state[d['active']];environment_z=np.broadcast_to(orig[None,:,2],state.shape[:2])[d['active']]
 if name=='tool':
  dims=[(np.array([2*geo['hinge_offset_m']+geo['strip_thickness_m'],geo['handle_width_m'],geo['handle_length_m']]),np.array([0,0,-geo['handle_length_m']/2]))]
 else:
  length=geo['length_m'];tip_length=geo.get('tip_length_m',.015)
  dims=[(np.array([geo['strip_thickness_m'],geo['strip_width_m'],length-tip_length]),np.array([0,0,(length-tip_length)/2])),(np.array([geo['strip_thickness_m'],min(.005,geo['strip_width_m']),tip_length]),np.array([0,0,length-tip_length/2]))]
 rot=Rotation.from_quat(np.roll(live_state[:,3:7],-1,axis=1)).as_matrix()
 import itertools
 for size,center in dims:
  vertices=np.array(list(itertools.product([-1,1],repeat=3)))*size/2+center
  z=np.einsum('ni,vi->nv',rot[:,2,:],vertices)+live_state[:,2,None]-environment_z[:,None]-.72
  tool_min=min(tool_min,float(z.min()));tool_bad+=int((z.min(-1)<-.0001).sum())
report['tool_table_source_box_audit']=dict(min_clearance_m=tool_min,below_table_body_samples=tool_bad,scope='all active control samples; exact authored strip, tip and straight handle boxes')
try:
 from pxr import Usd,UsdPhysics
 stage=Usd.Stage.Open(str(run/'final_scene.usda'));coupled=[];goals=[]
 for prim in stage.Traverse():
  if prim.GetName().startswith('Goal_'):
   goals.append(not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI))
  if prim.IsA(UsdPhysics.Joint):
   joint=UsdPhysics.Joint(prim);targets=[str(t) for t in list(joint.GetBody0Rel().GetTargets())+list(joint.GetBody1Rel().GetTargets())]
   if any('/Block' in t for t in targets) or (any('/Tweezers/' in t for t in targets) and any('/Robot/' in t for t in targets)):coupled.append(str(prim.GetPath()))
 report['USD_audit']=dict(forbidden_tool_robot_or_block_joints=coupled,goal_visual_count=len(goals),goal_markers_have_no_collision_or_rigid_body=all(goals))
 assert not coupled and all(goals)
except ImportError:
 import importlib.util
 spec=importlib.util.spec_from_file_location('existing_usd_helper',Path(__file__).resolve().parents[1]/'scripts/validate_rl_run.py')
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 helper.USD_AUDIT=r'''
import json,sys
sys.path.insert(0,sys.argv[2])
from pxr import Usd,UsdPhysics
s=Usd.Stage.Open(sys.argv[1]);goals=[];bad=[];dynamic=[]
for p in s.Traverse():
 path=str(p.GetPath())
 if p.GetName().startswith('Goal_'):goals.append(not p.HasAPI(UsdPhysics.CollisionAPI) and not p.HasAPI(UsdPhysics.RigidBodyAPI))
 if p.IsA(UsdPhysics.Joint):
  j=UsdPhysics.Joint(p);ends=[str(x) for x in list(j.GetBody0Rel().GetTargets())+list(j.GetBody1Rel().GetTargets())]
  if any('/Block' in x for x in ends) or (any('/Tweezers/' in x for x in ends) and any('/Robot/' in x for x in ends)):bad.append(path)
 if path.endswith('/Block') or ('/Tweezers/' in path and p.GetName() in ['Handle','Left','Right']):
  body=UsdPhysics.RigidBodyAPI(p);gravity=p.GetAttribute('physxRigidBody:disableGravity')
  dynamic.append(bool(body) and bool(body.GetRigidBodyEnabledAttr().Get()) and not bool(body.GetKinematicEnabledAttr().Get()) and not bool(gravity.Get() if gravity else False))
print('RL_USD_AUDIT='+json.dumps(dict(forbidden_tool_robot_or_block_joints=bad,goal_visual_count=len(goals),goal_markers_have_no_collision_or_rigid_body=all(goals),object_count=len(dynamic),all_objects_dynamic_with_gravity=all(dynamic))))
'''
 # A 64-environment exported scene takes longer to compose than a single demo.
 report['USD_audit']=helper.usd_audit(run/'final_scene.usda',timeout=300)
 u=report['USD_audit'];assert not u['forbidden_tool_robot_or_block_joints'] and u['goal_markers_have_no_collision_or_rigid_body'] and u['all_objects_dynamic_with_gravity'] and u['object_count']==4*r['episodes']
import hashlib
launch=json.loads((run/'launch.json').read_text())
report['archived_source_hashes_match']=all(hashlib.sha256((run/'source'/Path(name).name).read_bytes()).hexdigest()==expected for name,expected in launch['source_sha256'].items())
assert report['archived_source_hashes_match']
(run/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
assert report['all_claims_consistent'] and report['no_runtime_object_control_calls']
