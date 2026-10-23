"""Independent numeric review of the supplied robot and implemented controller FK/IK."""
from pathlib import Path
import sys,json,hashlib,math,xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kinematics import Kinematics,ASSET,stl_triangles,transform
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
k=Kinematics();doc=ET.parse(ASSET/'robot.urdf').getroot()
pose=json.loads((ROOT/'analysis/control_pose_b_clearance.json').read_text())
meta=json.loads((ROOT/'runs/free_feedback_b_05/metadata.json').read_text());runargs=meta['args'];rng=np.random.default_rng(93212)

def skew(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])

def rodrigues(axis,angle):
 a=np.asarray(axis,float);a/=np.linalg.norm(a);K=skew(a);return np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)

def reference_fk(q):
 T={'base_link':np.eye(4)};joints=list(doc.findall('joint'))
 while joints:
  done=[]
  for j in joints:
   p=j.find('parent').get('link');c=j.find('child').get('link')
   if p not in T:continue
   o=j.find('origin');r,pitch,y=np.fromstring(o.get('rpy','0 0 0'),sep=' ');loc=np.eye(4)
   loc[:3,:3]=rodrigues([0,0,1],y)@rodrigues([0,1,0],pitch)@rodrigues([1,0,0],r)
   loc[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
   if j.get('type')=='revolute':
    rot=np.eye(4);rot[:3,:3]=rodrigues(np.fromstring(j.find('axis').get('xyz'),sep=' '),q.get(j.get('name'),0.));loc=loc@rot
   T[c]=T[p]@loc;done.append(j)
  if not done:raise RuntimeError('Disconnected URDF')
  joints=[j for j in joints if j not in done]
 return T

fk_errors=[];orth_errors=[];limits=[]
for _ in range(32):
 q={n:float(rng.uniform(k.by_name[n]['lower'],k.by_name[n]['upper'])) for n in k.active}
 got=k.fk(q);ref=reference_fk(q)
 fk_errors += [float(np.max(np.abs(got[n]-ref[n]))) for n in ref]
 orth_errors += [float(np.linalg.norm(T[:3,:3].T@T[:3,:3]-np.eye(3))) for T in got.values()]
for label in ['q_open','q_closed']:
 for n,q in pose[label].items():
  j=k.by_name[n];limits.append({'pose':label,'joint':n,'position_rad':q,'lower':j['lower'],'upper':j['upper'],'within_limits':bool(j['lower']-1e-10<=q<=j['upper']+1e-10),'distance_to_limit_rad':min(q-j['lower'],j['upper']-q)})
# Known reachable targets with perturbed initial estimates: intentionally local IK validation.
ik_rows=[]
for _ in range(24):
 q=np.array([rng.uniform(-1.2,1.2),rng.uniform(-1.8,-.7),rng.uniform(.5,2),rng.uniform(-2.4,-.4),rng.uniform(-2,-.4),rng.uniform(-1.5,1.5)])
 tar=reference_fk(dict(zip(k.arm,q)))['right_hand_link'];sol=k.inverse_arm(tar,q+rng.normal(0,.06,6));got=reference_fk(sol['q'])['right_hand_link']
 ik_rows.append({'position_error_m':float(np.linalg.norm(got[:3,3]-tar[:3,3])),'rotation_error_rad':float(Rotation.from_matrix(tar[:3,:3].T@got[:3,:3]).magnitude()),'reported_error':sol['error'],'optimizer_success':sol['success']})
H=np.array(pose['T_hand_tool']);H[:3,3]+=(pose.get('tool_pinch_z',.075)-runargs['pinch_z'])*H[:3,2]
arm=np.array([meta['q_initial'][n] for n in k.arm]);path=[]
for z in np.r_[np.linspace(.98,.862,31),np.linspace(.862,.962,31)]:
 tar=transform([.48,0,z],np.diag([1.,-1.,-1.]))@np.linalg.inv(H);tar[:3,3]-=np.array([0,0,.72]);sol=k.inverse_arm(tar,arm);arm=np.array([sol['q'][n] for n in k.arm]);got=reference_fk(sol['q'])['right_hand_link'];path.append({'tool_z':float(z),'position_error_m':float(np.linalg.norm(got[:3,3]-tar[:3,3])),'rotation_error_rad':float(Rotation.from_matrix(tar[:3,:3].T@got[:3,:3]).magnitude()),'within_limits':all(k.by_name[n]['lower']<=sol['q'][n]<=k.by_name[n]['upper'] for n in k.arm)})
meshes={}
for e in doc.findall('link'):
 name=e.get('name');mesh=e.find('visual/geometry/mesh')
 if name.startswith('right_hand') and mesh is not None:meshes[name]=np.unique(stl_triangles(ASSET/mesh.get('filename')).reshape(-1,3),axis=0)
ideal=[]
for alpha in np.linspace(0,1,21):
 q={n:(1-alpha)*pose['q_open'][n]+alpha*pose['q_closed'][n] for n in k.hand};Ts=k.fk(q,root='right_hand_link');hand_world=transform([.48,0,.862],np.diag([1.,-1.,-1.]))@np.linalg.inv(H)
 for name,v in meshes.items():
  T=hand_world@Ts[name];clearance=float((v@T[2,:3]+T[2,3]).min()-.72);ideal.append({'alpha':float(alpha),'link':name,'clearance_m':clearance})
traj=json.loads((ROOT/'runs/free_feedback_b_05/trajectory.json').read_text());actual=[];jnames=meta['robot_joints']
for frame in traj:
 q=dict(zip(jnames,frame['joints']));Ts=k.fk(q)
 for name,v in meshes.items():
  T=Ts[name];clearance=float((v@T[2,:3]+T[2,3]).min());actual.append({'t':frame['t'],'link':name,'clearance_m':clearance})
unreachable=k.inverse_arm(transform([5.,5.,5.]),initial=np.array([meta['q_initial'][n] for n in k.arm]))
report={
 'scope':'Numerical FK/IK, static hand joint limits and hand STL table clearance for the current supplied URDF; not a dynamics or full collision certification.',
 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/kinematics.py',ROOT/'analysis/control_pose_b_clearance.json',ROOT/'analysis/validate_kinematics.py']},
 'asset_urdf':str(ASSET/'robot.urdf'),'asset_urdf_sha256':hashlib.sha256((ASSET/'robot.urdf').read_bytes()).hexdigest(),
 'independent_fk':{'method':'Independent XML traversal with manually composed Rz(yaw) Ry(pitch) Rx(roll), Rodrigues axis-angle joint rotations','sample_count':32,'link_comparisons':len(fk_errors),'max_transform_element_error':max(fk_errors),'max_rotation_orthogonality_error':max(orth_errors),'passed':max(fk_errors)<1e-10},
 'local_ik':{'sample_count':24,'max_position_error_m':max(r['position_error_m'] for r in ik_rows),'max_rotation_error_rad':max(r['rotation_error_rad'] for r in ik_rows),'all_optimizer_converged':all(r['optimizer_success'] for r in ik_rows),'samples':ik_rows},
 'task_nominal_ik':{'pinch_z_m':runargs['pinch_z'],'sample_count':len(path),'max_position_error_m':max(r['position_error_m'] for r in path),'max_rotation_error_rad':max(r['rotation_error_rad'] for r in path),'all_within_arm_limits':all(r['within_limits'] for r in path)},
 'hand_joint_limits':{'all_within_limits':all(r['within_limits'] for r in limits),'joint_pose_checks':limits,'interpolation_note':'All convex interpolations q_open..q_closed remain within the source URDF bounds; extrapolation beyond these endpoints is not covered.'},
 'table_clearance':{'mesh_source':'All 13 real hand visual STL meshes, exact vertices; root base is placed on table at z=.72m','assumptions':'Ideal nominal arm pose uses tool pivot at z=.862m, pinch_z=.04m, 21 hand closures from open to closed. Recorded trajectory uses FK of measured joint positions at 30Hz and TABLE base transform. PhysX collision hull inflation and between-frame extremes are not included.','ideal_minimum':min(ideal,key=lambda r:r['clearance_m']),'recorded_run':'runs/free_feedback_b_05','recorded_sample_count':len(traj),'recorded_minimum':min(actual,key=lambda r:r['clearance_m']),'all_sampled_hand_surfaces_above_table':min(r['clearance_m'] for r in ideal+actual)>0},
 'limitations':['The hardcoded inverse_arm bounds are slightly inside supplied URDF arm limits and assume its six-joint order.','Kinematics parser targets this provided revolute/fixed URDF; it is not a general prismatic/mimic/continuous-joint parser.','inverse_arm.success reports optimizer termination, not target reachability; callers must test error. Current runtime initial placement checks error<1e-4, but its motion loop does not yet reject excessive IK residual.','Joint limits include a few endpoints exactly at their upper/lower bound; modeled finite drive torque and contacts determine realized posture.','Recorded clearance uses FK and samples 30Hz; no claim of continuous swept collision-free motion or real-hardware calibration.'],
 'unreachable_target_probe':{'target_xyz':[5,5,5],'optimizer_success':unreachable['success'],'residual':unreachable['error'],'demonstrates_need_for_residual_guard':unreachable['error']>1e-4}
}
report['passed_numerical_checks']=report['independent_fk']['passed'] and report['local_ik']['max_position_error_m']<1e-6 and report['local_ik']['max_rotation_error_rad']<1e-6 and report['task_nominal_ik']['max_position_error_m']<1e-6 and report['hand_joint_limits']['all_within_limits'] and report['table_clearance']['all_sampled_hand_surfaces_above_table']
(ROOT/'analysis/kinematics_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({key:report[key] for key in ['passed_numerical_checks','independent_fk','task_nominal_ik','table_clearance','unreachable_target_probe']},indent=2))
