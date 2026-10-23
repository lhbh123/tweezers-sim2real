"""Independent numerical UR5e branch search with exact source-STL table extrema.

Preserves observed hand joint positions and all relative hand/tool rigid poses.
Only translates the entire grasp and optionally rotates it about WORLD Z.
No simulator state or shared source files are modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kinematics import Kinematics,ASSET,transform,xyz,stl_triangles


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(p):return transform(p[:3],Rotation.from_quat(np.asarray(p)[[4,5,6,3]]).as_matrix())
def quaternion(R):
 q=Rotation.from_matrix(R).as_quat();return np.r_[q[3],q[:3]]
def skew(a):
 x,y,z=a;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


class IndependentArm:
 def __init__(self):
  self.kin=Kinematics();self.chain=self.kin.joints[:7];self.meshes={};self.mesh_sources={}
  for link in self.kin.root.findall('link'):
   name=link.get('name')
   if name not in ['shoulder_link','upper_arm_link','forearm_link','wrist_1_link','wrist_2_link','wrist_3_link']:continue
   c=link.find('collision');m=c.find('geometry/mesh');o=c.find('origin');path=ASSET/m.get('filename')
   v=np.unique(stl_triangles(path).reshape(-1,3),axis=0)
   T=transform(xyz(o.get('xyz')),Rotation.from_euler('xyz',xyz(o.get('rpy'))).as_matrix())
   if m.get('scale'):v*=xyz(m.get('scale'))
   v=v@T[:3,:3].T+T[:3,3];self.meshes[name]=v[ConvexHull(v).vertices]
   self.mesh_sources[name]=dict(path=str(path),sha256=digest(path),original_unique_vertices=len(v),extreme_vertices=len(self.meshes[name]))
 def forward(self,q,jac=False):
  T=np.eye(4);points=[];axes=[];i=0
  for joint in self.chain:
   T=T@joint['T']
   if joint['kind']!='fixed':
    points.append(T[:3,3].copy());axes.append(T[:3,:3]@joint['axis'])
    a=q[i];R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    T=T@transform(rotation=R);i+=1
  if not jac:return T
  J=np.zeros((12,6))
  for i,(p,a) in enumerate(zip(points,axes)):
   J[:3,i]=4*np.cross(a,T[:3,3]-p);J[3:,i]=(skew(a)@T[:3,:3]).ravel()/np.sqrt(2)
  return T,J
 def solve(self,target,seed):
  def fun(q):
   T=self.forward(q);return np.r_[4*(T[:3,3]-target[:3,3]),((T[:3,:3]-target[:3,:3])/np.sqrt(2)).ravel()]
  result=least_squares(fun,seed,jac=lambda q:self.forward(q,True)[1],bounds=([-6.28,-6.28,-3.14,-6.28,-6.28,-6.28],[6.28,6.28,3.14,6.28,6.28,6.28]),max_nfev=100,xtol=1e-10,ftol=1e-10,gtol=1e-10)
  error=np.linalg.norm(fun(result.x));return result.x,error
 def clearance(self,q):
  fk=self.kin.fk(dict(zip(self.kin.arm,q)))
  return {n:float((v@fk[n][:3,:3].T+fk[n][:3,3])[:,2].min()) for n,v in self.meshes.items()}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--snapshot',type=Path,default=ROOT/'runs/tripod_seed57_preload02/c0.8_ix0.015_mid-0.04_snapshot.json')
 p.add_argument('--out',type=Path,default=ROOT/'reports/tripod_ik_workspace.json')
 p.add_argument('--seeds',type=int,default=32);p.add_argument('--yaw-deg',nargs='+',type=float,default=[0,5,-5,10,-10,15,-15,30,-30,60,-60,90,-90,180])
 p.add_argument('--x',nargs='+',type=float,default=[.48,.4,.45,.5,.55,.6,.65]);p.add_argument('--y',nargs='+',type=float,default=[0,-.05,.05,-.1,.1,-.15,.15])
 p.add_argument('--tip-z',type=float,default=.7458);p.add_argument('--tip-local-z',type=float,default=.1525);p.add_argument('--min-clearance',type=float,default=.01)
 p.add_argument('--stop-first',action='store_true');p.add_argument('--rng-seed',type=int,default=17)
 args=p.parse_args();s=json.loads(args.snapshot.read_text());arm=IndependentArm();qsource=dict(zip(s['joint_names'],s['q_actual']));q0=np.array([qsource[n] for n in arm.kin.arm])
 wrist=arm.kin.fk(qsource)['wrist_3_link'];base=np.array(s['arm_base_world_xyz']);wrist[:3,3]+=base
 bodies={n:matrix(s['rigid_body_states_13'][n]) for n in ['handle','left','right']}
 tip=.5*sum((bodies[n]@np.array([0,0,args.tip_local_z,1]))[:3] for n in ['left','right'])
 hand=matrix(s['world_hand_pose_7']);rng=np.random.default_rng(args.rng_seed)
 seeds=[q0]
 for i in range(args.seeds-1):seeds.append(rng.uniform([-np.pi,-np.pi,-3.1,-np.pi,-np.pi,-np.pi],[np.pi,np.pi,3.1,np.pi,np.pi,np.pi]))
 checkT,J=arm.forward(q0,True);errfk=float(np.max(np.abs(checkT-arm.kin.fk(qsource)['wrist_3_link'])))
 eps=1e-6;Jnum=np.empty_like(J)
 for i in range(6):
  dq=np.eye(6)[i]*eps;A=arm.forward(q0+dq);B=arm.forward(q0-dq);Jnum[:,i]=np.r_[4*(A[:3,3]-B[:3,3]),((A[:3,:3]-B[:3,:3])/np.sqrt(2)).ravel()]/(2*eps)
 errjac=float(np.max(np.abs(J-Jnum)));assert errfk<1e-10 and errjac<1e-7
 result=dict(snapshot=str(args.snapshot),snapshot_sha256=digest(args.snapshot),tip_local_z_m=args.tip_local_z,table_z_m=.72,base_xyz=base.tolist(),arm_links=list(arm.meshes),source_collision_meshes=arm.mesh_sources,method='Analytic Cartesian/matrix Jacobian numerical IK with bounded joints and random seeds; source collision STL extrema. Convex-hull vertices preserve exact global minimum Z for every rigid transform.',hand_joints_unchanged=True,relative_three_body_poses_unchanged=True,tool_tilt_and_web_up_unchanged_by_world_yaw=True,forward_check_max_error=errfk,jacobian_check_max_error=errjac,minimum_clearance_required_m=args.min_clearance,cases=[],feasible=[],complete_search=False)
 args.out.parent.mkdir(parents=True,exist_ok=True)
 def save():args.out.write_text(json.dumps(result,indent=2))
 start=time.monotonic();previous=[]
 for yaw in args.yaw_deg:
  R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
  for x in args.x:
   for y in args.y:
    target_tip=np.array([x,y,args.tip_z]);target=wrist.copy();target[:3,:3]=R@wrist[:3,:3];target[:3,3]=target_tip+R@(wrist[:3,3]-tip)-base
    solutions=[]
    for seed in previous+seeds:
     q,error=arm.solve(target,seed)
     if error>1e-5:continue
     if any(np.linalg.norm((q-prev['q']+np.pi)%(2*np.pi)-np.pi)<1e-3 for prev in solutions):continue
     clearance=arm.clearance(q);T=arm.forward(q)
     item=dict(q=q.tolist(),ik_residual=float(error),position_error_m=float(np.linalg.norm(T[:3,3]-target[:3,3])),rotation_error_deg=float(Rotation.from_matrix(T[:3,:3].T@target[:3,:3]).magnitude()*180/np.pi),clearance_by_link_m=clearance,min_clearance_m=min(clearance.values()))
     solutions.append(item)
    previous=[np.array(v['q']) for v in solutions]
    best=max(solutions,key=lambda z:z['min_clearance_m']) if solutions else None
    case=dict(tip_xyz=target_tip.tolist(),world_yaw_delta_deg=yaw,distinct_solutions=len(solutions),best=best,all_solutions=solutions)
    result['cases'].append(case)
    for item in solutions:
     if item['min_clearance_m']>=args.min_clearance:
      result['feasible'].append(dict(**item,tip_xyz=target_tip.tolist(),world_yaw_delta_deg=yaw,tool_quaternion_wxyz=quaternion(R@bodies['handle'][:3,:3]).tolist(),wrist_target_local=target.tolist(),web_up_component=float((R@hand[:3,:3])[2,0]),tool_down_angle_deg=float(np.degrees(np.arcsin(-(R@bodies['handle'][:3,:3])[2,2])))))
    print(json.dumps(dict(yaw=yaw,x=x,y=y,branches=len(solutions),best_mm=round(best['min_clearance_m']*1000,3) if best else None,feasible=len(result['feasible']),elapsed_s=round(time.monotonic()-start,2))),flush=True)
    result['elapsed_s']=time.monotonic()-start;save()
    if args.stop_first and result['feasible']:return
 result['complete_search']=True;save()

if __name__=='__main__':main()
