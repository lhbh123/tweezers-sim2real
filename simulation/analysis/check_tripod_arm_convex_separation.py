"""Supplement source-surface SAT with conservative solid-hull separation checks."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from kinematics import Kinematics,ASSET,xyz,transform,stl_triangles
from validate_rl_hand_geometry import load_meshes
r=json.loads((ROOT/'reports/tripod_arm_path_audit.json').read_text());s=json.loads(Path(r['snapshot']).read_text());qsource=dict(zip(s['joint_names'],s['q_actual']));kin=Kinematics();meshes,_=load_meshes(kin);vertices={n:np.unique(m.triangles.reshape(-1,3),axis=0) for n,m in meshes.items()}
for link in kin.root.findall('link'):
 n=link.get('name')
 if n not in r['frames'][0]['clearance_by_link_m']:continue
 c=link.find('collision');m=c.find('geometry/mesh');o=c.find('origin');v=np.unique(stl_triangles(ASSET/m.get('filename')).reshape(-1,3),axis=0);T=transform(xyz(o.get('xyz')),Rotation.from_euler('xyz',xyz(o.get('rpy'))).as_matrix());vertices[n]=v@T[:3,:3].T+T[:3,3]
hulls={n:ConvexHull(v) for n,v in vertices.items()};checks=[];candidate=[];start=time.monotonic();count=0;maxradius=-float('inf')
for i,frame in enumerate(r['frames']):
 q=dict(qsource);q.update(zip(kin.arm,frame['arm_q']));fk=kin.fk(q);equations={};bounds={}
 for n,h in hulls.items():
  T=fk[n];v=vertices[n][h.vertices]@T[:3,:3].T+T[:3,3];bounds[n]=(v.min(0),v.max(0));normal=h.equations[:,:3]@T[:3,:3].T;equations[n]=np.c_[normal,h.equations[:,3]-normal@T[:3,3]]
 for a,b in r['checked_pairs']:
  if np.any(np.maximum(bounds[a][0],bounds[b][0])>np.minimum(bounds[a][1],bounds[b][1])):continue
  eq=np.r_[equations[a],equations[b]];res=linprog([0,0,0,-1],A_ub=np.c_[eq[:,:3],np.ones(len(eq))],b_ub=-eq[:,3],bounds=[(None,None)]*4,method='highs');count+=1
  if not res.success:raise RuntimeError(res.message)
  radius=float(res.x[3]);maxradius=max(maxradius,radius)
  if radius>=-1e-9:candidate.append(dict(frame=i,links=[a,b],intersection_ball_radius_m=radius))
 if i%100==0:print(json.dumps(dict(frames=i+1,lp_checks=count,overlap_candidates=len(candidate),max_radius_m=maxradius,elapsed_s=time.monotonic()-start)),flush=True)
result=dict(source_audit='reports/tripod_arm_path_audit.json',method='Exact source vertices -> convex hulls -> maximum common ball LP (signed radius). Negative optimum proves disjoint solid hulls and rules out surface containment.',sampled_poses=len(r['frames']),nonadjacent_pairs_per_pose=len(r['checked_pairs']),aabb_overlapping_pairs_lp_checked=count,maximum_signed_intersection_ball_radius_m=maxradius,all_sampled_nonadjacent_solid_hulls_separated=len(candidate)==0,overlap_candidates=candidate,elapsed_s=time.monotonic()-start,limitations=['Convex hull overlap alone would not establish original mesh penetration.','Finite floating point linear programming, tolerance1nm; sampled poses only.'])
(ROOT/'reports/tripod_arm_convex_separation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result if len(candidate)<20 else {k:v for k,v in result.items() if k!='overlap_candidates'}))
