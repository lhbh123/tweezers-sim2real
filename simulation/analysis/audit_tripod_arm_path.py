"""Check one preserved tripod grasp IK branch over XY jitter and a vertical path.

Exact source mesh extrema for table clearance; BVH + triangle SAT for surfaces.
No new grasp fitting, physics, or shared-source edits.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from explore_tripod_ik_workspace import IndependentArm,matrix,digest
from validate_rl_hand_geometry import Mesh,mesh_sat,load_meshes

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kinematics import ASSET,transform,xyz,stl_triangles


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'reports/tripod_ik_workspace.json');p.add_argument('--out',type=Path,default=ROOT/'reports/tripod_arm_path_audit.json');p.add_argument('--z-step',type=float,default=.002)
 args=p.parse_args();input=json.loads(args.source.read_text());s=json.loads(Path(input['snapshot']).read_text());arm=IndependentArm();kin=arm.kin
 choice=min(input['feasible'],key=lambda f:abs(f['q'][0]));qseed=np.array(choice['q']);qsource=dict(zip(s['joint_names'],s['q_actual']));base=np.array(s['arm_base_world_xyz'])
 wrist=kin.fk(qsource)['wrist_3_link'];wrist[:3,3]+=base;bodies={n:matrix(s['rigid_body_states_13'][n]) for n in ['handle','left','right']};tip=.5*sum((bodies[n]@np.array([0,0,input['tip_local_z_m'],1]))[:3] for n in ['left','right'])
 meshes,sources=load_meshes(kin)
 for link in kin.root.findall('link'):
  name=link.get('name')
  if name not in arm.meshes:continue
  c=link.find('collision');m=c.find('geometry/mesh');o=c.find('origin');path=ASSET/m.get('filename');tri=stl_triangles(path)
  if m.get('scale'):tri*=xyz(m.get('scale'))
  T=transform(xyz(o.get('xyz')),Rotation.from_euler('xyz',xyz(o.get('rpy'))).as_matrix());tri=tri@T[:3,:3].T+T[:3,3]
  meshes[name]=Mesh(tri);sources.append(dict(link=name,path=str(path),sha256=digest(path),triangle_count=len(tri)))
 # Merge fixed-joint groups before deciding which surfaces share a mechanical joint.
 parent={link.get('name'):link.get('name') for link in kin.root.findall('link')}
 def find(n):
  while parent[n]!=n:n=parent[n]
  return n
 for j in kin.joints:
  if j['kind']=='fixed':parent[find(j['child'])]=find(j['parent'])
 adjacent={frozenset([find(j['parent']),find(j['child'])]) for j in kin.joints if j['kind']!='fixed'}
 links=list(meshes);pairs=[];excluded=[]
 for i,a in enumerate(links):
  for b in links[i+1:]:
   if a not in arm.meshes and b not in arm.meshes:continue
   if find(a)==find(b):excluded.append(dict(links=[a,b],reason='same fixed rigid-body group'));continue
   if frozenset([find(a),find(b)]) in adjacent:excluded.append(dict(links=[a,b],reason='joint-adjacent rigid-body groups'));continue
   pairs.append((a,b))
 start=time.monotonic();frames=[];worst=1.;intersections=[]
 zs=np.unique(np.r_[np.arange(-.018,.100+args.z_step/2,args.z_step),0.,.100]);offsets=[(x,y) for x in [-.002,0,.002] for y in [-.002,0,.002]]
 for ix,(dx,dy) in enumerate(offsets):
  seed=qseed.copy()
  for dz in zs:
   point=np.array(choice['tip_xyz'])+[dx,dy,dz];target=wrist.copy();target[:3,3]=point+(wrist[:3,3]-tip)-base
   q,error=arm.solve(target,seed)
   if error>1e-5:raise ValueError(f'IK failure at {point}: {error}')
   delta=float(np.max(np.abs((q-seed+np.pi)%(2*np.pi)-np.pi)));seed=q
   qfull=dict(qsource);qfull.update(zip(kin.arm,q));fk=kin.fk(qfull);posed={n:mesh.at_pose(fk[n]) for n,mesh in meshes.items()}
   clearance=arm.clearance(q);worst=min(worst,min(clearance.values()));hits=[];tests=0;overlapboxes=0
   for a,b in pairs:
    A,B=posed[a],posed[b]
    if np.any(np.maximum(A['lo'][0],B['lo'][0])>np.minimum(A['hi'][0],B['hi'][0])+1e-9):continue
    overlapboxes+=1;check=mesh_sat(A,B,max_witnesses=2);tests+=check['triangle_pair_tests']
    if check['surface_intersection_triangle_pairs']:hits.append(dict(links=[a,b],**check))
   row=dict(tip_xyz=point.tolist(),offset_xyz=[dx,dy,float(dz)],arm_q=q.tolist(),ik_residual=float(error),max_joint_delta_from_previous_rad=delta,clearance_by_link_m=clearance,min_clearance_m=min(clearance.values()),checked_pair_count=len(pairs),aabb_overlap_pair_count=overlapboxes,triangle_pair_tests=tests,surface_intersections=hits)
   frames.append(row)
   if hits:intersections.append(len(frames)-1)
  print(json.dumps(dict(x_offset=dx,y_offset=dy,poses=len(frames),worst_clearance_mm=worst*1000,intersecting_poses=len(intersections),elapsed_s=time.monotonic()-start)),flush=True)
 result=dict(source_search=str(args.source),source_search_sha256=digest(args.source),snapshot=input['snapshot'],snapshot_sha256=input['snapshot_sha256'],selected_branch=choice,source_meshes=sources,method='Original arm collision STL + original hand visual STL; exact min Z from rigidly transformed source extrema. Nonadjacent arm-arm/arm-hand BVH/triangle SAT with 1nm tolerance.',fixed_body_groups={n:find(n) for n in links},excluded_pairs=excluded,checked_pairs=[list(v) for v in pairs],hand_joints_preserved=True,relative_hand_tool_pose_preserved=True,world_yaw_delta_deg=0,table_z_m=.72,summary=dict(poses=len(frames),z_step_m=args.z_step,xy_offsets_m=[-.002,0,.002],z_range_m=[-.018,.1],minimum_arm_table_clearance_m=worst,all_poses_arm_clearance_ge_10mm=worst>=.01,frames_with_surface_intersections=len(intersections),intersecting_frame_indices=intersections,elapsed_s=time.monotonic()-start),frames=frames,limitations=['Sampled path in configuration space; no continuous swept-volume proof between samples.','Triangle surface intersection does not independently rule out fully contained disconnected shells.','Source meshes differ from PhysX convex cooking.','Adjacent rigid-body groups and all fixed-merged surfaces excluded because mechanical joint surfaces overlap by design.','This is geometry/IK verification only; dynamic stability and actual collision forces require simulation.'])
 args.out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(args.out),**result['summary'])))

if __name__=='__main__':main()
