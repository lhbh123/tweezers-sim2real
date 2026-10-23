"""Audit actual free-held source geometry, including every visible handle part.

Saved actual joint/body states determine geometry. Contact sensors and source
STL separation are reported separately because cooked convex shapes may differ.
"""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis')]
from kinematics import transform
from audit_distal_geometry import Auditor,pose_matrix
from tripod_tail_surface_gap import min_distance
from validate_human_grasp import triangle_box_overlap

def boxes_for_config(cfg):
 geo=cfg['tool_geometry'];width=geo['strip_width_m'];length=geo['length_m'];hlen=geo.get('handle_length_m',.006);hwidth=geo.get('handle_width_m',width)
 boxes=[('handle','bridge',[0,0,-hlen/2],[.006,hwidth/2,hlen/2],np.eye(4))]
 for n in ['left','right']:
  boxes +=[(n,'blade',[0,0,(length-.015)/2],[.0005,width/2,(length-.015)/2],np.eye(4)),(n,'tip',[0,0,length-.0075],[.0005,.0025,.0075],np.eye(4))]
 ext=geo.get('handle_extension')
 if ext:
  start=np.asarray(ext['start_m']);end=np.asarray(ext['end_m']);z=end-start;l=np.linalg.norm(z);z/=l;x=np.array([1.,0,0])-z*z[0];x/=np.linalg.norm(x);y=np.cross(z,x);E=transform(.5*(start+end),np.column_stack([x,y,z]));boxes.append(('handle','extension',[0,0,0],[ext['width_x_m']/2,ext['width_y_m']/2,l/2],E))
 return boxes

def audit_snapshot(path,a):
 s=json.loads(path.read_text());config_path=(ROOT/s['args']['pose']).resolve();cfg=json.loads(config_path.read_text());q=dict(zip(s['joint_names'],s['q_actual']));meshes,hulls=a.geometry(q);W=pose_matrix(s['world_hand_pose_7']);bodies={n:np.linalg.inv(W)@pose_matrix(v) for n,v in s['rigid_body_states_13'].items()};boxes=boxes_for_config(cfg);hits=[];gaps=[]
 for body,part,c,h,E in boxes:
  B=bodies[body]@E;c=np.asarray(c);h=np.asarray(h)
  if body=='handle':
   tri=(meshes['right_hand_link']-B[:3,3])@B[:3,:3];gap,pm,pb=min_distance(tri,c-h,c+h);count=int(triangle_box_overlap(tri,c,h).sum());pmh=B[:3,:3]@pm+B[:3,3];pbh=B[:3,:3]@pb+B[:3,3]
   gaps.append(dict(part=part,source_surface_distance_m=0. if count else float(gap),nearest_feature_distance_m=float(gap),source_surface_intersections=count,closest_palm_point_hand=pmh.tolist(),closest_tail_point_hand=pbh.tolist(),tail_to_palm_vector_hand=(pbh-pmh).tolist(),tail_to_palm_vector_world=(W[:3,:3]@(pbh-pmh)).tolist()))
  for link,mesh in meshes.items():
   tr=(mesh-B[:3,3])@B[:3,:3];broad=np.all((tr.max(1)>=c-h)&(tr.min(1)<=c+h),axis=1)
   if broad.any():
    count=int(triangle_box_overlap(tr[broad],c,h).sum())
    if count:hits.append(dict(link=link,body=body,part=part,source_triangle_intersections=count))
 best=min(gaps,key=lambda p:p['source_surface_distance_m']);qtarget=dict(zip(s['joint_names'],s['q_target']));ref=cfg['q_reference'];delta=cfg['closure_delta'];targetdif=np.array([qtarget[n]-ref[n] for n in a.kin.hand]);d=np.array([delta[n] for n in a.kin.hand]);closure=float(targetdif@d/max(d@d,1e-20));res=targetdif-closure*d
 return dict(snapshot=str(path.relative_to(ROOT)),snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),config=str(config_path.relative_to(ROOT)),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),tail_palm_source_surface_distance_m=best['source_surface_distance_m'],tail_palm_closest_visible_part=best['part'],tail_palm_source_surface_intersections=sum(p['source_surface_intersections'] for p in gaps),closest_palm_point_hand=best['closest_palm_point_hand'],closest_tail_point_hand=best['closest_tail_point_hand'],tail_to_palm_vector_hand=best['tail_to_palm_vector_hand'],tail_to_palm_vector_world=best['tail_to_palm_vector_world'],all_visible_handle_parts_source_clearance=gaps,self_collision=a.self_collision(meshes,hulls),actual_tool_hand_intersections=hits,hard_limit_min_margin_rad=float(min(min(q[n]-a.kin.by_name[n]['lower'],a.kin.by_name[n]['upper']-q[n]) for n in a.kin.hand)),control_reference_consistency=dict(best_fit_closure=closure,off_synergy_target_residual_max_rad=float(np.abs(res).max()),target_actual_difference_max_rad=float(max(abs(qtarget[n]-q[n]) for n in a.kin.hand)),note='Nonzero off-synergy residual can be a deliberately recorded independent finger preload; it must not be silently discarded.'),physics=s['tripod_hold'],limitations=['Source STL/box geometry uses actual recorded poses, not solver contact hulls.','Positive source gap and PhysX contact may coexist with convex decomposition; neither overrides the other.','No claim about unrecorded simulation steps or volume containment.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();a=Auditor();result=[]
 for f in args.run.glob('*snapshot.json'):
  r=audit_snapshot(f.resolve(),a);result.append(r);print(f.name,'closestpart',r['tail_palm_closest_visible_part'],'tailgapmm',r['tail_palm_source_surface_distance_m']*1000,'intersections',r['actual_tool_hand_intersections'],flush=True)
 (args.run/'source_geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
