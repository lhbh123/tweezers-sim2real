"""Export and source-audit straight-tail wide-face geometric initializations."""
from pathlib import Path
import sys,json,copy,argparse
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis')]
from kinematics import transform
from audit_distal_geometry import Auditor
from validate_human_grasp import triangle_box_overlap
from tripod_tail_surface_gap import min_distance

def bodies(T,angle):
 return {'handle':T,'left':T@transform([-.0045,0,0],Rotation.from_euler('y',-angle).as_matrix()),'right':T@transform([.0045,0,0],Rotation.from_euler('y',angle).as_matrix())}

def boxes(length,width,hlen):
 return [('handle','straight_tail',np.array([0,0,-hlen/2]),np.array([.005,width/2,hlen/2]))]+[(side,part,np.array(c),np.array(h)) for side in ['left','right'] for part,c,h in [('blade',[0,0,(length-.015)/2],[.0005,width/2,(length-.015)/2]),('tip',[0,0,length-.0075],[.0005,.0025,.0075])]]

def inspect(a,q,T,angle,length,width,hlen,Rh):
 mesh,hulls=a.geometry(q);B=bodies(T,angle);hits=[]
 for body,part,c,h in boxes(length,width,hlen):
  M=B[body]
  for name,tri in mesh.items():
   local=(tri-M[:3,3])@M[:3,:3];broad=np.all((local.max(1)>=c-h)&(local.min(1)<=c+h),axis=1)
   if broad.any():
    n=int(triangle_box_overlap(local[broad],c,h).sum())
    if n:hits.append(dict(body=body,part=part,link=name,triangle_intersections=n))
 intended={('right','right_hand_thumb_rota_link2'),('left','right_hand_index_rota_link2')}
 tip=.5*sum(M[:3,3]+M[:3,2]*(length-.0075) for name,M in B.items() if name!='handle')
 hand_z=.7276-(Rh@tip)[2]
 clear={n:float((tri.reshape(-1,3)@Rh[2]).min()+hand_z-.72) for n,tri in mesh.items()}
 palm_local=(mesh['right_hand_link']-T[:3,3])@T[:3,:3];c=np.array([0,0,-hlen/2]);h=np.array([.005,width/2,hlen/2]);gap,pm,pb=min_distance(palm_local,c-h,c+h)
 return dict(self_collision=a.self_collision(mesh,hulls),all_tool_hand_intersections=hits,
             unexpected_tool_hand_intersections=[x for x in hits if (x['body'],x['link']) not in intended],
             hand_table_clearance_by_link_m=clear,minimum_hand_table_clearance_m=min(clear.values()),tip_center_target_z_m=.7276,
             straight_tail_palm_source_gap_m=float(gap),closest_palm_point_hand=(T[:3,:3]@pm+T[:3,3]).tolist(),
             closest_tail_point_hand=(T[:3,:3]@pb+T[:3,3]).tolist())

def main(args):
 allfits=json.loads(args.candidates.read_text());old=json.loads((ROOT/'configs/human_grasp_tripod_stow_pick_v1.json').read_text());a=Auditor();out=[]
 for seed in args.seeds:
  fit=next(x for x in allfits if x['seed']==seed);T=np.array(fit['T_hand_tool']);Rh=np.array(fit['preferred_hand_world_R']);q=fit['q_reference'];names=a.kin.hand[:6]
  # Relaxed backup finger, not a forced load-bearing constraint.
  q['right_hand_mid_joint1']=1.0;q['right_hand_mid_joint2']=.75
  patches=fit['patches'];loc=np.array([p['point_local'] for p in patches]);nn=np.array([p['normal_local'] for p in patches]);links=[p['link'] for p in patches]
  def close_res(v,angle):
   qq={**q,**dict(zip(names,v[:6]))};F=a.kin.fk(qq,root='right_hand_link');B=bodies(T,angle);target=[];want=[]
   for side,sign,z in [('right',1,v[6]),('left',-1,v[7])]:
    M=B[side];target.append(M[:3,:3]@np.array([sign*.0005,0,z])+M[:3,3]);want.append(M[:3,:3]@np.array([-sign,0,0]))
   p=np.array([F[n][:3,:3]@pt+F[n][:3,3] for n,pt in zip(links,loc)]);n=np.array([F[name][:3,:3]@normal for name,normal in zip(links,nn)])
   return np.r_[((p-target)*900).ravel(),((n-want)*.3).ravel(),(v[:6]-np.array([q[n] for n in names]))*.3,(v[6:]-fit['contact_z'])*8]
  low=[fit['bounds'][n][0] for n in names]+[max(.02,z-.02) for z in fit['contact_z']];high=[fit['bounds'][n][1] for n in names]+[min(.13,z+.02) for z in fit['contact_z']]
  initial=np.r_[[q[n] for n in names],fit['contact_z']]
  closed=least_squares(close_res,np.clip(initial,np.array(low)+1e-7,np.array(high)-1e-7),args=(args.closed_angle,),bounds=(low,high),max_nfev=240)
  qc={**q,**dict(zip(names,closed.x[:6]))};delta={n:qc[n]-q[n] for n in a.kin.hand}
  width=.014;length=.18;hlen=args.handle_length
  audit=inspect(a,q,T,.05,length,width,hlen,Rh);caudit=inspect(a,qc,T,args.closed_angle,length,width,hlen,Rh)
  cfg={k:copy.deepcopy(old[k]) for k in ['human_joint_prior','human_evidence','palm_collision','arm_reset_seed','arm_reset_table_clearance_m'] if k in old}
  cfg.update(schema_version=4,method='Source-STL finger-pad fit to opposite outer broad faces of straight-tail tweezers; middle/web optional backups, not mandatory supports',q_reference=q,q_open=q,q_closed=qc,closure_delta=delta,T_hand_tool=T.tolist(),tool_pinch_z=float(np.mean(fit['contact_z'])),soft_bounds=copy.deepcopy(fit['bounds']),
   tool_geometry={'length_m':length,'hinge_offset_m':.0045,'open_angle_rad':.05,'strip_thickness_m':.001,'strip_width_m':width,'handle_width_m':width,'handle_length_m':hlen,'handle_mass_kg':.003,'jaw_mass_kg':.006},
   grasp_topology={'type':'thumb_index_outer_wide_face_pinch_optional_middle_web_backup','web_edge_axis_local':[1.,0,0],'web_anchor_local':[.04270149,.015,-.065],'palm_outward_axis_local':[0,1.,0],'tail_web_contact_claimed':False},
   web_contact_region={'frame':'right_hand_link','center_m':[.04270149,.015,-.065],'radius_m':.02},web_up_axis_hand=[1,0,0],preferred_tool_world_R=fit['preferred_tool_world_R'],preferred_hand_world_R=fit['preferred_hand_world_R'],
   face_pinch_assignment={'thumb':'right','index':'left'},face_pinch_geometry={'local_axes':'X thickness/closure, Y width, Z root-to-tip','thumb_expected_outward_normal_local':[1,0,0],'index_expected_outward_normal_local':[-1,0,0],'target_width_margin_m':width/2,'minimum_width_margin_m':.002,'minimum_axial_margin_m':.003,'maximum_face_error_m':.0004,'minimum_inward_normal_cosine':.866},
   contact_targets={role:{**p,'tool_body':side,'jaw_local_target_m':[sgn*.0005,0,z]} for role,p,side,sgn,z in zip(['thumb','index'],patches,['right','left'],[1,-1],fit['contact_z'])},
   geometry_validation={'open':audit,'closed_assumed_jaw_angle_rad':args.closed_angle,'closed':caudit,'physical_validation_pending':True},
   nominal_aperture={'definition':'Inner X-face gap at tip-box axial centre in root frame; actual dynamics must be measured','open_angle_rad':.05,'closed_angle_rad':args.closed_angle,'open_gap_m':.009+2*(length-.0075)*np.sin(.05)-.001*np.cos(.05),'closed_gap_m':.009+2*(length-.0075)*np.sin(args.closed_angle)-.001*np.cos(args.closed_angle),'maximum_joint_closure_change_rad':max(abs(v) for v in delta.values()),'closed_contact_axial_positions_m':closed.x[6:].tolist(),'closed_contact_fit_position_errors_m':(close_res(closed.x,args.closed_angle)[:6].reshape(2,3)/900).tolist()},source_fit=fit,
   limitations=['Geometric initialization, not a demonstrated free hold. Dynamic contacts must independently remain on each wide face with margins.','Source STL and cooked convex collision shapes can differ.','Middle/web are optional backups and are not asserted to carry load.','The straight tail has no bent extension or hidden support.','Penn reference constrains qualitative arrangement; monocular depth is not a 3-D target.'])
  for n in a.kin.hand:cfg['soft_bounds'][n]=[max(a.kin.by_name[n]['lower']+.02,min(cfg['soft_bounds'][n][0],q[n]-.04,qc[n]-.04)),min(a.kin.by_name[n]['upper']-.02,max(cfg['soft_bounds'][n][1],q[n]+.04,qc[n]+.04))]
  path=ROOT/f'configs/human_grasp_straight_face_pinch_{args.suffix}seed{seed}.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
  item={'config':str(path.relative_to(ROOT)),'seed':seed,'open_self_clear':audit['self_collision']['all_source_surfaces_disjoint'],'unexpected_open':audit['unexpected_tool_hand_intersections'],'min_open_table_clearance_m':audit['minimum_hand_table_clearance_m'],'tail_palm_gap_m':audit['straight_tail_palm_source_gap_m'],'closed_self_clear':caudit['self_collision']['all_source_surfaces_disjoint'],'unexpected_closed':caudit['unexpected_tool_hand_intersections']};out.append(item);print(json.dumps(item),flush=True)
 (ROOT/f'analysis/straight_face_pinch_{args.suffix}export_audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[12,13,17]);p.add_argument('--candidates',type=Path,default=ROOT/'analysis/straight_face_pinch_candidates.json');p.add_argument('--suffix',default='');p.add_argument('--closed-angle',type=float,default=.01);p.add_argument('--handle-length',type=float,default=.012);main(p.parse_args())
