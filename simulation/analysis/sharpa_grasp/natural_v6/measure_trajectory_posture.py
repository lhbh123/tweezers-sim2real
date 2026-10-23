"""Quantify natural W01 posture in saved control states, using actual gravity."""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import NaturalFit,transform,module

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(values):
 a=np.asarray([x for x in values if x is not None],float)
 return {'count':len(a),'min':float(a.min()),'max':float(a.max()),'median':float(np.median(a))} if len(a) else {'count':0,'min':None,'max':None,'median':None}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--pose',type=Path);a=p.parse_args()
 metadata=json.loads((a.run/'task.json').read_text());pose=a.pose or a.run/'source/sharpa_w01_natural_v6b.json';cfg=json.loads(pose.read_text())
 f=NaturalFit();f.cfg=cfg;f.audit=module.SourceMeshAudit(metadata);f.kin=f.audit.kin
 with np.load(a.run/'trajectory.npz',allow_pickle=False)as raw:d={k:raw[k]for k in ['q','tool','left','right','active','origins','phase','dt']}
 began=time.monotonic();rows=[];summaries=[]
 for env in range(d['active'].shape[1]):
  ids=np.flatnonzero(d['active'][:,env]);ev=[];angles=collections.defaultdict(list)
  for step in ids:
   q=d['q'][step,env];states={k:d[k][step,env]for k in ['tool','left','right']};origin=d['origins'][env];bounds=f.audit.state(q,states,origin,True)
   qs=dict(zip(f.audit.names,q));fk=f.kin.fk(qs);world_palm=transform(origin+[0,0,.72])@fk[f.kin.palm_link]
   f.Wpalm=world_palm[:3,:3];f.T=np.linalg.inv(world_palm)@module.pose_transform(states['tool']);s=f.support(qs)
   overlap=f.audit.examine(bounds,f.middle_distal_pairs);s['source_triangle_intersections']=overlap['intersections']
   if overlap['intersections']:
    # BVH nearest points during triangle overlap are not reliable contact
    # witnesses. Keep independent source centers, but never interpret those
    # points as a physical support location or a negative clearance magnitude.
    s['distance_m']=0.;s['pair']=overlap['nearest_pair']
    for key in ['points_root_m','points_palm_m','points_tool_m','tool_minus_middle_world_m','vertical_gap_m','horizontal_gap_m','middle_below_tool','tool_side_y_m','tool_axial_z_m','middle_witness_below_index_contact_m']:s[key]=None
   s.update(environment=env,control_step=int(step),time_s=float((step+1)*d['dt']),phase=int(d['phase'][step,env]));ev.append(s)
   for j in qs:
    if any(x in j for x in ['middle','ring','pinky']):angles[j].append(float(qs[j]))
  rows.extend(ev)
  if not ev:continue
  witness=[r['middle_witness_below_index_contact_m']for r in ev if r['middle_witness_below_index_contact_m'] is not None]
  summaries.append({'environment':env,'control_states':len(ev),'source_middle_triangle_intersection_states':sum(bool(r['source_triangle_intersections'])for r in ev),'nearest_pair_counts':dict(collections.Counter('/'.join(r['pair'])for r in ev)),'middle_to_blade_source_gap_m':stats([r['distance_m']for r in ev]),'middle_witness_below_index_contact_m':stats(witness),'middle_DP_center_below_index_DP_center_m':stats([r['middle_DP_center_below_index_DP_center_m']for r in ev]),'middle_below_local_blade_vertical_gap_m':stats([r['vertical_gap_m']for r in ev]),'middle_witness_below_index_contact_fraction_among_nonintersecting_states':float(np.mean(np.asarray(witness)>0)) if witness else None,'middle_DP_center_below_index_DP_fraction':float(np.mean([r['middle_DP_center_below_index_DP_center_m']>0 for r in ev])),'joint_angle_ranges_rad':{j:stats(v)for j,v in angles.items()}})
 result={'method':__doc__,'source_trajectory_sha256':sha(a.run/'trajectory.npz'),'source_pose':str(pose),'source_pose_sha256':sha(pose),'source_URDF_sha256':sha(f.audit.urdf),'script_sha256':sha(__file__),'sampling':'Every saved active control state, actual world gravity direction; not all physics substeps','summary':summaries,'samples':rows,'wall_seconds':time.monotonic()-began,'limitations':['Nearest-source-mesh clearance and relative vertical placement do not establish physical middle support force.','Original source FK ignores joint solver compliance, and collision meshes may differ from PhysX convex contact.','The nominal index contact point is transformed using actual index joint angles; it is not a measured contact-force centroid.','Middle DP centers are transformed local source AABB centers, used only to quantify the visible stair-step posture.','If a source intersection occurs, FCL nearest points are not a physically meaningful support witness; consult the independent intersection audit.']}
 out=a.run/'natural_posture_geometry.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'output':str(out),'summary':summaries,'seconds':result['wall_seconds']},indent=2),flush=True)

if __name__=='__main__':main()
