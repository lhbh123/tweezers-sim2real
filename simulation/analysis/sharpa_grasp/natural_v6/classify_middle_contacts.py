"""Classify measured middle contacts without equating merged DP with skin.

Tool-side and force signs come from raw PhysX substeps. Original source-part
proximity uses sensor-local points and the fixed DP-to-elastomer transform,
avoiding articulated joint FK errors in that distance attribution.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import NaturalFit,module

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):
 x=np.asarray(v,float)
 return {'count':len(x),'min':float(x.min()),'max':float(x.max()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95))}if len(x)else{'count':0}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();meta=json.loads((a.run/'task.json').read_text());dt=meta['physics_dt']
 normal=[];friction=[];steps=[];by_phase=collections.defaultdict(list)
 with np.load(a.run/'trajectory.npz',allow_pickle=False)as d:phases=d['phase'][:,0];active=d['active'][:,0]
 for control,line in enumerate((a.run/'grasp_substeps.jsonl').open()):
  block=json.loads(line)
  if control>=len(active)or not active[control]:continue
  for sub in block['substeps']:
   points=sub['grasp_contact_points'];row={'phase':int(phases[control]),'normal_up_N':0.,'friction_up_N':0.,'normal_contact_count':0,'friction_anchor_count':0}
   for channel,key,dest in [('normal','normal_contacts',normal),('friction','friction_anchors',friction)]:
    for c in points[key]:
     if c['role']!='middle':continue
     tool=c['tool'];pose=module.pose_transform(np.asarray(sub['tool'if tool=='handle'else tool]));R=pose[:3,:3]
     local=np.asarray(c['point_jaw_local_m'])if 'point_jaw_local_m'in c else R.T@(np.asarray(c['point_world'])-pose[:3,3])
     radial=float((R@np.array([local[0],local[1],0.]))[2]);force=np.asarray(c['force_on_tool_N']);record={'tool':tool,'robot_rigid_body':c['robot'],'jaw_local_m':local.tolist(),'radial_world_z_m':radial,'force_up_N':float(force[2]),'force_magnitude_N':float(np.linalg.norm(force)),'below_jaw_axis':radial<0,'positive_y_side':bool(local[1]>0),'phase':int(phases[control])}
     if 'separation_m'in c:record['separation_m']=c['separation_m']
     dest.append(record);row[channel+'_up_N']+=force[2];row['normal_contact_count'if channel=='normal'else'friction_anchor_count']+=1
   row['net_up_N']=row['normal_up_N']+row['friction_up_N'];steps.append(row);by_phase[row['phase']].append(row)
 f=NaturalFit();T=f.kin.fk(root='right_middle_DP')['right_middle_elastomer'];records={}
 with np.load(a.run/'tactile.npz',allow_pickle=False)as d:
  tactile_meta=json.loads(str(d['metadata_json']))
  external=a.run/'middle_contact_source_audit.json'
  if external.exists():
   part=json.loads(external.read_text());records=dict(part,independent_source_report=str(external),sha256=sha(external))
  else:
   for key in ['normal','friction']:
    r=d[key+'_records'];r=r[r[:,1]==2];control_ids=(r[:,0]//meta['decimation']).astype(int);keep=(control_ids<len(active));keep[keep]=active[control_ids[keep]];r=r[keep]
    p=r[:,6:9];pdp=p@T[:3,:3].T+T[:3,3];dist={}
    for name,points in [('right_middle_elastomer',p),('right_middle_DP',pdp)]:
     all_distance=[]
     for begin in range(0,len(points),512):all_distance.extend(trimesh.proximity.closest_point(f.audit.meshes[name],points[begin:begin+512])[1].tolist())
     dist[name]=np.asarray(all_distance)
    w=np.linalg.norm(r[:,9:12],axis=1);delta=dist['right_middle_elastomer']-dist['right_middle_DP'];labels=np.where(delta>.0001,'DP_source_closer',np.where(delta<-.0001,'elastomer_source_closer','within_0.1mm_ambiguous'))
    records[key]={'records':len(r),'source_DP_distance_m':stats(dist['right_middle_DP']),'source_elastomer_distance_m':stats(dist['right_middle_elastomer']),'proximity_class_counts':dict(collections.Counter(labels)),'force_magnitude_weighted_class_fractions':{label:float(w[labels==label].sum()/w.sum())if w.sum()>0 else None for label in ['DP_source_closer','elastomer_source_closer','within_0.1mm_ambiguous']},'old_taxel_map_distance_m':stats(r[:,14]),'mapped_record_fraction':float(np.mean(r[:,13]>=0))if len(r)else None}
 def contact_summary(rows):
  w=np.asarray([r['force_magnitude_N']for r in rows]);below=np.asarray([r['below_jaw_axis']for r in rows]);local=np.asarray([r['jaw_local_m']for r in rows]).reshape(-1,3)
  return {'count':len(rows),'body_pairs':dict(collections.Counter(r['robot_rigid_body']+'/'+r['tool']for r in rows)),'below_jaw_axis_fraction':float(below.mean())if len(rows)else None,'force_weighted_below_jaw_axis_fraction':float(w[below].sum()/w.sum())if w.sum()>0 else None,'jaw_x_m':stats(local[:,0]),'jaw_y_m':stats(local[:,1]),'jaw_z_m':stats(local[:,2]),'radial_world_z_m':stats([r['radial_world_z_m']for r in rows]),'source_solver_separation_m':stats([r['separation_m']for r in rows if 'separation_m'in r])}
 def step_summary(rows):return {'physics_steps':len(rows),'contact_fraction':float(np.mean([r['normal_contact_count']>0 for r in rows])),'positive_net_up_fraction':float(np.mean([r['net_up_N']>.003 for r in rows])),'mean_net_up_N':float(np.mean([r['net_up_N']for r in rows])),'normal_up_impulse_Ns':float(sum(r['normal_up_N']for r in rows)*dt),'friction_up_impulse_Ns':float(sum(r['friction_up_N']for r in rows)*dt),'net_up_N':stats([r['net_up_N']for r in rows])}
 result={'method':__doc__,'source_grasp_substeps_sha256':sha(a.run/'grasp_substeps.jsonl'),'source_tactile_sha256':sha(a.run/'tactile.npz'),'script_sha256':sha(__file__),'physics_dt_s':dt,'source_fixed_DP_to_elastomer':T.tolist(),'physical_middle_support':step_summary(steps),'by_phase':{str(k):step_summary(v)for k,v in by_phase.items()},'normal_contact_geometry':contact_summary(normal),'friction_anchor_geometry':contact_summary(friction),'contact_geometry_by_phase':{str(phase):{'normal':contact_summary([r for r in normal if r['phase']==phase]),'friction':contact_summary([r for r in friction if r['phase']==phase])}for phase in sorted(by_phase)},'source_part_proximity':records,'tactile_sensor_body_paths':tactile_meta.get('sensor_paths'),'limitations':['Source-part labels indicate closest original source surface, not the exact collider or hardware sensing element.','Merged DP rigid body includes the elastomer; its name alone is not skin attribution.','Taxel mapping is approximate nearest-surface projection; it does not prove a hardware tactile sensor measured the contact.','Normal/friction force signs reported here are on the tool from raw grasp contacts; source-distance weighting uses tactile force magnitudes only.','Physics substep records are filtered to active control intervals.','Raw grasp contact logging omits forces below 1e-8 N, whereas tactile records retain forces above 1e-9 N; near-zero record counts can differ without a meaningful force discrepancy.']};out=a.run/'middle_contact_classification.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'output':str(out),'physical_middle_support':result['physical_middle_support'],'normal_geometry':result['normal_contact_geometry'],'source_part_proximity':records},indent=2),flush=True)

if __name__=='__main__':main()
