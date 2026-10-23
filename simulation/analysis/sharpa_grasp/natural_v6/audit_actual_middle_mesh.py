"""Cross-check middle distal source/tool geometry using saved actual sensor poses.

The tactile sensor is the elastomer fixed frame on the merged distal rigid body.
This check removes articulated joint FK error for DP/elastomer geometry only;
it complements, and does not replace, the all-hand source audit.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
import fcl
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import module

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();meta=json.loads((a.run/'task.json').read_text());audit=module.SourceMeshAudit(meta);Tdp_skin=audit.kin.fk(root='right_middle_DP')['right_middle_elastomer'];pairs=[(f,t)for f in ['right_middle_DP','right_middle_elastomer']for t in audit.tool_meshes]
 with np.load(a.run/'trajectory.npz',allow_pickle=False)as d:states={k:d[k]for k in ['tool','left','right','active','phase','dt']}
 with np.load(a.run/'tactile.npz',allow_pickle=False)as d:sensors=d['substep_sensor_pose_w'];times=d['substep_time_s']
 assert states['active'].shape[1]==1,'Current source tactile log covers one environment'
 rows=[];began=time.monotonic()
 for step in np.flatnonzero(states['active'][:,0]):
  i=(int(step)+1)*int(meta['decimation'])-1;skin=module.pose_transform(sensors[i,2]);poses={'right_middle_elastomer':skin,'right_middle_DP':skin@np.linalg.inv(Tdp_skin)};poses.update({t:module.pose_transform(states[audit.tool_body[t]][step,0])for t in audit.tool_meshes});bounds={}
  for key,T in poses.items():
   audit.objects[key].setTransform(fcl.Transform(T[:3,:3],T[:3,3]));corners=audit.bounds_corners.get(key,audit.tool_corners.get(key));points=corners@T[:3,:3].T+T[:3,3];bounds[key]=np.stack([points.min(0),points.max(0)])
  r=audit.examine(bounds,pairs);r.update(control_step=int(step),time_s=float(times[i]),phase=int(states['phase'][step,0]));rows.append(r)
 hits=[r for r in rows if r['intersections']];byphase={str(phase):{'control_states':len(rs),'intersection_states':sum(bool(r['intersections'])for r in rs),'minimum_source_gap_m':min(r['min_distance_m']for r in rs)}for phase in sorted(set(r['phase']for r in rows))for rs in [[r for r in rows if r['phase']==phase]]}
 result={'method':__doc__,'run':str(a.run),'trajectory_sha256':sha(a.run/'trajectory.npz'),'tactile_sha256':sha(a.run/'tactile.npz'),'script_sha256':sha(__file__),'sampling':'Every active control endpoint; actual middle sensor pose at the final physical substep of that control interval','control_states':len(rows),'intersection_states':len(hits),'minimum_source_gap_m':min(r['min_distance_m']for r in rows),'intersection_pair_counts':dict(collections.Counter('/'.join(h['pair'])for r in hits for h in r['intersections'])),'first_intersection_time_s':hits[0]['time_s']if hits else None,'last_intersection_time_s':hits[-1]['time_s']if hits else None,'by_phase':byphase,'samples':rows,'wall_seconds':time.monotonic()-began,'limitations':['Tests original source triangles; does not reproduce PhysX convex/deformable geometry.','FCL witness depths are triangle witnesses, not maximum volumetric penetration depths.','Only distal middle DP/elastomer geometry uses actual recorded poses here; all-hand self collision remains in the independent FK source audit.']};out=a.run/'actual_middle_source_mesh_audit.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['samples']},indent=2),flush=True)
if __name__=='__main__':main()
