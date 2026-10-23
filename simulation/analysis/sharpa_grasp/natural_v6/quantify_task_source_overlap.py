"""Exact directed source-surface/box depth at every intersecting active control state.

Uses actual saved middle sensor poses, not joint FK, and leaves the original
intersection audit untouched. See surface_box_depth for the precise metric.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import module
from quantify_probe_source_overlap import surface_box_depth

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();meta=json.loads((a.run/'task.json').read_text());audit=module.SourceMeshAudit(meta);actual=a.run/'actual_middle_source_mesh_audit.json';old=json.loads(actual.read_text());Tdp_skin=audit.kin.fk(root='right_middle_DP')['right_middle_elastomer']
 with np.load(a.run/'trajectory.npz',allow_pickle=False)as d:states={k:d[k]for k in ['tool','left','right','active','phase','dt']}
 with np.load(a.run/'tactile.npz',allow_pickle=False)as d:sensors=d['substep_sensor_pose_w']
 assert states['active'].shape[1]==1
 rows=[];began=time.monotonic()
 for oldrow in old['samples']:
  step=oldrow['control_step'];skin=module.pose_transform(sensors[(step+1)*int(meta['decimation'])-1,2]);poses={'right_middle_elastomer':skin,'right_middle_DP':skin@np.linalg.inv(Tdp_skin)}
  for hit in oldrow['intersections']:
   hand,tool=hit['pair'];Tb=module.pose_transform(states[audit.tool_body[tool]][step,0]);r=surface_box_depth(audit.meshes[hand],poses[hand],Tb,audit.tool_meshes[tool]);r.update(control_step=step,time_s=oldrow['time_s'],phase=oldrow['phase'],pair=[hand,tool]);rows.append(r)
 summary={str(phase):{'intersection_surface_checks':len(rs),'max_exact_source_surface_inside_box_depth_m':max(r['max_source_surface_inside_box_depth_m']for r in rs),'median_exact_source_surface_inside_box_depth_m':float(np.median([r['max_source_surface_inside_box_depth_m']for r in rs]))}for phase in sorted(set(r['phase']for r in rows))for rs in [[r for r in rows if r['phase']==phase]]}
 result={'method':__doc__,'metric':'Maximum over every point on the source triangle surface of min(box_half_extent - abs(point-box_center)); positive values quantify source surface inside the rectangular blade solid. Exact triangle-barycentric linear optimization, millimeter scaling, HiGHS solver.','source_actual_middle_audit_sha256':sha(actual),'trajectory_sha256':sha(a.run/'trajectory.npz'),'tactile_sha256':sha(a.run/'tactile.npz'),'script_sha256':sha(__file__),'geometry_helper_sha256':sha(Path(__file__).with_name('quantify_probe_source_overlap.py')),'active_control_states':old['control_states'],'intersecting_control_states':old['intersection_states'],'exact_intersecting_surface_checks':len(rows),'max_exact_surface_inside_box_depth_m':max((r['max_source_surface_inside_box_depth_m']for r in rows),default=0.),'by_phase':summary,'samples':rows,'wall_seconds':time.monotonic()-began,'limitations':['This is directed source-surface penetration into the rectangular tool volume, not intersection volume or rigid-body minimum separation translation.','The original source triangles differ from the simulator convex contact surface. This check neither silently exempts contacts nor changes any solver geometry.','Only active control endpoints are audited; the all-physics-step raw force classification is separate.']};out=a.run/'source_overlap_depth.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='samples'},indent=2),flush=True)
if __name__=='__main__':main()
