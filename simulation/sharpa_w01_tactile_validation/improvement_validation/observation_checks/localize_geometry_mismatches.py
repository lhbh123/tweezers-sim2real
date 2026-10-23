"""Localize strict CPU/GPU ray discrepancies without relaxing thresholds."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file());sys.path.insert(0,str(ROOT/'analysis'))
from validate_sharpa_tacmap import ray_box_depth
OUT=Path(__file__).resolve().parent
run=ROOT/'sharpa_w01_tactile_validation/improvement_validation/runs/compliant_v1_full_validation_v2';folder=OUT/(run.name+'_ambiguity')
report=json.loads((folder/'report.json').read_text());pred=np.load(folder/'diagnostic_masks.npz');z=np.load(run/'tacmap.npz');meta=json.loads(str(z['metadata_json']));names=z['finger_names'].tolist();raw=z['diagnostic_raw_cpd_depth_m'];gpu_first=z['diagnostic_first_ray_depth_m'];gpu_rev=z['diagnostic_reverse_ray_depth_m'];rows=[]
for row in report['rows']:
 if row['geometry_match_depth_le_tolerance_and_gray_le_1']:continue
 frame=row['frame'];finger=row['finger'];k=names.index(finger);cpu=pred[f'frame{frame}_{finger}_CPU_CPD_m'];error=abs(cpu-raw[frame,k]);indices=np.argwhere(error>1e-6);cfg=meta['sensors'][k]['config']
 groups=[]
 for j,(target,pose) in enumerate(zip(meta['ray_targets'],z['ray_target_pose_w'][frame])):
  r=Rotation.from_quat(np.roll(pose[3:],-1)).as_matrix();vv=np.asarray(target['vertices_body_m'],np.float32).astype(float)
  for m in range(0,len(vv),8):
   lo,hi=vv[m:m+8].min(0),vv[m:m+8].max(0);groups.append({'body':j,'component':m//8,'center':((lo+hi)/2)@r.T+pose[:3],'half':(hi-lo)/2,'rotation':r})
 sp=z['sensor_pose_w'][frame,k];r=Rotation.from_quat(np.roll(sp[3:],-1)).as_matrix();ns=z['surface_normals'][k].astype(np.float32);ns=-ns/(np.linalg.norm(ns,axis=-1,keepdims=True)+np.float32(1e-12));pts=z['surface_points_m'][k].astype(np.float32)
 for ij0 in indices:
  ij=tuple(ij0);s=(pts[ij]+np.float32(cfg['pts_offsets'])*ns[ij])@r.T+sp[:3];n=ns[ij]@r.T
  hits=np.array([ray_box_depth(s,n,b['center'],b['half'],b['rotation'],cfg['max_distance']) for b in groups]);bi=int(hits.argmin());b=groups[bi];first=hits[bi];local=(s-b['center'])@b['rotation'];di=n@b['rotation'];hitlocal=local+first*di;axis=int(np.argmin(abs(abs(hitlocal)-b['half'])));face_cos=float(abs(di[axis]));source_dist=[]
  for box in groups:
   q=(s-box['center'])@box['rotation'];delta=np.abs(q)-box['half'];inside=(delta<=0).all();dist=float((-delta).min()) if inside else float(np.linalg.norm(np.maximum(delta,0)));source_dist.append({'body':box['body'],'component':box['component'],'inside':bool(inside),'distance_to_surface_um':dist*1e6})
  nearest=min(source_dist,key=lambda x:x['distance_to_surface_um'])
  category='near_start_surface_entry_exit_branch' if nearest['distance_to_surface_um']<.1 and error[ij]>1e-4 else 'grazing_ray_sensitive_distance' if face_cos<.05 else 'unresolved'
  rows.append({'frame':frame,'time_s':float(z['frame_time_s'][frame]),'finger':finger,'pixel_row_col':[int(x) for x in ij],'CPU_CPD_um':float(cpu[ij]*1e6),'GPU_CPD_um':float(raw[frame,k][ij]*1e6),'absolute_CPD_error_um':float(error[ij]*1e6),'CPU_first_ray_um':float(first*1e6),'GPU_first_ray_um':float(gpu_first[frame,k][ij]*1e6),'GPU_reverse_ray_um':float(gpu_rev[frame,k][ij]*1e6) if np.isfinite(gpu_rev[frame,k][ij]) else None,'CPU_first_hit_body':b['body'],'CPU_first_hit_component':b['component'],'CPU_first_hit_face_axis':axis,'absolute_ray_dot_hit_face_normal':face_cos,'nearest_start_surface':nearest,'classification':category,'world_start_m':s.tolist(),'world_direction':n.tolist()})
result={'scope':'Strict failures retained. This is a coordinate/conditioning diagnosis using reconstructed float64 rays, not exact replay of unrecorded GPU start/direction tensors.','depth_threshold_um':1,'gray_threshold_code':1,'sampled_finger_frames':len(report['rows']),'failed_finger_frames':sum(not r['geometry_match_depth_le_tolerance_and_gray_le_1'] for r in report['rows']),'pixels_above_depth_threshold':len(rows),'counts_by_class':{c:sum(r['classification']==c for r in rows) for c in sorted(set(r['classification'] for r in rows))},'rows':rows,'source_sha256':{'script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sample_report':hashlib.sha256((folder/'report.json').read_bytes()).hexdigest()}}
(folder/'strict_failure_localization.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
