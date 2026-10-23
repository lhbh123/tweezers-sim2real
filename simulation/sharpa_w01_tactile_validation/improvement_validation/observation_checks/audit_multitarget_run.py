"""Offline box-ray ambiguity diagnostics; preserves every official output unchanged."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file())
sys.path.insert(0,str(ROOT/'analysis'))
from validate_sharpa_tacmap import ray_box_depth,quantize_and_blur
P=argparse.ArgumentParser();P.add_argument('run');P.add_argument('--frames',nargs='+',type=int);P.add_argument('--fingers',nargs='+',default=['thumb','index','middle']);P.add_argument('--boundary-tolerance-um',type=float,default=1.);args=P.parse_args();run=Path(args.run).resolve();out=Path(__file__).resolve().parent/(run.name+'_ambiguity');out.mkdir(exist_ok=True)
with np.load(run/'tacmap.npz') as z:
 meta=json.loads(str(z['metadata_json']));names=[str(x) for x in z['finger_names']];poses=z['sensor_pose_w'];targetposes=z['ray_target_pose_w'];times=z['frame_time_s'];steps=z['sample_control_step'];images=z['deformation_uint8'];gpu_raw=z['diagnostic_raw_cpd_depth_m'];gpu_first=z['diagnostic_first_ray_depth_m'];gpu_reverse=z['diagnostic_reverse_ray_depth_m'];points=z['surface_points_m'].astype(np.float32);normals=z['surface_normals'].astype(np.float32)
phase=None
if (run/'trajectory.npz').is_file():
 with np.load(run/'trajectory.npz') as z:
  if 'phase' in z.files:phase=z['phase'][steps,0]
selected=set(args.frames or [0,len(times)-1])
if not args.frames:
 for finger in ['thumb','index']:
  selected.add(int(images[:,names.index(finger)].max(axis=(-1,-2)).argmax()))
 if phase is not None:
  for value in [4,5,6]:
   ix=np.flatnonzero(phase==value)
   if len(ix):selected.update([int(ix[0]),int(ix[len(ix)//2]),int(ix[-1])])
frames=sorted(selected);finger_indices=[names.index(n) for n in args.fingers];tol=args.boundary_tolerance_um*1e-6;rows=[];saved={};peak_records=[]
for frame in frames:
 grouped=[]
 for target,pose in zip(meta['ray_targets'],targetposes[frame]):
  r=Rotation.from_quat(np.roll(pose[3:],-1)).as_matrix();vv=np.asarray(target['vertices_body_m'],np.float32).astype(float);group=[]
  for j in range(0,len(vv),8):
   lo,hi=vv[j:j+8].min(0),vv[j:j+8].max(0);group.append({'center':((lo+hi)/2)@r.T+pose[:3],'half':(hi-lo)/2,'rotation':r})
  grouped.append(group)
 for k in finger_indices:
  cfg=meta['sensors'][k]['config'];normal=-normals[k]/(np.linalg.norm(normals[k],axis=-1,keepdims=True)+np.float32(1e-12));sp=poses[frame,k];sr=Rotation.from_quat(np.roll(sp[3:],-1)).as_matrix();starts=(points[k]+np.float32(cfg['pts_offsets'])*normal)@sr.T+sp[:3];directions=normal@sr.T
  def cast(s,n,limit):return np.stack([np.minimum.reduce([ray_box_depth(s,n,b['center'],b['half'],b['rotation'],limit) for b in group]) for group in grouped])
  fb=cast(starts,directions,cfg['max_distance']);first=fb.min(0);fi=fb.argmin(0);safe=np.where(np.isfinite(first),np.maximum(first,0.)-1e-4,0.);rb=cast(starts+safe[...,None]*directions,-directions,cfg['cpd_max_dist']);reverse=rb.min(0);ri=rb.argmin(0);kept=np.isfinite(first)&np.isfinite(reverse);cpd=np.where(kept,first,0.).astype(np.float32);same_reverse=np.take_along_axis(rb,fi[None],axis=0)[0];cross=kept&~np.isfinite(same_reverse)
  inside=np.zeros(first.shape,bool);outside_distance=np.full(first.shape,np.inf);inside_clearance=np.full(first.shape,-np.inf)
  for group in grouped:
   for b in group:
    local=(starts-b['center'])@b['rotation'];excess=np.abs(local)-b['half'];inside|=(excess<=0).all(-1);outside_distance=np.minimum(outside_distance,np.linalg.norm(np.maximum(excess,0),axis=-1));inside_clearance=np.maximum(inside_clearance,(-excess).min(-1))
  raw=gpu_raw[frame,k];positive=raw>0;boundary=np.where(inside,inside_clearance<=tol,outside_distance<=tol);outside=positive&~inside;robust_outside=outside&(outside_distance>tol);rawerr=abs(raw-cpd);matched=rawerr<=tol;pred=quantize_and_blur(cpd,cfg['resolution_step']);grayerr=abs(pred.astype(int)-images[frame,k].astype(int));raw_positive_cross=positive&cross
  def depth_comparison(cpu,gpu):
   both=np.isfinite(cpu)&np.isfinite(gpu);return {'finite_hit_mismatch_pixels':int(np.logical_xor(np.isfinite(cpu),np.isfinite(gpu)).sum()),'finite_both_max_distance_error_um':float(abs(cpu[both]-gpu[both]).max()*1e6) if both.any() else None}
  row={'frame':frame,'time_s':float(times[frame]),'control_step':int(steps[frame]),'phase_index':int(phase[frame]) if phase is not None else None,'finger':names[k],'raw_positive_pixels':int(positive.sum()),'GPU_raw_CPD_peak_um':float(raw.max()*1e6),'CPU_CPD_peak_um':float(cpd.max()*1e6),'max_CPD_error_um':float(rawerr.max()*1e6),'max_gray_error_code':int(grayerr.max()),'geometry_match_depth_le_tolerance_and_gray_le_1':bool(rawerr.max()<=tol and grayerr.max()<=1),'first_ray_comparison':depth_comparison(first,gpu_first[frame,k]),'reverse_ray_comparison':depth_comparison(reverse,gpu_reverse[frame,k]),'positive_outside_all_boxes_pixels':int(outside.sum()),'positive_outside_by_more_than_tolerance_pixels':int(robust_outside.sum()),'positive_boundary_uncertain_pixels':int((positive&boundary).sum()),'positive_first_reverse_different_nearest_body_pixels':int((positive&kept&(fi!=ri)).sum()),'positive_cross_target_only_pixels':int(raw_positive_cross.sum()),'positive_cross_target_only_with_matched_CPD_and_robust_outside_pixels':int((raw_positive_cross&matched&robust_outside).sum()),'cross_target_only_body_pairs':{str(i)+'/'+str(j):int((raw_positive_cross&(fi==i)&(ri==j)).sum()) for i in range(len(grouped)) for j in range(len(grouped)) if (raw_positive_cross&(fi==i)&(ri==j)).any()}}
  rows.append(row);prefix=f'frame{frame}_{names[k]}'
  for key,value in [('CPU_CPD_m',cpd),('GPU_positive_mask',positive),('CPU_inside_any_box',inside),('boundary_uncertain_mask',positive&boundary),('outside_robust_mask',robust_outside),('cross_target_only_mask',raw_positive_cross),('CPU_first_body_id',np.where(np.isfinite(first),fi,-1).astype(np.int8)),('CPU_reverse_body_id',np.where(np.isfinite(reverse),ri,-1).astype(np.int8))]:saved[prefix+'_'+key]=value
  peak_records.append((row,images[frame,k],raw,raw_positive_cross,robust_outside,positive&boundary))
np.savez_compressed(out/'diagnostic_masks.npz',**saved)
result={'scope':'Selected archived frames, independent analytic closed-box ray reconstruction; official arrays remain unchanged.','run':str(run),'boundary_tolerance_um':args.boundary_tolerance_um,'geometry_gray_threshold_code':1,'selected_frames':frames,'body_id_order':[t['body_path'] for t in meta['ray_targets']],'method':'Saved surface/pose -> analytic per-body forward/reverse intersections and exact box containment. Same-body reverse is a diagnostic condition only.','limitations':['World rays reconstructed in float64 from archived poses can differ slightly from original GPU float32 rays. Boundary pixels and CPU/GPU mismatches are reported, not silently resolved.','Different nearest forward/reverse bodies alone are not proof of invalid CPD: same-body reverse may also exist. cross_target_only requires absence of a same-body reverse intersection.','The masks are diagnostic evidence, never a default image repair or replacement for official raw CPD.','This checks target geometry only, not cooked skin collision shapes or force calibration.'],'rows':rows,'source_sha256':{'script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tacmap_npz':hashlib.sha256((run/'tacmap.npz').read_bytes()).hexdigest()}}
# Prefer strong diagnostic cases and distinct fingers; when there is no
# ambiguity, the fallback shows informative nonzero images, not empty reset.
ranked=sorted(peak_records,key=lambda r:(r[0]['positive_cross_target_only_with_matched_CPD_and_robust_outside_pixels'],r[0]['positive_outside_by_more_than_tolerance_pixels'],r[0]['GPU_raw_CPD_peak_um']),reverse=True)
chosen=[]
for entry in ranked:
 if not any(x[0]['finger']==entry[0]['finger'] for x in chosen):chosen.append(entry)
 if len(chosen)==2:break
result['plotted_cases']=[{'frame':x[0]['frame'],'finger':x[0]['finger']} for x in chosen]
(out/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
fig,ax=plt.subplots(len(chosen),4,figsize=(13,4*len(chosen)),squeeze=False)
for i,(row,gray,raw,cross,outside,boundary) in enumerate(chosen):
 for col,(value,title,cmap,maximum) in enumerate([(gray,'Official gray','gray',255),(raw*1e6,'Original GPU CPD (um)','magma',max(1,float(raw.max()*1e6))),(cross,'Independent cross-target-only mask','gray',1),(outside,'Positive origins outside >1 um','gray',1)]):
  im=ax[i,col].imshow(value,vmin=0,vmax=maximum,cmap=cmap);ax[i,col].set_title(f'{row["finger"]} f{row["frame"]}: {title}',fontsize=10);ax[i,col].set_xticks([]);ax[i,col].set_yticks([]);fig.colorbar(im,ax=ax[i,col],fraction=.045)
fig.suptitle(run.name+': diagnostic ambiguity masks; raw data unchanged',fontsize=14);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'diagnostic_masks.png',dpi=140);plt.close(fig)
print(json.dumps({'output':str(out),'selected_finger_frames':len(rows),'robust_cross_target_total':sum(r['positive_cross_target_only_with_matched_CPD_and_robust_outside_pixels'] for r in rows),'geometry_failures':sum(not r['geometry_match_depth_le_tolerance_and_gray_le_1'] for r in rows)},ensure_ascii=False))
