"""Per-body closed-box CPD diagnostic, preserving official values unchanged."""
from pathlib import Path
import json,sys,hashlib,argparse
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file());sys.path.insert(0,str(ROOT/'analysis'))
from validate_sharpa_tacmap import ray_box_depth,quantize_and_blur
ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--frame',type=int,default=399);ap.add_argument('--fingers',nargs='+',default=['thumb','index']);args=ap.parse_args();run=Path(args.run).resolve();frame=args.frame;out=Path(__file__).resolve().parent/(run.name+'_frame'+str(frame));out.mkdir(exist_ok=True)
z=np.load(run/'tacmap.npz');meta=json.loads(str(z['metadata_json']));names=z['finger_names'].tolist();rows=[];arrays={}
groups=[]
for target,pose in zip(meta['ray_targets'],z['ray_target_pose_w'][frame]):
 r=Rotation.from_quat(np.roll(pose[3:],-1)).as_matrix();vv=np.asarray(target['vertices_body_m'],np.float32).astype(float);group=[]
 for j in range(0,len(vv),8):
  lo,hi=vv[j:j+8].min(0),vv[j:j+8].max(0);group.append({'center':((lo+hi)/2)@r.T+pose[:3],'half':(hi-lo)/2,'rotation':r})
 groups.append(group)
fig,ax=plt.subplots(len(args.fingers),4,figsize=(14,4*len(args.fingers)),squeeze=False)
for k,finger in enumerate(args.fingers):
 i=names.index(finger);cfg=meta['sensors'][i]['config'];points=z['surface_points_m'][i].astype(np.float32);normals=z['surface_normals'][i].astype(np.float32);normal=-normals/(np.linalg.norm(normals,axis=-1,keepdims=True)+np.float32(1e-12));sp=z['sensor_pose_w'][frame,i];sr=Rotation.from_quat(np.roll(sp[3:],-1)).as_matrix();s=(points+np.float32(cfg['pts_offsets'])*normal)@sr.T+sp[:3];n=normal@sr.T
 def onecast(st,di,group,limit):return np.minimum.reduce([ray_box_depth(st,di,b['center'],b['half'],b['rotation'],limit) for b in group])
 f=np.stack([onecast(s,n,g,cfg['max_distance']) for g in groups]);first=f.min(0);fi=f.argmin(0)
 q=s+np.where(np.isfinite(first),np.maximum(first,0)-1e-4,0)[...,None]*n
 rev=np.stack([onecast(q,-n,g,cfg['cpd_max_dist']) for g in groups]);reverse=rev.min(0);ri=rev.argmin(0)
 kept=np.isfinite(first)&np.isfinite(reverse);cpu=np.where(kept,first,0).astype(np.float32);ownreverse=np.take_along_axis(rev,fi[None],axis=0)[0];cross=kept&~np.isfinite(ownreverse);restricted=np.where(kept&np.isfinite(ownreverse),first,0).astype(np.float32)
 owncpds=[]
 for j,g in enumerate(groups):
  qj=s+np.where(np.isfinite(f[j]),np.maximum(f[j],0)-1e-4,0)[...,None]*n
  rj=onecast(qj,-n,g,cfg['cpd_max_dist']);owncpds.append(np.where(np.isfinite(f[j])&np.isfinite(rj),f[j],np.inf))
 owncpds=np.stack(owncpds);paired=owncpds.min(0);paired=np.where(np.isfinite(paired),paired,0).astype(np.float32)
 inside=np.zeros(first.shape,bool);distance=np.full(first.shape,np.inf);per_inside=[]
 for g in groups:
  ins=np.zeros(first.shape,bool)
  for b in g:
   delta=np.abs((s-b['center'])@b['rotation'])-b['half'];ins|=(delta<=0).all(-1);distance=np.minimum(distance,np.linalg.norm(np.maximum(delta,0),axis=-1))
  inside|=ins;per_inside.append(ins)
 raw=z['diagnostic_raw_cpd_depth_m'][frame,i];gray=z['deformation_uint8'][frame,i];positive=raw>0;matched=abs(cpu-raw)<=1e-6;robust=positive&cross&~inside&(distance>1e-6)&matched
 pairgray=quantize_and_blur(paired,cfg['resolution_step']);restrictedgray=quantize_and_blur(restricted,cfg['resolution_step']);cpu_gray=quantize_and_blur(cpu,cfg['resolution_step'])
 ij=np.unravel_index(np.argmax(np.where(robust,raw,-1)),raw.shape)
 example={'pixel_row_col':list(map(int,ij)),'GPU_CPD_um':float(raw[ij]*1e6),'CPU_CPD_um':float(cpu[ij]*1e6),'outside_distance_um':float(distance[ij]*1e6),'inside_by_body':[bool(v[ij]) for v in per_inside],'first_by_body_um':[float(v[ij]*1e6) if np.isfinite(v[ij]) else None for v in f],'reverse_from_global_first_by_body_um':[float(v[ij]*1e6) if np.isfinite(v[ij]) else None for v in rev],'first_body_id':int(fi[ij]),'reverse_body_id':int(ri[ij]),'per_body_paired_CPD_um':[float(v[ij]*1e6) if np.isfinite(v[ij]) else None for v in owncpds],'world_start_m':s[ij].tolist(),'world_direction':n[ij].tolist()}
 row={'finger':finger,'official_raw_peak_um':float(raw.max()*1e6),'official_gray_peak':int(gray.max()),'official_positive_pixels':int(positive.sum()),'positive_start_inside_any_box':int((positive&inside).sum()),'positive_start_outside_gt_1um':int((positive&~inside&(distance>1e-6)).sum()),'robust_cross_target_only_pixels':int(robust.sum()),'CPU_official_algorithm_max_error_um':float(abs(cpu-raw).max()*1e6),'CPU_official_algorithm_max_gray_error':int(abs(cpu_gray.astype(int)-gray.astype(int)).max()),'same_first_body_reverse_reference':{'positive_pixels':int((restricted>0).sum()),'raw_peak_um':float(restricted.max()*1e6),'gray_peak':int(restrictedgray.max())},'independent_per_body_pair_reference':{'positive_pixels':int((paired>0).sum()),'raw_peak_um':float(paired.max()*1e6),'gray_peak':int(pairgray.max()),'vs_restricted_first_max_difference_um':float(abs(paired-restricted).max()*1e6)},'example_peak_cross_target_pixel':example}
 if not robust.any():row['example_peak_cross_target_pixel']=None
 rows.append(row)
 for key,value in [('official_gray',gray),('official_raw_CPD_m',raw),('CPU_official_CPD_m',cpu),('cross_target_robust_mask',robust),('per_body_pair_CPD_m',paired),('per_body_pair_reference_gray',pairgray),('inside_any_box',inside)]:arrays[finger+'_'+key]=value
 for j,(val,title,cm,vmax) in enumerate([(gray,'Official raw image','gray',255),(robust,'Cross-target-only diagnostic mask','gray',1),(pairgray,'Per-body-pair reference (not applied)','gray',255),(abs(gray.astype(int)-pairgray.astype(int)),'Official vs reference absolute delta','magma',255)]):
  im=ax[k,j].imshow(val,cmap=cm,vmin=0,vmax=vmax);ax[k,j].set_title(finger+': '+title,fontsize=10);ax[k,j].set_xticks([]);ax[k,j].set_yticks([]);fig.colorbar(im,ax=ax[k,j],fraction=.045)
report={'run':str(run),'frame':frame,'time_s':float(z['frame_time_s'][frame]),'body_order':[x['body_path'] for x in meta['ray_targets']],'depth_match_threshold_um':1.,'gray_match_threshold_code':1,'method':'Independent analytic closed boxes at saved poses. Global two-pass official logic compared with reverse restricted to global first body, and separately with a forward/reverse pair per body (minimum accepted positive CPD across bodies).','scope':'Diagnostic references only; no official data, controller, physics, or sensor algorithm is changed. Raw peak differences are not force estimates or established physical calibration.','rows':rows,'sources_sha256':{'script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tacmap_npz':hashlib.sha256((run/'tacmap.npz').read_bytes()).hexdigest()}}
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(out/'diagnostics.npz',**arrays)
fig.suptitle(f'Frame {frame}, t={report["time_s"]:.4f} s: cross-target CPD ambiguity; original outputs preserved',fontsize=14);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'diagnostic_comparison.png',dpi=150);print(json.dumps(report,ensure_ascii=False,indent=2))
