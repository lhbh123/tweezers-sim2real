"""Independent analytic box ray reference for archived single-blade experiments."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file())
sys.path.insert(0,str(ROOT/'analysis'))
from validate_sharpa_tacmap import closed_box_cpd,quantize_and_blur
OUT=Path(__file__).resolve().parent;results=[];selected=[]
for case in ['rigid_rest0','rigid_restneg','soft_rest0','soft_restneg']:
 run=ROOT/'sharpa_w01_tactile_validation/improvement_validation/single_blade/run_2x2'/case
 with np.load(run/'trajectory.npz') as archive:d={key:archive[key] for key in archive.files}
 with np.load(run/'geometry.npz') as archive:g={key:archive[key] for key in archive.files}
 points=g['surface_points_m'];normals=g['surface_normals'];dirs=-normals/(np.linalg.norm(normals,axis=-1,keepdims=True)+np.float32(1e-12));vv=g['blade_vertices_m'];lo,hi=vv.min(0),vv.max(0);half=(hi-lo)/2;center=(hi+lo)/2
 assert len(vv)==8 and all(np.all(np.isclose(v,lo)|np.isclose(v,hi)) for v in vv)
 gray_peak_frame=int(d['gray_u8'].max(axis=(-2,-1)).argmax());depth_peak_frame=int(d['raw_CPD_m'].max(axis=(-2,-1)).argmax());rows=[]
 for frame in range(len(d['time_s'])):
  sp=d['sensor_pose_w'][frame];bp=d['blade_pose_w'][frame];sr=Rotation.from_quat(np.roll(sp[3:],-1)).as_matrix();br=Rotation.from_quat(np.roll(bp[3:],-1)).as_matrix();box={'center_m':center@br.T+bp[:3],'half_extents_m':half,'rotation_matrix':br};cpd=closed_box_cpd(points@sr.T+sp[:3],dirs@sr.T,[box],.015,.5);gray=quantize_and_blur(cpd,1);rawerr=abs(cpd-d['raw_CPD_m'][frame]);err=abs(gray.astype(int)-d['gray_u8'][frame].astype(int))
  row={'frame':frame,'time_s':float(d['time_s'][frame]),'CPU_CPD_peak_um':float(cpd.max()*1e6),'saved_recomputed_GPU_CPD_peak_um':float(d['raw_CPD_m'][frame].max()*1e6),'CPU_gray_peak':int(gray.max()),'official_GPU_gray_peak':int(d['gray_u8'][frame].max()),'maximum_raw_depth_error_um':float(rawerr.max()*1e6),'maximum_gray_code_error':int(err.max()),'mean_gray_code_error':float(err.mean()),'pass_gray_le_1_code':bool(err.max()<=1),'pass_depth_le_1um':bool(rawerr.max()<=1e-6)};rows.append(row)
  if frame==depth_peak_frame:selected.append((case,row,d['gray_u8'][frame],gray,err))
 result={'case':case,'frames':len(rows),'gray_threshold_code':1,'depth_threshold_um':1,'strict_all_frame_pass':all(r['pass_gray_le_1_code'] and r['pass_depth_le_1um'] for r in rows),'maximum_raw_depth_error_um':max(r['maximum_raw_depth_error_um'] for r in rows),'maximum_gray_code_error':max(r['maximum_gray_code_error'] for r in rows),'failed_frames':[r for r in rows if not(r['pass_gray_le_1_code'] and r['pass_depth_le_1um'])],'peak_frames':{'gray':rows[gray_peak_frame],'CPD':rows[depth_peak_frame]},'all_frame_rows':rows,'input_sha256':{name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['geometry.npz','trajectory.npz','executed_script.py']}}
 results.append(result);print(json.dumps({k:result[k] for k in ['case','frames','strict_all_frame_pass','maximum_raw_depth_error_um','maximum_gray_code_error'] }),flush=True)
report={'scope':'Independent CPU analytic closed-box intersections against all 424 saved frames in each of four single-blade physics runs. No simulation launched.','method':'Float64 poses and analytic slab rays, official two-pass CPD criterion, published piecewise quantization, independent SciPy 9x9 sigma1.5 Gaussian.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['gray_u8 is the official sensor output. raw_CPD_m in the source experiment was independently re-run on GPU by its diagnostic function, not tapped from the actual quantizer input.','This is a constrained blade coupon with disabled gravity and a force-bounded drive; not a free-hand task or a force calibration.'],'cases':results}
(OUT/'single_blade_independent_CPU_validation.json').write_text(json.dumps(report,indent=2)+'\n')
fig,ax=plt.subplots(4,3,figsize=(10,12))
for i,(name,row,gpu,cpu,err) in enumerate(selected):
 for j,(array,title,cmap,vmax) in enumerate([(gpu,'Official GPU gray','gray',255),(cpu,'Analytic CPU gray','gray',255),(err,'Absolute difference','inferno',1)]):
  im=ax[i,j].imshow(array,cmap=cmap,vmin=0,vmax=vmax);ax[i,j].set_title(f'{name}: {title}\nframe {row["frame"]}, peak {int(array.max())}');ax[i,j].set_xticks([]);ax[i,j].set_yticks([]);fig.colorbar(im,ax=ax[i,j],fraction=.045)
fig.suptitle('Single-blade 2x2 contact: independent reference at peak CPD',fontsize=15);fig.tight_layout(rect=[0,.02,1,.96]);fig.text(.5,.01,'Gray scales fixed 0-255; difference fixed 0-1 code. Quantitative checks cover every frame.',ha='center',fontsize=10);fig.savefig(OUT/'single_blade_independent_CPU_validation.png',dpi=140);plt.close(fig)
