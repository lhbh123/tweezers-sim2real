"""CPU checks of saved original GPU intermediates; does not start simulation."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file())
sys.path.insert(0,str(ROOT/'analysis'))
from validate_sharpa_tacmap import quantize_and_blur
parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();run=Path(args.run).resolve();out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(run/'tacmap.npz') as z:
 data={k:z[k] for k in ['frame_time_s','sample_control_step','deformation_uint8','diagnostic_first_ray_depth_m','diagnostic_reverse_ray_depth_m','diagnostic_raw_cpd_depth_m','diagnostic_preblur_uint8']};meta=json.loads(str(z['metadata_json']))
shape=data['deformation_uint8'].shape;rows=[];cpd_bad=[];quant_bad=[];blur_bad=[]
for frame in range(shape[0]):
 for k in range(shape[1]):
  first=data['diagnostic_first_ray_depth_m'][frame,k];second=data['diagnostic_reverse_ray_depth_m'][frame,k];raw=data['diagnostic_raw_cpd_depth_m'][frame,k];expected=np.where(np.isfinite(first)&np.isfinite(second),first,0)
  if not np.array_equal(expected,raw):cpd_bad.append([frame,k,int((expected!=raw).sum())])
  mm=raw*np.float32(1000);pre=np.clip(np.where(mm<np.float32(.5),mm/np.float32(.005),(mm-np.float32(.5))/np.float32(.03)+np.float32(100)),0,255).astype(np.uint8)
  if not np.array_equal(pre,data['diagnostic_preblur_uint8'][frame,k]):quant_bad.append([frame,k,int(abs(pre.astype(int)-data['diagnostic_preblur_uint8'][frame,k].astype(int)).max())])
  gray=quantize_and_blur(raw,meta['sensors'][k]['config']['resolution_step']);err=abs(gray.astype(int)-data['deformation_uint8'][frame,k].astype(int));maximum=int(err.max())
  if maximum:blur_bad.append([frame,k,maximum,float(err.mean())])
  rows.append({'frame':frame,'finger':meta['finger_names'][k],'time_s':float(data['frame_time_s'][frame]),'raw_CPD_max_um':float(raw.max()*1e6),'positive_CPD_pixels':int((raw>0).sum()),'preblur_peak_code':int(pre.max()),'official_gray_peak_code':int(data['deformation_uint8'][frame,k].max()),'blur_max_code_error':maximum})
 print('DIAGNOSTICS_CPU_PROGRESS',frame+1,'/',shape[0],flush=True) if frame%200==199 else None
force=json.loads((run/'tactile.json').read_text());mapping=[]
for k,sensor in enumerate(meta['sensors']):
 fs=force['surface_assets'][k];mapping.append({'finger':sensor['finger'],'point_equal':fs['point_sha256']==sensor['surface_assets'][0]['sha256'],'normal_equal':fs['normal_sha256']==sensor['surface_assets'][1]['sha256']})
launch=json.loads((run/'launch.json').read_text());source_rows=[]
for name,expected in launch['source_sha256'].items():
 archived=run/'source'/Path(name).name;live=Path(name) if Path(name).is_absolute() else ROOT/name
 source_rows.append({'path':name,'launch_sha256':expected,'archived_copy_sha256':sha(archived) if archived.exists() else None,'live_now_sha256':sha(live) if live.exists() else None,'archive_matches_launch':archived.exists() and sha(archived)==expected})
provenance=run/'execution_source_provenance.json'
summary={'run':str(run),'scope':'Saved GPU diagnostic consistency check. Original GPU ray depths and pre-quantization tensors are not independently ray-cast here.','frames':shape[0],'finger_frames':shape[0]*shape[1],'CPD_rule_exact':not cpd_bad,'CPD_rule_mismatches':cpd_bad,'preblur_quantization_exact':not quant_bad,'preblur_mismatches':quant_bad,'independent_Gaussian_max_code_error':max([r['blur_max_code_error'] for r in rows]),'independent_Gaussian_nonexact_frames':blur_bad,'gray_threshold_code':1,'strict_internal_diagnostics_pass':not cpd_bad and not quant_bad and max(r['blur_max_code_error'] for r in rows)<=1,'force_and_tacmap_surface_sources':mapping,'archive_all_copies_match_launch':all(r['archive_matches_launch'] for r in source_rows),'source_rows':source_rows,'execution_source_provenance':json.loads(provenance.read_text()) if provenance.exists() else {'status':'not_recorded','interpretation':'Run launched before boundary hash instrumentation. Matching launch snapshots does not by itself establish the exact live code imported during execution.'},'official_sources':meta['official_sources'],'source_sha256':{'validator':sha(__file__),'tacmap_npz':sha(run/'tacmap.npz'),'tactile_json':sha(run/'tactile.json')},'all_frame_rows':rows}
summary['finger_peaks']=[max([r for r in rows if r['finger']==name],key=lambda r:r['official_gray_peak_code']) for name in meta['finger_names']]
(out/(run.name+'_diagnostics_validation.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['run','frames','finger_frames','CPD_rule_exact','preblur_quantization_exact','independent_Gaussian_max_code_error','strict_internal_diagnostics_pass','archive_all_copies_match_launch','finger_peaks']},ensure_ascii=False),flush=True)
