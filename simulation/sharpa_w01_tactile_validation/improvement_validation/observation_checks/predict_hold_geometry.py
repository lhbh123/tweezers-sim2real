"""Independent CPU CPD predictions for archived hold; no simulation or GPU output."""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file())
sys.path.insert(0,str(ROOT/'analysis'))
from validate_sharpa_tacmap import closed_box_cpd,quantize_and_blur,ray_box_depth
OUT=Path(__file__).resolve().parent
run=ROOT/'sharpa_w01_tactile_validation/improvement_validation/runs/hold_soft_rest0'
old=ROOT/'runs/sharpa_w01_official_tacmap_matched_v2_video'
meta=json.loads(str(np.load(old/'tacmap.npz')['metadata_json']))
report=json.loads((run/'report.json').read_text());pose=json.loads((run/'source/soft_rest0.json').read_text())
oldpose=json.loads(Path(json.loads((old/'launch.json').read_text())['arguments']['pose']).read_text())
assert oldpose['tool_geometry']==pose['tool_geometry'], 'Cannot borrow legacy triangle boxes for changed geometry'
spec=importlib.util.spec_from_file_location('archived_kinematics',run/'source/kinematics.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
kin=module.Kinematics(Path(pose['robot_asset_dir'])/'robot.urdf',palm_link=pose['palm_link'],hand_root_link=pose['hand_root_link'])
d=np.load(run/'timeseries.npz');names=report['joint_names'];origins=np.asarray(report['env_origins'])
rows=[];stored={};fk_checks=[]
for env in [0,2,3]:
 name=report['candidates'][env]['name'];snap=json.loads((run/(name+'_snapshot.json')).read_text())
 q=dict(zip(names,d['q'][-1,env]));Ts=kin.fk(q);base=np.asarray(snap['arm_base_world_xyz'])+origins[env]
 wrist=Ts['wrist_3_link'].copy();wrist[:3,3]+=base
 expected=np.asarray(snap['world_wrist_pose_7']);wr=Rotation.from_quat(np.roll(expected[3:],-1)).as_matrix()
 check={'env':env,'position_error_um':float(np.linalg.norm(wrist[:3,3]-expected[:3]-origins[env])*1e6),'rotation_error_deg':float(Rotation.from_matrix(wrist[:3,:3]@wr.T).magnitude()*180/np.pi)};fk_checks.append(check)
 boxes=[];grouped=[]
 for target,target_pose in zip(meta['ray_targets'],d['bodies'][-1,env,:,:7]):
  r=Rotation.from_quat(np.roll(target_pose[3:],-1)).as_matrix();vv=np.asarray(target['vertices_body_m'],np.float32).astype(float);group=[]
  for j in range(0,len(vv),8):
   lo,hi=vv[j:j+8].min(0),vv[j:j+8].max(0);group.append({'center_m':((lo+hi)/2)@r.T+target_pose[:3],'half_extents_m':(hi-lo)/2,'rotation_matrix':r})
  boxes.extend(group);grouped.append(group)
 for k,finger in enumerate(['thumb','index']):
  cfg=meta['sensors'][k]['config'];points=np.load(cfg['points_npy']).astype(np.float32)*np.float32(cfg['correction_scale']);normals=np.load(cfg['normals_npy']).astype(np.float32);direction=-normals/(np.linalg.norm(normals,axis=-1,keepdims=True)+np.float32(1e-12));points+=np.float32(cfg['pts_offsets'])*direction
  T=Ts['right_'+finger+'_elastomer'];starts=points@T[:3,:3].T+T[:3,3]+base;dirs=direction@T[:3,:3].T
  depth=closed_box_cpd(starts,dirs,boxes,cfg['max_distance'],cfg['cpd_max_dist']);gray=quantize_and_blur(depth,1)
  mm=depth*np.float32(1000);pre=np.clip(np.where(mm<.5,mm/.005,(mm-.5)/.03+100),0,255).astype(np.uint8)
  separate=[closed_box_cpd(starts,dirs,b,cfg['max_distance'],cfg['cpd_max_dist']) for b in grouped]
  independent_single_body=np.maximum.reduce(separate)
  extra=(depth>0)&(independent_single_body==0)
  inside_any=np.zeros(depth.shape,bool)
  for box in boxes:
   local=(starts-box['center_m'])@box['rotation_matrix'];inside_any|=(np.abs(local)<=box['half_extents_m']).all(-1)
  first_by_body=np.stack([np.minimum.reduce([ray_box_depth(starts,dirs,b['center_m'],b['half_extents_m'],b['rotation_matrix'],cfg['max_distance']) for b in group]) for group in grouped])
  first=first_by_body.min(0);first_body=first_by_body.argmin(0);safe=np.where(np.isfinite(first),np.maximum(first,0)-1e-4,0)
  reverse_starts=starts+dirs*safe[...,None]
  reverse_by_body=np.stack([np.minimum.reduce([ray_box_depth(reverse_starts,-dirs,b['center_m'],b['half_extents_m'],b['rotation_matrix'],cfg['cpd_max_dist']) for b in group]) for group in grouped])
  reverse_body=reverse_by_body.argmin(0)
  selected_reverse=np.take_along_axis(reverse_by_body,first_body[None],axis=0)[0]
  same_body_depth=np.where(np.isfinite(first)&np.isfinite(selected_reverse),first,0).astype(np.float32)
  same_body_gray=quantize_and_blur(same_body_depth,1)
  row={'env':env,'candidate':name,'finger':finger,'time_s':float(d['time'][-1]),'raw_CPD_peak_um':float(depth.max()*1e6),'positive_depth_pixels':int((depth>0).sum()),'preblur_peak_code':int(pre.max()),'CPU_predicted_blur_peak_code':int(gray.max()),'normal_body_resultant_N':float(np.linalg.norm(d['role_normal_force_w'][-1,env,k])),'friction_body_resultant_N':float(np.linalg.norm(d['role_friction_force_w'][-1,env,k])),'assigned_face_inward_force_N':float(d['broad_face_inward_force_N'][-1,env,k,k]),'all_assigned_normal_inward_force_N':float(d['broad_face_all_normal_inward_force_N'][-1,env,k,k]),'single_body_CPD_peaks_um':{n:float(v.max()*1e6) for n,v in zip(['handle','left','right'],separate)},'positive_multi_target_CPD_without_any_single_body_CPD_pixels':int(((depth>0)&(independent_single_body==0)).sum())}
  stored[f'env{env}_{finger}_multi_target_only_mask']=extra;stored[f'env{env}_{finger}_same_body_reference_gray_u8']=same_body_gray;stored[f'env{env}_{finger}_raw_CPD_m']=depth;stored[f'env{env}_{finger}_predicted_gray_u8']=gray;rows.append(row)
  row['multi_target_only_points']={'count':int(extra.sum()),'outside_all_target_boxes_count':int((extra&~inside_any).sum()),'CPD_range_um':([float(depth[extra].min()*1e6),float(depth[extra].max()*1e6)] if extra.any() else []),'first_and_reverse_body_pairs':{a+'/'+b:int((extra&(first_body==i)&(reverse_body==j)).sum()) for i,a in enumerate(['handle','left','right']) for j,b in enumerate(['handle','left','right']) if (extra&(first_body==i)&(reverse_body==j)).any()},'same_body_reverse_reference_gray_peak':int(same_body_gray.max()),'same_body_reverse_reference_max_gray_change':int(abs(gray.astype(int)-same_body_gray.astype(int)).max()),'interpretation':'Diagnostic only; same-body reverse restriction changes the official criterion and is not applied to the saved official-style prediction.'}
np.savez_compressed(OUT/'hold_soft_rest0_CPU_predicted_maps.npz',**stored)
source_paths=[run/'timeseries.npz',run/'report.json',run/'source/kinematics.py',run/'source/soft_rest0.json',old/'tacmap.npz',Path(pose['robot_asset_dir'])/'robot.urdf',Path(__file__)]
result={'scope':'Offline CPU predictions using final archived joint/object states. This hold run did not record official GPU TacMap; these images are predictions, not observed GPU output.','method':'Full URDF FK plus archived identical tool box geometry; independent analytic slab intersections, original two-pass criterion, published quantization and SciPy Gaussian convolution.','geometry_matches_historical_tool_config':True,'finger_surface_directory':pose['tacmap_surface_directory'],'FK_vs_archived_wrist':fk_checks,'rows':rows,'sources_sha256':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},'limitations':['FK reconstructs surface poses from measured joint angles; it is checked against saved wrist pose but no exact live distal sensor pose was archived.','Body force data merge skin and shell contacts; these arrays cannot assign actual contact to one collider.','Geometric CPD is not calibrated skin compression or force.','Do not report these predicted images as official GPU observations or use them alone to declare a runtime algorithm pass.']}
(OUT/'hold_soft_rest0_CPU_prediction.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
fig,ax=plt.subplots(3,4,figsize=(13,10));vmax=max(v.max()*1e6 for key,v in stored.items() if 'CPD' in key)
for ri,env in enumerate([0,2,3]):
 for k,finger in enumerate(['thumb','index']):
  row=next(r for r in rows if r['env']==env and r['finger']==finger);a=ax[ri,k];im=a.imshow(stored[f'env{env}_{finger}_raw_CPD_m']*1e6,cmap='magma',vmin=0,vmax=vmax);a.set_title(f'env{env} {finger}: CPD {row["raw_CPD_peak_um"]:.1f} um\n{row["positive_depth_pixels"]} positive pixels');fig.colorbar(im,ax=a,fraction=.045)
  a=ax[ri,2+k];im=a.imshow(stored[f'env{env}_{finger}_predicted_gray_u8'],cmap='gray',vmin=0,vmax=255);a.set_title(f'CPU predicted gray {row["CPU_predicted_blur_peak_code"]}\nbody resultant {row["normal_body_resultant_N"]:.3f} N');fig.colorbar(im,ax=a,fraction=.045)
 for a in ax[ri]:a.set_xticks([]);a.set_yticks([])
fig.suptitle('Soft hold: offline CPU prediction at 6.0 s (no GPU TacMap recorded)',fontsize=16);fig.text(.5,.01,'Measured q/body poses + matched_v2 surface | same 0-255 gray scale | body forces do not identify skin vs shell',ha='center',fontsize=10);fig.tight_layout(rect=[0,.03,1,.96]);fig.savefig(OUT/'hold_soft_rest0_CPU_prediction.png',dpi=140);plt.close(fig)
print(json.dumps({'rows':rows,'FK_checks':fk_checks},indent=2))
