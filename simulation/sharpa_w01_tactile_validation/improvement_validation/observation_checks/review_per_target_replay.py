"""Independent selected-frame and semantics audit of the offline per-target output."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file());OUT=Path(__file__).resolve().parent
base=ROOT/'sharpa_w01_tactile_validation/improvement_validation';run=base/'runs/compliant_v1_full_validation_v2';folder=base/'per_target_replay/full_trajectory';r=json.loads((folder/'report.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
z=np.load(folder/'per_target_tacmap.npz');frames=z['source_frame_index'];names=z['finger_names'].tolist();depth=z['min_positive_cpd_depth_m'];gray=z['min_positive_gray_uint8'];checks=[]
for frame in [262,582,757]:
 output_frame=int(np.flatnonzero(frames==frame)[0]);ref=np.load(OUT/(run.name+'_frame'+str(frame))/'diagnostics.npz')
 for finger in ['thumb','index','middle']:
  k=names.index(finger)
  for field,actual,expected in [('CPD_m',depth[output_frame,k],ref[finger+'_per_body_pair_CPD_m']),('gray',gray[output_frame,k],ref[finger+'_per_body_pair_reference_gray'])]:
   checks.append({'frame':frame,'finger':finger,'field':field,'array_equal':bool(np.array_equal(actual,expected)),'maximum_absolute_difference':float(np.max(abs(actual.astype(float)-expected.astype(float))))})
packed=z['per_body_robust_outside_origin_mask_packbits'];popcount=np.array([int(i).bit_count() for i in range(256)],np.uint8);outside_by_finger={n:int(popcount[packed[:,k]].sum(dtype=np.int64)) for k,n in enumerate(names)}
source_hash=sha(run/'tacmap.npz')
spec=importlib.util.spec_from_file_location('reviewed_replay',folder/'source/replay_per_target_tacmap.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
box=lambda x,h:(np.array([x,0.,0.]),np.array([h,.001,.001]),np.eye(3))
cases=[('inside',[[box(0,.001)]],.001,.001),('air_gap',[[box(-.005,.0005)],[box(.005,.0005)]],0.,0.),('overlap',[[box(0,.001)],[box(.0005,.001)]],.001,.0015),('thin_epsilon',[[box(0,.00002)]],0.,0.),('intra_body_gap',[[box(-.005,.0005),box(.005,.0005)]],.0045,.0045)]
semantic=[]
for name,groups,mn,mx in cases:
 values=[float(mod.per_body_cpd(np.array([[0.,0.,0.]]),np.array([[1.,0.,0.]]),g,.015,.5)[0][0]) for g in groups];nonzero=[x for x in values if x>0];actual=[min(nonzero) if nonzero else 0,max(values)];passed=bool(np.allclose(actual,[mn,mx],rtol=0,atol=1e-9));assert passed;semantic.append({'case':name,'expected_min_max_m':[mn,mx],'actual_min_max_m':actual,'passed':passed})
summary={'scope':'Independent saved-frame reconstruction and explicit geometry semantics checks of the archived per-target implementation. No full trajectory recomputation or physics run performed here.','selected_frames':[262,582,757],'selected_frame_phase_labels':{'262':'lift','582':'lower','757':'settle_at_goal'},'selected_fingers':['thumb','index','middle'],'array_checks':checks,'all_18_arrays_bitwise_equal':all(c['array_equal'] for c in checks),'per_body_robust_outside_positive_pixel_frames_from_saved_masks':outside_by_finger,'saved_outside_masks_match_report':all(v==r['summary'][k]['per_body_robust_outside_positive_pixel_frames'] for k,v in outside_by_finger.items()),'original_tacmap_current_sha256':source_hash,'original_tacmap_hash_matches_replay_before_and_after':source_hash==r['source_tacmap_npz_sha256_before']==r['source_tacmap_npz_sha256'],'implementation_semantics_checks':semantic,'implementation_builtin_checks':mod.geometry_checks(),'interpretation':['The 9 independent finger-frame references and archived output match bitwise. This does not convert the CPU extension into official GPU output.','Per-body own-box outside-origin masks contain zero >1um positive pixel-frames throughout all 758x5x3 bodies in this replay, by independent popcount of the saved masks.','The 100um epsilon limitation and same-body disconnected-component ambiguity remain. No inside override was introduced.','The original official multi-target reconstruction remains strict-false on 11/39 sampled finger-frames as documented separately; this review does not erase those boundary cases.'],'source_sha256':{'review_script':sha(__file__),'per_target_archive':sha(folder/'per_target_tacmap.npz'),'per_target_report':sha(folder/'report.json')}}
(OUT/'per_target_replay_independent_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['all_18_arrays_bitwise_equal','per_body_robust_outside_positive_pixel_frames_from_saved_masks','saved_outside_masks_match_report','original_tacmap_hash_matches_replay_before_and_after']},indent=2))
