"""Join immutable source, actual-contact and exact-depth evidence for one video.

Review completeness is not a zero-penetration statement or a new contact filter.
Task success, force direction, source intersections, and geometry approximation
are reported separately.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import module

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
 x=np.asarray(x,float);return {'count':len(x),'min':float(x.min()),'max':float(x.max()),'median':float(np.median(x))}if len(x)else{'count':0}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();paths={k:a.run/v for k,v in {'source':'source_mesh_trajectory_audit.json','actual_middle':'actual_middle_source_mesh_audit.json','exact_depth':'source_overlap_depth.json','contact':'middle_contact_classification.json','source_part':'middle_contact_source_audit.json','posture':'natural_posture_geometry.json','evaluation':'evaluation.json','task':'task.json'}.items()};reports={k:json.loads(v.read_text())for k,v in paths.items()};source=reports['source'];actual=reports['actual_middle'];depth=reports['exact_depth'];meta=reports['task']
 with np.load(a.run/'trajectory.npz',allow_pickle=False)as d:active=d['active'];phases=d['phase'];dt=float(d['dt'])
 assert active.shape[1]==1,'Review currently covers single-video tactile provenance'
 allowed={(h,t)for h in ['right_middle_DP','right_middle_elastomer']for t in ['left_blade','right_blade']};source_by={r['control_step']:r for r in source['samples']};actual_by={r['control_step']:r for r in actual['samples']};exact_by={(r['control_step'],tuple(r['pair'])):r for r in depth['samples']};union=collections.defaultdict(set);other=[];self_rows=[]
 for r in source['samples']:
  if r['self']['intersections']:self_rows.append(r['control_step'])
  for h in r['tool']['nontarget_intersections']:
   pair=tuple(h['pair'])
   if pair not in allowed:other.append({'control_step':r['control_step'],'pair':list(pair)})
   else:union[r['control_step']].add(pair)
 for r in actual['samples']:
  for h in r['intersections']:
   pair=tuple(h['pair'])
   if pair not in allowed:other.append({'control_step':r['control_step'],'pair':list(pair),'pose_source':'actual_sensor'})
   else:union[r['control_step']].add(pair)
 raw_by={}
 for step,line in enumerate((a.run/'grasp_substeps.jsonl').open()):
  if step not in union:continue
  block=json.loads(line);raw_by[step]=block['substeps']
 rows=[];missing=[]
 for step in sorted(union):
  block=raw_by.get(step);sub=block[-1]if block else None;geometry=actual_by.get(step);observed=[]
  if sub:
   for channel,key in [('normal','normal_contacts'),('friction','friction_anchors')]:
    for c in sub['grasp_contact_points'][key]:
     if c['role']!='middle':continue
     tool=c['tool'];T=module.pose_transform(np.asarray(sub['tool'if tool=='handle'else tool]));local=np.asarray(c.get('point_jaw_local_m',T[:3,:3].T@(np.asarray(c['point_world'])-T[:3,3])));radial=float((T[:3,:3]@np.r_[local[:2],0.])[2]);observed.append({'channel':channel,'robot_merged_body':c['robot'],'tool_body':tool,'point_tool_local_m':local.tolist(),'radial_world_z_m':radial,'force_on_tool_N':c['force_on_tool_N'],'separation_m':c.get('separation_m')})
  for pair in sorted(union[step]):
   key=(step,pair);tool_body=pair[1].split('_')[0];matches=[c for c in observed if c['tool_body']==tool_body];lower=[c for c in matches if c['radial_world_z_m']<0];actual_pairs={tuple(h['pair'])for h in geometry['intersections']}if geometry else set();exact=exact_by.get(key);covered=geometry is not None and(pair not in actual_pairs or exact is not None)
   substep_rows=[]
   for sample in block or []:
    contacts=[c for name in ['normal_contacts','friction_anchors']for c in sample['grasp_contact_points'][name]if c['role']=='middle'and c['tool']==tool_body];substep_rows.append({'time_within_control_s':sample['time_within_control_s'],'middle_contact_records':len(contacts),'max_record_force_magnitude_N':max((float(np.linalg.norm(c['force_on_tool_N']))for c in contacts),default=0.)})
   if not covered:missing.append({'control_step':step,'pair':list(pair)})
   r={'control_step':step,'time_s':(step+1)*dt,'phase':int(phases[step,0]),'pair':list(pair),'source_FK_triangle_intersection':any(tuple(h['pair'])==pair for h in source_by[step]['tool']['nontarget_intersections']),'actual_sensor_pose_triangle_intersection':pair in actual_pairs,'exact_depth_covered':covered,'actual_source_surface_inside_tool_depth_m':exact['max_source_surface_inside_box_depth_m']if exact else(0. if covered else None),'matched_measured_middle_contact':bool(matches),'matched_lower_side_measured_contact':bool(lower),'net_tool_up_from_matching_middle_contacts_N':sum(c['force_on_tool_N'][2]for c in matches),'physx_normal_separation_m':stats([c['separation_m']for c in matches if c['separation_m']is not None]),'observed_middle_contacts':matches,'same_control_physics_substeps':substep_rows,'matched_any_physics_substep_middle_contact':any(x['middle_contact_records']for x in substep_rows)};rows.append(r)
 active_count=int(active.sum());coverage={'all_active_control_states':active_count,'source_exact_control_states':source['sampling']['exact_samples'],'actual_middle_control_states':actual['control_states'],'all_active_control_states_covered':source['sampling']['exact_samples']==active_count==actual['control_states'],'source_FK_middle_intersection_states':len({r['control_step']for r in rows if r['source_FK_triangle_intersection']}),'actual_pose_middle_intersection_states':actual['intersection_states'],'union_middle_intersection_states':len(union),'reviewed_intersection_pair_states':len(rows),'exact_depth_missing_pair_states':len(missing),'all_intersection_depths_covered':not missing,'matched_measured_middle_contact_pair_states':sum(r['matched_measured_middle_contact']for r in rows),'matched_lower_side_contact_pair_states':sum(r['matched_lower_side_measured_contact']for r in rows)}
 phase_summary={str(phase):{'reviewed_pair_states':len(rs),'max_exact_source_depth_m':max((r['actual_source_surface_inside_tool_depth_m']or 0.)for r in rs),'physx_normal_separation_m':stats([c['separation_m']for r in rs for c in r['observed_middle_contacts']if c['separation_m']is not None]),'net_tool_up_at_intersection_endpoints_N':stats([r['net_tool_up_from_matching_middle_contacts_N']for r in rs]),'all_matched_measured_lower_side':all(r['matched_lower_side_measured_contact']for r in rs)}for phase in sorted(set(r['phase']for r in rows))for rs in [[r for r in rows if r['phase']==phase]]}
 unmatched=[r for r in rows if not r['matched_measured_middle_contact']]
 missing_interval=[r for r in unmatched if not r['matched_any_physics_substep_middle_contact']]
 unmatched_summary={'endpoint_unmatched_states':len({r['control_step']for r in unmatched}),'all_8_physics_substeps_contact_free_states':len({r['control_step']for r in missing_interval}),'time_s':stats([r['time_s']for r in unmatched]),'source_depth_m':stats([r['actual_source_surface_inside_tool_depth_m']for r in unmatched]),'phase_pair_state_counts':dict(collections.Counter(str(r['phase'])for r in unmatched)),'endpoint_unmatched_but_contact_in_earlier_substep_states':[r['control_step']for r in unmatched if r['matched_any_physics_substep_middle_contact']]}
 urdf_matches=source['source_urdf']['matches_run_hash']is True
 total_states=set(self_rows)|{r['control_step']for r in other}|set(union)
 valid_reviewed_states={step for step in union if all(r['exact_depth_covered']and r['matched_lower_side_measured_contact']for r in rows if r['control_step']==step)}-set(self_rows)-{r['control_step']for r in other}
 complete=urdf_matches and coverage['all_active_control_states_covered']and not self_rows and not other and not missing and all(r['matched_lower_side_measured_contact']for r in rows)
 result={'schema_version':1,'method':__doc__,'review_complete':bool(complete),'source_urdf_matches_run_hash':urdf_matches,'reviewed_intersection_states':len(valid_reviewed_states),'total_intersection_states':len(total_states),'original_fk_tool_intersection_states':int(source['summary']['sampled_nontarget_tool_intersection_samples']),'zero_source_penetration_claim':False,'task_success':bool(reports['evaluation']['successes']==reports['evaluation']['episodes']),'self_intersection_states':len(set(self_rows)),'other_hand_tool_intersection_states':len({r['control_step']for r in other}),'other_hand_tool_intersections':other,'reviewed_middle_pairs':dict(collections.Counter('/'.join(r['pair'])for r in rows)),'coverage':coverage,'max_actual_source_surface_inside_tool_depth_m':depth['max_exact_surface_inside_box_depth_m'],'depth_metric':depth['metric'],'phase_summary':phase_summary,'independent_contact_force_summary':reports['contact']['physical_middle_support'],'independent_source_part_attribution':{'report':str(paths['source_part']),'sha256':sha(paths['source_part']),'interpretation':reports['source_part']['interpretation']},'source_reports':{k:{'path':str(v),'sha256':sha(v)}for k,v in paths.items()},'script_sha256':sha(__file__),'missing_exact_depth':missing,'unmatched_contact_summary':unmatched_summary,'intersection_pair_state_review':rows,'limitations':['review_complete means the listed intersection cases have been individually quantified and matched to measured lower-side middle contacts, while other source intersection classes are zero; it does not assert zero source penetration or an arbitrary acceptable depth.','Support force can change sign during unloading/retraction. Actual contact and positive upward support are not synonymous; endpoint and all-substep force summaries are retained.','A merged middle DP rigid body includes shell and elastomer. Independent actual-sensor-frame surface distances, not the rigid-body name, describe closest original source part.','The exact depth metric concerns source surface inside the blade box; it is not collision volume, material strain, or a modification of simulator collision filters.','Geometry coverage is every active control endpoint; measured contact force classification separately covers every recorded active physics substep.']};out=a.run/'contact_geometry_review.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='intersection_pair_state_review'},indent=2),flush=True)
if __name__=='__main__':main()
