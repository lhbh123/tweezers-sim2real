"""Parallel exhaustive source audit with explicit gap-free physics-sample coverage."""
import argparse,concurrent.futures,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
task=json.loads((a.run/'task.json').read_text());z=np.load(a.run/'trajectory.npz');n=len(z['q'])*task['decimation'];z.close()
bounds=np.linspace(0,n,a.jobs+1,dtype=int).tolist()
def work(i):
 output=a.run/f'substep_geometry_part{i}.json'
 with (a.run/f'substep_geometry_part{i}.log').open('w') as log:
  cmd=[sys.executable,'analysis/audit_trained_tripod_geometry.py',str(a.run),'--sample-start',str(bounds[i]),'--sample-end',str(bounds[i+1]),'--output',str(output)]
  subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 return json.loads(output.read_text())
with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:parts=list(pool.map(work,range(a.jobs)))
for i,r in enumerate(parts):
 assert r['audited_sample_range']==bounds[i:i+2] and r['total_recorded_physics_samples']==n
 assert r['artifacts_sha256']==parts[0]['artifacts_sha256']
 assert r['nonadjacent_hand_self_collision']['checked_pair_instances']==r['audited_actual_samples']*r['nonadjacent_hand_self_collision']['pairs_per_sample']
robot_min=min(v['minimum_clearance_m'] for r in parts for v in r['moving_robot_table_clearance'].values())
tool_min=min(v['minimum_clearance_m'] for r in parts for v in r['actual_tool_box_table_clearance'].values())
checks=dict(all_samples_covered_exactly_once=sum(r['audited_actual_samples'] for r in parts)==n,
 all_joint_limits=all(r['checks']['all_18_joint_limits'] for r in parts),source_robot_above_table=robot_min>0,
 tool_table_penetration_below_0_1mm=tool_min>=-.0001,
 all_hand_nonadjacent_surfaces_disjoint=all(r['checks']['nonadjacent_hand_source_surfaces_disjoint'] for r in parts),
 all_orientations_within_task_bounds=all(r['checks']['all_recorded_substep_orientations_within_task_bounds'] for r in parts))
report=dict(run=str(a.run),physics_substeps=n,control_steps=n//task['decimation'],partition_bounds=bounds,checks=checks,
 minimum_robot_table_clearance_m=robot_min,minimum_tool_table_clearance_m=tool_min,
 contact_tolerance_m=.0001,contact_tolerance_note='Placement can touch the tabletop. Raw signed clearance is reported; penetration deeper than 0.1 mm fails. This is not a claim of strictly positive tool clearance.',
 hand_pair_instances=sum(r['nonadjacent_hand_self_collision']['checked_pair_instances'] for r in parts),source_artifact_sha256=parts[0]['artifacts_sha256'],limitations=parts[0]['limitations'])
(a.run/'substep_geometry_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert all(checks.values())
