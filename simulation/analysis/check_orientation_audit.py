"""Adversarial checks of the independent audit using a real saved preview.

Mutations stay in memory. No trajectory or simulated result is altered on disk.
"""
from pathlib import Path
from copy import deepcopy
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from validate_rl_run import orientation_audit,tool_geometry_audit


def main():
    run=ROOT/'runs/natural_orientation_pose_preview'
    rows=json.loads((run/'trajectory.json').read_text())
    task=json.loads((run/'task.json').read_text())
    launch=json.loads((run/'launch.json').read_text())
    usd=json.loads((run/'rl_validation.json').read_text())['USD_structural_evidence']
    active=np.ones(len(rows),dtype=bool)
    outcome={}
    good,checks,evidence=orientation_audit(rows,task,usd,active)
    assert good.all() and all(checks.values())
    outcome['actual_preview_matches_independent_FK_and_both_axes']='pass'

    changed=deepcopy(rows);changed[3]['orientation_acceptable']=False
    good,checks,_=orientation_audit(changed,task,usd,active)
    assert good.all() and not checks['orientation_pass_flag_agrees_with_independent_geometry']
    outcome['lying_pass_flag_is_not_used_as_geometry']='pass'

    changed=deepcopy(rows);changed[3]['palm_normal_w'][2]*=-1
    good,checks,_=orientation_audit(changed,task,usd,active)
    assert good.all() and not checks['independent_orientation_matches_palm_normal_w']
    outcome['tampered_palm_vector_rejected']='pass'

    changed=deepcopy(rows)
    older=json.loads((ROOT/'runs/rl_distal_open250_render_final/trajectory.json').read_text())[0]
    changed[3]['joint_pos']=older['joint_pos']
    good,checks,_=orientation_audit(changed,task,usd,active)
    assert not good[3] and not checks['palm_faces_down_in_all_active_samples']
    outcome['actual_upward_palm_rejected_despite_logged_downward_normal']='pass'

    changed=deepcopy(rows);quat=Rotation.from_euler('y',165,degrees=True).as_quat()
    for name in ['left','right']:changed[3][name][3:7]=np.r_[quat[3],quat[:3]].tolist()
    good,checks,_=orientation_audit(changed,task,usd,active)
    assert not good[3] and not checks['both_tool_axes_in_angle_range_in_all_active_samples']
    outcome['actual_steep_jaws_rejected_even_with_good_handle_angle']='pass'

    changed=deepcopy(rows);del changed[0]['left']
    good,checks,_=orientation_audit(changed,task,usd,active)
    assert not checks['orientation_telemetry_and_actual_jaw_states_complete']
    outcome['required_actual_jaw_state_cannot_be_omitted']='pass'

    checks,_=tool_geometry_audit(task,launch,usd)
    assert all(checks.values())
    wrong=deepcopy(task);wrong['tool_geometry']['length_m']+=.01
    checks,_=tool_geometry_audit(wrong,launch,usd)
    assert not checks['USD_tool_collision_dimensions_match_task_geometry']
    outcome['metadata_length_change_without_USD_geometry_rejected']='pass'

    old_run=ROOT/'runs/rl_distal_open250_render_final'
    old_rows=json.loads((old_run/'trajectory.json').read_text())
    old_task=json.loads((old_run/'task.json').read_text())
    old_usd=json.loads((old_run/'rl_validation.json').read_text())['USD_structural_evidence']
    good,checks,evidence=orientation_audit(old_rows,old_task,old_usd,np.ones(len(old_rows),dtype=bool))
    assert good.all() and all(checks.values()) and not evidence['orientation_requirements_enabled']
    outcome['legacy_run_does_not_gain_unrecorded_orientation_requirements']='pass'
    result=dict(tests=outcome,count=len(outcome),all_passed=True,
                mutation_scope='In-memory copies only; original physical trajectories unchanged.',
                real_preview='runs/natural_orientation_pose_preview')
    (ROOT/'reports/orientation_audit_regression.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
