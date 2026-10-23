"""Summarize observed W01 pick traces and isolate orientation-gate failures.

The optional wider-angle calculation is a diagnostic on the recorded trace,
never a replacement for rerunning a modified task or a counted success.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def longest(mask, dt):
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return float(np.max(edges[1::2] - edges[::2], initial=0) * dt)


def audit(run):
    result = json.loads((run/'evaluation.json').read_text())
    task = result['task']
    d = np.load(run/'trajectory.npz')
    dt = float(d['dt'])
    reports = []
    for i in range(result['episodes']):
        active = d['active'][:, i].astype(bool)
        phase = d['phase'][:, i]
        tool_angle = d['normal_angle'][:, i]
        root_tip = d['tip'][:, i] - d['tool'][:, i, :3]
        root_angle = np.degrees(np.arctan2(-root_tip[:, 2], np.linalg.norm(root_tip[:, :2], axis=-1)))
        palm_z = d['palm_normal'][:, i, 2]
        web_z = d['web_axis'][:, i, 2]
        frame_ok = np.ones_like(active)
        if task.get('require_palm_down'):
            frame_ok &= palm_z < -.05
        if task.get('require_web_up'):
            frame_ok &= web_z >= task['min_web_up_component']
        if task.get('max_palm_up_component') is not None:
            frame_ok &= palm_z <= task['max_palm_up_component']
        retained = (d['retained'][:, i] < .035) & (d['orientation_error'][:, i] < .3)
        face = d['face'][:, i].astype(bool)
        block = d['block'][:, i]
        place = task['place_config']
        stable_base = active & (phase == 9) & (d['goal_error'][:, i] < place['goal_tolerance_m'])
        stable_base &= np.abs(d['lift'][:, i]) < .0015
        stable_base &= np.linalg.norm(block[:, 7:10], axis=-1) < .01
        stable_base &= np.linalg.norm(block[:, 10:13], axis=-1) < .5
        stable_base &= np.max(d['forces'][:, i], axis=-1) < .003
        stable_base &= d['tip'][:, i, 2] - block[:, 2] > .035
        stable_base &= retained & face & frame_ok
        angles = {}
        for upper in sorted({float(task['tool_down_angle_range_deg'][1]), 85.}):
            lower = float(task['tool_down_angle_range_deg'][0])
            angle_ok = (tool_angle >= lower) & (tool_angle <= upper) & (root_angle >= lower) & (root_angle <= upper)
            lifting = active & (d['lift'][:, i] > .045) & d['both'][:, i].astype(bool)
            lifting &= retained & face & frame_ok & angle_ok
            angles[str(upper)] = {
                'lift_bilateral_valid_grasp_longest_s': longest(lifting, dt),
                'final_stability_longest_s_without_lift_history_gate': longest(stable_base & angle_ok, dt),
                'orientation_valid_fraction': float((angle_ok & frame_ok)[active].mean()),
            }
        last = np.flatnonzero(active)[-1]
        stats = lambda x: [float(x[active].min()), float(x[active].max())]
        old_upper = str(float(task['tool_down_angle_range_deg'][1]))
        gates = {
            'phase9': bool(phase[last] == 9),
            'lift_history': angles[old_upper]['lift_bilateral_valid_grasp_longest_s'] >= task['success_definition'].get('continuous_bilateral_lift_s', .25),
            'goal_xy': bool(d['goal_error'][last, i] < place['goal_tolerance_m']),
            'table_height': bool(abs(d['lift'][last, i]) < .0015),
            'linear_speed': bool(np.linalg.norm(block[last, 7:10]) < .01),
            'angular_speed': bool(np.linalg.norm(block[last, 10:13]) < .5),
            'no_jaw_contact': bool(np.max(d['forces'][last, i]) < .003),
            'tip_withdrawn': bool(d['tip'][last, i, 2] - block[last, 2] > .035),
            'retained': bool(retained[last]), 'face': bool(face[last]),
            'orientation': bool(d['orientation_ok'][last, i]),
        }
        phase_entries = {}
        for p, name in enumerate(task['phases']):
            selected = np.flatnonzero(active & (phase == p))
            if not len(selected):
                continue
            k = selected[0]
            phase_entries[name] = {'time_s': float((k+1)*dt), 'block_xy_w': block[k, :2].tolist(),
                                  'tip_xy_w': d['tip'][k, i, :2].tolist(),
                                  'goal_xy_error_m': float(d['goal_error'][k, i]),
                                  'lift_m': float(d['lift'][k, i]), 'tool_angle_deg': float(tool_angle[k]),
                                  'jaw_forces_N': d['forces'][k, i].tolist()}
        reports.append({
            'environment': i, 'recorded_success': bool(result['success_per_env'][i]),
            'duration_s': float(active.sum()*dt),
            'phase_duration_s': {name: float(np.count_nonzero(active & (phase == p))*dt) for p, name in enumerate(task['phases'])},
            'max_lift_m': float(d['lift'][active, i].max()),
            'final_xy_error_m': float(d['goal_error'][last, i]),
            'tool_angle_range_deg': stats(tool_angle), 'root_to_tip_angle_range_deg': stats(root_angle),
            'final_tool_angle_deg': float(tool_angle[last]), 'final_root_to_tip_angle_deg': float(root_angle[last]),
            'grip_error_range_m': stats(d['retained'][:, i]),
            'relative_rotation_error_range_sin_half_angle_squared': stats(d['orientation_error'][:, i]),
            'face_contact_fraction': float(face[active].mean()),
            'web_z_range': stats(web_z), 'palm_z_range': stats(palm_z),
            'final_gate_pass': gates, 'failed_final_gates': [name for name, value in gates.items() if not value],
            'final_gate_values': {'block_linear_speed_m_s': float(np.linalg.norm(block[last, 7:10])),
                                  'block_angular_speed_rad_s': float(np.linalg.norm(block[last, 10:13])),
                                  'jaw_forces_N': d['forces'][last, i].tolist(),
                                  'tip_above_block_m': float(d['tip'][last, i, 2] - block[last, 2]),
                                  'block_lift_m': float(d['lift'][last, i])},
            'phase_entries': phase_entries,
            'angle_gate_diagnostics': angles,
            'max_final_stability_excluding_absolute_angle_and_lift_history_s': longest(stable_base, dt),
        })
    successful_errors = np.asarray([e['final_xy_error_m'] for e in reports if e['recorded_success']])
    finite = {name: bool(np.isfinite(d[name]).all()) for name in d.files}
    return {'run': str(run), 'recorded_evaluation': str(run/'evaluation.json'),
            'task_angle_limits_deg': task['tool_down_angle_range_deg'], 'episodes': reports,
            'recorded_success_count': int(sum(e['recorded_success'] for e in reports)),
            'all_trajectory_arrays_finite': all(finite.values()),
            'nonfinite_arrays': [name for name, ok in finite.items() if not ok],
            'successful_final_xy_error_mm': {'min': float(successful_errors.min()*1000),
                                             'median': float(np.median(successful_errors)*1000),
                                             'max': float(successful_errors.max()*1000)} if len(successful_errors) else None,
            'warning': 'Any 85-degree calculation differing from the recorded configuration is counterfactual diagnostic only, not a successful experiment. Rerun the task to validate a changed configuration.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    text = json.dumps(result, indent=2)+'\n'
    if args.out:
        args.out.write_text(text)
    print(text)
