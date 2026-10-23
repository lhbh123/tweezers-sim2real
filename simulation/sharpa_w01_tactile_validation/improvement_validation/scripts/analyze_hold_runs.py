"""Read saved probe outputs, independently recompute stability and draw comparisons.

CPU only. Does not import or change the simulator. The recorded contact forces
are reused; this script does not independently solve PhysX or derive pressure.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_preheld_candidate import load_limits, limit_audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pose(run, report):
    name = Path(report['arguments']['pose']).name
    source = run / 'source' / name
    if not source.exists(): source = Path(report['arguments']['pose'])
    return source, json.loads(source.read_text())


def quaternion_range_deg(q):
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.rad2deg(2 * np.arccos(np.clip(np.abs(q @ q[-1]), 0, 1)))


def analyze_run(run, out):
    run = Path(run).resolve()
    report_path = run / 'report.json'
    data_path = run / 'timeseries.npz'
    report = json.loads(report_path.read_text())
    data = np.load(data_path)
    pose_path, pose = read_pose(run, report)
    names = report['joint_names']
    urdf, bounds = load_limits(pose, names)
    t = data['time'].astype(float).ravel()
    # Match the probe's explicit final-window rule, including short cold runs.
    final_duration = min(1., report['arguments']['settle_seconds'] or float(t[-1]))
    tail = t >= float(t[-1]) - final_duration - 1e-6
    assignment = report['face_pinch_assignment']
    point_mode = report.get('face_definition', {}).get('contact_point_mode', 'raw')
    pairs = [(['left', 'right'].index(assignment[role]), ['thumb', 'index'].index(role))
             for role in ('thumb', 'index')]
    finite = all(np.isfinite(data[k]).all() for k in data.files)
    results = []
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, .9, max(len(report['candidates']), 2)))
    for i, candidate in enumerate(report['candidates']):
        name = candidate['name']
        body = data['bodies'][:, i, 0]
        position_range = np.ptp(body[tail, :3], axis=0)
        angle = quaternion_range_deg(body[tail, 3:7])
        inward = np.stack([data['broad_face_inward_force_N'][:, i, j, r] for j, r in pairs], axis=-1)
        both = (inward > .003).all(-1)
        retained = bool((data['grip_error_m'][:, i] < .035).all()
                        and (data['grip_orientation_error'][:, i] < .3).all()
                        and data['tip_clearance_bound_m'][:, i].min() > .005
                        and data['cube_contact_force_N'][:, i].max() < 1e-6)
        stable = bool(retained and position_range.max() < .001 and angle.max() < 1.)
        broad = bool(stable and both[tail].mean() >= .95)
        target_limits = limit_audit(data['q_target'][:, i], bounds, names)
        actual_limits = limit_audit(data['q'][:, i], bounds, names)
        tracking = data['q_target'][:, i] - data['q'][:, i]
        speed = np.linalg.norm(body[tail, 7:10], axis=-1)
        angular_speed = np.linalg.norm(body[tail, 10:13], axis=-1)
        result = {
            'run': str(run), 'candidate': name, 'environment_index': i,
            'input_report_retained': candidate['retained'],
            'input_report_stable_hold': candidate['stable_hold'],
            'input_report_broad_face_hold': candidate['broad_face_hold'],
            'recomputed_retained': retained, 'recomputed_stable_hold': stable,
            'recomputed_broad_face_hold': broad,
            'all_recorded_arrays_finite': bool(finite),
            'recorded_physics_time_s': [float(t[0]), float(t[-1])],
            'tail_duration_s': final_duration, 'tail_frame_count': int(tail.sum()),
            'tail_position_range_mm': (position_range * 1e3).tolist(),
            'tail_rotation_to_final_max_deg': float(angle.max()),
            'tail_linear_speed_max_mm_s': float(speed.max() * 1e3),
            'tail_angular_speed_max_rad_s': float(angular_speed.max()),
            'tail_both_assigned_face_fraction': float(both[tail].mean()),
            'tail_assigned_inward_mean_N': inward[tail].mean(0).tolist(),
            'tail_assigned_inward_min_N': inward[tail].min(0).tolist(),
            'tail_gap_mean_mm': float(data['tip_gap_m'][tail, i].mean() * 1e3),
            'tail_gap_range_mm': float(np.ptp(data['tip_gap_m'][tail, i]) * 1e3),
            'tail_joint_tracking_rms_rad': float(np.sqrt(np.mean(tracking[tail] ** 2))),
            'tail_joint_tracking_max_abs_rad': float(np.abs(tracking[tail]).max()),
            'full_run_target_limits': target_limits,
            'full_run_actual_limits': actual_limits,
            'full_run_grip_error_max_mm': float(data['grip_error_m'][:, i].max() * 1e3),
            'full_run_min_tip_clearance_bound_mm': float(data['tip_clearance_bound_m'][:, i].min() * 1e3),
            'full_run_max_cube_contact_force_N': float(data['cube_contact_force_N'][:, i].max()),
            'final_tool_angle_deg': float(data['tool_angle_deg'][-1, i]),
        }
        result['independent_diagnostic_pass'] = bool(
            finite and broad and target_limits['max_violation_rad'] <= 1e-6)
        result['report_gate_agreement'] = bool(retained == candidate['retained']
                                              and stable == candidate['stable_hold']
                                              and broad == candidate['broad_face_hold'])
        snapshot = run / (name + '_snapshot.json')
        if snapshot.exists():
            snap = json.loads(snapshot.read_text())
            source = snap.get('zero_velocity_variant', snap)
            result['snapshot_sha256'] = sha(snapshot)
            result['snapshot_q_actual_matches_last_frame_max_abs_rad'] = float(
                np.max(np.abs(np.asarray(source['q_actual']) - data['q'][-1, i])))
            result['snapshot_q_target_matches_last_frame_max_abs_rad'] = float(
                np.max(np.abs(np.asarray(source['q_target']) - data['q_target'][-1, i])))
            result['snapshot_zero_velocity_length_correct'] = len(source.get('q_velocity', [])) == len(names)
        results.append(result)
        color = colors[i % len(colors)]
        label = name + (' [PASS]' if result['independent_diagnostic_pass'] else ' [FAIL]')
        axs[0, 0].plot(t, data['grip_error_m'][:, i] * 1e3, color=color, label=label, lw=1)
        axs[0, 1].plot(t, inward[:, 0], color=color, label=label, lw=1)
        axs[0, 2].plot(t, inward[:, 1], color=color, label=label, lw=1)
        axs[1, 0].plot(t, data['tip_gap_m'][:, i] * 1e3, color=color, label=label, lw=1)
        axs[1, 1].plot(t, data['closure'][:, i], color=color, label=label, lw=1)
        axs[1, 2].plot(t, np.sqrt(np.mean(tracking ** 2, axis=-1)), color=color, label=label, lw=1)
    titles = ['Tool displacement relative to wrist (recorded)', 'Thumb assigned broad-face normal force',
              'Index assigned broad-face normal force', 'Projected tip gap',
              'Extra closure relative to snapshot target', 'Joint target tracking RMS']
    units = ['mm', 'N', 'N', 'mm', 'closure increment', 'rad']
    for ax, title, unit in zip(axs.flat, titles, units):
        ax.set_title(title, fontsize=11); ax.set_xlabel('Simulation time (s)'); ax.set_ylabel(unit)
        ax.grid(alpha=.22)
        ax.axvspan(float(t[tail][0]), float(t[-1]), color='gray', alpha=.09)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(run.name + ' | face point mode: ' + point_mode +
                 '\nRecorded physics; no temporal smoothing. Gray band = acceptance window.', fontsize=15)
    png = out / (run.name + '_hold_diagnostics.png')
    fig.savefig(png, dpi=160)
    plt.close(fig)
    result = {
        'run': str(run), 'inputs_sha256': {str(p): sha(p) for p in (report_path, data_path, pose_path, urdf)},
        'source_script_sha256': sha(__file__), 'face_contact_point_mode': point_mode,
        'criteria': {'retained': 'all recorded grip_error <35 mm, orientation error <0.3; tip bound >5 mm; no cube contact',
                     'stable': 'retained AND last-window max Cartesian coordinate range <1 mm AND rotation-to-final <1 deg',
                     'broad_face': 'stable AND both assigned inward forces >3 mN for >=95% of last window',
                     'diagnostic_pass': 'finite AND broad_face AND every recorded target within URDF limits (1e-6 rad tolerance)'},
        'limitations': [
            'Acceptance is a static preheld-tool diagnostic, not successful block pickup.',
            'Contact forces and wrist-relative errors are recorded probe inputs; physical contacts are not independently solved here.',
            'Body trajectory, rotation stability, target/actual limits and snapshot consistency are recomputed independently.',
            'Solver actual-joint overshoot is reported separately; it is not silently clipped or silently made a target failure.',
            'A positive static gate still requires independent cold reset and then task validation.',
            'Material coefficients and geometry differ between experiments; gray brightness is not a force calibration.'
        ],
        'candidates': results, 'figure': str(png)}
    (out / (run.name + '_independent_hold_audit.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    results = [analyze_run(Path(run), out) for run in args.runs]
    rows = []
    for result in results:
        for c in result['candidates']:
            rows.append({'run': Path(c['run']).name, 'candidate': c['candidate'],
                         'diagnostic_pass': c['independent_diagnostic_pass'],
                         'retained': c['recomputed_retained'], 'broad_face_hold': c['recomputed_broad_face_hold'],
                         'tail_position_max_range_mm': max(c['tail_position_range_mm']),
                         'tail_rotation_max_deg': c['tail_rotation_to_final_max_deg'],
                         'tail_both_face_fraction': c['tail_both_assigned_face_fraction'],
                         'thumb_inward_N': c['tail_assigned_inward_mean_N'][0],
                         'index_inward_N': c['tail_assigned_inward_mean_N'][1],
                         'tip_gap_mm': c['tail_gap_mean_mm'],
                         'target_limit_violation_rad': c['full_run_target_limits']['max_violation_rad'],
                         'actual_limit_violation_rad': c['full_run_actual_limits']['max_violation_rad']})
    with (out / 'hold_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'output': str(out), 'candidates': rows}, ensure_ascii=False))


if __name__ == '__main__': main()
