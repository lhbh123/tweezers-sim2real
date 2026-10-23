"""Reconstruct release-middle commands from saved control states.

The recorded phase is post-step while the blend and applied action were computed
from pre-step phase and measurements. No simulation or controller is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run):
    task = json.loads((run / 'task.json').read_text())
    cfg = task['place_config']
    with np.load(run / 'trajectory.npz', allow_pickle=False) as data:
        d = {name: data[name] for name in data.files}
    if 'release_middle_blend' not in d:
        raise ValueError('This archive has no release_middle_blend telemetry')
    dt = float(d['dt']); steps, envs = d['phase'].shape
    deltas = cfg.get('release_middle_joint_deltas') or {}
    enabled = bool(deltas); tracking = bool(cfg.get('release_track_tip', False))
    ramp = np.float32(cfg.get('release_middle_ramp_seconds', .5))
    delay = np.float32(cfg.get('release_middle_delay_seconds', .8))
    if ramp <= 0 or delay < 0:
        raise ValueError('Invalid release timing')
    expected_blend = np.zeros((steps, envs), dtype=np.float32)
    old_phases = np.zeros((steps, envs), dtype=np.int64)
    command_error = np.zeros((steps, envs))
    target_error = np.full((steps, envs), np.nan)
    counters = np.zeros(envs, dtype=np.int64)
    old_phase = np.zeros(envs, dtype=np.int64)
    offsets = np.zeros((envs, 3), dtype=np.float32); offsets[:, 2] = -.0015
    goals = d['goal'] + d['origins']
    max_step = float(task['max_wrist_step'])
    for i in range(steps):
        old_phases[i] = old_phase
        clock = counters.astype(np.float32) * np.float32(dt)
        releasing = old_phase == 7
        if enabled:
            expected_blend[i, releasing] = np.clip((clock[releasing] - delay) / ramp, 0, 1)
            withdrawing = old_phase >= 8
            if i and withdrawing.any():
                height = d['tip'][i-1, :, 2] - d['block'][i-1, :, 2]
                fraction = (height - np.float32(.020)) / np.float32(.035)
                expected_blend[i, withdrawing] = 1 - np.clip(fraction[withdrawing], 0, 1)
        if i and releasing.any():
            target = goals.copy() - offsets
            target[:, 2] = d['origins'][:, 2] + .72 + .0076
            error = target - d['tip'][i-1]
            target_error[i, releasing] = np.linalg.norm(error[releasing], axis=1)
            nominal = .5 * error
            lengths = np.linalg.norm(nominal, axis=1)
            nominal *= np.minimum(1., cfg['max_speed_m_s'] * dt / np.maximum(lengths, 1e-8))[:, None]
            previous = d['applied'][i-1, :, :3] * max_step
            nominal = np.clip(nominal, previous - .00008, previous + .00008)
            if not tracking:
                nominal[:] = 0
            applied = nominal / max_step
            applied += d['residual'][i, :, :3] * cfg['residual_xyz_m'] / max_step
            applied = np.clip(applied, -1, 1) * max_step
            command_error[i, releasing] = np.max(
                np.abs(applied[releasing] - d['applied'][i, releasing, :3] * max_step), axis=1)
        changed = d['phase'][i] != old_phase
        lifted = changed & (old_phase == 4) & (d['phase'][i] == 5)
        offsets[lifted] = d['block'][i, lifted, :3] - d['tip'][i, lifted]
        counters = np.where(changed, 0, counters + 1)
        old_phase = d['phase'][i].astype(np.int64)
    errors = np.abs(expected_blend - d['release_middle_blend'])
    rows = []
    for env in range(envs):
        active = d['active'][:, env]
        phase7 = active & (old_phases[:, env] == 7)
        phase8 = active & (old_phases[:, env] >= 8)
        onsets = np.flatnonzero(active & (d['release_middle_blend'][:, env] > 0))
        live = np.flatnonzero(active); end = live[-1]
        speeds = np.linalg.norm(d['applied'][phase7, env, :3] * max_step, axis=1) / dt
        max_error = float(errors[active, env].max())
        max_command_error = float(command_error[phase7, env].max()) if phase7.any() else 0.
        row = {'environment': env, 'recorded_success': bool(d['success'][end, env]),
               'max_blend_reconstruction_error': max_error,
               'blend_formula_pass': max_error <= 3e-6,
               'max_release_wrist_command_reconstruction_error_m': max_command_error,
               'wrist_tracking_formula_pass': max_command_error <= 1e-6,
               'release_command_samples': int(phase7.sum()),
               'withdraw_command_samples': int(phase8.sum()),
               'first_nonzero_blend_record_time_s': float((onsets[0]+1)*dt) if len(onsets) else None,
               'maximum_blend': float(d['release_middle_blend'][active, env].max()),
               'final_blend': float(d['release_middle_blend'][end, env]),
               'final_success_has_restored_middle_command': bool(not d['success'][end, env] or
                                                                 abs(d['release_middle_blend'][end, env]) <= 3e-6),
               'release_peak_command_speed_m_s': float(speeds.max()) if len(speeds) else None,
               'release_command_path_length_m': float(speeds.sum()*dt) if len(speeds) else 0.,
               'release_max_precommand_tip_target_error_m': float(np.nanmax(target_error[phase7, env])) if phase7.any() else None,
               'actual_release_joint_ranges_rad': {}}
        for name in deltas:
            joint = task['joint_names'].index(name)
            values = d['q'][phase7, env, joint]
            row['actual_release_joint_ranges_rad'][name] = [float(values.min()), float(values.max())] if len(values) else None
        rows.append(row)
    return {'schema_version': 1, 'run': str(run), 'configured_joint_deltas_rad': deltas,
            'configured_delay_s': float(delay), 'configured_ramp_s': float(ramp),
            'configured_release_track_tip': tracking,
            'all_command_reconstructions_pass': all(r['blend_formula_pass'] and r['wrist_tracking_formula_pass'] and
                                                    r['final_success_has_restored_middle_command'] for r in rows),
            'rows': rows,
            'source_sha256': {str(run/name): sha(run/name) for name in ['trajectory.npz', 'task.json']},
            'script_sha256': sha(__file__),
            'limitations': ['Command-level reconstruction, not a claim of exact actual-joint response time.',
                            'Phase in telemetry is post-step; commands are checked against previous phase and measurements.',
                            'Wrist-command comparison includes recorded policy residuals and existing component acceleration smoothing.',
                            'No success/force threshold is changed; physical placement and source geometry have separate audits.',
                            'Reported command speed is an observation, not a claimed strict physical velocity bound.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    output = args.out or args.run / 'independent_release_middle_audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'report': str(output), 'all_command_reconstructions_pass': result['all_command_reconstructions_pass']}))
