#!/usr/bin/env python3
"""Validate training transitions and exact RGB/state alignment without Isaac Sim."""
import argparse
import json
from pathlib import Path

import numpy as np


def validate(path, decode=True):
    path = Path(path)
    schema = json.loads((path / 'training_schema.json').read_text())
    report = json.loads((path / 'evaluation.json').read_text())
    provenance = json.loads((path / 'execution_source_provenance.json').read_text())
    recording = json.loads((path / 'camera_recording.json').read_text())
    frames = [json.loads(s) for s in (path / 'camera_frames.jsonl').read_text().splitlines()]
    with np.load(path / 'training_transitions.npz', allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    with np.load(path / 'trajectory.npz', allow_pickle=False) as archive:
        old = {k: archive[k] for k in archive.files}
    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise ValueError('Training data check failed: ' + name)

    t = len(d['action_done'])
    j = len(schema['joint_names'])
    require('nonempty_unique_joint_order', t > 0 and j == len(set(schema['joint_names'])) and j > 6)
    for key, values in d.items():
        require('finite_' + key, np.isfinite(values).all())
        if key.startswith('observation_'):
            require('length_' + key, len(values) == t + 1)
        if key.startswith('action_'):
            require('length_' + key, len(values) == t)
    for key in ['observation_qpos_rad', 'observation_qvel_rad_s', 'action_joint_position_target_rad']:
        require('shape_' + key, d[key].shape[-1] == j)
    times = d['observation_time_s']
    require('initial_time_zero', abs(times[0]) < 1e-8)
    require('time_increments', np.allclose(np.diff(times), schema['control_dt_s'], atol=2e-7))
    require('done_only_at_end', d['action_done'][-1] and not d['action_done'][:-1].any())
    require('command_buffer_exact', np.array_equal(d['action_joint_position_target_rad'], d['action_actuator_target_buffer_rad']))
    # Cross-check independently saved legacy trace to detect action/state shifts.
    pairs = [('observation_qpos_rad', 'q'), ('observation_tool_state_w', 'tool'),
             ('observation_block_state_w', 'block')]
    for new, legacy in pairs:
        require('legacy_' + legacy, np.array_equal(d[new][1:], old[legacy][:, 0]))
    for new, legacy in [('action_policy_residual', 'residual'), ('action_applied_control', 'applied')]:
        require('legacy_' + legacy, np.array_equal(d[new], old[legacy][:, 0]))
    predicted = np.clip(d['action_nominal_control'] + np.asarray(schema['residual_scale'], dtype=np.float32)
                        * d['action_policy_residual'], -1, 1)
    predicted[d['observation_phase'][:-1] == 0, :3] = 0
    require('nominal_plus_residual', np.allclose(predicted, d['action_applied_control'], atol=2e-7))
    indices = d['frame_observation_index']
    expected = list(range(0, t + 1, 2))
    if expected[-1] != t:
        expected.append(t)
    require('initial_regular_terminal_frame_indices', indices.tolist() == expected)
    require('frame_counts', len(frames) == len(indices) == recording['frames'])
    camera_manifest = json.loads((path / 'camera_manifest.json').read_text())
    require('camera_joint_order', camera_manifest['joint_names'] == schema['joint_names'])
    for f, (row, oi) in enumerate(zip(frames, indices)):
        require(f'frame_{f}_index', row['frame_index'] == f and row['observation_index'] == oi
                and row['control_step'] == oi - 1)
        require(f'frame_{f}_time', abs(row['elapsed_simulation_time_s'] - times[oi]) < 2e-7)
        require(f'frame_{f}_qpos', np.allclose(row['joint_positions_rad'], d['observation_qpos_rad'][oi], atol=1e-7))
        require(f'frame_{f}_tcp', np.allclose(row['T_world_TCP'], d['observation_T_world_tcp'][oi], atol=1e-7))
        for key in ['T_world_Cw', 'T_world_Cg']:
            require(f'frame_{f}_finite_{key}', np.isfinite(row[key]).all())
    require('global_camera_fixed', np.allclose([f['T_world_Cg'] for f in frames], frames[0]['T_world_Cg'], atol=1e-9))
    tcp_camera = np.asarray(camera_manifest['rig']['cameras']['wrist']['T_TCP_Cw'])
    require('wrist_mount_fixed', np.allclose(np.asarray([f['T_world_TCP'] for f in frames]) @ tcp_camera,
                                            np.asarray([f['T_world_Cw'] for f in frames]), atol=1e-7))
    require('provenance_complete', provenance['complete'] and provenance['status'] == 'consistent_at_observed_boundaries')
    require('terminal_success_consistent', bool(d['action_success'][-1]) == bool(report['successes']))
    decoded = {}
    if decode:
        import imageio.v2 as imageio
        for camera in ['wrist', 'global']:
            count = 0
            means = []
            with imageio.get_reader(path / (camera + '.mp4')) as reader:
                require(camera + '_fps', abs(reader.get_meta_data()['fps'] - 30) < .01)
                for rgb in reader:
                    require(camera + '_image_shape', rgb.shape == (480, 640, 3))
                    if count in [0, len(indices) // 2, len(indices) - 1]:
                        means.append(float(rgb.mean()))
                    count += 1
            require(camera + '_decoded_count', count == len(indices))
            require(camera + '_not_black_or_white', all(2 < m < 253 for m in means))
            decoded[camera] = {'frames': count, 'sample_image_means': means}
    samples = []
    for frame_index, oi in enumerate(indices):
        oi = int(oi)
        samples.append(dict(frame_index=frame_index, observation_index=oi,
                            time_s=float(times[oi]), next_action_index=oi if oi < t else None,
                            available_future_actions=t - oi, terminal_observation=oi == t))
    (path / 'training_samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    limits = np.asarray(schema['joint_soft_limits_rad'])
    target = d['action_joint_position_target_rad']
    outside = (target < limits[:, 0] - 1e-5) | (target > limits[:, 1] + 1e-5)
    task_success = bool(report['successes'])
    final_flags = {k: bool(d['action_' + k][-1]) for k in ['failure', 'phase_timeout', 'time_limit']}
    result = dict(data_valid=True, task_success=task_success,
                  training_candidate=task_success and not outside.any() and not any(final_flags.values()),
                  transition_count=t, observation_count=t + 1, joint_count=j, frame_count=len(indices),
                  duration_s=float(times[-1]), final_xy_error_m=report['final_xy_error_m'][0],
                  max_phase=report['max_phase'][0], terminal_flags=final_flags,
                  target_soft_limit_violations=int(outside.sum()),
                  max_joint_target_step_rad=float(np.abs(np.diff(target, axis=0)).max()),
                  max_measured_joint_step_rad=float(np.abs(np.diff(d['observation_qpos_rad'], axis=0)).max()),
                  max_tracking_error_rad=float(np.abs(target - d['observation_qpos_rad'][1:]).max()),
                  checks_passed=len(checks), videos=decoded,
                  scope='Data/command/timing/video consistency and existing task gates; not visual policy training, joint-command replay, or real execution')
    (path / 'training_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    p.add_argument('--skip-video-decode', action='store_true')
    a = p.parse_args()
    print(json.dumps(validate(a.episode, not a.skip_video_decode), indent=2))
