"""Describe complete active pick trajectories without adding success gates.

Stage targets and CPU arm-command replay explain where a recorded timeout
occurred. The replay is a numerical diagnostic, not another physics run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from arm_command import ArmCommandKinematics
from kinematics import Kinematics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run):
    task = json.loads((run / 'task.json').read_text())
    evaluation = json.loads((run / 'evaluation.json').read_text())
    with np.load(run / 'trajectory.npz', allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    dt = float(d['dt']); count, envs = d['phase'].shape
    kin = Kinematics(Path(task['robot_asset_dir']) / 'robot.urdf', palm_link=task['palm_link'])
    arm = ArmCommandKinematics(kin, 'cpu')
    ai = [task['joint_names'].index(n) for n in kin.arm]
    snapshot_path = run / 'source' / Path(task['reset_snapshot']).name
    snapshot = json.loads(snapshot_path.read_text())
    source = snapshot.get('zero_velocity_variant', snapshot)
    bias = {n: b-a for n, a, b in zip(snapshot['joint_names'], source['q_actual'], source['q_target'])}
    initial = [task['arm_reset_IK_evidence'][0]['q'][n]+bias[n] for n in kin.arm]
    command_q = torch.tensor(initial, dtype=torch.float32)[None].repeat(envs, 1)
    reference = arm.fk_jacobian(command_q)[0]
    lower = reference[:, :3, 3] - torch.tensor([.35, .35, .09])
    upper = reference[:, :3, 3] + torch.tensor([.35, .35, .10])
    actual_q = torch.tensor(d['q'][:, :, ai].reshape(-1, 6))
    actual_transforms, actual_jacobians = arm.fk_jacobian(actual_q)
    actual_wrist = actual_transforms[:, :3, 3].numpy().reshape(count, envs, 3)
    minimum_singular = torch.linalg.svdvals(actual_jacobians)[:, -1].numpy().reshape(count, envs)
    command_wrist = np.zeros_like(actual_wrist)
    command_joint_error = np.zeros((count, envs))
    clamped = np.zeros((count, envs, 3), dtype=bool)
    old_phase = np.zeros(envs, dtype=np.int64)
    pre_phase = np.zeros_like(d['phase'])
    offsets = np.zeros((envs, 3), dtype=np.float32); offsets[:, 2] = -.0015
    target = np.full_like(d['tip'], np.nan)
    goal = d['goal'] + d['origins']
    floor = d['origins'][:, 2] + .72
    with torch.no_grad():
        for step in range(count):
            delta = torch.tensor(d['applied'][step, :, :3] * task['max_wrist_step'])
            current = arm.fk_jacobian(command_q)[0][:, :3, 3]
            desired = current + delta
            clamped[step] = ((desired < lower) | (desired > upper)).numpy()
            command_q = arm.increment(command_q, delta, reference, lower, upper)
            command_wrist[step] = arm.fk_jacobian(command_q)[0][:, :3, 3].numpy()
            command_joint_error[step] = np.max(np.abs(command_q.numpy()-d['q'][step][:, ai]), axis=1)
            pre_phase[step] = old_phase
            if step:
                pre_tip = d['tip'][step-1]; pre_block = d['block'][step-1, :, :3]
                target[step] = pre_tip
                mask = np.isin(old_phase, [1, 2, 3, 4])
                target[step, mask, :2] = pre_block[mask, :2]
                travel = np.isin(old_phase, [1, 4, 5, 8, 9])
                target[step, travel, 2] = floor[travel] + task['place_config']['travel_height_m'] + .0061
                descending = np.isin(old_phase, [2, 3])
                target[step, descending, 2] = pre_block[descending, 2] + .0015
                goal_phase = old_phase >= 5
                target[step, goal_phase, :2] = goal[goal_phase, :2] - offsets[goal_phase, :2]
                lowering = np.isin(old_phase, [6, 7])
                target[step, lowering, 2] = floor[lowering] + .0076
            lifted = (old_phase == 4) & (d['phase'][step] == 5)
            offsets[lifted] = d['block'][step, lifted, :3] - d['tip'][step, lifted]
            old_phase = d['phase'][step].astype(np.int64)
    distance = np.linalg.norm(target-d['tip'], axis=2)
    wrist_error = np.linalg.norm(command_wrist-actual_wrist, axis=2)
    rows = []
    for env in range(envs):
        active = d['active'][:, env]; ix = np.flatnonzero(active); end = ix[-1]
        force = d['forces'][ix, env]
        fi, jaw = np.unravel_index(force.argmax(), force.shape); peak = ix[fi]
        gi = ix[np.argmax(d['retained'][ix, env])]
        retained = (d['retained'][:, env] < .035) & (d['orientation_error'][:, env] < .3)
        phase_rows = []
        for p, name in enumerate(task['phases']):
            ids = np.flatnonzero(active & (pre_phase[:, env] == p))
            if not len(ids):
                continue
            fpeak = ids[np.argmax(d['forces'][ids, env].max(axis=1))]
            phase_rows.append({'phase': p, 'name': name, 'duration_s': len(ids)*dt,
                'peak_jaw_block_net_force_N': float(d['forces'][fpeak, env].max()),
                'peak_force_time_s': float((fpeak+1)*dt),
                'face_valid_fraction': float(d['face'][ids, env].mean()),
                'orientation_valid_fraction': float(d['orientation_ok'][ids, env].mean()),
                'retained_fraction': float(retained[ids].mean()),
                'max_grip_error_m': float(d['retained'][ids, env].max()),
                'max_commanded_actual_wrist_distance_m': float(wrist_error[ids, env].max()),
                'workspace_clamped_command_samples': int(clamped[ids, env].any(axis=1).sum()),
                'minimum_actual_arm_jacobian_singular_value': float(minimum_singular[ids, env].min()),
                'duration_actual_arm_jacobian_min_singular_below_0_005_s': float((minimum_singular[ids, env] < .005).sum()*dt),
                'minimum_target_tip_distance_m': float(np.nanmin(distance[ids, env])) if p else None,
                'final_target_tip_distance_m': float(distance[ids[-1], env]) if p else None,
                'diagnostic_samples_every_2s': [
                    {'time_s': float((j+1)*dt), 'target_tip_distance_m': float(distance[j, env]) if p else None,
                     'tip_local_m': (d['tip'][j, env]-d['origins'][env]).tolist(),
                     'command_wrist_local_to_base_m': command_wrist[j, env].tolist(),
                     'actual_wrist_local_to_base_m': actual_wrist[j, env].tolist(),
                     'minimum_actual_arm_jacobian_singular_value': float(minimum_singular[j, env]),
                     'command_actual_wrist_distance_m': float(wrist_error[j, env]),
                     'command_actual_arm_joint_error_rad': float(command_joint_error[j, env])}
                    for j in ids[::max(1, round(2/dt))]]})
        final_phase = int(d['phase'][end, env])
        final_ids = np.flatnonzero(active & (d['phase'][:, env] == final_phase))
        last_second = ix[ix >= end-round(1/dt)+1]
        tail_path = np.linalg.norm(np.diff(d['tip'][last_second, env], axis=0), axis=1).sum()
        row = {'environment': env, 'recorded_success': bool(evaluation['success_per_env'][env]),
            'duration_s': len(ix)*dt, 'final_phase': final_phase, 'final_phase_name': task['phases'][final_phase],
            'final_phase_duration_s': len(final_ids)*dt,
            'timeout_stage': task['phases'][final_phase] if not evaluation['success_per_env'][env] and len(final_ids)*dt > 12 else None,
            'max_grip_error_m': float(d['retained'][gi, env]), 'peak_grip_error_time_s': float((gi+1)*dt),
            'peak_grip_error_phase': int(pre_phase[gi, env]),
            'retained_fraction': float(retained[ix].mean()), 'face_valid_fraction': float(d['face'][ix, env].mean()),
            'orientation_valid_fraction': float(d['orientation_ok'][ix, env].mean()),
            'tool_angle_range_deg': [float(d['normal_angle'][ix, env].min()), float(d['normal_angle'][ix, env].max())],
            'peak_jaw_block_net_force_N': float(force[fi, jaw]), 'peak_force_time_s': float((peak+1)*dt),
            'peak_force_phase': int(pre_phase[peak, env]), 'peak_force_jaw': ['left', 'right'][jaw],
            'peak_force_pair_N': d['forces'][peak, env].tolist(),
            'final_target_tip_error_m': (target[end, env]-d['tip'][end, env]).tolist(),
            'final_target_tip_distance_m': float(distance[end, env]),
            'final_commanded_wrist_workspace_margin_lower_m': (command_wrist[end, env]-lower[env].numpy()).tolist(),
            'final_commanded_wrist_workspace_margin_upper_m': (upper[env].numpy()-command_wrist[end, env]).tolist(),
            'final_actual_wrist_workspace_margin_upper_m': (upper[env].numpy()-actual_wrist[end, env]).tolist(),
            'final_commanded_actual_wrist_distance_m': float(wrist_error[end, env]),
            'max_commanded_actual_wrist_distance_m': float(wrist_error[ix, env].max()),
            'max_commanded_actual_arm_joint_error_rad': float(command_joint_error[ix, env].max()),
            'final_commanded_actual_arm_joint_error_rad': float(command_joint_error[end, env]),
            'final_command_speed_m_s': float(np.linalg.norm(d['applied'][end, env, :3])*task['max_wrist_step']/dt),
            'last_second_actual_tip_path_length_m': float(tail_path),
            'final_jaw_block_net_force_N': d['forces'][end, env].tolist(),
            'phases': phase_rows}
        rows.append(row)
    archived_arm = run/'source'/'arm_command.py'
    return {'schema_version': 1, 'run': str(run), 'successes': evaluation['successes'], 'episodes': rows,
        'all_active_control_samples_checked': True,
        'arm_command_source_matches_archive': sha(archived_arm) == sha(ROOT/'src'/'arm_command.py'),
        'source_sha256': {str(run/n): sha(run/n) for n in ['task.json', 'trajectory.npz', 'evaluation.json']},
        'script_sha256': sha(__file__),
        'definitions': ['No extra process-wide success criteria are introduced; success is checked separately against the recorded task gates.',
            'Jaw/block force is the magnitude of each recorded net contact vector at the last physics substep of a control frame; it is not an all-substep peak or a normal-only force.',
            'Pre-command phase is reconstructed because recorded phase is post-step.',
            'Arm target replay uses archived reset IK solution plus snapshot target bias, recorded commands and CPU float32 kinematics; small CPU/GPU numerical differences are possible.',
            'Workspace margins refer to configured wrist-command bounds, not source-mesh collision clearance.',
            'Jacobian singular values use the controller 6-by-6 spatial Jacobian with mixed metre/radian rows; 0.005 is a diagnostic marker, not a new task gate.',
            'Nonzero transient face/orientation failure does not erase a later valid continuous success interval.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit(args.run)
    out = args.out or args.run/'independent_process_audit.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'report': str(out), 'successes': result['successes'],
                      'failed': [{'env': r['environment'], 'stage': r['final_phase_name']} for r in result['episodes'] if not r['recorded_success']]}))
