"""Collect verified W01 simulation evidence without changing acceptance results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def summarize(run):
    evaluation = read(run / 'evaluation.json')
    validation = read(run / 'validation.json')
    assert validation['all_claims_consistent']
    assert validation['no_runtime_object_control_calls']
    assert validation['archived_source_hashes_match']
    geometry = validation['source_mesh_table_audit']
    assert geometry['below_table_link_samples'] == 0
    assert geometry['joint_limit_violations'] == 0
    assert validation['tool_table_source_box_audit']['below_table_body_samples'] == 0
    with np.load(run / 'trajectory.npz') as data:
        active = data['active'].astype(bool)
        angles = data['normal_angle'][active]
        face = data['face'][active]
        relative_error = data['retained'][active]
        failures = []
        for index, success in enumerate(evaluation['success_per_env']):
            if success:
                continue
            last = int(np.flatnonzero(active[:, index])[-1])
            failures.append({
                'environment': index, 'last_phase': int(data['phase'][last, index]),
                'final_jaw_forces_N': data['forces'][last, index].tolist(),
                'final_tool_down_angle_deg': float(data['normal_angle'][last, index]),
                'final_tool_relative_position_error_mm': float(data['retained'][last, index] * 1000),
                'final_extra_opening': float(data['release_extra_opening'][last, index])
                    if 'release_extra_opening' in data else None,
            })
    errors = np.asarray(evaluation['final_xy_error_m']) * 1000
    successful_errors = errors[np.asarray(evaluation['success_per_env'], dtype=bool)]
    result = {
        'run': str(run), 'episodes': evaluation['episodes'],
        'successes': evaluation['successes'],
        'success_per_env': evaluation['success_per_env'],
        'final_xy_error_mm': errors.tolist(),
        'mean_final_xy_error_mm': float(errors.mean()),
        'max_final_xy_error_mm': float(errors.max()),
        'successful_final_xy_error_mm': successful_errors.tolist(),
        'successful_final_xy_error_range_mm':
            [float(successful_errors.min()), float(successful_errors.max())]
            if len(successful_errors) else None,
        'max_lift_mm_per_env': (np.asarray(evaluation['max_lift_m']) * 1000).tolist(),
        'duration_s_per_env': evaluation['duration_s'],
        'actual_tool_down_angle_range_deg': [float(angles.min()), float(angles.max())],
        'active_control_samples_broad_face_fraction': float(face.mean()),
        'max_tool_relative_position_error_mm': float(relative_error.max() * 1000),
        'independent_validation': validation,
        'failed_episodes': failures,
    }
    if evaluation['task']['place_config'].get('release_contact_clear_seconds', 0) > 0:
        feedback = read(run / 'independent_release_feedback_audit.json')
        result['release_feedback_audit'] = feedback
        result['controller_audit_warnings'] = [
            {'environment': row['environment'], 'failed_check': name}
            for row in feedback['rows'] for name, value in row.items()
            if value is False and name not in ['recorded_success', 'withdrawal_reached']
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--video-run', type=Path, required=True)
    parser.add_argument('--regression', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--contact-review', type=Path,
                        help='Independent classification and depth review of measured support-contact source intersections')
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/sharpa_w01_final.json'))
    args = parser.parse_args()
    batch = summarize(args.evaluation)
    video = summarize(args.video_run)
    tactile = read(args.video_run / 'tactile_validation.json')
    assert tactile['valid'] and tactile['task_success']
    composition = read(args.media.with_suffix('.json'))
    assert hashlib.sha256(args.media.read_bytes()).hexdigest() == composition['output_sha256']
    launch = read(args.video_run / 'launch.json')
    task = read(args.video_run / 'task.json')
    mesh_audit = read(args.video_run / 'source_mesh_trajectory_audit.json')
    assert mesh_audit['source_urdf']['matches_run_hash']
    assert mesh_audit['summary']['sampled_nonmechanism_self_intersection_samples'] == 0
    contact_review = None
    intersection_states = mesh_audit['summary']['sampled_nontarget_tool_intersection_samples']
    if args.contact_review:
        contact_review = read(args.contact_review)
        assert contact_review['source_urdf_matches_run_hash']
        assert contact_review['self_intersection_states'] == 0
        assert contact_review['other_hand_tool_intersection_states'] == 0
        assert contact_review['original_fk_tool_intersection_states'] == intersection_states
        assert contact_review['total_intersection_states'] >= intersection_states
        assert contact_review['coverage']['all_active_control_states_covered']
        assert contact_review['coverage']['all_intersection_depths_covered']
        assert 0 <= contact_review['reviewed_intersection_states'] <= contact_review['total_intersection_states']
        for evidence in contact_review['source_reports'].values():
            assert hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest() == evidence['sha256']
    else:
        assert intersection_states == 0, 'Nonzero source intersections require a complete measured-contact geometry review'
    report = {
        'model': task['robot_model'], 'date': '2026-10-19',
        'status': 'completed' if batch['successes'] == batch['episodes'] and video['successes'] == 1 else 'completed_with_test_failures',
        'controller': 'Nominal geometric phase controller with contact-guided release; no W01 PPO training or reuse of XHand weights.',
        'hand_active_dof': 22, 'arm_active_dof': 6,
        'physics_hz': 480, 'control_hz': 60,
        'official_assets': {
            'robot': 'https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e/wave_01',
            'tactile': 'https://github.com/sharpa-robotics/sharpa-tacmap/tree/e4f8827407e15625368ef0b9cf5a72c44304e16b',
        },
        'task_adaptations': {
            'tool_down_angle_range_deg': task['tool_down_angle_range_deg'],
            'explanation': 'W01 uses its own working-angle range; the historical XHand limit was 20 to 70 degrees. All lift, contact, release and final stability thresholds are retained.',
            'torsional_patch': task['finger_torsional_patch_settings'],
            'starting_condition': 'Pre-held tweezers, dynamically validated cold reset. Picking up the tweezers is outside this experiment.',
            'release_feedback': {key: task['place_config'][key] for key in
                                 ['release_extra_opening_max', 'release_extra_opening_rate',
                                  'release_contact_clear_seconds', 'withdraw_initial_speed_m_s', 'retreat_closure']},
            'middle_release': {key: task['place_config'].get(key) for key in
                               ['release_middle_joint_deltas', 'release_middle_delay_seconds',
                                'release_middle_ramp_seconds', 'release_track_tip']},
        },
        'randomized_evaluation': batch, 'recorded_episode': video,
        'tactile': {key: tactile[key] for key in ['valid', 'physics_substeps', 'tactile_frames', 'recorded_duration_s', 'force_channels', 'surface_mapping_threshold_m', 'warnings', 'tactile_npz_sha256']},
        'source_mesh_trajectory_audit': {
            'path': str(args.video_run / 'source_mesh_trajectory_audit.json'),
            'sha256': hashlib.sha256((args.video_run / 'source_mesh_trajectory_audit.json').read_bytes()).hexdigest(),
            'summary': mesh_audit['summary'], 'sampling': mesh_audit['sampling'],
            'limitations': mesh_audit['limitations'],
        },
        'media': composition, 'reproduction_arguments': launch['arguments'],
        'limitations': [
            'Finite simulation experiments; no hardware validation or full-workspace guarantee.',
            'Material friction, actuator gains and rigid contact models are uncalibrated simulation assumptions.',
            'Tactile maps show actual PhysX normal/friction forces projected to official W01 surfaces in N/bin, not pressure or hardware sensor images.',
            'The older official tactile map differs locally from the current hand mesh; out-of-map force remains in the raw records.',
            'No runtime object teleportation, robot-tool attachment or external grasp force is used.',
            'The initial withdrawal speed setting is a nominal target before acceleration smoothing, not a strict speed cap. Any observed exceedance is retained in the feedback audit.',
        ],
    }
    if contact_review is not None:
        report['measured_support_contact_geometry_review'] = {
            'path': str(args.contact_review),
            'sha256': hashlib.sha256(args.contact_review.read_bytes()).hexdigest(),
            'review': contact_review,
        }
        if contact_review['total_intersection_states']:
            report['limitations'].append('Source mesh intersections occur in the middle support region. Their original counts and independently measured depths are retained; this is not a claim of zero source penetration or continuous upward support.')
        if not contact_review['review_complete']:
            unmatched = contact_review['total_intersection_states'] - contact_review['reviewed_intersection_states']
            report['contact_geometry_status'] = 'source_intersections_not_fully_matched_by_solver_contacts'
            report['unmatched_source_intersection_states'] = unmatched
            report['limitations'].append(f'{unmatched} source-intersection control states lack a matching measured middle contact. Exact source geometry is fully measured, but the stricter contact-matching review remains false; these states are preserved as a source/solver discrepancy, not counted as verified support.')
        else:
            report['contact_geometry_status'] = 'all_source_intersections_reviewed_with_matching_measured_contacts'
    if args.regression:
        report['regression_on_previous_eight_cases'] = summarize(args.regression)
        report['regression_note'] = 'Same eight starts/goals as the development baseline, rerun after the release-control correction; not a held-out test.'
        regression = report['regression_on_previous_eight_cases']
        if regression['successes'] != regression['episodes']:
            report['status'] = 'completed_with_test_failures'
        report['combined_batch_results'] = {
            'successes': batch['successes'] + regression['successes'],
            'episodes': batch['episodes'] + regression['episodes'],
            'note': 'Combines a fresh held-out batch with a development regression batch; these are not sixteen held-out cases.',
        }
        if regression['failed_episodes'] or batch['failed_episodes']:
            report['limitations'].append('Failed cases remain failures under the original task criteria. See each failed_episodes entry and the independent gate audit for the final phase and measured forces; being near the goal alone is not success.')
    if args.baseline:
        report['development_baseline'] = summarize(args.baseline)
        report['baseline_failure_diagnosis'] = (
            'The baseline is retained with its original acceptance results. '
            'See development_baseline.failed_episodes for each recorded final phase, '
            'jaw forces, tool angle and relative grip error; placing the cube near '
            'its goal does not turn an unfinished release/withdrawal into success.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'evaluation': [batch['successes'], batch['episodes']],
                      'video': video['successes'], 'report': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
