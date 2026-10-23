#!/usr/bin/env python3
"""Summarize separately validated configurations without pooling their rates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'


def load(path):
    return json.loads(path.read_text())


def entry(name, condition):
    path = RUNS / name
    if not (path / 'validation.json').exists():
        return {'run': name, 'condition': condition, 'status': 'pending_validation'}
    validation = load(path / 'validation.json')
    metadata = load(path / 'metadata.json')
    result = load(path / 'result.json')
    args = metadata['args']
    source_hashes = metadata.get('source_sha256', {})
    matches = {filename: (path / filename).exists() and hashlib.sha256((path / filename).read_bytes()).hexdigest() == digest
               for filename, digest in source_hashes.items()}
    return {
        'run': name, 'condition': condition, 'status': 'complete',
        'strict_hold_success': validation['success'],
        'controller_fault': result.get('controller_fault'),
        'max_ik_error': result.get('max_ik_error'),
        'failed_hold_checks': [name for name, ok in validation['checks'].items() if not ok],
        'arguments': args,
        'min_final_hold_lift_m': validation['min_lift_m'],
        'bilateral_contact_fraction': validation['bilateral_contact_fraction'],
        'block_xyz_range_m': validation['block_xyz_range_m'],
        'position_range_diagonal_m': validation['block_bounding_box_diagonal_m'],
        'physics_reported_speed_m_s': validation['physics_reported_speed_m_s'],
        'position_finite_difference_speed_m_s': validation['position_finite_difference_speed_m_s'],
        'position_100ms_difference_speed_m_s': validation['position_100ms_difference_speed_m_s'],
        'fixture_disabled_after_reset': validation['checks']['no_fixture_after_reset_phase'],
        'final_usd_fixture_disabled': validation['checks']['final_usd_fixture_disabled_if_available'],
        'source_sha256': source_hashes, 'archived_source_hash_matches': matches,
        'all_declared_source_hashes_match': bool(matches) and all(matches.values()),
        'validation_path': str(path.relative_to(ROOT) / 'validation.json'),
        'evidence_sha256': validation['sha256'],
    }


def suite(identifier, material, names):
    rows = [entry(name, condition) for name, condition in names]
    complete = [row for row in rows if row['status'] == 'complete']
    passed = sum(row['strict_hold_success'] for row in complete)
    return {
        'id': identifier, 'material_configuration': material,
        'physics_dt_s': 1 / 240, 'controller_update_period_s': 4 / 240,
        'nominal_block_mass_kg': .002, 'block_dimensions_m': [.012] * 3,
        'spring_stiffness_Nm_per_rad': 1.5, 'finger_closure_fraction': .25,
        'reset_fixture_release_s': .6, 'planned_conditions': len(rows),
        'completed_conditions': len(complete), 'passed_conditions': passed,
        'failed_conditions': len(complete) - passed,
        'pass_fraction_among_completed_conditions': passed / len(complete) if complete else None,
        'interpretation': 'One deterministic run per listed condition; this is a condition pass fraction, not a randomized success-rate estimate.',
        'runs': rows,
    }


def release_control():
    row = entry('release_negative', 'Intentional finger opening at t=7s, release_close=-2.0')
    samples = load(RUNS / 'release_negative' / 'telemetry.json')
    release = row['arguments']['release_at']
    before = [r for r in samples if release - .5 <= r['t'] < release]
    final = [r for r in samples if r['t'] >= samples[-1]['t'] - .5]
    before_lift = np.array([r['block'][2] for r in before]) - .726
    final_lift = np.array([r['block'][2] for r in final]) - .726
    before_contact = (np.linalg.norm(np.array([r['force'] for r in before]), axis=-1) > .001).all(axis=-1)
    final_contact = (np.linalg.norm(np.array([r['force'] for r in final]), axis=-1) > .001).all(axis=-1)
    checks = {
        'held_before_command': bool(before_lift.min() > .075 and before_contact.mean() >= .95),
        'returned_to_table': bool(np.abs(final_lift).max() < .002),
        'bilateral_contact_lost': bool(final_contact.mean() <= .05),
        'no_fixture_after_reset': row['fixture_disabled_after_reset'],
        'final_usd_fixture_disabled': row['final_usd_fixture_disabled'],
    }
    row.update({'classification': 'expected_negative_control', 'expected_release_behavior_passed': all(checks.values()),
                'release_checks': checks, 'pre_release_min_lift_m': float(before_lift.min()),
                'final_half_second_max_absolute_lift_m': float(np.abs(final_lift).max()),
                'final_half_second_bilateral_contact_fraction': float(final_contact.mean()),
                'included_in_nominal_hold_denominator': False,
                'interpretation': 'The hold validator correctly fails after commanded release. Falling onto the table after finger opening is the expected negative-control outcome, not a nominal task failure.'})
    return row


def main():
    high = {'finger_mu': 1.4, 'steel_jaw_mu': 1.4, 'plastic_mu': .65,
            'friction_combine_mode': 'max', 'effective_finger_steel_mu': 1.4,
            'effective_steel_plastic_mu': 1.4, 'static_dynamic_coefficients_equal': True}
    final = {**high, 'steel_jaw_mu': .5, 'effective_steel_plastic_mu': .65}
    nominal = [('nominal', 'Baseline, 2 g'), ('xminus', 'Initial block x offset -2 mm'),
               ('xplus', 'Initial block x offset +2 mm'), ('yminus', 'Initial block y offset -2 mm'),
               ('yplus', 'Initial block y offset +2 mm'), ('mass', 'Block mass 3 g')]
    final_names = [('demo_v2' if key == 'nominal' else 'v2_' + key, condition) for key, condition in nominal]
    high_names = [('demo_final' if key == 'nominal' else 'eval_' + key, condition) for key, condition in nominal]
    stress = entry('dt480_corrected', '480 Hz physics and 120 Hz control, remaining final-v2 parameters unchanged')
    stress.update({'classification': 'out_of_nominal_configuration_stress_test',
                   'physics_dt_s': 1 / 480, 'controller_update_period_s': 4 / 480,
                   'tool_estimate_filter_time_constant_s': .13,
                   'tool_estimate_filter_coefficient_per_control_update': float(1 - np.exp(-(4 / 480) / .13)),
                   'included_in_nominal_hold_denominator': False,
                   'interpretation': 'Failed. The archived controller updates every four physics steps, so this changes physics rate and controller sampling rate together. The tool-pose filter coefficient is correctly recomputed as 1-exp(-4*dt/0.13), preserving its 0.13 s time constant. This is not an isolated physics-timestep convergence test.'})
    friction = entry('eval_friction', 'Preliminary shared finger/jaw friction reduced from 1.4 to 1.0')
    friction.update({'classification': 'preliminary_material_sensitivity_test',
                     'material_configuration': {**high, 'finger_mu': 1., 'steel_jaw_mu': 1.,
                                                'effective_finger_steel_mu': 1., 'effective_steel_plastic_mu': 1.},
                     'included_in_nominal_hold_denominator': False,
                     'interpretation': 'Failed under a different shared-material configuration. Distinct from final-v2, which retains finger friction 1.4 and lowers only steel/plastic effective friction.'})
    final_friction = entry('v2_friction1', 'Final-v2 materials, only finger friction reduced from 1.4 to 1.0')
    final_friction.update({'classification': 'final_configuration_finger_friction_stress_test',
                           'material_configuration': {**final, 'finger_mu': 1., 'effective_finger_steel_mu': 1.},
                           'included_in_nominal_hold_denominator': False,
                           'interpretation': 'Failed to lift. With lower finger friction the tool grip was not maintained; the controller later detected an unreachable IK target and held its last safe joint target. Steel-jaw friction remains 0.5, plastic remains 0.65, and their effective friction remains 0.65.'})
    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Contact-physics demonstration in Isaac Sim 5.0.0 using the supplied UR5e + right XHand asset.',
        'controller': {'type': 'Deterministic feedback controller with inverse kinematics and finger joint targets',
                       'observations': 'Privileged simulator tool pose and block pose; no camera-perception or hardware validation.',
                       'robot_gravity_compensation': True,
                       'tool': 'Two passive torsional-spring jaws; no commanded jaw actuators.',
                       'initialization': 'Tool is pre-positioned in the hand; reset fixture lasts 0.6 s and is disabled before approach.',
                       'block': 'Dynamic gravity-enabled rigid body with no joint constraints.'},
        'strict_hold_criteria': {'minimum_lift_m': .075, 'continuous_final_duration_s': 2.,
                                 'minimum_bilateral_contact_fraction': .95, 'minimum_per_jaw_normal_force_N': .001,
                                 'maximum_position_range_diagonal_m': .003, 'fixture_disabled_after_reset': True,
                                 'dynamic_block_without_constraints': True},
        'configuration_groups': [suite('final_v2_separate_materials', final, final_names),
                                 suite('preliminary_shared_high_friction', high, high_names)],
        'primary_demonstration': entry('final_demo', 'Repeat of final-v2 nominal physics with IK residual guard and joint-limit clipping; excluded from the six-condition denominator'),
        'stress_tests': [stress, final_friction, friction],
        'negative_controls': [release_control()],
        'pooled_success_rate': None,
        'pooling_policy': 'No overall success rate is computed across material versions, stress tests, repeated demonstrations, or intentional-release controls.',
        'limitations': ['Only the explicitly listed deterministic conditions were tested.',
                        'High-frequency contact-solver jitter remains; both physics-reported velocities and position-derived velocities are retained per run.',
                        'The task starts with a tool in the hand; acquisition of an unheld tweezer from the table is not demonstrated.',
                        'Controller robustness at other physics/control rates is not established; the 480/120 Hz test failed.',
                        'Reducing final-v2 finger friction from 1.4 to 1.0 failed and triggered the IK safeguard; friction robustness is limited.'],
        'validator_sha256': hashlib.sha256((ROOT / 'scripts/validate_run.py').read_bytes()).hexdigest(),
    }
    path = ROOT / 'reports/evaluation.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'path': str(path), 'groups': [{k: group[k] for k in ['id', 'passed_conditions', 'completed_conditions', 'planned_conditions']}
                                                 for group in report['configuration_groups']],
                      'primary_demo_pass': report['primary_demonstration']['strict_hold_success'],
                      'expected_release_pass': report['negative_controls'][0]['expected_release_behavior_passed']}, indent=2))


if __name__ == '__main__':
    main()
