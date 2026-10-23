"""Independently reconstruct recorded tripod support from raw contact points.

For substep-mean telemetry, geometry and force classification are reconstructed
at *each* source-FK pose before the values (including counts) are averaged. The
support gate is applied to those means, not to averaged Boolean pass flags.
This checks recorded telemetry, not cooked collision fidelity or provenance.
"""
from collections import defaultdict
import numpy as np


MEAN_AGGREGATION = 'mean_over_every_physics_substep_in_control_interval'
ROLES = ('thumb', 'index', 'middle', 'wrist_palm', 'other')
SUPPORT_KEYS = ('role_normal_force_w', 'role_friction_force_w',
                'role_contact_strength_N', 'role_support_z_N',
                'web_contact_count', 'web_normal_force_w',
                'web_friction_force_w', 'middle_under_contact_count')


def _finite(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Invalid ' + label)
    return result


def _classify(name, known_links):
    if name not in known_links:
        raise ValueError('Unknown source URDF robot link: ' + str(name))
    if name.startswith('right_hand_thumb_'):
        return 'thumb'
    if name.startswith('right_hand_index_'):
        return 'index'
    if name.startswith('right_hand_mid_'):
        return 'middle'
    return 'wrist_palm' if name == 'wrist_3_link' else 'other'


def _inside_region(point, region):
    if 'bbox_min_m' in region:
        return bool(((point >= region['bbox_min_m']) &
                     (point <= region['bbox_max_m'])).all())
    return bool(np.linalg.norm(point - region['center_m']) <= region['radius_m'])


def _support_gate(values, contact_threshold, vertical_threshold):
    return bool((values['role_contact_strength_N'][:3] > contact_threshold).all()
                and values['middle_under_contact_count'] > 0
                and values['role_support_z_N'][2] > vertical_threshold
                and values['web_contact_count'] > 0
                and np.linalg.norm(values['web_normal_force_w']) > contact_threshold
                and (values['web_normal_force_w'] +
                     values['web_friction_force_w'])[2] > vertical_threshold)


def _reconstruct_sample(sample, names, base, kin, rotation, region, known_links, errors):
    q = _finite(sample['joint_pos'], (len(names),), 'contact sample joint state')
    hand = base @ kin.fk(dict(zip(names, q)))['right_hand_link']
    bodies = {n: _finite(sample['tool' if n == 'handle' else n], (13,),
                         'contact sample ' + n + ' state')
              for n in ('handle', 'left', 'right')}
    matrices = {n: rotation(s[None, 3:7])[0] for n, s in bodies.items()}
    pair_normal = defaultdict(lambda: np.zeros(3))
    normal = np.zeros((5, 3)); friction = np.zeros_like(normal)
    web_normal = np.zeros(3); web_friction = np.zeros(3)
    web_count = middle_count = 0
    raw = sample['grasp_contact_points']
    if type(raw['environment']) is not int or raw['environment'] != 0:
        raise ValueError('Wrong environment in contact audit')
    for kind, key in (('normal', 'normal_contacts'), ('friction', 'friction_anchors')):
        if not isinstance(raw[key], list):
            raise ValueError('Invalid raw contact list: ' + key)
        for point in raw[key]:
            p = _finite(point['point_world'], (3,), 'raw point_world')
            force = _finite(point['force_on_tool_N'], (3,), 'raw contact force')
            saved_hand = _finite(point['point_hand'], (3,), 'raw point_hand')
            robot, body = point['robot'], point['tool']
            role_name = _classify(robot, known_links)
            role = ROLES.index(role_name)
            if body not in bodies:
                raise ValueError('Unknown tool body in contact audit: ' + str(body))
            errors['role_classification_disagreements'] += int(point['role'] != role_name)
            ph = (p - hand[:3, 3]) @ hand[:3, :3]
            errors['hand_point_coordinates_m'] = max(errors['hand_point_coordinates_m'],
                                                     float(np.max(np.abs(ph - saved_hand))))
            web = body == 'handle' and robot == 'wrist_3_link' and _inside_region(ph, region)
            if type(point['within_web_region']) is not bool:
                raise ValueError('Web classification must be Boolean')
            errors['web_classification_disagreements'] += int(web != point['within_web_region'])
            if kind == 'normal':
                normal[role] += force
                pair_normal[(body, robot)] += force
                axis = matrices[body][:, 2]
                delta = p - bodies[body][:3]
                radial_z = delta[2] - axis[2] * (delta @ axis)
                saved_radial = _finite(point['radial_offset_world_z_m'], (), 'radial offset')
                errors['radial_offset_m'] = max(errors['radial_offset_m'],
                                               float(abs(radial_z - saved_radial)))
                # The raw reader excludes zero normal points, as does the GPU count.
                nonzero = np.linalg.norm(force) > 1e-8
                if role == 2 and radial_z < -.0001 and nonzero:
                    middle_count += 1
                if web and nonzero:
                    web_normal += force
                    web_count += 1
            else:
                friction[role] += force
                if web:
                    web_friction += force
    strength = np.zeros(5)
    for (_, robot), force in pair_normal.items():
        strength[ROLES.index(_classify(robot, known_links))] += np.linalg.norm(force)
    return dict(role_normal_force_w=normal, role_friction_force_w=friction,
                role_contact_strength_N=strength, role_support_z_N=(normal + friction)[:, 2],
                web_contact_count=float(web_count), web_normal_force_w=web_normal,
                web_friction_force_w=web_friction, middle_under_contact_count=float(middle_count))


def audit_support(rows, task, usd, active, kin, rotations_from_wxyz):
    if not bool(task.get('require_tripod_support', False)):
        return np.ones(len(rows), bool), {}, {'required': False}
    if not rows or not all('grasp_contact_points' in r and 'grasp_support' in r and
                           'grasp_support_acceptable' in r for r in rows):
        raise ValueError('Tripod acceptance requires raw contact points and support telemetry in every frame')
    active = np.asarray(active, dtype=bool)
    if active.shape != (len(rows),) or not active.any():
        raise ValueError('Support audit requires at least one active frame')
    mode = task.get('grasp_support_temporal_aggregation')
    mean_mode = mode == MEAN_AGGREGATION
    if mode not in (None, 'instantaneous', MEAN_AGGREGATION):
        raise ValueError('Unsupported contact temporal aggregation: ' + str(mode))
    if not mean_mode and any('grasp_contact_substeps' in r for r in rows):
        raise ValueError('Substep records require explicit mean temporal-aggregation metadata')
    decimation = task.get('decimation')
    dt = float(task.get('physics_dt', 0))
    if mean_mode and (type(decimation) is not int or decimation <= 0 or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Invalid physics timestep or decimation for substep averaging')
    region = dict(task['web_contact_region'])
    if region.get('frame') != 'right_hand_link':
        raise ValueError('Web contact ROI must be in right_hand_link source-FK frame')
    if 'bbox_min_m' in region:
        low = _finite(region['bbox_min_m'], (3,), 'ROI minimum')
        high = _finite(region['bbox_max_m'], (3,), 'ROI maximum')
        if not (low < high).all():
            raise ValueError('Empty web contact ROI')
    else:
        _finite(region['center_m'], (3,), 'ROI center')
        radius = _finite(region['radius_m'], (), 'ROI radius')
        if radius <= 0:
            raise ValueError('Web contact ROI radius must be positive')
    base = _finite(usd['robot_base_world_matrix_gf_row_vectors'], (4, 4), 'robot base transform').T
    names = task['joint_names']
    if len(names) != len(set(names)) or set(names) != set(kin.active):
        raise ValueError('Contact FK requires exactly the source URDF active joint names')
    known_links = {e.get('name') for e in kin.root.findall('link')}
    ct = float(task['grasp_contact_threshold_N']); zt = float(task['support_vertical_threshold_N'])
    if not np.isfinite([ct, zt]).all() or ct < 0 or zt < 0:
        raise ValueError('Invalid support force thresholds')
    tolerance = 2e-5
    errors = defaultdict(float)
    computed = []; role_contact = []; middle_up = []; web_up = []
    substep_pass = []; substep_counts = []; recorded = []
    for row in rows:
        if type(row['grasp_support_acceptable']) is not bool:
            raise ValueError('Logged support acceptance must be Boolean')
        if mean_mode:
            samples = row.get('grasp_contact_substeps')
            if not isinstance(samples, list) or len(samples) != decimation:
                raise ValueError('Missing/extra physics substeps: expected ' + str(decimation))
            times = _finite([s['time_within_control_s'] for s in samples],
                            (decimation,), 'physics substep timestamps')
            clock_error = float(np.max(np.abs(times - dt * np.arange(1, decimation + 1))))
            errors['substep_clock_s'] = max(errors['substep_clock_s'], clock_error)
            if clock_error > 1e-10:
                raise ValueError('Missing, duplicated, or unordered physics substep timestamps')
            for key in ('joint_pos', 'tool', 'left', 'right'):
                expected_shape = (len(names),) if key == 'joint_pos' else (13,)
                final = _finite(samples[-1][key], expected_shape, 'last substep ' + key)
                endpoint = _finite(row[key], expected_shape, 'row endpoint ' + key)
                errors['last_substep_endpoint'] = max(errors['last_substep_endpoint'],
                                                       float(np.max(np.abs(final - endpoint))))
            errors['last_substep_raw_contact_disagreements'] += int(
                samples[-1]['grasp_contact_points'] != row['grasp_contact_points'])
        else:
            samples = [row]
        values = [_reconstruct_sample(s, names, base, kin, rotations_from_wxyz,
                                      region, known_links, errors) for s in samples]
        expected = {key: np.mean([v[key] for v in values], axis=0) for key in SUPPORT_KEYS}
        for key, value in expected.items():
            actual = _finite(row['grasp_support'][key], np.shape(value), 'support telemetry ' + key)
            errors[key] = max(errors[key], float(np.max(np.abs(actual - value))))
        ok = _support_gate(expected, ct, zt)
        computed.append(ok); recorded.append(row['grasp_support_acceptable'])
        role_contact.append(expected['role_contact_strength_N'][:3] > ct)
        middle_up.append(expected['role_support_z_N'][2])
        web_up.append((expected['web_normal_force_w'] + expected['web_friction_force_w'])[2])
        substep_pass.append(np.mean([_support_gate(v, ct, zt) for v in values]))
        substep_counts.append(len(samples))
    computed = np.asarray(computed, dtype=bool)
    checks = {
        'raw_contact_roles_match_source_URDF_links': errors['role_classification_disagreements'] == 0,
        'raw_contact_hand_coordinates_match_source_FK': errors['hand_point_coordinates_m'] < tolerance,
        'raw_contact_radial_offsets_match_actual_tool_bodies': errors['radial_offset_m'] < tolerance,
        'raw_web_contacts_are_tail_in_anatomical_ROI': errors['web_classification_disagreements'] == 0,
        'support_force_sums_match_raw_normal_and_friction_points': all(errors[k] < tolerance for k in SUPPORT_KEYS),
        'support_acceptance_recomputed_matches_log': bool(np.array_equal(computed, recorded)),
    }
    if mean_mode:
        checks.update({
            'support_mean_covers_every_physics_substep': all(n == decimation for n in substep_counts),
            'support_last_substep_matches_recorded_endpoint': errors['last_substep_endpoint'] < 1e-9,
            'support_last_substep_raw_contacts_match_endpoint': errors['last_substep_raw_contact_disagreements'] == 0,
        })
    evidence = {
        'required': True,
        'temporal_aggregation': mode if mean_mode else 'legacy_instantaneous',
        'gate_semantics': 'Reconstruct forces/counts per sample, arithmetic-average values, then apply support thresholds. Boolean gates are never averaged for acceptance.',
        'physics_samples_reconstructed': int(sum(substep_counts)),
        'samples_per_control_interval_min_max': [min(substep_counts), max(substep_counts)],
        'max_reconstruction_errors': dict(errors),
        'active_support_fraction': float(computed[active].mean()),
        'active_contact_fraction_thumb_index_middle': np.asarray(role_contact)[active].mean(0).tolist(),
        'active_middle_upward_force_N_min_max': [float(np.min(np.asarray(middle_up)[active])), float(np.max(np.asarray(middle_up)[active]))],
        'active_web_upward_force_N_min_max': [float(np.min(np.asarray(web_up)[active])), float(np.max(np.asarray(web_up)[active]))],
        'active_individual_sample_gate_fraction_diagnostic_only': float(np.asarray(substep_pass)[active].mean()),
        'limitations': [
            'Contact data describe cooked collision shapes; source STL surface fidelity is audited separately.',
            ('Acceptance uses the control-interval mean. It does not assert simultaneous support in every individual physics substep.'
             if mean_mode else 'Legacy raw samples are at the control rate, not every physics substep.'),
            'Internal consistency checks cannot establish authenticity of coherently altered raw telemetry.'
        ]}
    return computed, {k: bool(v) for k, v in checks.items()}, evidence
