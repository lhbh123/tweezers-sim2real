"""Independent numpy reconstruction of opposing broad-face normal pressure.

Uses raw world contact positions/forces and actual jaw body poses. No imported
runtime contact-classification code and no assumption of middle/web contact.
"""
from collections import defaultdict
import numpy as np

KEYS = ('broad_face_inward_force_N', 'broad_face_contact_count',
        'broad_face_closing_torque_Nm')


def point_values(p, f, sign, geometry, criteria):
    """p/f are expressed in the actual blade frame, never the handle frame."""
    p, f = np.asarray(p, float), np.asarray(f, float)
    if p.shape != (3,) or f.shape != (3,) or not np.isfinite([p, f]).all():
        raise ValueError('Invalid contact vector')
    length = float(geometry['length_m'])
    end = length-float(geometry.get('tip_length_m', .015))
    tip = p[2] > end
    width = float(geometry['strip_width_m'])
    if tip: width = min(.005, width)
    margin = width/2-abs(p[1])
    axial = min(p[2]-(end if tip else 0), (length if tip else end)-p[2])
    error = abs(sign*p[0]-float(geometry['strip_thickness_m'])/2)
    inward = -sign*f[0]
    cosine = inward/max(np.linalg.norm(f), 1e-12)
    ok = (margin >= criteria['width_margin_m'] and axial >= criteria['axial_margin_m']
          and error <= criteria['face_tolerance_m'] and inward > 1e-8
          and cosine >= criteria['min_inward_cos']
          and (criteria.get('allow_tip_contact', False) or not tip))
    return np.array([inward, 1., -sign*np.cross(p, f)[1]])*bool(ok)


def reconstruct_sample(sample, geometry, criteria, known_links, rotations, errors=None):
    values = np.zeros((3, 2, 2))
    raw = sample['grasp_contact_points']
    if type(raw['environment']) is not int or raw['environment'] != 0:
        raise ValueError('Wrong raw contact environment')
    for point in raw['normal_contacts']:
        name, jaw = point['robot'], point['tool']
        if name not in known_links:
            raise ValueError('Unknown source URDF contact link: '+name)
        if jaw not in ('handle', 'left', 'right'):
            raise ValueError('Unknown tool body: '+jaw)
        role = (0 if name.startswith('right_hand_thumb_') else
                1 if name.startswith('right_hand_index_') else None)
        if role is None or jaw == 'handle': continue
        state = np.asarray(sample[jaw], float)
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError('Invalid actual jaw pose')
        R = rotations(state[None, 3:7])[0]
        p = (np.asarray(point['point_world'])-state[:3]) @ R
        f = np.asarray(point['force_on_tool_N']) @ R
        j = ('left', 'right').index(jaw)
        contribution = point_values(p, f, -1 if j == 0 else 1, geometry, criteria)
        if errors is not None:
            for key,value in [('point_jaw_local_m',p),('force_jaw_local_N',f)]:
                saved = np.asarray(point[key],float)
                if saved.shape != (3,) or not np.isfinite(saved).all():
                    raise ValueError('Invalid raw local face diagnostic: '+key)
                errors[key] = max(errors[key],float(np.max(np.abs(saved-value))))
            if type(point['broad_face_qualified']) is not bool:
                raise ValueError('Nonboolean point face classification')
            errors['point_gate_disagreement'] += bool(contribution[1]) != point['broad_face_qualified']
        values[:, j, role] += contribution
    return dict(zip(KEYS, values))


def audit_face_pinch(rows, task, active, kin, rotations):
    if not task.get('require_face_pinch', False):
        return np.ones(len(rows), bool), {}, {'required': False}
    if task.get('require_tripod_support', False):
        raise ValueError('Optional-support task cannot require continuous tripod contact')
    if task.get('grasp_support_temporal_aggregation') != 'mean_over_every_physics_substep_in_control_interval':
        raise ValueError('Broad-face audit requires every physics substep')
    criteria, geometry = task['face_pinch_criteria'], task['tool_geometry']
    expected = task['face_pinch_assignment']
    if set(expected) != {'thumb','index'} or set(expected.values()) != {'left','right'}:
        raise ValueError('Thumb/index must press different blades')
    if any(not np.isfinite(criteria[k]) or criteria[k] < 0 for k in
           ('width_margin_m','axial_margin_m','face_tolerance_m','min_inward_cos')):
        raise ValueError('Invalid finite-face thresholds')
    known = {link.get('name') for link in kin.root.findall('link')}
    decimation, dt = task['decimation'], task['physics_dt']
    if type(decimation) is not int or decimation < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid contact sampling clock')
    errors = defaultdict(float); gates = []; forces = []; samples_count = 0
    for row in rows:
        samples = row['grasp_contact_substeps']
        if len(samples) != decimation: raise ValueError('Missing or duplicated physics samples')
        if not np.allclose([s['time_within_control_s'] for s in samples],
                           dt*np.arange(1,decimation+1),rtol=0,atol=1e-10):
            raise ValueError('Incorrect substep clock')
        for key in ('joint_pos','tool','left','right'):
            errors['endpoint'] = max(errors['endpoint'],float(np.max(np.abs(
                np.asarray(samples[-1][key])-np.asarray(row[key])))))
        errors['raw_endpoint'] += samples[-1]['grasp_contact_points'] != row['grasp_contact_points']
        computed = [reconstruct_sample(s,geometry,criteria,known,rotations,errors) for s in samples]
        means = {key:np.mean([c[key] for c in computed],axis=0) for key in KEYS}
        samples_count += len(samples)
        for key,value in means.items():
            actual = np.asarray(row['broad_face_contacts'][key],float)
            if actual.shape != (2,2) or not np.isfinite(actual).all():
                raise ValueError('Invalid broad face telemetry')
            errors[key] = max(errors[key],float(np.max(np.abs(actual-value))))
        pairs = [(('left','right').index(expected[role]),r) for r,role in enumerate(('thumb','index'))]
        inward = np.asarray([means[KEYS[0]][pair] for pair in pairs])
        counts = np.asarray([means[KEYS[1]][pair] for pair in pairs])
        gate = bool((inward>task['grasp_contact_threshold_N']).all() and (counts>0).all())
        if type(row['face_pinch_acceptable']) is not bool: raise ValueError('Nonboolean face gate')
        errors['gate_disagreement'] += gate != row['face_pinch_acceptable']
        errors['selected_force'] = max(errors['selected_force'],float(np.max(np.abs(
            inward-np.asarray(row['face_pinch_inward_force_N'])))))
        gates.append(gate); forces.append(inward)
    gates,active=np.asarray(gates,bool),np.asarray(active,bool)
    checks = {
        'face_normal_pressure_reconstructed_from_world_points':all(errors[k]<2e-5 for k in KEYS),
        'face_assignment_and_gate_reconstructed':errors['gate_disagreement']==0 and errors['selected_force']<2e-5,
        'raw_face_local_coordinates_and_flags_reconstructed':errors['point_gate_disagreement']==0 and errors['point_jaw_local_m']<2e-5 and errors['force_jaw_local_N']<2e-5,
        'face_all_substeps_and_endpoint_match':errors['endpoint']<1e-9 and errors['raw_endpoint']==0,
        'middle_web_contact_optional_in_task':task.get('middle_web_support_requirement')=='optional_backup_contacts_not_rewarded_or_required',
        'tool_tail_has_no_bent_extension':not geometry.get('handle_extension'),
    }
    evidence = dict(required=True,physics_samples_reconstructed=samples_count,criteria=criteria,
        assignment=expected,max_reconstruction_errors=dict(errors),
        active_broad_face_pinch_fraction=float(gates[active].mean()),
        active_inward_normal_force_mean_N=np.asarray(forces)[active].mean(0).tolist(),
        force_scope='Qualified normal forces only; friction excluded from closing criterion.',
        middle_web='Optional contacts; their absence does not invalidate the pinch.',
        limitations=['Control-interval mean pressure does not assert every substep has simultaneous contact.',
                    'Collision mesh fidelity and real finger-pad geometry need separate source-mesh audit.'])
    return gates, {k:bool(v) for k,v in checks.items()}, evidence
