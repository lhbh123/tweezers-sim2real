"""Audit measured optional middle-finger support without running a simulator.

Examples:
  python3 analysis/sharpa_grasp/audit_natural_support.py RUN
  python3 analysis/sharpa_grasp/audit_natural_support.py RUN \
      --baseline-window .8 1 --window loose 1.5 3 --compare-envs 0 1

Probe input: report.json + timeseries.npz, with every physics-step role forces.
Task input: evaluation.json + trajectory.npz + grasp_substeps.jsonl. Current
task JSONL records env 0 only; raw points are streamed and summed per substep.
No contact absence, configuration field, or rendered pose proves catch ability.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rotate(q, v):
    """Quaternion wxyz; normalize the recorded quaternion before use."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Invalid zero quaternion in recorded rigid state')
    q = q / norm
    v = np.broadcast_to(np.asarray(v), q.shape[:-1] + (3,))
    t = 2. * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


def mass_model(geo):
    """Uniform density within the authored box colliders of each rigid body."""
    length = float(geo['length_m'])
    tip = float(geo.get('tip_length_m', .015))
    width = float(geo['strip_width_m'])
    thick = float(geo['strip_thickness_m'])
    handle_len = float(geo.get('handle_length_m', .006))
    handle_width = float(geo.get('handle_width_m', width))
    bridge = 2. * float(geo.get('hinge_offset_m', .0045)) + thick
    volume = bridge * handle_width * handle_len
    handle_com = volume * np.array([0., 0., -.5 * handle_len])
    if geo.get('handle_extension'):
        ext = geo['handle_extension']
        start, end = [np.asarray(ext[k], dtype=float) for k in ['start_m', 'end_m']]
        added = np.linalg.norm(end-start) * ext['width_x_m'] * ext['width_y_m']
        handle_com += added * .5 * (start+end)
        volume += added
    handle_com /= volume
    blade = length-tip
    volumes = np.array([blade*width*thick, tip*min(.005, width)*thick])
    jaw_z = float(volumes @ np.array([.5*blade, length-.5*tip]) / volumes.sum())
    masses = np.array([geo.get('handle_mass_kg', .003),
                       geo.get('jaw_mass_kg', .006), geo.get('jaw_mass_kg', .006)], dtype=float)
    if np.any(masses <= 0):
        raise ValueError('Tool body masses must be positive')
    return masses, np.array([handle_com, [0., 0., jaw_z], [0., 0., jaw_z]])


def world_com(bodies, masses, centres):
    bodies = np.asarray(bodies, dtype=float)
    positions = bodies[..., :3] + rotate(bodies[..., 3:7], centres)
    return np.sum(positions*masses[:, None], axis=-2) / masses.sum()


def probe_pose(run, report):
    name = Path(report['arguments']['pose'])
    for path in [run/'source'/name.name, run/name.name, name]:
        if path.is_file():
            return read_json(path), path
    raise FileNotFoundError('Cannot find the probe pose configuration')


def load_probe(run):
    report = read_json(run/'report.json')
    pose, pose_path = probe_pose(run, report)
    d = np.load(run/'timeseries.npz')
    dt = float(report['physics_dt_s'])
    t = np.asarray(d['time'], dtype=float)
    if len(t) < 2 or not np.allclose(np.diff(t), dt, atol=max(1e-7, dt*1e-3)):
        raise ValueError('Probe records must contain every physics step')
    masses, centres = mass_model(pose['tool_geometry'])
    com = world_com(d['bodies'], masses, centres)
    normal = d['role_normal_force_w'][:, :, 2, 2]
    friction = d['role_friction_force_w'][:, :, 2, 2]
    net = d['role_support_z_N'][:, :, 2]
    if not np.allclose(normal+friction, net, atol=1e-5, rtol=1e-5):
        raise ValueError('Recorded middle net force does not match normal + friction')
    rows = []
    for env, candidate in enumerate(report['candidates']):
        rows.append(dict(environment=env, name=candidate['name'], time=t, dt=np.full(len(t), dt),
            normal_z=normal[:, env], friction_z=friction[:, env], net_z=net[:, env],
            strength=d['role_contact_strength_N'][:, env, 2],
            under=d['middle_under_contact_count'][:, env], com=com[:, env],
            grip=d['grip_error_m'][:, env], rotation=d['grip_orientation_error'][:, env],
            recorded_result={k:candidate[k] for k in ['retained', 'stable_hold', 'broad_face_hold'] if k in candidate},
            phase=None, link_points=None))
    return rows, pose, masses, centres, dict(kind='probe',
        files=[run/'report.json', run/'timeseries.npz', pose_path],
        physics_dt_s=dt,
        force_scope='Role tensors at every physics step; no temporal subsampling.',
        retention_scope='Probe measurements at every physics step.',
        contact_link_scope='Probe tensors group all middle links; individual fingertip/MP attribution is not inferred.',
        simulation_object_writes=report.get('runtime_object_state_writes'))


def middle_raw(points):
    """Reconstruct the reader's role vectors and sum of body/link magnitudes."""
    pair_normal = defaultdict(lambda: np.zeros(3))
    normal = np.zeros(3)
    friction = np.zeros(3)
    under = 0
    links = defaultdict(int)
    for point in points['normal_contacts']:
        if point.get('role') != 'middle':
            continue
        force = np.asarray(point['force_on_tool_N'], dtype=float)
        normal += force
        pair_normal[(point['tool'], point['robot'])] += force
        links[point['robot']] += 1
        under += int(point['radial_offset_world_z_m'] < -.0001 and np.linalg.norm(force) > 1e-8)
    for point in points['friction_anchors']:
        if point.get('role') == 'middle':
            friction += np.asarray(point['force_on_tool_N'], dtype=float)
    strength = sum(float(np.linalg.norm(v)) for v in pair_normal.values())
    return float(normal[2]), float(friction[2]), strength, under, links


def middle_aggregated(values):
    """Accept explicit single-env tensors, or env 0 of batched tensors."""
    def role(key, vector=False):
        a = np.asarray(values[key], dtype=float)
        if a.ndim == (3 if vector else 2):
            a = a[0]
        return a[2, 2] if vector else a[2]
    under = float(np.asarray(values['middle_under_contact_count']).reshape(-1)[0])
    return (float(role('role_normal_force_w', True)), float(role('role_friction_force_w', True)),
            float(role('role_contact_strength_N')), under)


def load_task(run):
    result = read_json(run/'evaluation.json')
    task = result['task']
    d = np.load(run/'trajectory.npz')
    path = run/'grasp_substeps.jsonl'
    if not path.is_file():
        raise FileNotFoundError('Task support forces unavailable: rerun with --record-substeps. Missing data is not zero force.')
    control_dt = float(d['dt'])
    physics_dt = float(task['physics_dt'])
    decimation = int(task['decimation'])
    masses, centres = mass_model(task['tool_geometry'])
    rows = defaultdict(list)
    link_counts = defaultdict(int)
    sources = defaultdict(int)
    lines = 0
    with path.open() as stream:
        for k, text in enumerate(stream):
            record = json.loads(text)
            lines += 1
            if k >= len(d['active']) or abs(float(record['time'])-(k+1)*control_dt) > 1e-5:
                raise ValueError(f'JSONL/control trajectory time mismatch at line {k+1}')
            steps = record['substeps']
            if len(steps) != decimation:
                raise ValueError(f'Missing physics steps at line {k+1}')
            if not bool(d['active'][k, 0]):
                continue
            for j, step in enumerate(steps):
                offset = float(step['time_within_control_s'])
                if abs(offset-(j+1)*physics_dt) > 1e-8:
                    raise ValueError('Unexpected physics-substep time offset')
                aggregate = step.get('grasp_support', step)
                if all(key in aggregate for key in ['role_normal_force_w', 'role_friction_force_w',
                                                     'role_contact_strength_N', 'middle_under_contact_count']):
                    nz, fz, strength, under = middle_aggregated(aggregate)
                    sources['aggregated_role_tensors'] += 1
                else:
                    points = step['grasp_contact_points']
                    if int(points.get('environment', 0)) != 0:
                        raise ValueError('Current task recorder supports environment 0 only')
                    nz, fz, strength, under, links = middle_raw(points)
                    for name, count in links.items():
                        link_counts[name] += count
                    sources['streamed_raw_contact_points'] += 1
                rows['time'].append(k*control_dt+offset)
                rows['dt'].append(physics_dt)
                rows['normal_z'].append(nz)
                rows['friction_z'].append(fz)
                rows['net_z'].append(nz+fz)
                rows['strength'].append(strength)
                rows['under'].append(under)
                rows['com'].append(world_com([step[name] for name in ['tool', 'left', 'right']], masses, centres))
                rows['grip'].append(float(d['retained'][k, 0]))
                rows['rotation'].append(float(d['orientation_error'][k, 0]))
                rows['phase'].append(int(d['phase'][k, 0]))
    if lines != len(d['active']):
        raise ValueError('JSONL ends before the saved control trajectory')
    row = {key:np.asarray(value) for key,value in rows.items()}
    row.update(environment=0, name='environment_0', link_points=dict(link_counts),
               recorded_result={'task_success': bool(result['success_per_env'][0])})
    return [row], task, masses, centres, dict(kind='task',
        files=[run/'evaluation.json', run/'trajectory.npz', path],
        physics_dt_s=physics_dt, control_dt_s=control_dt,
        force_scope='Every active env-0 physics substep, read one JSONL control record at a time.',
        retention_scope='Associated control-end grip/orientation measurements repeated over its substeps; not independent substep retention.',
        contact_link_scope='Actual raw normal-contact source links; aggregated-only records cannot identify individual links.',
        force_record_sources=dict(sources), recorded_environment_count=1,
        task_environment_count=result['episodes'],
        simulation_object_writes=task.get('object_runtime_writes'))


def longest_duration(mask, dt):
    best = current = 0.
    for yes, duration in zip(mask, dt):
        current = current+float(duration) if yes else 0.
        best = max(best, current)
    return best


def window_summary(row, mask, baseline_z, weight, detection, substantial):
    t, dt = row['time'], row['dt']
    if not mask.any():
        return {'samples': 0, 'status': 'no_samples_in_window'}
    duration = float(dt[mask].sum())
    mean = lambda value: float(np.sum(np.asarray(value)[mask]*dt[mask])/duration)
    net = row['net_z']
    contact = row['strength'] > detection
    under = row['under'] > 0
    support = contact & under & (net > detection)
    load = contact & under & (net > substantial)
    drop = baseline_z-row['com'][:, 2]
    speed = np.gradient(row['com'][:, 2], t)
    retained = (row['grip'] < .035) & (row['rotation'] < .3)
    last = int(np.flatnonzero(mask)[-1])
    return dict(samples=int(mask.sum()), sampled_duration_s=duration,
        first_sample_end_s=float(t[mask][0]), last_sample_end_s=float(t[last]),
        middle_contact_fraction=mean(contact), middle_under_contact_fraction=mean(under),
        middle_net_upward_mean_N=mean(net), middle_normal_upward_mean_N=mean(row['normal_z']),
        middle_friction_upward_mean_N=mean(row['friction_z']),
        middle_net_upward_min_N=float(net[mask].min()), middle_net_upward_max_N=float(net[mask].max()),
        middle_mean_load_fraction_of_tool_weight=mean(net)/weight,
        middle_net_vertical_impulse_Ns=float(np.sum(net[mask]*dt[mask])),
        middle_positive_vertical_impulse_Ns=float(np.sum(np.maximum(0., net[mask])*dt[mask])),
        qualifying_under_support_fraction=mean(support),
        qualifying_under_support_duration_s=float(dt[mask & support].sum()),
        longest_continuous_under_support_s=longest_duration(mask & support, dt),
        substantial_under_support_duration_s=float(dt[mask & load].sum()),
        longest_continuous_substantial_support_s=longest_duration(mask & load, dt),
        support_while_tool_retained_duration_s=float(dt[mask & support & retained].sum()),
        longest_continuous_support_while_tool_retained_s=longest_duration(mask & support & retained, dt),
        COM_drop_from_baseline_max_mm=float(drop[mask].max()*1000),
        COM_drop_from_baseline_mean_mm=mean(drop)*1000,
        COM_drop_from_baseline_final_mm=float(drop[last]*1000),
        COM_vertical_speed_mean_mm_s=mean(speed)*1000,
        COM_vertical_speed_final_mm_s=float(speed[last]*1000),
        tool_retained_fraction=mean(retained), final_tool_retained=bool(retained[last]),
        grip_error_max_m=float(row['grip'][mask].max()),
        grip_rotation_error_max_sin_half_squared=float(row['rotation'][mask].max()))


def audit(args):
    run = args.run
    kind = args.kind
    if kind == 'auto':
        kind = 'probe' if (run/'timeseries.npz').is_file() else 'task'
    rows, metadata, masses, centres, source = (load_probe if kind == 'probe' else load_task)(run)
    output = dict(schema_version=1, run=str(run), evidence_label=args.evidence_label,
        generated_by='analysis/sharpa_grasp/audit_natural_support.py',
        audit_script_sha256=sha256(Path(__file__)),
        thresholds={'contact_and_net_upward_N': args.contact_threshold,
                    'under_axis_radial_world_z_less_than_m': -.0001,
                    'substantial_net_upward_N': args.substantial_threshold,
                    'sustained_substantial_support_s': args.sustained_seconds,
                    'retained_grip_error_m': .035, 'retained_rotation_sin_half_squared': .3},
        tool_mass_kg=float(masses.sum()), tool_weight_N=float(masses.sum()*9.81),
        COM_model={'body_order':['handle','left','right'], 'masses_kg':masses.tolist(),
                   'local_COM_m':centres.tolist(),
                   'assumption':'Uniform density over authored straight-tool box volumes, no custom COM/inertia override.',
                   'velocity':'Finite differences of reconstructed world COM position.'},
        geometric_backup={'declared_in_config':metadata.get('backup_support_geometry'),
            'independently_verified_by_this_script':False,
            'meaning':'Configuration geometry is a separate claim. Zero measured force does not verify or disprove a suitable backup position; measured support does not by itself validate source-mesh clearance.'},
        source={k:v for k,v in source.items() if k != 'files'},
        source_sha256={str(p):sha256(p) for p in source['files']}, rows=[],
        limitations=[
            'Only saved data is analyzed. This script applies no simulation forces, state writes, or controller changes.',
            'Force vectors act on the tool; net upward force is normal plus friction. Signed and positive-only impulses are both reported.',
            'Under-axis contact counts and net middle load are role-level evidence. Other simultaneous middle contacts can contribute to the net force.',
            'World COM descent includes commanded arm motion. It proves tool slip only with a suitable matched, approximately fixed-wrist comparison.',
            'A sustained-support flag is diagnostic, not a changed task success gate or proof of robust catch ability.',
            'Task current JSONL covers environment 0 only. Probe role tensors cannot distinguish a middle MP contact from a fingertip sensor contact.',
            'Time windows use sample-end times in (start, end]; impulses use the full corresponding physics-step duration.'])
    for row in rows:
        for key in ['time','dt','normal_z','friction_z','net_z','strength','under','com','grip','rotation']:
            if not np.isfinite(row[key]).all():
                raise ValueError(f'Nonfinite {key} in environment {row["environment"]}')
        if len(row['time']) < 2 or np.any(np.diff(row['time']) <= 0):
            raise ValueError('Insufficient or nonmonotonic sample times')
        t = row['time']
        baseline = (t > args.baseline_window[0]+1e-9) & (t <= args.baseline_window[1]+1e-7)
        if not baseline.any():
            raise ValueError('The requested baseline window has no recorded samples')
        baseline_z = float(np.average(row['com'][baseline, 2], weights=row['dt'][baseline]))
        windows = {'whole_run':np.ones(len(t), dtype=bool), 'baseline':baseline,
                   'last_1s':t > t[-1]-1.+1e-9}
        for name, start, end in args.window:
            if name in windows or float(end) <= float(start):
                raise ValueError(f'Invalid or repeated named window {name}')
            windows[name] = (t > float(start)+1e-9) & (t <= float(end)+1e-7)
        if row['phase'] is not None:
            for p in np.unique(row['phase']):
                label = metadata.get('phases', [])[int(p)] if metadata.get('phases') else str(p)
                windows['phase_'+label] = row['phase'] == p
        stats = {name:window_summary(row, mask, baseline_z, output['tool_weight_N'],
                                    args.contact_threshold, args.substantial_threshold)
                 for name,mask in windows.items()}
        whole = stats['whole_run']
        output['rows'].append(dict(environment=row['environment'], name=row['name'],
            recorded_result=row['recorded_result'], baseline_window_s=args.baseline_window,
            baseline_COM_world_z_m=baseline_z, windows=stats,
            actual_middle_under_support_observed=whole['qualifying_under_support_duration_s'] > 0,
            sustained_substantial_support_observed=whole['longest_continuous_substantial_support_s'] >= args.sustained_seconds,
            raw_middle_normal_contact_point_counts_by_link=row['link_points']))
    if args.compare_envs is not None:
        lookup = {r['environment']:r for r in output['rows']}
        first, second = [lookup[i] for i in args.compare_envs]
        output['paired_comparison'] = {'environment_order':args.compare_envs,
            'interpretation':'First minus second. The user must establish that only the intended middle posture differs.',
            'windows':{name:{key+'_first_minus_second':first['windows'][name][key]-second['windows'][name][key]
                            for key in ['COM_drop_from_baseline_max_mm','COM_drop_from_baseline_mean_mm',
                                        'middle_net_upward_mean_N','qualifying_under_support_duration_s']}
                       for name in first['windows'] if first['windows'][name]['samples'] and second['windows'][name]['samples']}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--kind', choices=['auto','probe','task'], default='auto')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--baseline-window', type=float, nargs=2, default=[0., .5])
    parser.add_argument('--window', nargs=3, action='append', default=[], metavar=('NAME','START','END'))
    parser.add_argument('--compare-envs', type=int, nargs=2)
    parser.add_argument('--contact-threshold', type=float, default=.003)
    parser.add_argument('--substantial-threshold', type=float, default=.03)
    parser.add_argument('--sustained-seconds', type=float, default=.1)
    parser.add_argument('--evidence-label', default='Recorded run only; no transfer of evidence to another pose/version.')
    args = parser.parse_args()
    if not (0 < args.contact_threshold <= args.substantial_threshold and args.sustained_seconds > 0
            and 0 <= args.baseline_window[0] < args.baseline_window[1]):
        parser.error('Invalid force, duration or baseline thresholds')
    result = audit(args)
    path = args.out or args.run/'natural_support_audit.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report':str(path), 'kind':result['source']['kind'],
                     'rows':[{'environment':r['environment'], 'name':r['name'],
                              'middle_under_support_observed':r['actual_middle_under_support_observed'],
                              'longest_support_s':r['windows']['whole_run']['longest_continuous_under_support_s']}
                             for r in result['rows']]}, indent=2))


if __name__ == '__main__':
    main()
