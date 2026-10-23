"""Independent source-geometry audit of a recorded selected PPO environment.

Uses every recorded physics substep when available, original URDF collisions,
actual tool rigid states, and authored USD tool boxes. Does not run simulation,
change task thresholds, or infer contact forces from geometric overlap.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'analysis')]
from kinematics import ASSET, Kinematics, stl_triangles, transform, xyz
from validate_rl_hand_geometry import Mesh, mesh_sat


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def origin(element):
    node = element.find('origin')
    return np.eye(4) if node is None else transform(
        xyz(node.get('xyz')), Rotation.from_euler('xyz', xyz(node.get('rpy'))).as_matrix())


def rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-4:
        raise ValueError('Invalid recorded or authored quaternion')
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def load_geometry(kin):
    """Retain exact mesh support vertices, cylinders, and all hand triangles."""
    meshes, supports, cylinders, provenance = {}, {}, {}, []
    hand_links = set(kin.fk(root='right_hand_link'))
    mounted = {'base_link', 'base_link_inertia'}
    for link in kin.root.findall('link'):
        name = link.get('name')
        if name in mounted:
            continue
        triangles = []
        for ci, collision in enumerate(link.findall('collision')):
            geometry, matrix = collision.find('geometry'), origin(collision)
            mesh = geometry.find('mesh')
            if mesh is not None:
                path = ASSET / mesh.get('filename')
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                tri = stl_triangles(path) * scale
                tri = tri @ matrix[:3, :3].T + matrix[:3, 3]
                triangles.append(tri)
                provenance.append(dict(link=name, collision_index=ci, path=str(path),
                                       sha256=sha(path), scale=scale.tolist(), origin=matrix.tolist()))
            elif geometry.find('cylinder') is not None:
                node = geometry.find('cylinder')
                cylinders.setdefault(name, []).append((matrix, float(node.get('radius')), float(node.get('length')) / 2))
                provenance.append(dict(link=name, collision_index=ci, primitive='cylinder',
                                       radius_m=float(node.get('radius')), length_m=float(node.get('length')),
                                       origin=matrix.tolist()))
            else:
                raise ValueError(f'Unhandled source collision geometry: {name} #{ci}')
        if triangles:
            tri = np.concatenate(triangles)
            vertices = np.unique(tri.reshape(-1, 3), axis=0)
            # Plane support is exact at these vertices; no hull collision proxy.
            supports[name] = vertices[ConvexHull(vertices).vertices]
            if name in hand_links:
                meshes[name] = Mesh(tri)
    return meshes, supports, cylinders, provenance, sorted(mounted)


def recorded_samples(rows, task):
    samples = []
    mean_mode = task.get('grasp_support_temporal_aggregation') == 'mean_over_every_physics_substep_in_control_interval'
    for frame, row in enumerate(rows):
        substeps = row.get('grasp_contact_substeps')
        if substeps is None:
            if mean_mode:
                raise ValueError(f'Frame {frame} is missing physics substeps')
            samples.append(dict(row, time_s=float(row['time']), frame_index=frame, substep_index=None))
            continue
        if len(substeps) != task['decimation']:
            raise ValueError(f'Frame {frame}: wrong physics substep count')
        for j, substep in enumerate(substeps):
            expected = (j + 1) * task['physics_dt']
            if abs(substep['time_within_control_s'] - expected) > 1e-10:
                raise ValueError(f'Frame {frame}, substep {j}: incorrect timestamp')
            sample = dict(substep, time_s=float(row['time']) - task['physics_dt'] * task['decimation'] + expected,
                          frame_index=frame, substep_index=j)
            for key in ['joint_pos', 'tool', 'left', 'right']:
                if j == len(substeps) - 1 and not np.array_equal(substep[key], row[key]):
                    raise ValueError(f'Frame {frame}: endpoint {key} differs from final substep')
            samples.append(sample)
    if np.any(np.diff([s['time_s'] for s in samples]) <= 0):
        raise ValueError('Recorded sample timestamps are not strictly increasing')
    return samples, mean_mode


def elevated_endpoint_clearance(rows, task, usd, tool_corners):
    """Cube states exist at control endpoints, so make no substep lift claim."""
    threshold = .9 * task['target_lift']
    table = usd['table_top_world_z_m']
    rest_height = table + usd['block_size_m'][2] / 2 + .0001
    selected, minima = [], []
    for index, row in enumerate(rows):
        if float(row['cube'][2]) - rest_height < threshold:
            continue
        values = []
        for name, corners in tool_corners.items():
            body = {'Handle': 'tool', 'Left': 'left', 'Right': 'right'}[name.split('/')[0]]
            state = np.asarray(row[body])
            values.append(float((corners @ rotation(state[3:7])[2] + state[2]).min() - table))
        selected.append(index)
        minima.append(min(values))
    return dict(cube_lift_threshold_m=threshold, qualifying_control_endpoint_count=len(selected),
                minimum_all_tool_parts_table_clearance_m=min(minima) if minima else None,
                first_qualifying_time_s=rows[selected[0]]['time'] if selected else None,
                last_qualifying_time_s=rows[selected[-1]]['time'] if selected else None,
                scope='Control endpoints only: cube states are not saved in each raw contact substep.')


def main(args):
    started = time.perf_counter()
    run = args.run
    task_path, trajectory_path, usd_path = run / 'task.json', run / 'trajectory.json', run / 'final_scene.usda'
    task = json.loads(task_path.read_text())
    data = json.loads(trajectory_path.read_text())
    rows = data if isinstance(data, list) else data['frames']
    samples, mean_mode = recorded_samples(rows, task)
    total_recorded_samples=len(samples)
    sample_begin=args.sample_start or 0
    sample_end=total_recorded_samples if args.sample_end is None else args.sample_end
    if not 0<=sample_begin<sample_end<=total_recorded_samples:raise ValueError('Invalid audit slice')
    samples=samples[sample_begin:sample_end]
    names = task['joint_names']
    kin = Kinematics()
    if len(names) != 18 or len(set(names)) != 18:
        raise ValueError('Expected 18 unique recorded joint names')
    spec = importlib.util.spec_from_file_location('usd_validator', ROOT / 'scripts/validate_rl_run.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    validator.USD_AUDIT = validator.USD_AUDIT.replace("env='/World/envs/env_0'", f"env='/World/envs/env_{args.env}'")
    usd = validator.usd_audit(usd_path)
    base = np.asarray(usd['robot_base_world_matrix_gf_row_vectors']).T
    table = usd['table_top_world_z_m']
    meshes, supports, cylinders, provenance, mounted = load_geometry(kin)
    adjacent = {frozenset((j['parent'], j['child'])) for j in kin.joints}
    pairs = [(a, b) for a, b in itertools.combinations(meshes, 2) if frozenset((a, b)) not in adjacent]
    tool_corners = {}
    for name, box in usd['tool_collision_boxes'].items():
        half = np.asarray(box['size_m']) / 2
        corners = np.array(list(itertools.product(*[[-h, h] for h in half])))
        tool_corners[name] = corners @ rotation(box['local_quaternion_wxyz']).T + box['local_translation_m']
    link_names = sorted(set(supports) | set(cylinders))
    clearances = np.full((len(samples), len(link_names)), np.inf)
    tool_clearances = np.full((len(samples), len(tool_corners)), np.inf)
    qvalues = np.array([s['joint_pos'] for s in samples], dtype=float)
    if qvalues.shape != (len(samples), 18) or not np.isfinite(qvalues).all():
        raise ValueError('Invalid actual joint-state array')
    limit_margin = np.stack([np.minimum(qvalues[:, j] - kin.by_name[n]['lower'],
                                         kin.by_name[n]['upper'] - qvalues[:, j]) for j, n in enumerate(names)], axis=1)
    orientations, collision_frames, maximum_raw_penetration = [], [], 0.
    for i, sample in enumerate(samples):
        q = dict(zip(names, qvalues[i]))
        fk = kin.fk(q)
        for li, name in enumerate(link_names):
            matrix = base @ fk[name]
            values = []
            if name in supports:
                values.append(float((supports[name] @ matrix[2, :3] + matrix[2, 3]).min() - table))
            for local, radius, half_length in cylinders.get(name, []):
                C = matrix @ local
                values.append(float(C[2, 3] - half_length * abs(C[2, 2]) - radius * np.linalg.norm(C[2, :2]) - table))
            clearances[i, li] = min(values)
        body_matrices = {}
        for body in ['tool', 'left', 'right']:
            state = np.asarray(sample[body], dtype=float)
            if state.shape != (13,) or not np.isfinite(state).all():
                raise ValueError(f'Invalid actual {body} state at sample {i}')
            body_matrices[body] = transform(state[:3], rotation(state[3:7]))
        for ti, (name, corners) in enumerate(tool_corners.items()):
            body = {'Handle': 'tool', 'Left': 'left', 'Right': 'right'}[name.split('/')[0]]
            M = body_matrices[body]
            tool_clearances[i, ti] = (corners @ M[2, :3] + M[2, 3]).min() - table
        hand = base @ fk['right_hand_link']
        axis = body_matrices['tool'][:3, 2]
        tip_z = task['tool_geometry']['length_m'] - .0075
        tip = sum(B[:3, 3] + B[:3, 2] * tip_z for key, B in body_matrices.items() if key in ['left', 'right']) / 2
        tip_axis = tip - body_matrices['tool'][:3, 3]
        tip_axis /= np.linalg.norm(tip_axis)
        web = hand[:3, :3] @ np.asarray(task['grasp_topology']['web_edge_axis_local'])
        angles = [np.degrees(np.arctan2(-a[2], np.linalg.norm(a[:2]))) for a in [axis, tip_axis]]
        orientations.append([web[2], hand[2, 1], *angles])
        for contact in sample.get('grasp_contact_points', {}).get('normal_contacts', []):
            if sample.get('grasp_contact_points', {}).get('environment') != args.env:
                raise ValueError('Selected environment does not match contact record')
            maximum_raw_penetration = max(maximum_raw_penetration, -float(contact['separation_m']))
        if not args.skip_self:
            relative = kin.fk(q, root='right_hand_link')
            posed = {n: mesh.at_pose(relative[n]) for n, mesh in meshes.items()}
            hits = []
            for a, b in pairs:
                result = mesh_sat(posed[a], posed[b], args.tolerance, max_witnesses=2)
                if result['surface_intersection_triangle_pairs']:
                    hits.append(dict(links=[a, b], **result))
            if hits:
                collision_frames.append(dict(sample_index=i, frame_index=sample['frame_index'],
                                             substep_index=sample['substep_index'], time_s=sample['time_s'], pairs=hits))
        if i % 120 == 0:
            print(json.dumps(dict(processed=i + 1, total=len(samples), elapsed_s=time.perf_counter() - started,
                                  robot_clearance_min_m=float(clearances[:i+1].min()),
                                  tool_clearance_min_m=float(tool_clearances[:i+1].min()),
                                  self_collision_samples=len(collision_frames))), flush=True)

    def summarize(values, labels):
        result = {}
        for j, name in enumerate(labels):
            arr = values[:, j]
            lowest = int(arr.argmin())
            bad = np.flatnonzero(arr < 0)
            state = samples[lowest]
            result[name] = dict(minimum_clearance_m=float(arr[lowest]), minimum_sample_index=lowest,
                                minimum_time_s=state['time_s'], minimum_frame_index=state['frame_index'],
                                minimum_substep_index=state['substep_index'], negative_sample_count=int(len(bad)),
                                first_negative_time_s=samples[bad[0]]['time_s'] if len(bad) else None,
                                last_negative_time_s=samples[bad[-1]]['time_s'] if len(bad) else None)
        return result

    O = np.asarray(orientations)
    lo, hi = task['tool_down_angle_range_deg']
    acceptable = ((O[:, 0] >= task['min_web_up_component']) & (O[:, 1] <= task['max_palm_up_component']) &
                  (O[:, 2:] >= lo).all(1) & (O[:, 2:] <= hi).all(1))
    checks = dict(all_18_joint_limits=bool((limit_margin >= -1e-6).all()),
                  all_source_moving_robot_collisions_above_table=bool((clearances > 0).all()),
                  all_actual_tool_boxes_above_table=bool((tool_clearances > 0).all()),
                  all_recorded_substep_orientations_within_task_bounds=bool(acceptable.all()),
                  nonadjacent_hand_source_surfaces_disjoint=None if args.skip_self else not collision_frames)
    report = dict(environment=args.env, artifacts_sha256={str(p.name): sha(p) for p in [trajectory_path, task_path, usd_path]},
                  urdf=str(ASSET / 'robot.urdf'), urdf_sha256=sha(ASSET / 'robot.urdf'),
                  trajectory_control_rows=len(rows), audited_actual_samples=len(samples),
                  audited_sample_range=[sample_begin,sample_end],total_recorded_physics_samples=total_recorded_samples,
                  sample_source=('every_recorded_physics_substep' if sample_begin==0 and sample_end==total_recorded_samples else 'explicit_contiguous_recorded_physics_substep_slice') if mean_mode else 'recorded_samples',
                  physics_dt_s=task['physics_dt'], first_sample_time_s=samples[0]['time_s'], last_sample_time_s=samples[-1]['time_s'],
                  first_done_control_frame=next((i for i,r in enumerate(rows) if r.get('done')), None),
                  joint_mapping=names, checks=checks,
                  joint_minimum_limit_margin_rad=dict(zip(names, limit_margin.min(0).tolist())),
                  moving_robot_table_clearance=summarize(clearances, link_names),
                  actual_tool_box_table_clearance=summarize(tool_clearances, list(tool_corners)),
                  elevated_control_endpoint_tool_clearance=elevated_endpoint_clearance(rows, task, usd, tool_corners),
                  orientation_ranges=dict(zip(['web_side_z', 'palm_normal_z', 'tool_table_angle_deg', 'root_tip_table_angle_deg'],
                                              np.stack([O.min(0), O.max(0)], axis=1).tolist())),
                  nonadjacent_hand_self_collision=dict(method='Original URDF collision STL, BVH and triangle SAT',
                     metric_tolerance_m=args.tolerance, skipped=args.skip_self, pairs_per_sample=len(pairs),
                     checked_pair_instances=0 if args.skip_self else len(pairs) * len(samples),
                     excluded_parent_child_pairs=[sorted(p) for p in adjacent if p.issubset(meshes)],
                     intersecting_sample_count=len(collision_frames), intersecting_samples=collision_frames),
                  maximum_recorded_PhysX_contact_penetration_m=maximum_raw_penetration,
                  source_collision_geometry=provenance, omitted_table_mounted_base_links=mounted,
                  USD_evidence=usd, elapsed_s=time.perf_counter() - started,
                  limitations=[
                    'This selected-environment geometric audit does not estimate policy success rate or independently certify learned action provenance.',
                    'Raw support reconstruction and task-success gates belong to the separate contact/main audits; geometry does not substitute for contact forces.',
                    'Source STL collision surfaces differ from PhysX cooked convex decomposition. Reported solver separation and source intersections are different measurements.',
                    'Table clearance uses the infinite tabletop support plane. Mounted base links are intentionally excluded; the adapter cylinder is included analytically.',
                    'Self-intersection tests exclude directly adjacent URDF links. They detect triangle-surface intersections, not complete shell containment or penetration volume.',
                    'All saved physics states are checked, but states inside an individual solver step are unrecorded.',
                    'Tool-hand contact intersections, grasp-tail overlap depth and hand-arm self-collision are outside this hand-to-hand self-collision audit.',
                    'A passive spring drive is an idealized elastic joint; source geometry and idealized friction do not establish real-world feasibility.'
                  ])
    output = args.output or run / f'env{args.env}_trained_geometry_audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), checks=checks, samples=len(samples),
                          minimum_robot_clearance_m=float(clearances.min()), minimum_tool_clearance_m=float(tool_clearances.min()),
                          elapsed_s=report['elapsed_s']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--env', type=int, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--skip-self', action='store_true', help='Explicit partial audit; report self-collision check as unperformed.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-start',type=int)
    parser.add_argument('--sample-end',type=int)
    main(parser.parse_args())
