"""Audit actual recorded XHand poses against the original URDF/STL surfaces.

Example:
  python analysis/validate_rl_hand_geometry.py runs/RUN/trajectory.json --frames 20
Older runs without joint names require an explicit mapping, for example
  --joint-names-json analysis/rl_scene_probe.json

This performs continuous triangle SAT, not vertex sampling or a convex-hull
collision surrogate. It checks sampled source-mesh surfaces, not PhysX cooking
or the unsampled time between frames, and does not prove volume non-containment.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import ASSET, Kinematics, stl_triangles, transform, xyz


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triangle_sat(a, b, tolerance=1e-9):
    """Paired (N,3,3) triangle SAT with unit axes and a metric tolerance.

    In-plane axes additionally resolve coplanar triangles. Degenerate source
    triangles are removed when loading. Crossings require both triangles to
    straddle the other's plane beyond tolerance; touching still counts as a
    surface intersection but is reported separately from transverse crossing.
    """
    ea = np.roll(a, -1, axis=1) - a
    eb = np.roll(b, -1, axis=1) - b
    na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
    axes = [na, nb]
    axes += [np.cross(ea[:, i], eb[:, j]) for i in range(3) for j in range(3)]
    axes += [np.cross(na, ea[:, i]) for i in range(3)]
    axes += [np.cross(nb, eb[:, i]) for i in range(3)]
    intersects = np.ones(len(a), dtype=bool)
    # Translate first to keep projection precision independent of world origin.
    offset = a[:, :1]
    a, b = a - offset, b - offset
    for axis in axes:
        length = np.linalg.norm(axis, axis=1)
        valid = length > 1e-20
        axis = axis / np.maximum(length[:, None], 1e-30)
        pa, pb = np.einsum('nij,nj->ni', a, axis), np.einsum('nij,nj->ni', b, axis)
        overlap = (pa.max(1) >= pb.min(1) - tolerance) & (pb.max(1) >= pa.min(1) - tolerance)
        intersects &= overlap | ~valid
    na /= np.maximum(np.linalg.norm(na, axis=1, keepdims=True), 1e-30)
    nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-30)
    da = np.einsum('nij,nj->ni', a - b[:, :1], nb)
    db = np.einsum('nij,nj->ni', b - a[:, :1], na)
    transverse = intersects & (da.min(1) < -tolerance) & (da.max(1) > tolerance)
    transverse &= (db.min(1) < -tolerance) & (db.max(1) > tolerance)
    return intersects, transverse


class Mesh:
    def __init__(self, triangles, leaf_size=24):
        area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
        valid = np.isfinite(triangles).all((1, 2)) & (area2 > 1e-16)
        self.source_indices = np.flatnonzero(valid)
        self.triangles = triangles[valid]
        self.removed_degenerate = int((~valid).sum())
        if not len(self.triangles):
            raise ValueError('Mesh contains no valid triangles')
        tmin, tmax = self.triangles.min(1), self.triangles.max(1)
        centroids = self.triangles.mean(1)
        self.nodes = []

        def split(indices):
            lo, hi = tmin[indices].min(0), tmax[indices].max(0)
            node = len(self.nodes)
            self.nodes.append(None)
            if len(indices) <= leaf_size:
                children, leaf = None, indices
            else:
                axis = np.ptp(centroids[indices], axis=0).argmax()
                ordered = indices[np.argsort(centroids[indices, axis], kind='stable')]
                half = len(ordered) // 2
                children, leaf = (split(ordered[:half]), split(ordered[half:])), None
            self.nodes[node] = (lo, hi, children, leaf)
            return node

        split(np.arange(len(self.triangles)))
        self.centres = np.array([(n[0] + n[1]) * .5 for n in self.nodes])
        self.extents = np.array([(n[1] - n[0]) * .5 for n in self.nodes])

    def at_pose(self, matrix):
        rotation, translation = matrix[:3, :3], matrix[:3, 3]
        centres = self.centres @ rotation.T + translation
        extents = self.extents @ np.abs(rotation).T
        triangles = self.triangles @ rotation.T + translation
        return dict(mesh=self, triangles=triangles, lo=centres - extents,
                    hi=centres + extents, tlo=triangles.min(1), thi=triangles.max(1))


def mesh_sat(a, b, tolerance=1e-9, max_witnesses=12):
    stack = [(0, 0)]
    count, transverse_count, tested, broad_tests = 0, 0, 0, 0
    witnesses = []
    while stack:
        ia, ib = stack.pop()
        broad_tests += 1
        if np.any(np.maximum(a['lo'][ia], b['lo'][ib]) > np.minimum(a['hi'][ia], b['hi'][ib]) + tolerance):
            continue
        ca, la = a['mesh'].nodes[ia][2:]
        cb, lb = b['mesh'].nodes[ib][2:]
        if ca is None and cb is None:
            candidates = np.all((a['thi'][la, None] + tolerance >= b['tlo'][lb]) &
                                (b['thi'][lb] + tolerance >= a['tlo'][la, None]), axis=-1)
            ai, bi = np.nonzero(candidates)
            ai, bi = la[ai], lb[bi]
            if not len(ai):
                continue
            tested += len(ai)
            hits, transverse = triangle_sat(a['triangles'][ai], b['triangles'][bi], tolerance)
            count += int(hits.sum())
            transverse_count += int(transverse.sum())
            for index in np.flatnonzero(hits)[:max(0, max_witnesses - len(witnesses))]:
                witnesses.append(dict(triangle_a=int(a['mesh'].source_indices[ai[index]]),
                                      triangle_b=int(b['mesh'].source_indices[bi[index]]),
                                      transverse=bool(transverse[index]),
                                      triangle_a_vertices_m=a['triangles'][ai[index]].tolist(),
                                      triangle_b_vertices_m=b['triangles'][bi[index]].tolist()))
        elif cb is None or (ca is not None and np.prod(a['hi'][ia] - a['lo'][ia]) >= np.prod(b['hi'][ib] - b['lo'][ib])):
            stack.extend((child, ib) for child in ca)
        else:
            stack.extend((ia, child) for child in cb)
    return dict(triangle_pair_tests=tested, bvh_node_pair_tests=broad_tests,
                surface_intersection_triangle_pairs=count,
                transverse_crossing_triangle_pairs=transverse_count,
                witnesses=witnesses)


def load_meshes(kinematics):
    descendants = kinematics.fk(root='right_hand_link')
    meshes, sources = {}, []
    for link in kinematics.root.findall('link'):
        name = link.get('name')
        if name not in descendants:
            continue
        triangles = []
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            path = ASSET / mesh.get('filename')
            tri = stl_triangles(path)
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            origin = visual.find('origin')
            matrix = np.eye(4) if origin is None else transform(xyz(origin.get('xyz')), Rotation.from_euler('xyz', xyz(origin.get('rpy'))).as_matrix())
            tri = (tri * scale) @ matrix[:3, :3].T + matrix[:3, 3]
            triangles.append(tri)
            sources.append(dict(link=name, path=str(path), sha256=sha256(path), triangle_count=len(tri)))
        if triangles:
            meshes[name] = Mesh(np.concatenate(triangles))
    return meshes, sources


def names_from_json(value):
    if isinstance(value, list) and all(isinstance(n, str) for n in value):
        return value
    if isinstance(value, dict):
        if 'joint_names' in value:
            return names_from_json(value['joint_names'])
        if 'robot_joints' in value:
            return names_from_json(value['robot_joints'])
        if 'task' in value:
            return names_from_json(value['task'])
    return None


def resolve_names(data, trajectory_path, explicit):
    if isinstance(data, dict):
        names = names_from_json(data)
        if names:
            return names, {'source': str(trajectory_path), 'sha256': sha256(trajectory_path)}
    files = [explicit] if explicit else [trajectory_path.parent / 'task.json', trajectory_path.parent / 'evaluation.json', trajectory_path.parent / 'metadata.json']
    for path in files:
        if path and path.is_file():
            names = names_from_json(json.loads(path.read_text()))
            if names:
                return names, {'source': str(path), 'sha256': sha256(path), 'external_mapping': bool(explicit)}
    raise ValueError('No recorded joint_names. Supply --joint-names-json with the actual simulator order; URDF order must not be assumed.')


def audit(args):
    started = time.perf_counter()
    data = json.loads(args.trajectory.read_text())
    frames = data if isinstance(data, list) else data.get('frames', data.get('trajectory'))
    if not isinstance(frames, list) or not frames:
        raise ValueError('Expected a nonempty trajectory list, or frames/trajectory field')
    eligible = list(range(len(frames)))
    first_done = next((i for i, f in enumerate(frames) if bool(f.get('done', False))), None)
    if first_done is not None and not args.include_after_done:
        eligible = eligible[:first_done + 1]
    eligible = [i for i in eligible if (args.start_time is None or float(frames[i].get('time', frames[i].get('t'))) >= args.start_time)
                and (args.end_time is None or float(frames[i].get('time', frames[i].get('t'))) <= args.end_time)]
    if not eligible:
        raise ValueError('No frames in the requested time interval')
    count = min(args.frames, len(eligible))
    selected = np.linspace(0, len(eligible)-1, count).round().astype(int) if args.sampling == 'uniform' else np.sort(np.random.default_rng(args.seed).choice(len(eligible), count, replace=False))
    selected = [eligible[i] for i in selected]
    kin = Kinematics()
    if all(isinstance(frames[i].get('joint_pos', frames[i].get('joints')), dict) for i in selected):
        names, mapping = None, {'source': 'per-frame joint_pos dictionaries'}
    else:
        names, mapping = resolve_names(data, args.trajectory, args.joint_names_json)
        if len(set(names)) != len(names) or not set(kin.hand).issubset(names):
            raise ValueError('Joint-name mapping must contain every hand joint exactly once')
    meshes, mesh_sources = load_meshes(kin)
    adjacent = {frozenset((j['parent'], j['child'])) for j in kin.joints}
    links = list(meshes)
    pairs = [(a, b) for i, a in enumerate(links) for b in links[i+1:] if frozenset((a, b)) not in adjacent]
    frame_reports = []
    for index in selected:
        frame = frames[index]
        q = frame.get('joint_pos', frame.get('joints'))
        if not isinstance(q, dict):
            if len(q) != len(names):
                raise ValueError(f'Frame {index}: {len(q)} joint positions != {len(names)} names')
            q = dict(zip(names, q))
        hand_q = {n: float(q[n]) for n in kin.hand}
        report = dict(frame_index=index, time_s=frame.get('time', frame.get('t')),
                      done=bool(frame.get('done', False)), hand_joint_positions_rad=hand_q)
        if not np.isfinite(list(hand_q.values())).all():
            report.update(status='invalid_nonfinite_joint_state', pairs=[])
            frame_reports.append(report)
            continue
        matrices = kin.fk(hand_q, root='right_hand_link')
        posed = {n: mesh.at_pose(matrices[n]) for n, mesh in meshes.items()}
        checks = []
        for a, b in pairs:
            checks.append(dict(links=[a, b], **mesh_sat(posed[a], posed[b], args.tolerance)))
        hit_pairs = [p for p in checks if p['surface_intersection_triangle_pairs']]
        crossing_pairs = [p for p in checks if p['transverse_crossing_triangle_pairs']]
        hard_limit_violations = []
        for name, value in hand_q.items():
            lo, hi = kin.by_name[name]['lower'], kin.by_name[name]['upper']
            if value < lo - 1e-6 or value > hi + 1e-6:
                hard_limit_violations.append(dict(joint=name, actual_rad=value, lower_rad=lo, upper_rad=hi))
        report.update(status='surface_intersections_detected' if hit_pairs else 'sampled_surfaces_disjoint',
                      checked_nonadjacent_link_pairs=len(checks),
                      surface_intersecting_link_pairs=len(hit_pairs),
                      transverse_crossing_link_pairs=len(crossing_pairs),
                      triangle_pair_tests=sum(p['triangle_pair_tests'] for p in checks),
                      hard_joint_limit_violations=hard_limit_violations,
                      pairs=checks if args.all_pairs else hit_pairs)
        frame_reports.append(report)
        print(json.dumps({k: report[k] for k in ['frame_index', 'time_s', 'status', 'surface_intersecting_link_pairs', 'transverse_crossing_link_pairs', 'triangle_pair_tests']}), flush=True)
    finite_reports = [f for f in frame_reports if f['status'] != 'invalid_nonfinite_joint_state']
    hits = [f for f in finite_reports if f['surface_intersecting_link_pairs']]
    crossings = [f for f in finite_reports if f['transverse_crossing_link_pairs']]
    report = dict(trajectory=str(args.trajectory), trajectory_sha256=sha256(args.trajectory),
                  joint_mapping=mapping, joint_names=names,
                  urdf=str(ASSET / 'robot.urdf'), urdf_sha256=sha256(ASSET / 'robot.urdf'),
                  mesh_sources=mesh_sources,
                  removed_degenerate_triangle_counts={n: m.removed_degenerate for n, m in meshes.items()},
                  method='URDF FK + original visual STL + local BVH AABB/triangle AABB broad phase + normalized-axis continuous triangle SAT',
                  tolerance_m=args.tolerance, coordinate_frame='right_hand_link',
                  excluded_directly_adjacent_pairs=[sorted(p) for p in adjacent if p.issubset(meshes)],
                  selection=dict(mode=args.sampling, seed=args.seed, requested=args.frames,
                                 total_trajectory_frames=len(frames), first_done_index=first_done,
                                 start_time=args.start_time, end_time=args.end_time,
                                 include_after_done=args.include_after_done, sampled_indices=selected),
                  summary=dict(sampled_frames=len(frame_reports), finite_frames=len(finite_reports),
                               frames_with_surface_intersections=len(hits), frames_with_transverse_crossings=len(crossings),
                               intersecting_frame_fraction=len(hits)/max(1, len(finite_reports)),
                               all_sampled_surfaces_disjoint=bool(finite_reports) and not hits and len(finite_reports) == len(frame_reports),
                               elapsed_s=time.perf_counter()-started),
                  frames=frame_reports,
                  limitations=[
                      'Only sampled actual joint poses; no guarantee for unsampled frames or continuous motion.',
                      'Directly parent-child URDF links are excluded because joint geometry can overlap by design.',
                      'Surface intersection is geometric evidence; it does not quantify penetration volume or contact force.',
                      'Touching/coplanar surfaces are included in intersections; transverse crossings are reported separately.',
                      'Disjoint surfaces do not rule out fully contained disconnected shells.',
                      'Source visual STL surfaces differ from PhysX convex decomposition/cooking.',
                      'The hand is reconstructed relative to its base; rigid arm motion does not change inter-hand-link intersections.',
                      'Finite floating-point SAT uses the stated metric tolerance; this is not exact rational arithmetic.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(args.output), **report['summary']}))


def self_test():
    a = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    transverse = np.array([[[.2, .2, -.2], [.2, .2, .2], [.8, .2, .2]]])
    assert triangle_sat(a, transverse)[0].all()
    assert triangle_sat(a, transverse)[1].all()
    assert not triangle_sat(a, a + [0, 0, .01])[0].any()
    assert not triangle_sat(a, a + [2, 0, 0])[0].any()
    assert triangle_sat(a, a + [.1, .1, 0])[0].all()
    assert not triangle_sat(a, a + [.1, .1, 0])[1].any()
    matrix = transform([.4, -.2, .8], Rotation.from_euler('xyz', [.4, .8, 1.2]).as_matrix())
    result = mesh_sat(Mesh(a).at_pose(matrix), Mesh(transverse).at_pose(matrix))
    assert result['surface_intersection_triangle_pairs'] == result['transverse_crossing_triangle_pairs'] == 1
    # Compare BVH candidate traversal against brute-force SAT on a random soup.
    rng = np.random.default_rng(3)
    x, y = rng.random((39, 3, 3)), rng.random((47, 3, 3))
    brute = triangle_sat(np.repeat(x, len(y), axis=0), np.tile(y, (len(x), 1, 1)))
    result = mesh_sat(Mesh(x, 4).at_pose(matrix), Mesh(y, 4).at_pose(matrix))
    assert result['surface_intersection_triangle_pairs'] == int(brute[0].sum())
    assert result['transverse_crossing_triangle_pairs'] == int(brute[1].sum())
    print(json.dumps({'triangle_sat': 'pass', 'coplanar_separation': 'pass',
                      'rigid_transform_invariance': 'pass', 'bvh_vs_all_pairs': 'pass'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', nargs='?', type=Path)
    parser.add_argument('--joint-names-json', type=Path)
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--sampling', choices=['uniform', 'random'], default='uniform')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tolerance', type=float, default=1e-9)
    parser.add_argument('--include-after-done', action='store_true')
    parser.add_argument('--start-time', type=float)
    parser.add_argument('--end-time', type=float)
    parser.add_argument('--all-pairs', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.trajectory:
        if args.frames < 1 or args.tolerance < 0:
            parser.error('--frames must be positive and --tolerance nonnegative')
        args.output = args.output or args.trajectory.parent / 'hand_geometry_audit.json'
        audit(args)
    elif not args.self_test:
        parser.error('trajectory is required unless --self-test is used')
