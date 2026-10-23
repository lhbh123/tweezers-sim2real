"""Source collision-triangle audits at commanded and actual settled poses.

No config or dynamics file is changed. Static rest-tool intersections at a
closed command are not used to infer the deflection of a passive blade.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import ASSET, Kinematics, stl_triangles, transform, xyz
from check_mesh_pair import mesh_intersections
from validate_human_grasp import triangle_box_overlap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_matrix(pose):
    return transform(pose[:3], Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix())


class Auditor:
    def __init__(self):
        self.kin = Kinematics()
        self.adjacent = {frozenset((j['parent'], j['child'])) for j in self.kin.joints}
        self.meshes = {}
        self.hulls = {}
        self.source_files = {}
        for link in self.kin.root.findall('link'):
            name = link.get('name')
            if not name.startswith('right_hand'):
                continue
            collision = link.find('collision')
            mesh = link.find('collision/geometry/mesh')
            if mesh is None:
                continue
            path = ASSET / mesh.get('filename')
            triangles = stl_triangles(path)
            origin = collision.find('origin')
            if origin is not None:
                T = transform(xyz(origin.get('xyz')), Rotation.from_euler('xyz', xyz(origin.get('rpy'))).as_matrix())
                triangles = triangles @ T[:3, :3].T + T[:3, 3]
            self.meshes[name] = triangles
            self.hulls[name] = ConvexHull(np.unique(triangles.reshape(-1, 3), axis=0)).equations
            self.source_files[name] = {'path': str(path), 'sha256': sha(path), 'triangles': len(triangles)}

    def geometry(self, q):
        Ts = self.kin.fk(q, root='right_hand_link')
        meshes, hulls = {}, {}
        for name, triangles in self.meshes.items():
            T = Ts[name]
            meshes[name] = triangles @ T[:3, :3].T + T[:3, 3]
            equations = self.hulls[name]
            normal = equations[:, :3] @ T[:3, :3].T
            hulls[name] = np.column_stack([normal, equations[:, 3] - normal @ T[:3, 3]])
        return meshes, hulls

    def self_collision(self, meshes, hulls):
        names = list(meshes)
        bounds = {n: (tri.min(axis=(0, 1)), tri.max(axis=(0, 1))) for n, tri in meshes.items()}
        checks = []
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                if frozenset((first, second)) in self.adjacent:
                    continue
                low_a, high_a = bounds[first]
                low_b, high_b = bounds[second]
                if np.any(np.maximum(low_a, low_b) > np.minimum(high_a, high_b)):
                    checks.append({'links': [first, second], 'separated': True, 'method': 'AABB'})
                    continue
                eq = np.concatenate([hulls[first], hulls[second]])
                result = linprog([0, 0, 0, -1], A_ub=np.column_stack([eq[:, :3], np.ones(len(eq))]),
                                 b_ub=-eq[:, 3], bounds=[(None, None)] * 4, method='highs')
                separated = bool(result.success and result.x[3] <= 1e-7)
                check = {'links': [first, second], 'separated': separated,
                         'method': 'hull halfspaces', 'solver_success': bool(result.success),
                         'hull_intersection_ball_radius_m': float(result.x[3]) if result.success else None}
                if not separated:
                    exact = mesh_intersections(meshes[first], meshes[second])
                    check['triangle_pair_tests'] = exact['triangle_pair_tests']
                    check['triangle_surface_intersection_pairs'] = exact['intersection_pair_count']
                checks.append(check)
        return {'checked_nonadjacent_pairs': len(checks),
                'all_source_surfaces_disjoint': all(c['separated'] or c.get('triangle_surface_intersection_pairs') == 0 for c in checks),
                'hull_overlap_candidates': [c for c in checks if not c['separated']]}

    def tool_collision(self, meshes, bodies):
        # Dimensions agree with the actual PhysX primitive colliders in the hold probe.
        boxes = [('handle', 'bridge', [0, 0, -.003], [.006, .0045, .003]),
                 ('left', 'blade', [0, 0, .0625], [.0005, .0045, .0625]),
                 ('right', 'blade', [0, 0, .0625], [.0005, .0045, .0625]),
                 ('left', 'tip', [0, 0, .1325], [.0005, .0025, .0075]),
                 ('right', 'tip', [0, 0, .1325], [.0005, .0025, .0075])]
        intended = {'right_hand_thumb_rota_link2', 'right_hand_index_rota_link2'}
        hits = []
        for body, part, center, half in boxes:
            T = bodies[body]
            for link, mesh in meshes.items():
                tri = (mesh - T[:3, 3]) @ T[:3, :3]
                # Cheap broad phase before the continuous 13-axis triangle test.
                broad = np.all((tri.max(axis=1) >= np.asarray(center) - half) &
                               (tri.min(axis=1) <= np.asarray(center) + half), axis=1)
                if not broad.any():
                    continue
                count = int(triangle_box_overlap(tri[broad], center, half).sum())
                if count:
                    hits.append({'link': link, 'tool_body': body, 'tool_part': part,
                                 'intersecting_source_triangles': count,
                                 'intended_contact_link': link in intended})
        return {'triangle_intersections': hits,
                'no_noncontact_triangle_intersections': all(h['intended_contact_link'] for h in hits)}

    def bounds(self, q, config):
        hard = {n: min(q[n] - self.kin.by_name[n]['lower'], self.kin.by_name[n]['upper'] - q[n]) for n in self.kin.hand}
        soft = {n: min(q[n] - config['soft_bounds'][n][0], config['soft_bounds'][n][1] - q[n]) for n in self.kin.hand}
        return {'minimum_URDF_limit_margin_rad': float(min(hard.values())),
                'minimum_soft_limit_margin_rad': float(min(soft.values())),
                'hard_limit_violations': {n: float(m) for n, m in hard.items() if m < -1e-7},
                'soft_limit_violations': {n: float(m) for n, m in soft.items() if m < -1e-7}}

    def config_sweep(self, path):
        config = json.loads(path.read_text())
        T = np.asarray(config['T_hand_tool'])
        bodies = {'handle': T,
                  'left': T @ transform([-.0045, 0, 0], Rotation.from_euler('y', -.05).as_matrix()),
                  'right': T @ transform([.0045, 0, 0], Rotation.from_euler('y', .05).as_matrix())}
        samples = []
        for closure in np.round(np.arange(.6, 1.50001, .1), 6):
            q = {n: v + closure * config['closure_delta'][n] for n, v in config['q_reference'].items()}
            meshes, hulls = self.geometry(q)
            sample = {'closure': float(closure), 'joint_bounds': self.bounds(q, config),
                      'self_collision': self.self_collision(meshes, hulls),
                      'tool_rest_envelope': self.tool_collision(meshes, bodies)}
            samples.append(sample)
            print(path.name, closure, 'self_clear', sample['self_collision']['all_source_surfaces_disjoint'],
                  'other_tool_clear', sample['tool_rest_envelope']['no_noncontact_triangle_intersections'], flush=True)
        return {'config': str(path.relative_to(ROOT)), 'config_sha256': sha(path),
                'closure_samples': samples,
                'all_sampled_hand_source_surfaces_disjoint': all(s['self_collision']['all_source_surfaces_disjoint'] for s in samples),
                'all_sampled_noncontact_links_clear_of_rest_tool': all(s['tool_rest_envelope']['no_noncontact_triangle_intersections'] for s in samples),
                'all_sampled_joint_targets_inside_hard_and_soft_limits': all(not s['joint_bounds']['hard_limit_violations'] and not s['joint_bounds']['soft_limit_violations'] for s in samples)}

    def snapshot(self, path):
        snap = json.loads(path.read_text())
        config = json.loads((ROOT / snap['args']['pose']).read_text())
        q = dict(zip(snap['joint_names'], snap['q_actual']))
        meshes, hulls = self.geometry(q)
        T_hand_world = np.linalg.inv(pose_matrix(snap['world_hand_pose_7']))
        bodies = {n: T_hand_world @ pose_matrix(s) for n, s in snap['rigid_body_states_13'].items()}
        wrist_fk = self.kin.fk(q)['right_hand_link']
        wrist_fk[:3, 3] += snap['arm_base_world_xyz']
        Tsnapshot = pose_matrix(snap['world_hand_pose_7'])
        tip_left = bodies['left'] @ np.array([0, 0, .14, 1])
        tip_right = bodies['right'] @ np.array([0, 0, .14, 1])
        distance = np.linalg.norm(tip_right[:3] - tip_left[:3])
        return {'snapshot': str(path.relative_to(ROOT)), 'snapshot_sha256': sha(path),
                'actual_joint_bounds': self.bounds(q, config),
                'target_joint_bounds': self.bounds(dict(zip(snap['joint_names'], snap['q_target'])), config),
                'hand_pose_FK_position_error_m': float(np.linalg.norm(wrist_fk[:3, 3] - Tsnapshot[:3, 3])),
                'hand_pose_FK_rotation_error_rad': float(Rotation.from_matrix(wrist_fk[:3, :3].T @ Tsnapshot[:3, :3]).magnitude()),
                'self_collision': self.self_collision(meshes, hulls),
                'actual_tool_geometry': self.tool_collision(meshes, bodies),
                'tip_center_separation_m': float(distance),
                'tip_center_separation_minus_1mm_thickness_m': float(distance - .001),
                'fixture_active': snap['fixture_active'], 'hold_metrics': snap['natural_hold']}


if __name__ == '__main__':
    auditor = Auditor()
    report = {
        'method': 'Source collision STL triangles with continuous triangle/box and triangle/triangle separating-axis tests; hull and AABB broad phases.',
        'tool_geometry_override_m': {'blade_width': .009, 'blade_length': .125, 'tip_width': .005, 'tip_length': .015, 'thickness': .001, 'handle_dimensions': [.012, .009, .006]},
        'source_meshes': auditor.source_files,
        'config_sweeps': [auditor.config_sweep(ROOT / 'configs' / name) for name in ['human_grasp_distal_suspend30.json', 'human_grasp_distal_suspend30_10.json']],
        'settled_snapshot_audits': [auditor.snapshot(ROOT / 'runs' / run / 'stable_snapshot.json') for run in ['natural_hold_distal_open060', 'natural_hold_distal_cold110']],
        'limitations': [
            'Closure is sampled at increments of 0.1, not a certified continuous joint-space sweep.',
            'Commanded closed fingers intersect the static rest-angle tool by design; only unintended links are judged against that rest envelope. Actual passive deflection is checked separately in settled snapshots.',
            'Source triangle surfaces are not the cooked PhysX convex decomposition; runtime requires the documented palm-index and palm-thumb collision filters.',
            'Directly adjacent URDF links are excluded from hand self-collision checks because mechanical joint meshes overlap by construction.',
            'Surface tests do not separately certify volume containment of disconnected closed shells.',
            'Audits cover the hand assembly. They do not certify all arm/tool/table trajectories or learned-policy states.',
            'Tip center distance minus 1 mm is a simple opening descriptor, not the minimum 3-D clearance for an axis-aligned cube.'
        ]}
    output = ROOT / 'analysis/distal_geometry_closure_audit.json'
    output.write_text(json.dumps(report, indent=2))
    print(output)
