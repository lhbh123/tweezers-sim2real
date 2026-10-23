"""CPU-only URDF/FK candidates for real palm-down/side-down tweezer poses.

This analyzes existing evidence and geometry; it does not run a simulator or
modify a shared task/config. Changed gravity orientations are NOT certified
dynamically stable by a rigid geometric transform.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import ASSET, Kinematics, stl_triangles, transform, xyz
from human_retarget import mesh_axis_surface
from validate_human_grasp import triangle_box_overlap


def unit(v):
    return np.asarray(v) / np.linalg.norm(v)


def matrix(state):
    return transform(state[:3], R.from_quat(np.asarray(state)[[4, 5, 6, 3]]).as_matrix())


def pose(T):
    q = R.from_matrix(T[:3, :3]).as_quat()
    return np.r_[T[:3, 3], q[3], q[:3]].tolist()


def inclination(axis):
    return float(np.rad2deg(np.arctan2(-axis[2], np.linalg.norm(axis[:2]))))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_meshes(kin):
    meshes = {}
    for link in kin.root.findall('link'):
        collision = link.find('collision')
        if collision is None:
            continue
        mesh = collision.find('geometry/mesh')
        if mesh is None:
            # Adapter bounding cylinder is handled analytically in clearance().
            continue
        tri = stl_triangles(ASSET / mesh.get('filename'))
        origin = collision.find('origin')
        T = np.eye(4) if origin is None else transform(xyz(origin.get('xyz')), R.from_euler('xyz', xyz(origin.get('rpy'))).as_matrix())
        meshes[link.get('name')] = tri @ T[:3, :3].T + T[:3, 3]
    return meshes


def clearance(kin, meshes, joints, base):
    Ts = kin.fk(joints)
    minima = {}
    for name, triangles in meshes.items():
        if name == 'base_link_inertia':
            continue  # The fixed robot mounting base intentionally sits on the table.
        T = Ts[name]
        minima[name] = float(np.min(triangles @ T[2, :3] + T[2, 3] + base[2]) - base[2])
    T = Ts['xhand_adapter']
    center = T[:3, 3] + T[:3, :3] @ [0, 0, .01]
    lower = center[2] - .01 * abs(T[2, 2]) - .03 * np.linalg.norm(T[2, :2])
    minima['xhand_adapter'] = float(lower)
    hand = {n: z for n, z in minima.items() if n.startswith('right_hand')}
    arm = {n: z for n, z in minima.items() if not n.startswith('right_hand')}
    return {'hand_min_clearance_m': min(hand.values()), 'hand_lowest_link': min(hand, key=hand.get),
            'moving_arm_min_clearance_m': min(arm.values()), 'moving_arm_lowest_link': min(arm, key=arm.get),
            'all_link_plane_clearances_m': minima,
            'method': 'Minimum transformed source collision-triangle vertex z; exact for a horizontal infinite plane. Adapter cylinder support function; fixed mounting base excluded.'}


def tool_boxes(bodies, length):
    return [('handle', 'bridge', bodies['handle'], np.array([0, 0, -.003]), np.array([.006, .0045, .003])),
            *[(side, part, bodies[side], np.array(center), np.array(half))
              for side in ['left', 'right']
              for part, center, half in [
                  ('blade', [0, 0, (length-.015)/2], [.0005, .0045, (length-.015)/2]),
                  ('tip', [0, 0, length-.0075], [.0005, .0025, .0075])]]]


def tip_centres(bodies, length):
    return {side: (bodies[side] @ [0, 0, length-.0075, 1])[:3] for side in ['left', 'right']}


def gravity_estimate(bodies, length, contact_points):
    """Uniform-volume COM estimate, not a measured PhysX contact-wrench audit."""
    blade_length = length-.015
    arm_com_z = (.009*blade_length*(blade_length/2)+.005*.015*(length-.0075)) / (.009*blade_length+.005*.015)
    masses = {'handle': .003, 'left': .006, 'right': .006}
    weighted = []
    for name, mass in masses.items():
        p = bodies[name] @ [0, 0, -.003 if name == 'handle' else arm_com_z, 1]
        weighted.append(mass*p[:3])
    com = sum(weighted)/sum(masses.values())
    midpoint = .5*(contact_points[0]+contact_points[1])
    line = unit(contact_points[1]-contact_points[0])
    force = np.array([0., 0., -sum(masses.values())*9.81])
    torque = np.cross(com-midpoint, force)
    return {'assumed_total_mass_kg': sum(masses.values()), 'COM_world_m': com.tolist(),
            'contact_proxy_midpoint_world_m': midpoint.tolist(), 'COM_minus_contact_midpoint_m': (com-midpoint).tolist(),
            'gravity_torque_about_midpoint_Nm': torque.tolist(),
            'gravity_torque_norm_Nm': float(np.linalg.norm(torque)),
            'gravity_torque_about_two_contact_line_Nm': float(torque@line),
            'limitation': 'COM assumes uniform-density primitive shape volumes and fixed 3/6/6 g masses. Contact proxies are the configured source-mesh points transformed by actual FK, not recorded PhysX contact patch centres. Two ideal point contacts cannot counter a torque about their joining line; finite contact patches may, so this is a mechanism hypothesis rather than a measured stability proof.'}


def tool_clearance(bodies, length, table):
    result = {}
    for body, part, T, center, half in tool_boxes(bodies, length):
        p = T[:3, 3] + T[:3, :3] @ center
        result[f'{body}/{part}'] = float(p[2] - np.abs(T[2, :3]) @ half - table)
    return result


def local_tool_intersections(kin, meshes, q, hand_world, bodies, length):
    Ts = kin.fk(q, root='right_hand_link')
    world_mesh = {n: tri @ (hand_world @ Ts[n])[:3, :3].T + (hand_world @ Ts[n])[:3, 3]
                  for n, tri in meshes.items() if n in Ts}
    hits = []
    for body, part, T, center, half in tool_boxes(bodies, length):
        for name, tri in world_mesh.items():
            local = (tri - T[:3, 3]) @ T[:3, :3]
            broad = np.all((local.max(1) >= center-half) & (local.min(1) <= center+half), axis=1)
            count = int(triangle_box_overlap(local[broad], center, half).sum()) if broad.any() else 0
            if count:
                hits.append({'hand_link': name, 'tool_body': body, 'part': part, 'triangles': count})
    return hits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, default=ROOT/'runs/natural_hold_distal_open060/stable_snapshot.json')
    p.add_argument('--config', type=Path, default=ROOT/'configs/human_grasp_distal_suspend30.json')
    p.add_argument('--output', type=Path, default=ROOT/'analysis/palm_orientation_candidates.json')
    a = p.parse_args()
    kin = Kinematics(); meshes = source_meshes(kin)
    snap = json.loads(a.snapshot.read_text()); cfg = json.loads(a.config.read_text())
    base = np.asarray(snap['arm_base_world_xyz']); table = float(base[2])
    q = dict(zip(snap['joint_names'], snap['q_actual']))
    qtarget = dict(zip(snap['joint_names'], snap['q_target']))
    H = kin.fk(q)['right_hand_link']; H[:3, 3] += base
    bodies = {n: matrix(s) for n, s in snap['rigid_body_states_13'].items()}
    normal = H[:3, :3] @ [0, 1, 0]
    hand_fk = kin.fk(q, root='right_hand_link')
    contact_points = []
    for finger in ['thumb', 'index']:
        contact = cfg['contact_targets'][finger]
        contact_points.append((H@hand_fk[contact['link']]@np.r_[contact['point_local'], 1])[:3])
    surface_samples = []
    for x, z in [(0, -.06), (0, -.04), (-.02, -.07), (.02, -.07)]:
        for face in [False, True]:
            point, n = mesh_axis_surface(ASSET/'sources/spider/assets/right_hand_link.STL', 1, (0, 2), (x, z), face)
            surface_samples.append({'side': '+Y' if face else '-Y', 'point_hand_m': point.tolist(), 'surface_normal_hand': n.tolist()})
    flexion = {}
    for finger, joint, tip in [('index', 'right_hand_index_joint1', 'right_hand_index_rota_tip'),
                               ('middle', 'right_hand_mid_joint1', 'right_hand_mid_tip'),
                               ('ring', 'right_hand_ring_joint1', 'right_hand_ring_tip'),
                               ('pinky', 'right_hand_pinky_joint1', 'right_hand_pinky_tip')]:
        v = kin.fk({joint: .1}, root='right_hand_link')[tip][:3, 3] - kin.fk(root='right_hand_link')[tip][:3, 3]
        flexion[finger] = {'joint_increment_rad': .1, 'tip_displacement_hand_m': v.tolist()}
    candidates = []
    length_audits = []
    for length in [.14, .18, .20]:
        tips = tip_centres(bodies, length); centre = .5*(tips['left']+tips['right'])
        source_x = unit(tips['right']-tips['left'])
        source_z = unit(bodies['left'][:3, 2]+bodies['right'][:3, 2])
        source_z = unit(source_z-source_x*(source_z@source_x))
        source_basis = np.column_stack([source_x, np.cross(source_z, source_x), source_z])
        length_audits.append({'length_m': length, 'hand_tool_source_triangle_intersections': local_tool_intersections(kin, meshes, q, H, bodies, length)})
        for degrees in [30, 35, 40, 45, 50]:
            alpha = np.deg2rad(degrees)
            target_x = np.array([0., -1., 0.]); target_z = np.array([np.cos(alpha), 0., -np.sin(alpha)])
            target_basis = np.column_stack([target_x, np.cross(target_z, target_x), target_z])
            # Align actual spread, not assumed handle X, so both tips have equal height.
            D = target_basis @ source_basis.T
            target_tip = np.array([.48, 0., table+.0078])
            offset = target_tip-D@centre
            hand = transform(D@H[:3, 3]+offset, D@H[:3, :3])
            transformed = {n: transform(D@T[:3, 3]+offset, D@T[:3, :3]) for n, T in bodies.items()}
            local_hand = hand.copy(); local_hand[:3, 3] -= base
            sol = kin.inverse_arm(local_hand, [q[n] for n in kin.arm])
            actual = {**q, **sol['q']}
            target = dict(qtarget)
            for n in kin.arm:
                target[n] = actual[n] + qtarget[n]-q[n]
            approach_hand = local_hand.copy(); approach_hand[2, 3] += .02
            approach = kin.inverse_arm(approach_hand, list(sol['q'].values()))
            fk = kin.fk(actual); wrist = fk['wrist_3_link'].copy(); wrist[:3, 3] += base
            palm = hand[:3, :3] @ [0, 1, 0]
            ts = tip_centres(transformed, length)
            gap = float(np.linalg.norm(ts['right']-ts['left']))
            half = np.array([.0005, .0025, .0075])
            projected_half = sum(float(np.abs(transformed[n][:3, :3].T@target_x)@half) for n in ['left', 'right'])
            clear = clearance(kin, meshes, actual, base)
            state13 = {n: pose(T)+[0.]*6 for n, T in transformed.items()}
            item = {'id': f'length{round(length*1000)}_down{degrees}_leveljaws',
                    'construction': 'Rigidly rotate the actual settled grasp so its measured tip-spread axis is world -Y; place its orthogonalized mean-blade axis downward at the named table inclination. This changes real physics orientation, not camera.',
                    'tool_length_m': length, 'blade_length_m': length-.015, 'tip_length_m': .015,
                    'blade_width_m': .009, 'tip_width_m': .005, 'thickness_m': .001,
                    'named_axis_inclination_to_table_deg': degrees,
                    'actual_handle_axis_inclination_to_table_deg': inclination(transformed['handle'][:3, 2]),
                    'actual_mean_blade_axis_inclination_to_table_deg': inclination(unit(transformed['left'][:3, 2]+transformed['right'][:3, 2])),
                    'palm_normal_world': palm.tolist(), 'palm_normal_dot_table_up': float(palm[2]),
                    'palm_normal_angle_to_table_up_deg': float(np.rad2deg(np.arccos(np.clip(palm[2], -1, 1)))),
                    'hand_world_pose_7_wxyz': pose(hand), 'wrist_world_pose_7_wxyz': pose(wrist),
                    'handle_world_rotation': transformed['handle'][:3, :3].tolist(),
                    'rigid_rotation_from_open060': D.tolist(),
                    'T_hand_handle_settled_preserved': (np.linalg.inv(hand)@transformed['handle']).tolist(),
                    'joint_names': snap['joint_names'],
                    'q_actual': [actual[n] for n in snap['joint_names']], 'q_target': [target[n] for n in snap['joint_names']],
                    'q_velocity': [0.]*len(snap['joint_names']), 'tool_rigid_body_states_13': state13,
                    'tool_tip_center_world_m': target_tip.tolist(),
                    'tip_height_difference_m': float(ts['right'][2]-ts['left'][2]),
                    'tip_center_gap_m': gap, 'projected_inner_tip_gap_world_y_m': gap-projected_half,
                    'at_grasp_height': clear,
                    'tool_part_clearances_at_grasp_height_m': tool_clearance(transformed, length, table),
                    'gravity_moment_estimate': gravity_estimate(transformed, length, [D@point+offset for point in contact_points]),
                    'arm_IK_residual': sol['error'], 'arm_IK_position_error_m': float(np.linalg.norm(fk['right_hand_link'][:3, 3]-local_hand[:3, 3])),
                    'arm_IK_rotation_error_rad': float(R.from_matrix(fk['right_hand_link'][:3, :3].T@local_hand[:3, :3]).magnitude()),
                    'approach_20mm_above': {'arm_joint_targets': approach['q'], 'IK_residual': approach['error'], 'all_tool_body_translation_m': [0., 0., .02]},
                    'kinematically_reachable_and_hand_clear': bool(sol['error']<1e-4 and approach['error']<1e-4 and clear['hand_min_clearance_m']>.005 and clear['moving_arm_min_clearance_m']>.005),
                    'dynamically_verified': False,
                    'tool_mass_inertia_note': '140 mm retains the source geometry. Longer blades require regenerated body inertia and renewed dynamics testing; no stability is inherited.'}
            candidates.append(item)
    baseline = {'open060': {'palm_normal_world': normal.tolist(), 'palm_normal_dot_table_up': float(normal[2]),
                            'palm_normal_angle_to_table_up_deg': float(np.rad2deg(np.arccos(normal[2]))),
                            'actual_handle_axis_world': bodies['handle'][:3, 2].tolist(),
                            'actual_handle_axis_inclination_to_table_deg': inclination(bodies['handle'][:3, 2]),
                            'original_command_tilt_from_downward_vertical_deg': 30,
                            'original_command_inclination_to_table_deg': 60,
                            'gravity_moment_estimate': gravity_estimate(bodies, .14, contact_points)}}
    trajectory_path = ROOT/'runs/rl_distal_open250_render_final/trajectory.json'
    if trajectory_path.exists():
        trajectory = json.loads(trajectory_path.read_text())
        rows = []
        for t in [0, .5, 1., 2., 3., 4.9]:
            row = min(trajectory, key=lambda r: abs(r['time']-t))
            # Training uses the same articulation joint order, verified via task.json.
            task = json.loads((trajectory_path.parent/'task.json').read_text())
            T = kin.fk(dict(zip(task['joint_names'], row['joint_pos'])))['right_hand_link']
            n = T[:3, :3]@[0, 1, 0]
            rows.append({'time_s': row['time'], 'palm_normal_world': n.tolist(),
                         'tool_handle_inclination_to_table_deg': inclination(matrix(row['tool'])[:3, 2])})
        baseline['delivered_ppo_video_samples'] = rows
    report = {'palm_definition': {'local_frame': 'right_hand_link', 'palm_outward_normal': [0., 1., 0.],
                                'reason': 'The +Y broad STL surface faces the positive flexion motion of all four long fingers; -Y is the dorsal side.',
                                'STL_surface_ray_samples': surface_samples, 'positive_flexion_FK_checks': flexion},
              'source_config': str(a.config), 'source_config_sha256': sha(a.config),
              'source_snapshot': str(a.snapshot), 'source_snapshot_sha256': sha(a.snapshot),
              'baseline': baseline, 'source_tool_length_audits': length_audits,
              'candidates': candidates,
              'recommended_first_geometry': ['length140_down40_leveljaws', 'length140_down45_leveljaws'],
              'long_tool_clearance_alternatives': ['length180_down35_leveljaws', 'length200_down30_leveljaws'],
              'reference_image': 'references/human/keyframes/KXDHa39Z-2s_218.009s.jpg',
              'limitations': [
                  'The source video is monocular and uncalibrated; its overhead appearance does not determine a unique 3-D palm normal or tool/table inclination. The 30–50 degree range is a requested design target, not a recovered measurement.',
                  'No GPU or physics simulation is run by this script. FK reachability and plane clearance do not establish frictional holding stability after gravity reorientation.',
                  'The old open060 gravity equilibrium is not valid after a rigid rotation; actual contact forces must be re-established with legal robot actuation and tested after support removal.',
                  'Source hand self-intersection distances are unchanged by rigid pose rotation; full arm self-collision and all motion trajectories are not certified here.',
                  'Longer blades increase reach/hand clearance but change inertia, gravity torque and opening. They must be rebuilt and retested rather than replaying an old dynamic snapshot as if unchanged.',
                  'The original q_reference/closure and shared src files remain unchanged. Each candidate exports actual 18-joint pose, target, wrist pose and three independent dynamic tool-body transforms.'
              ]}
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    for x in candidates:
        print(x['id'], 'palm_z', round(x['palm_normal_dot_table_up'], 3), 'tool_deg', round(x['actual_handle_axis_inclination_to_table_deg'], 2),
              'hand_mm', round(x['at_grasp_height']['hand_min_clearance_m']*1000, 2), 'reachable', x['kinematically_reachable_and_hand_clear'])
    print(a.output)


if __name__ == '__main__':
    main()
