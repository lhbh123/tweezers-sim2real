"""CPU audit of a settled palm-down snapshot; no simulation or source edits."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from kinematics import Kinematics
from audit_distal_geometry import Auditor, sha, pose_matrix
from compute_palm_orientation_candidates import source_meshes, clearance, tip_centres, tool_clearance, inclination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, default=ROOT/'runs/natural_orientation_spring5_sweep/table40_x-1_close0.2_snapshot.json')
    p.add_argument('--output', type=Path, default=ROOT/'analysis/palmdown_spring5_snapshot_audit.json')
    a = p.parse_args()
    a.snapshot = a.snapshot.resolve()
    snap = json.loads(a.snapshot.read_text())
    config_path = ROOT/snap['args']['pose']; config = json.loads(config_path.read_text())
    k = Kinematics(); q = dict(zip(snap['joint_names'], snap['q_actual'])); target = dict(zip(snap['joint_names'], snap['q_target']))
    base = np.asarray(snap['arm_base_world_xyz']); table = base[2]
    Ts = k.fk(q); meshes = source_meshes(k)
    auditor = Auditor(); source = auditor.snapshot(a.snapshot)
    fk_consistency = {}
    for link, field in [('wrist_3_link', 'world_wrist_pose_7'), ('right_hand_link', 'world_hand_pose_7')]:
        T = Ts[link].copy(); T[:3, 3] += base; saved = pose_matrix(snap[field])
        fk_consistency[link] = {'position_error_m': float(np.linalg.norm(T[:3, 3]-saved[:3, 3])),
                                'rotation_error_rad': float(R.from_matrix(T[:3, :3].T@saved[:3, :3]).magnitude())}
    hard = {n: float(min(q[n]-k.by_name[n]['lower'], k.by_name[n]['upper']-q[n])) for n in snap['joint_names']}
    bodies = {name: pose_matrix(state) for name, state in snap['rigid_body_states_13'].items()}
    tips = tip_centres(bodies, .14); centre = .5*(tips['left']+tips['right'])
    delta = np.array([.48, 0., table+.0078])-centre
    hand = Ts['right_hand_link'].copy(); hand[:3, 3] += base
    local_target = hand.copy(); local_target[:3, 3] += delta-base
    sol = k.inverse_arm(local_target, [q[n] for n in k.arm])
    q_grasp = {**q, **sol['q']}
    moved_bodies = {name: T.copy() for name, T in bodies.items()}
    for T in moved_bodies.values(): T[:3, 3] += delta
    joint_comparison = {}
    for n in k.hand:
        joint_comparison[n] = {'image_mapped_prior_rad': config['human_joint_prior'][n],
                               'contact_refined_reference_rad': config['q_reference'][n],
                               'actual_rad': q[n], 'target_rad': target[n],
                               'actual_minus_reference_deg': float(np.rad2deg(q[n]-config['q_reference'][n])),
                               'actual_minus_mapped_prior_deg': float(np.rad2deg(q[n]-config['human_joint_prior'][n]))}
    actual = np.array([q[n] for n in k.hand]); ref = np.array([config['q_reference'][n] for n in k.hand])
    prior = np.array([config['human_joint_prior'][n] for n in k.hand]); dq = np.array([config['closure_delta'][n] for n in k.hand])
    saved_target = np.array([target[n] for n in k.hand]); estimated_closure = float((saved_target-ref)@dq/(dq@dq))
    palm = hand[:3, :3]@[0, 1, 0]
    result = {'snapshot': str(a.snapshot), 'snapshot_sha256': sha(a.snapshot),
              'source_config': str(config_path), 'source_config_sha256': sha(config_path),
              'geometry_audit': source, 'FK_saved_pose_consistency': fk_consistency,
              'all_18_actual_joints_inside_URDF_limits': min(hard.values())>=0,
              'URDF_joint_limit_margins_rad': hard,
              'source_height_table_clearance': clearance(k, meshes, q, base),
              'palm_normal_hand': [0., 1., 0.], 'palm_normal_world': palm.tolist(),
              'palm_normal_dot_table_up': float(palm[2]),
              'handle_axis_inclination_down_from_table_deg': inclination(bodies['handle'][:3, 2]),
              'tip_center_world_m': centre.tolist(),
              'repositioned_grasp_height': {'desired_tip_center_world_m': [.48, 0., float(table+.0078)],
                                           'rigid_translation_m': delta.tolist(), 'arm_IK_error': sol['error'],
                                           'arm_joint_positions': sol['q'],
                                           'robot_clearance': clearance(k, meshes, q_grasp, base),
                                           'tool_part_plane_clearances_m': tool_clearance(moved_bodies, .14, table)},
              'estimated_saved_target_closure': estimated_closure,
              'maximum_target_synergy_reconstruction_error_rad': float(np.max(np.abs(saved_target-(ref+estimated_closure*dq)))),
              'actual_minus_reference_joint_RMS_deg': float(np.rad2deg(np.sqrt(np.mean((actual-ref)**2)))),
              'actual_minus_image_mapped_prior_joint_RMS_deg': float(np.rad2deg(np.sqrt(np.mean((actual-prior)**2)))),
              'joint_comparison': joint_comparison,
              'limitations': ['FK checks verify snapshot bookkeeping consistency; saved wrist/hand poses are themselves generated from FK in probe_natural_orientation.py, not an independent external pose measurement.',
                             'Source hand triangle audit excludes directly adjacent links; it does not reproduce PhysX convex decomposition or certify all arm self-collisions.',
                             'Moving a stable pose to the grasp height here is only an IK/clearance calculation, not a new dynamic test.',
                             'Reference angle differences are in robot joint coordinates. Image proxies are view-dependent and partly regularized, not reliable human 3-D anatomical angles.',
                             'The planar table clearance is measured from actual mesh vertices and primitive support functions; contact margins and whole approach trajectories still require simulator validation.']}
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({key: result[key] for key in ['snapshot','all_18_actual_joints_inside_URDF_limits','palm_normal_world','handle_axis_inclination_down_from_table_deg','estimated_saved_target_closure','actual_minus_reference_joint_RMS_deg','actual_minus_image_mapped_prior_joint_RMS_deg']}, indent=2))
    print('Hand surfaces disjoint:', source['self_collision']['all_source_surfaces_disjoint'])
    print('Hand/tool source triangle intersections:', source['actual_tool_geometry']['triangle_intersections'])
    print('Grasp height clearance:', result['repositioned_grasp_height']['robot_clearance'])
    print(a.output)


if __name__ == '__main__': main()
