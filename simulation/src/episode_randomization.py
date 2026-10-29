"""Small bounded preheld-grasp randomization; no simulator imports at module load."""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def local_delta(translation_m, rotvec_deg):
    xyz, rotation = np.asarray(translation_m, dtype=float), np.asarray(rotvec_deg, dtype=float)
    if xyz.shape != (3,) or rotation.shape != (3,) or not np.isfinite(np.r_[xyz, rotation]).all():
        raise ValueError('Mount perturbations must be finite 3-vectors')
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.deg2rad(rotation)).as_matrix()
    T[:3, 3] = xyz
    return T


def perturb_camera_rig(original, settings):
    """Right-compose local mounting errors; preserve the physical link/TCP chain."""
    rig = copy.deepcopy(original)
    cameras = rig['cameras']
    wrist = cameras['wrist']
    T_link_tcp = np.asarray(wrist['T_link_Cw']) @ np.linalg.inv(wrist['T_TCP_Cw'])
    for name, key in [('wrist', 'T_TCP_Cw'), ('global', 'T_B_Cg')]:
        row = settings[name]
        changed = np.asarray(cameras[name][key]) @ local_delta(row['translation_m'], row['rotation_vector_deg'])
        cameras[name][key] = changed.tolist()
        if name == 'wrist':
            cameras[name]['T_link_Cw'] = (T_link_tcp @ changed).tolist()
    rig['synthetic_mount_perturbation'] = settings
    return rig


def sample_episode(profile, base_layout, hand_names, seed):
    rng = np.random.default_rng(seed)
    layout = copy.deepcopy(base_layout)
    bounds = [np.asarray(profile[k], dtype=float) for k in
              ['start_xy_lower_m', 'start_xy_upper_m', 'goal_xy_lower_m', 'goal_xy_upper_m']]
    sl, su, gl, gu = bounds
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in bounds) or np.any(sl >= su) or np.any(gl >= gu):
        raise ValueError('Invalid XY sampling bounds')
    scalar_keys = ['home_height_radius_m', 'tool_yaw_radius_deg', 'hand_target_radius_rad', 'initial_settle_seconds', 'minimum_transfer_m']
    if any(not np.isfinite(profile[k]) or profile[k] < 0 for k in scalar_keys):
        raise ValueError('Randomization magnitudes must be finite and nonnegative')
    if profile['hand_target_radius_rad'] > .003 or profile['initial_settle_seconds'] < .2:
        raise ValueError('Preheld v1 requires hand offsets <=0.003 rad and settling >=0.2 seconds')
    for _ in range(1000):
        start, goal = rng.uniform(sl, su), rng.uniform(gl, gu)
        if np.linalg.norm(start - goal) >= profile['minimum_transfer_m']:
            break
    else:
        raise ValueError('Cannot sample the requested transfer distance')
    centre, size = np.asarray(layout['table_center_xy_m']), np.asarray(layout['table_size_m'][:2])
    for xy in [start, goal]:
        if np.any(np.abs(xy - centre) >= size / 2 - .03):
            raise ValueError('Sample lies outside tabletop margin')
    radius = np.asarray(profile['home_xy_radius_m'], dtype=float)
    if radius.shape != (2,) or not np.isfinite(radius).all() or np.any(radius < 0) or np.any(radius > .02):
        raise ValueError('HOME XY radii must be in [0,0.02] m')
    offset = rng.uniform(-radius, radius)
    layout['home_xy_m'] = (np.asarray(layout['home_xy_m']) + offset).tolist()
    dz = float(rng.uniform(-profile['home_height_radius_m'], profile['home_height_radius_m']))
    dyaw = float(rng.uniform(-profile['tool_yaw_radius_deg'], profile['tool_yaw_radius_deg']))
    layout['initial_tip_clearance_m'] = .075 + dz
    if not .04 <= layout['initial_tip_clearance_m'] <= .10:
        raise ValueError('Initial clearance outside conservative v1 range')
    layout['tool_yaw_deg'] += dyaw
    if abs(dyaw) > 3:
        raise ValueError('Initial yaw perturbation outside conservative v1 range')
    hand = {name: float(rng.uniform(-profile['hand_target_radius_rad'], profile['hand_target_radius_rad'])) for name in hand_names}
    cameras = {}
    for name, spec in profile['cameras'].items():
        tr, rr = spec['local_translation_radius_m'], spec['local_rotvec_component_radius_deg']
        if not np.isfinite([tr, rr]).all() or not 0 <= tr <= .01 or not 0 <= rr <= 1:
            raise ValueError('Camera perturbation exceeds v1 bounds')
        cameras[name] = dict(translation_m=rng.uniform(-tr, tr, 3).tolist(),
                             rotation_vector_deg=rng.uniform(-rr, rr, 3).tolist())
    if set(cameras) != {'wrist', 'global'}:
        raise ValueError('Both wrist and global mount perturbations required')
    return dict(schema_version=1, seed=seed, start_xy_m=start.tolist(), goal_xy_m=goal.tolist(),
                layout=layout, home_offset_xy_m=offset.tolist(), home_height_offset_m=dz,
                tool_yaw_offset_deg=dyaw, hand_target_offsets_rad=hand,
                initial_settle_seconds=profile['initial_settle_seconds'], cameras=cameras,
                semantics='Arm IK moves the entire preheld grasp; hand target offsets act through physics, not independent finger teleports; cameras right-compose local mount errors')


def initialize_preheld_episode(env, settings, out):
    """Perturb hand commands and physically settle BEFORE episode recording starts."""
    import torch

    names = env.scene.kin.hand
    offsets = settings['hand_target_offsets_rad']
    if set(offsets) != set(names):
        raise ValueError('Hand offset names must match active hand joints exactly')
    delta = torch.tensor([offsets[n] for n in names], device=env.device, dtype=env.ref.dtype)
    if not bool(torch.isfinite(delta).all()) or float(delta.abs().max()) > .003:
        raise ValueError('Invalid hand command offsets')
    previous = env.q_target[:, env.hand].clone()
    env.ref = env.ref + delta
    env.hand_target = (previous + delta).clamp(env.lo, env.hi)
    env.q_target[:, env.hand] = env.hand_target
    duration = float(settings['initial_settle_seconds'])
    if not .2 <= duration <= 2:
        raise ValueError('Initial settling must be in [0.2,2] seconds')
    steps = int(np.ceil(duration / env.dt))
    env.scene.step(env.q_target, decimation=env.cfg.decimation * steps)
    m = env.measurements()
    state = m['state']
    finite = bool(torch.isfinite(torch.cat([state['joint_pos'], state['joint_vel'], state['tool_root_state_w']], -1)).all())
    retained = bool((m['grip_error'] < .035).all() and (m['orientation_error'] < .3).all())
    grasp = bool(m['face_pinch_ok'].all() and m['orientation_ok'].all())
    above = bool((m['tip'][:, 2] > env.scene.table_height + .04).all())
    report = dict(valid=finite and retained and grasp and above,
                  checks=dict(finite=finite, retained=retained, grasp=grasp, tip_clearance=above),
                  requested=settings, settling_control_intervals=steps, settling_duration_s=steps*env.dt,
                  hand_joint_names=names,
                  effective_hand_target_offsets_rad=(env.hand_target-previous)[0].cpu().tolist(),
                  actual_joint_names=env.scene.robot.joint_names,
                  actual_initial_qpos_rad=state['joint_pos'][0].cpu().tolist(),
                  actual_initial_qvel_rad_s=state['joint_vel'][0].cpu().tolist(),
                  actual_initial_tool_state_w=state['tool_root_state_w'][0].cpu().tolist(),
                  actual_initial_block_state_w=state['block_root_state_w'][0].cpu().tolist(),
                  actual_initial_tip_xyz_m=m['tip'][0].cpu().tolist(),
                  grip_error_m=m['grip_error'][0].item(), orientation_error=m['orientation_error'][0].item())
    (Path(out)/'initial_conditions.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['valid']:
        raise ValueError('Perturbed initial grasp failed validation; see initial_conditions.json')
    return env.observe(m)
