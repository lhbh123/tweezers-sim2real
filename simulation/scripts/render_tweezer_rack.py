"""Build and physically settle a rack-mounted, tips-down tweezer start scene."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', required=True)
parser.add_argument('--config', default=str(ROOT/'configs/tweezer_rack_scene.json'))
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=False)
cfg = json.loads(Path(args.config).read_text())
app = AppLauncher(args).app
try:
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from scipy.spatial.transform import Rotation
    from tweezer_rack_scene import TweezerRackScene
    from probe_render import ProbeRenderer

    yaw = float(cfg['tool_yaw_deg'])
    rotation = Rotation.from_euler('z', yaw, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    snapshot = json.loads((ROOT.parent/'models/snapshots/xhand_stable_grasp.json').read_text())
    reference_rotation = Rotation.from_euler('z', cfg['grasp_reference_layout_yaw_deg'], degrees=True).as_matrix() @ Rotation.from_quat(
        np.roll(snapshot['rigid_body_states_13']['handle'][3:7], -1)).as_matrix()
    scene = TweezerRackScene(
        rack_root_xyz=cfg['rack_tool_root_xyz_m'], rack_yaw_deg=cfg['rack_yaw_deg'],
        rack_guides=cfg['rack_guides'], num_envs=1, device=args.device,
        block_xy=cfg.get('block_xy_m', [0., .65]),
        pose_json=ROOT/'configs/xhand_local.json',
        tool_world_R=reference_rotation, tool_position=cfg['robot_reference_tool_xyz_m'],
        table_height=cfg['table_height_m'], table_size=cfg['table_size_m'],
        table_center_xy=cfg['table_center_xy_m'], spring=5.)
    q = scene.default_joint_pos.clone()
    # A visibly open, stationary hand. Existing arm IK remains unchanged.
    for index in scene.hand_joint_indices:
        low, high = scene.robot.data.soft_joint_pos_limits[0, index].tolist()
        q[:, index] = min(high, max(low, 0.1))
    # Tool and rack headings are independent. Start 0.5 mm above support
    # and let gravity seat the tool on the tongue.
    root = cfg['rack_tool_root_xyz_m']
    quat = np.roll(Rotation.from_matrix(rotation).as_quat(), 1)
    pose = torch.tensor([[root[0], root[1], root[2]+.0005, *quat]], dtype=torch.float32, device=scene.device)
    blockpos = [*cfg.get('block_xy_m', [0., .65]), scene.table_height+.0061]
    scene.reset(q=q, tool_rootpose=pose, blockpos=torch.tensor([blockpos], device=scene.device))
    history = []
    body_history = {name: [] for name in ['left', 'right']}
    for _ in range(round(cfg['settle_seconds']/scene.dt)):
        state = scene.step(q)
        history.append(state['tool_root_state_w'][0].cpu().numpy().copy())
        for name in body_history:
            body_history[name].append(state[name+'_root_state_w'][0].cpu().numpy().copy())
    states = np.asarray(history)
    final = states[-1]
    down = Rotation.from_quat(final[[4,5,6,3]]).apply([0.,0.,1.])
    angle = float(np.degrees(np.arccos(np.clip(-down[2], -1., 1.))))
    drift = float(np.linalg.norm(states[-240:,:3]-states[-1,:3], axis=1).max())
    final_rotation = Rotation.from_quat(final[[4,5,6,3]]).as_matrix()
    axis_error = float(np.degrees(np.arccos(np.clip(np.dot(final_rotation[:,0], reference_rotation[:,0]), -1., 1.))))
    valid = bool(np.isfinite(states).all() and angle < 5 and drift < .002
                 and abs(final[2]-root[2]) < .003)
    report = dict(config=cfg, rack_parts=scene.rack_parts,
                  support='Gravity and static rack contact only; no fixed joint or kinematic tool',
                  settled_tool_state_13=final.tolist(), tilt_from_down_deg=angle,
                  jaw_opening_axis_world=final_rotation[:,0].tolist(),
                  reference_jaw_axis_world=reference_rotation[:,0].tolist(),
                  jaw_axis_error_deg=axis_error,
                  last_second_max_position_deviation_m=drift, stable_on_rack=valid,
                  final_linear_speed_m_s=float(np.linalg.norm(final[7:10])),
                  final_angular_speed_rad_s=float(np.linalg.norm(final[10:13])),
                  validation_scope='Three-second seating and root pose retention only; contact jitter, grasping and removal are not validated',
                  joint_names=scene.joint_names, joint_positions_rad=scene.robot.data.joint_pos[0].cpu().tolist(),
                  cameras='Presentation cameras, not calibrated training cameras')
    (out/'scene_report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez(out/'settling_trajectory.npz', tool_state=states, dt=scene.dt,
             **{name+'_state': np.asarray(values) for name, values in body_history.items()})
    (out/'views.json').write_text(json.dumps(cfg['views'], indent=2))
    renderer = ProbeRenderer(scene, out/'views.json', out)
    for _ in range(40):
        scene.sim.render()
    for view, camera in zip(cfg['views'], renderer.cameras):
        rgba = camera.get_rgba()
        if rgba is None or not rgba.size:
            raise RuntimeError('Camera returned no image')
        imageio.imwrite(out/(view['name']+'.png'), rgba[:,:,:3])
    renderer.close()
    # Export authored stage for inspection; exact measured states are recorded
    # above because Fabric simulation transforms need not be written into USD.
    from isaaclab.sim.utils import export_prim_to_file
    export_prim_to_file(str(out/'scene.usd'), '/World')
    print('RACK_SCENE_RESULT', json.dumps(report), flush=True)
    if not valid:
        raise RuntimeError('Rack stability check failed; inspect scene_report.json')
finally:
    if 'scene' in locals():
        scene.close()
    # Static PNG capture has no asynchronous Replicator writers to drain.
    app.close(wait_for_replicator=False)
