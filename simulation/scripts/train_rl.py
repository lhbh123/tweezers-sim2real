"""Train or evaluate the from-scratch, human-reference-conditioned PPO policy."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
parser = argparse.ArgumentParser()
parser.add_argument('--out', default='runs/rl_precision_s0')
parser.add_argument('--pose', default='configs/human_grasp_precision_v2.json')
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dt', type=float, default=1/240)
parser.add_argument('--decimation', type=int, default=4)
parser.add_argument('--tilt', type=float, default=30.)
parser.add_argument('--tool-pitch', type=float, help='Handle +Z downward angle measured from the TABLE plane, degrees; also rotates snapshot resets')
parser.add_argument('--tool-yaw', type=float, default=0., help='World horizontal tool-axis azimuth, degrees')
parser.add_argument('--tool-roll', type=float, default=0., help='Roll around tool +Z, degrees; 180 reverses the jaw spread direction')
parser.add_argument('--tool-quaternion', nargs=4, type=float, metavar=('W','X','Y','Z'))
parser.add_argument('--require-palm-down', action='store_true')
parser.add_argument('--require-web-up', action='store_true', help='Require the hand radial/thumb-index side to face upward; separate from the palm normal')
parser.add_argument('--min-web-up', type=float, default=.35)
parser.add_argument('--max-palm-up', type=float, help='Optional maximum vertical component of the outward palm normal')
parser.add_argument('--require-tripod-support', action='store_true', help='Require actual thumb/index/middle contacts and middle-under/web-region support')
parser.add_argument('--monitor-grasp-contacts', action='store_true', help='Record optional hand/tool contacts without requiring middle or web support')
parser.add_argument('--require-face-pinch', action='store_true', help='Require inward thumb/index pressure on opposite broad jaw faces; middle and web support are optional')
parser.add_argument('--support-reward', type=float, default=.5)
parser.add_argument('--independent-support-actions',action='store_true',help='Add bounded index MCP and middle PIP corrections to the human-reference hand synergy')
parser.add_argument('--command-space-arm',action='store_true',help='Apply Cartesian increments to actuator targets, preserving the grasp preload when holding still')
parser.add_argument('--index-action-scale',type=float,default=.03,help='Index MCP residual range in radians')
parser.add_argument('--middle-action-scale',type=float,default=.12,help='Middle PIP residual range in radians')
parser.add_argument('--tool-angle-range', nargs=2, type=float, help='Permitted actual downward angle from table for task success')
parser.add_argument('--friction', type=float, default=1.4)
parser.add_argument('--spring', type=float, default=1.5)
parser.add_argument('--spring-damping', type=float, default=.003)
parser.add_argument('--hand-stiffness', type=float, default=10.)
parser.add_argument('--hand-damping', type=float, default=.3)
parser.add_argument('--tool-width', type=float)
parser.add_argument('--tool-length', type=float, help='Physical tweezer root-to-tip length in metres')
parser.add_argument('--bridge-width', type=float)
parser.add_argument('--solver-velocity-iterations', type=int, default=2)
parser.add_argument('--contact-offset', type=float, default=.0003)
parser.add_argument('--self-collisions', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--simplify-arm-collisions', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--filter-mechanical-self-pairs', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--solver-iterations', type=int, default=32)
parser.add_argument('--clearance', type=float, default=.002)
parser.add_argument('--max-wrist-descent',type=float,default=.08,help='Maximum downward Cartesian command displacement from reset, metres')
parser.add_argument('--xy-range', type=float, default=.002)
parser.add_argument('--work-xy',type=float,nargs=2,default=[.48,0.],help='Block and initial tool-tip workspace centre on the table')
parser.add_argument('--closure-bias', type=float, default=.4)
parser.add_argument('--closure-scale', type=float, default=1.1)
parser.add_argument('--closure-min', type=float, default=-.2)
parser.add_argument('--closure-max', type=float, default=2.)
parser.add_argument('--mode', choices=['hold', 'lift'], default='lift')
parser.add_argument('--episode-seconds', type=float, default=5.)
parser.add_argument('--resume')
parser.add_argument('--initialize-policy', help='Load only policy weights/statistics; start new PPO optimizer and sample count, with initialization provenance')
parser.add_argument('--initialize-std',type=float,help='Explicitly override warm-start exploration std; never affects deterministic actor weights')
parser.add_argument('--initial-std', type=float, default=.35)
parser.add_argument('--freeze-observation-normalizer',action='store_true',help='Preserve demonstration-fitted observation coordinates during PPO fine-tuning')
parser.add_argument('--learning-rate',type=float,default=3e-4)
parser.add_argument('--target-kl',type=float,default=.035)
parser.add_argument('--reset-snapshot')
parser.add_argument('--aperture-reward', type=float, default=0.)
parser.add_argument('--evaluate', help='Checkpoint file, or zero for constant zero policy')
parser.add_argument('--render-video', action='store_true')
parser.add_argument('--wide-closeup', action='store_true', help='Keep the full hand and tool in view during a larger lift')
parser.add_argument('--orientation-views', action='store_true', help='Render an actual vertical top view and horizontal side view in addition to overview/closeup')
parser.add_argument('--orthographic-top', action='store_true', help='Use calibrated vertical orthographic top view to avoid perspective enlargement of the robot wrist')
parser.add_argument('--camera-config',help='JSON list of name, eye, target, optional focal_length_m and orthographic_width_m; positions relative to environment origin')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.require_face_pinch and args.require_tripod_support:
    parser.error('Face pinch uses optional middle/web support; do not combine with continuous tripod support')
monitor_grasp = args.monitor_grasp_contacts or args.require_face_pinch or args.require_tripod_support
args.headless = True
args.multi_gpu = False
if args.render_video:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from rl_scene import BatchedTweezersScene
from rl_task import TweezerTask, TaskConfig
from ppo import PPO, PPOConfig, ActorCritic

torch.set_num_threads(4)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
source_hashes = {}
source_names=['ppo.py', 'rl_task.py', 'rl_scene.py', 'kinematics.py', 'reset_reference.py','arm_clearance.py']
if args.command_space_arm:source_names.append('arm_command.py')
if monitor_grasp: source_names.append('grasp_contacts.py')
for f in [ROOT/'src'/n for n in source_names] + [Path(__file__), ROOT/args.pose]:
    source_hashes[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    shutil.copy2(f, out/f.name)
(out/'launch.json').write_text(json.dumps({'arguments': vars(args), 'source_sha256': source_hashes}, indent=2, default=str))
theta = np.deg2rad(args.tilt)
rotation = np.column_stack(([0, 1, 0], [np.cos(theta), 0, np.sin(theta)],
                           [np.sin(theta), 0, -np.cos(theta)]))
explicit_orientation = args.tool_pitch is not None or args.tool_quaternion is not None
if args.reset_snapshot and not explicit_orientation:
    snapshot = json.loads(Path(args.reset_snapshot).read_text())
    snapshot_source = snapshot.get('zero_velocity_variant',snapshot)
    q = snapshot_source['rigid_body_states_13']['handle'][3:7]
    rotation = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
if args.tool_pitch is not None:
    pitch,yaw = np.deg2rad([args.tool_pitch,args.tool_yaw])
    axis = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), -np.sin(pitch)])
    across = np.array([-np.sin(yaw), np.cos(yaw), 0.])
    rotation = np.column_stack([across,np.cross(axis,across),axis]) @ Rotation.from_euler('z',args.tool_roll,degrees=True).as_matrix()
if args.tool_quaternion is not None:
    w,x,y,z = args.tool_quaternion
    rotation = Rotation.from_quat([x,y,z,w]).as_matrix()
rotation_xyzw = Rotation.from_matrix(rotation).as_quat()
reset_orientation = tuple(np.r_[rotation_xyzw[3],rotation_xyzw[:3]]) if explicit_orientation else None
scene_pose = ROOT/args.pose
if args.tool_width is not None or args.tool_length is not None:
    pose_override = json.loads(scene_pose.read_text())
    if args.tool_width is not None: pose_override['tool_geometry']['strip_width_m'] = args.tool_width
    if args.tool_length is not None: pose_override['tool_geometry']['length_m'] = args.tool_length
    pose_override['runtime_geometry_override'] = {'strip_width_m': args.tool_width, 'length_m':args.tool_length,
        'source_pose': str(scene_pose), 'source_pose_sha256': hashlib.sha256(scene_pose.read_bytes()).hexdigest(),
        'note': 'Original geometric audit predates this width change; dynamics must be revalidated.'}
    scene_pose = out/'runtime_pose.json'
    scene_pose.write_text(json.dumps(pose_override, indent=2))
scene = BatchedTweezersScene(num_envs=args.num_envs, device=args.device, dt=args.dt,
                             pose_json=scene_pose, tool_world_R=rotation,
                             tool_position=(*args.work_xy,.95),
                             finger_friction=args.friction, spring=args.spring,
                             spring_damping=args.spring_damping, hand_stiffness=args.hand_stiffness,
                             hand_damping=args.hand_damping, self_collisions=args.self_collisions,
                             filter_mechanical_self_pairs=args.filter_mechanical_self_pairs,
                             simplify_arm_collisions=args.simplify_arm_collisions,
                             enable_grasp_support=monitor_grasp,
                             enable_face_metrics=args.require_face_pinch,
                             bridge_width=args.bridge_width, solver_velocity_iterations=args.solver_velocity_iterations,
                             contact_offset=args.contact_offset,
                             solver_iterations=args.solver_iterations)
cfg = TaskConfig(decimation=args.decimation, initial_tip_clearance=args.clearance,
                 max_wrist_descent=args.max_wrist_descent,
                 reset_xy_range=args.xy_range, closure_bias=args.closure_bias,
                 closure_scale=args.closure_scale, seed=args.seed, mode=args.mode,
                 closure_min=args.closure_min, closure_max=args.closure_max,
                 reset_snapshot=args.reset_snapshot,
                 work_xy=tuple(args.work_xy),
                 reset_tool_quaternion_wxyz=reset_orientation,
                 require_palm_down=args.require_palm_down,
                 require_web_up=args.require_web_up,
                 min_web_up_component=args.min_web_up,
                 max_palm_up_component=args.max_palm_up,
                 require_tripod_support=args.require_tripod_support,
                 monitor_grasp_contacts=args.monitor_grasp_contacts,
                 require_face_pinch=args.require_face_pinch,
                 support_reward_weight=args.support_reward,
                 independent_support_actions=args.independent_support_actions,
                 command_space_arm=args.command_space_arm,
                 index_support_action_rad=args.index_action_scale,
                 middle_support_action_rad=args.middle_action_scale,
                 tool_down_angle_range_deg=tuple(args.tool_angle_range) if args.tool_angle_range else None,
                 aperture_reward_weight=args.aperture_reward,
                 episode_seconds=args.episode_seconds)
env = TweezerTask(scene, cfg)
(out/'task.json').write_text(json.dumps(env.metadata(), indent=2))
print('RL_TASK_READY', env.num_envs, env.observation_size, env.action_size, flush=True)

if not args.evaluate:
    trainer = PPO(env, out, PPOConfig(initial_std=args.initial_std,initialize_std_override=args.initialize_std,learning_rate=args.learning_rate,
        target_kl=args.target_kl,freeze_observation_normalizer=args.freeze_observation_normalizer),
        seed=args.seed,resume=args.resume,initialize=args.initialize_policy)
    trainer.train(args.iterations)
else:
    if args.evaluate != 'zero':
        saved = torch.load(args.evaluate, map_location=scene.device, weights_only=False)
        policy = ActorCritic(saved['observation_size'], saved['action_size']).to(scene.device)
        policy.load_state_dict(saved['policy'])
        policy.eval()
    cameras, writers = [], []
    if args.render_video:
        from pxr import UsdGeom, UsdLux, UsdShade, Sdf, Gf, UsdPhysics
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.sensors.camera')
        app.update()
        from isaacsim.sensors.camera import Camera
        import imageio.v2 as imageio
        light = UsdLux.DomeLight.Define(scene.stage, '/World/RLDome')
        light.CreateIntensityAttr(850.)
        key = UsdLux.DistantLight.Define(scene.stage, '/World/RLKey')
        key.CreateIntensityAttr(1200.)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-25, 30, 0))
        for name, color, metallic in [('RLSteel', (.5, .55, .62), .8), ('RLPlastic', (.95, .22, .03), 0),
                                      ('RLTable', (.16, .21, .27), .1)]:
            mat = UsdShade.Material.Get(scene.stage, '/World/Materials/'+name)
            shader = UsdShade.Shader.Define(scene.stage, str(mat.GetPath())+'/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.35)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        for e in range(scene.num_envs):
            for suffix, name in [('/Table', 'RLTable'), ('/Block/Shape', 'RLPlastic'),
                                  ('/Tweezers/Handle/Shape', 'RLSteel'), ('/Tweezers/Left/Blade', 'RLSteel'),
                                  ('/Tweezers/Right/Blade', 'RLSteel'), ('/Tweezers/Left/Tip', 'RLSteel'),
                                  ('/Tweezers/Right/Tip', 'RLSteel'),('/Tweezers/Handle/TailExtension','RLSteel')]:
                prim = scene.stage.GetPrimAtPath(scene.env_paths[e]+suffix)
                if not prim: continue
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(scene.stage, '/World/Materials/'+name))
        origin = scene.env_origins[0].cpu().numpy()
        closeup_eye, closeup_target = ([1.25, .75, 1.38], [.44, 0, .93]) if args.wide_closeup else ([.94, .48, 1.16], [.44, 0, .83])
        views = [('overview', [1.18, -1.05, 1.25], [.30, 0, .94]),
                 ('closeup', closeup_eye, closeup_target)]
        if args.orientation_views:
            # Cover the entire approach/lift path; retain actual top and horizontal side views.
            views += [('top', [.40,-.08,1.8], [.40,-.08,.87]),
                      ('side', [.40,-1.0,.93], [.40,0,.93])]
        view_settings={}
        if args.camera_config:
            configured=json.loads(Path(args.camera_config).read_text())
            view_settings={v['name']:v for v in configured}
            if len(view_settings)!=len(configured):raise ValueError('Camera names must be unique')
            views=[(v['name'],v['eye'],v['target']) for v in configured]
        camera_manifest=[]
        for name, eye, target in views:
            eye, target = np.array(eye)+origin, np.array(target)+origin
            z = eye-target; z /= np.linalg.norm(z)
            up = [0,1,0] if abs(z[2])>.999 else [0,0,1]
            x = np.cross(up, z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            q = Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()
            camera = Camera('/World/RL_'+name, resolution=(960, 720), frequency=30)
            camera.set_world_pose(position=eye, orientation=np.r_[q[3], q[:3]], camera_axes='usd')
            camera.set_horizontal_aperture(.036); camera.set_focal_length(.028)
            if name == 'side': camera.set_focal_length(.032)
            if name == 'top' and args.orthographic_top:
                camera.set_projection_mode('orthographic')
                camera.set_horizontal_aperture(.55)
            settings=view_settings.get(name,{})
            if 'focal_length_m' in settings:camera.set_focal_length(settings['focal_length_m'])
            if 'orthographic_width_m' in settings:
                camera.set_projection_mode('orthographic')
                camera.set_horizontal_aperture(settings['orthographic_width_m'])
            camera.set_clipping_range(.01, 100.)
            camera.initialize()
            cameras.append(camera)
            writers.append(imageio.get_writer(out/(name+'.mp4'), fps=30, codec='libx264', quality=8))
            camera_manifest.append({'name':name,'eye_world':eye.tolist(),'target_world':target.tolist(),
                                    'quaternion_wxyz':np.r_[q[3],q[:3]].tolist(),'custom_settings':settings})
        (out/'camera_manifest.json').write_text(json.dumps(camera_manifest,indent=2))
        for _ in range(4): scene.sim.render()
    obs = env.reset()
    scene.capture_grasp_substeps = bool(monitor_grasp)
    live = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    successes = torch.zeros_like(live)
    max_lift = torch.zeros(env.num_envs, device=env.device)
    min_grip_error = torch.full_like(max_lift, 100.)
    episode_lengths = torch.zeros_like(max_lift)
    contact_count = torch.zeros_like(max_lift)
    retained_count = torch.zeros_like(max_lift)
    natural_orientation_count = torch.zeros_like(max_lift)
    grasp_support_count = torch.zeros_like(max_lift)
    face_pinch_count = torch.zeros_like(max_lift)
    middle_contact_count = torch.zeros_like(max_lift)
    web_contact_count = torch.zeros_like(max_lift)
    palm_z_min = torch.full_like(max_lift, float('inf'))
    palm_z_max = torch.full_like(max_lift, -float('inf'))
    web_z_min = torch.full_like(max_lift, float('inf'))
    web_z_max = torch.full_like(max_lift, -float('inf'))
    tool_angle_min = torch.full_like(max_lift, float('inf'))
    tool_angle_max = torch.full_like(max_lift, -float('inf'))
    invalid_orientation_count = torch.zeros_like(max_lift)
    trajectory = []
    done_at = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    with torch.no_grad():
        for step in range(env.max_steps):
            action = torch.zeros((env.num_envs, env.action_size), device=env.device) if args.evaluate == 'zero' else policy.act(obs.nan_to_num(0., posinf=0., neginf=0.))
            action[~live] = 0
            obs, reward, done, info = env.step(action, auto_reset=False)
            m = info['measurements']
            successes |= env.episode_success & live
            finite_lift = m['lift'].nan_to_num(0., posinf=0., neginf=0.)
            max_lift = torch.maximum(max_lift, torch.where(live, finite_lift, max_lift))
            episode_lengths += live
            contact_count += m['both'] & live
            retained_count += (m['grip_error'] < .035) & live
            natural_orientation_count += m['orientation_ok'] & live
            grasp_support_count += m['grasp_support_ok'] & live
            face_pinch_count += m['face_pinch_ok'] & live
            if m['grasp_support'] is not None:
                middle_contact_count += (m['grasp_support']['role_contact_strength_N'][:,2]>.003) & live
                web_contact_count += (m['grasp_support']['web_contact_count']>0) & live
            palm_z = m['palm_normal_w'][:,2]
            finite_orientation = torch.isfinite(palm_z)&torch.isfinite(m['tool_down_angle_deg'])&torch.isfinite(m['root_to_tip_down_angle_deg'])
            invalid_orientation_count += live&~finite_orientation
            valid_orientation = live&finite_orientation
            palm_z_min = torch.minimum(palm_z_min,torch.where(valid_orientation,palm_z,palm_z_min))
            palm_z_max = torch.maximum(palm_z_max,torch.where(valid_orientation,palm_z,palm_z_max))
            web_z = m['web_side_axis_w'][:,2]
            web_z_min = torch.minimum(web_z_min,torch.where(live&torch.isfinite(web_z),web_z,web_z_min))
            web_z_max = torch.maximum(web_z_max,torch.where(live&torch.isfinite(web_z),web_z,web_z_max))
            tool_angle_min = torch.minimum(tool_angle_min,torch.where(valid_orientation,m['tool_down_angle_deg'],tool_angle_min))
            tool_angle_max = torch.maximum(tool_angle_max,torch.where(valid_orientation,m['tool_down_angle_deg'],tool_angle_max))
            just_done = done & live
            done_at[just_done] = step
            live &= ~done
            trajectory.append({'time': (step+1)*env.dt, 'action': action[0].cpu().tolist(),
                               'joint_pos': m['state']['joint_pos'][0].cpu().tolist(),
                               'cube': m['state']['block_root_state_w'][0].cpu().tolist(),
                               'tool': m['state']['tool_root_state_w'][0].cpu().tolist(),
                               'left': m['state']['left_root_state_w'][0].cpu().tolist(),
                               'right': m['state']['right_root_state_w'][0].cpu().tolist(),
                               'palm_normal_w': m['palm_normal_w'][0].cpu().tolist(),
                               'web_side_axis_w': m['web_side_axis_w'][0].cpu().tolist(),
                               'tool_axis_w': m['tool_axis_w'][0].cpu().tolist(),
                               'tool_down_angle_deg': float(m['tool_down_angle_deg'][0]),
                               'root_to_tip_axis_w':m['root_to_tip_axis_w'][0].cpu().tolist(),
                               'root_to_tip_down_angle_deg':float(m['root_to_tip_down_angle_deg'][0]),
                               'orientation_acceptable':bool(m['orientation_ok'][0]),
                               'cube_lift_m': float(m['lift'][0]), 'tool_grip_error_m': float(m['grip_error'][0]),
                               'jaw_contact_N': m['forces'][0].cpu().tolist(), 'reward': float(reward[0]),
                               'fixture_active': False, 'done': bool(done[0])})
            if m['grasp_support'] is not None:
                support=m['grasp_support']
                trajectory[-1]['grasp_support']={key:support[key][0].cpu().tolist() for key in
                    ['role_normal_force_w','role_friction_force_w','role_contact_strength_N','role_support_z_N',
                     'web_contact_count','web_normal_force_w','web_friction_force_w','middle_under_contact_count']}
                trajectory[-1]['grasp_support_acceptable']=bool(m['grasp_support_ok'][0])
                if args.require_face_pinch:
                    trajectory[-1]['face_pinch_acceptable']=bool(m['face_pinch_ok'][0])
                    trajectory[-1]['face_pinch_inward_force_N']=m['face_pinch_inward_force_N'][0].cpu().tolist()
                    trajectory[-1]['broad_face_contacts']={key:value[0].cpu().tolist() for key,value in support.items() if key.startswith('broad_face_')}
                trajectory[-1]['grasp_contact_points']=scene.grasp_contact_reader.audit_env_points(
                    m['state'],scene.dt,env_id=0)
                trajectory[-1]['grasp_contact_substeps']=scene.grasp_contact_substeps
            if cameras and step % max(1, round(1/(30*env.dt))) == 0:
                scene.sim.render()
                for camera, writer in zip(cameras, writers):
                    rgba = camera.get_rgba()
                    if rgba is not None and rgba.size:
                        writer.append_data(np.asarray(rgba)[..., :3].astype(np.uint8))
            if not bool(live.any()):
                break
            # Ended rows are excluded from all scores; reset their physics so an
            # invalid row cannot contaminate another environment's evaluation.
            if bool(just_done.any()):
                obs = env.reset(just_done.nonzero(as_tuple=False).flatten())
    for writer in writers: writer.close()
    finite_list = lambda value: [float(v) if np.isfinite(v) else None for v in value.cpu().tolist()]
    report = {'checkpoint': str(args.evaluate), 'episodes': env.num_envs,
              'successes': int(successes.sum()), 'success_rate': float(successes.float().mean()),
              'success_per_env':successes.tolist(), 'episode_length_steps':episode_lengths.tolist(),
              'max_lift_m': max_lift.tolist(), 'contact_fraction': (contact_count/episode_lengths.clamp_min(1)).tolist(),
              'retention_fraction': (retained_count/episode_lengths.clamp_min(1)).tolist(),
              'natural_orientation_fraction':(natural_orientation_count/episode_lengths.clamp_min(1)).tolist(),
              'grasp_support_fraction':(grasp_support_count/episode_lengths.clamp_min(1)).tolist() if args.require_tripod_support else None,
              'face_pinch_fraction':(face_pinch_count/episode_lengths.clamp_min(1)).tolist() if args.require_face_pinch else None,
              'optional_middle_contact_fraction':(middle_contact_count/episode_lengths.clamp_min(1)).tolist() if monitor_grasp else None,
              'optional_web_contact_fraction':(web_contact_count/episode_lengths.clamp_min(1)).tolist() if monitor_grasp else None,
              'palm_normal_z_min':finite_list(palm_z_min), 'palm_normal_z_max':finite_list(palm_z_max),
              'web_side_z_min':finite_list(web_z_min), 'web_side_z_max':finite_list(web_z_max),
              'tool_down_angle_min_deg':finite_list(tool_angle_min), 'tool_down_angle_max_deg':finite_list(tool_angle_max),
              'invalid_orientation_samples':invalid_orientation_count.tolist(),
              'done_step': done_at.tolist(), 'task': env.metadata()}
    (out/'evaluation.json').write_text(json.dumps(report, indent=2))
    (out/'trajectory.json').write_text(json.dumps(trajectory))
    scene.stage.GetRootLayer().Export(str(out/'final_scene.usda'))
    print('RL_EVALUATION', json.dumps(report), flush=True)
scene.close()
app.close()
