"""Evaluate live rack pickup -> frozen tweezer pick/place without a physics reset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# Register verified camera schemas before AppLauncher imports USD.
CAMERA_EXAMPLES=ROOT.parent/'calibration/examples'
sys.path.insert(0,str(CAMERA_EXAMPLES))
from isaac_camera_bootstrap import register_lens_schemas
register_lens_schemas()
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(p)
p.add_argument('--out',required=True)
p.add_argument('--pickup-checkpoint',required=True,help='Fixed-start v2 checkpoint, or zero for controller diagnosis')
p.add_argument('--place-checkpoint',default=str(ROOT.parent/'models/policies/xhand_pick_place_ppo.pt'))
p.add_argument('--task-config',default=str(ROOT/'configs/rack_pickup_train.json'))
p.add_argument('--scene-config',default=str(ROOT/'configs/tweezer_rack_scene.json'))
p.add_argument('--rack-offset-y',type=float,default=.10,help='World +Y translation in metres applied to both rack and initial tweezers; 0 restores the training layout')
p.add_argument('--scene-layout',default=str(ROOT/'configs/calibrated_demo_y_table.json'))
p.add_argument('--cases',default=str(ROOT/'configs/calibrated_demo_y_case.json'))
p.add_argument('--calibrated-rig',default=str(ROOT.parent/'calibration/results/camera_rig_isaac.json'))
p.add_argument('--world-base',default=str(ROOT.parent/'calibration/results/world_base_isaac.json'))
p.add_argument('--goal-xy',type=float,nargs=2,help='Optional override; default is the original calibrated demo goal')
p.add_argument('--episodes',type=int,default=1)
p.add_argument('--place-seconds',type=float,default=36.)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--video',action='store_true')
p.add_argument('--preview-only',action='store_true',help='Render the initialized calibrated scene and exit without executing or evaluating the task')
a=p.parse_args()
import math
if a.episodes<1 or not math.isfinite(a.place_seconds) or a.place_seconds<=0:p.error('Positive episodes and place-seconds required')
if a.goal_xy is not None and not all(math.isfinite(x) for x in a.goal_xy):p.error('Goal must be finite')
if not math.isfinite(a.rack_offset_y):p.error('Rack offset must be finite')
if a.preview_only and a.episodes!=1:p.error('Scene preview requires exactly one episode')
a.headless=True;a.enable_cameras=a.video or a.preview_only;a.multi_gpu=False
if a.enable_cameras:a.kit_args+=' --enable omni.usd.schema.omni_lens_distortion'
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'source').mkdir();(out/'checkpoints').mkdir()
scene_cfg=json.loads(Path(a.scene_config).read_text())
scene_cfg['rack_tool_root_xyz_m'][1]+=a.rack_offset_y
task_cfg=json.loads(Path(a.task_config).read_text());task_cfg['seed']=a.seed
layout_path=Path(a.scene_layout)
layout=json.loads(layout_path.read_text())
cases=json.loads(Path(a.cases).read_text())
if len(cases.get('start',[]))!=1 or len(cases.get('goal',[]))!=1:p.error('Single-environment pipeline requires exactly one start and goal')
start_xy=cases['start'][0];goal_xy=cases['goal'][0] if a.goal_xy is None else a.goal_xy
if any(len(xy)!=2 or not all(math.isfinite(float(x)) for x in xy) for xy in [start_xy,goal_xy]):p.error('Start and goal must be finite XY pairs')
for xy in [start_xy,goal_xy]:
    if any(abs(x-c)+.006>s/2 for x,c,s in zip(xy,layout['table_center_xy_m'],layout['table_size_m'])):p.error('Block start/goal lies outside table')
pose_path=ROOT/'configs/xhand_local.json'
snapshot_path=ROOT.parent/'models/snapshots/xhand_stable_grasp.json'
tracked=[Path(__file__),Path(a.task_config),Path(a.scene_config),layout_path,pose_path,snapshot_path,Path(a.cases),
    Path(a.calibrated_rig),Path(a.world_base),CAMERA_EXAMPLES/'isaac_camera_bootstrap.py',CAMERA_EXAMPLES/'isaac_camera_rig.py']
tracked += [ROOT/'src'/name for name in ['continuous_pick_place.py','rack_pickup_task.py','tweezer_rack_scene.py',
    'rl_scene.py','rl_task.py','pick_place_task.py','ppo.py','reset_reference.py','grasp_contacts.py',
    'arm_command.py','kinematics.py','arm_clearance.py','calibrated_demo_render.py']]
hashes={}
for i,path in enumerate(tracked):
    shutil.copy2(path,out/'source'/f'{i:02d}_{path.name}')
    hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
# Read/copy each checkpoint once; later replacement of a training latest.pt
# cannot change the model used by this test.
checkpoints={}
for label,path in [('pickup',a.pickup_checkpoint),('place',a.place_checkpoint)]:
    if path=='zero' and label=='pickup':continue
    data=Path(path).read_bytes();dest=out/'checkpoints'/f'{label}.pt';dest.write_bytes(data)
    checkpoints[label]={'source':str(Path(path).resolve()),'sha256':hashlib.sha256(data).hexdigest()}
(out/'launch.json').write_text(json.dumps({'args':vars(a),'source_sha256':hashes,'checkpoints':checkpoints},indent=2))
app=AppLauncher(a).app
scene=None;renderer=None
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from tweezer_rack_scene import TweezerRackScene
    from rack_pickup_task import RackPickupTask,RackPickupConfig,PHASES as PICKUP_PHASES
    from continuous_pick_place import ContinuousPickPlaceTask,demo_joint_baseline
    from rl_task import TaskConfig
    from pick_place_task import PickPlaceConfig,PHASES as PLACE_PHASES
    from ppo import ActorCritic
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    R=Rotation.from_euler('z',scene_cfg['tool_yaw_deg'],degrees=True).as_matrix()@np.diag([1.,-1.,-1.])
    # Match run_calibrated_demo.sh / run_pick_place.py physics from the start.
    # The rack and physical pickup are added; nothing is swapped at handoff.
    physics=dict(spring=5.,spring_damping=.02,hand_stiffness=10.,hand_damping=.3,
        finger_friction=1.4,solver_iterations=64,solver_velocity_iterations=8,
        contact_offset=.0002,simplify_arm_collisions=False,enable_grasp_support=True,enable_face_metrics=True)
    scene=TweezerRackScene(rack_root_xyz=scene_cfg['rack_tool_root_xyz_m'],rack_yaw_deg=scene_cfg['rack_yaw_deg'],
        rack_guides=scene_cfg['rack_guides'],block_xy=start_xy,
        num_envs=1,device=a.device,dt=1/480,pose_json=pose_path,tool_world_R=R,
        tool_position=scene_cfg['rack_tool_root_xyz_m'],table_height=scene_cfg['table_height_m'],
        table_size=layout['table_size_m'],table_center_xy=layout['table_center_xy_m'],**physics)
    (out/'scene_layout.json').write_text(json.dumps(dict(
        table_size_m=scene.table_size.tolist(),table_center_xy_m=scene.table_center_xy.tolist(),
        table_height_m=scene.table_height,home_xy_m=layout['home_xy_m'],start_xy_m=[start_xy],goal_xy_m=[goal_xy],
        rack_root_xyz_m=scene_cfg['rack_tool_root_xyz_m'],rack_offset_y_m=a.rack_offset_y,physics=physics,
        cameras='Original calibrated wrist/global camera rig; no recentering',
        physics_reference='run_calibrated_demo.sh and run_pick_place.py defaults',
        pickup_distribution_shift='Original calibrated-demo physics and block start, plus explicit rack/tool Y translation, differ from pickup training'),indent=2))
    # Original demo goal marker: visual bars only, no collision or body API.
    from pxr import UsdGeom,Gf
    for j,(offset,size) in enumerate([((-.013,0),(.002,.028)),((.013,0),(.002,.028)),((0,-.013),(.028,.002)),((0,.013),(.028,.002))]):
        cube=UsdGeom.Cube.Define(scene.stage,f'/World/PipelineGoal_{j}');cube.CreateSizeAttr(1.)
        xf=UsdGeom.Xformable(cube);xf.AddTranslateOp().Set(Gf.Vec3d(goal_xy[0]+offset[0],goal_xy[1]+offset[1],scene.table_height+.0005))
        xf.AddScaleOp().Set(Gf.Vec3f(size[0],size[1],.0004));cube.CreateDisplayColorAttr([Gf.Vec3f(.1,.95,.3)])
    pickup=RackPickupTask(scene,RackPickupConfig(**task_cfg),snapshot_path,layout)
    pickup_meta=pickup.metadata();(out/'pickup_task.json').write_text(json.dumps(pickup_meta,indent=2))
    def load_policy(path,obs_size,action_size):
        saved=torch.load(path,map_location=scene.device,weights_only=False)
        if (saved['observation_size'],saved['action_size'])!=(obs_size,action_size):raise ValueError('Policy dimensions mismatch')
        policy=ActorCritic(obs_size,action_size).to(scene.device);policy.load_state_dict(saved['policy']);policy.eval()
        return saved,policy
    pickup_policy=None
    if 'pickup' in checkpoints:
        saved,pickup_policy=load_policy(out/'checkpoints/pickup.pt',pickup.observation_size,pickup.action_size)
        previous=saved.get('env_config',{})
        for key in ['task_kind','implementation_signature','home_tool_position','home_tool_quaternion']:
            if previous.get(key)!=pickup_meta[key]:raise ValueError('Pickup checkpoint mismatch: '+key)
        # Permit only the declared common Y translation; geometry and rotation
        # must still match the trained rack, including every individual part.
        old_parts=previous.get('rack_parts',[]);new_parts=pickup_meta['rack_parts']
        if len(old_parts)!=len(new_parts):raise ValueError('Pickup checkpoint mismatch: rack parts')
        for old_part,new_part in zip(old_parts,new_parts):
            old_geometry={k:v for k,v in old_part.items() if k!='center_xyz_m'}
            new_geometry={k:v for k,v in new_part.items() if k!='center_xyz_m'}
            expected=np.asarray(old_part['center_xyz_m'])+[0,a.rack_offset_y,0]
            if old_geometry!=new_geometry or not np.allclose(expected,new_part['center_xyz_m'],rtol=0,atol=1e-9):
                raise ValueError('Pickup checkpoint mismatch: rack beyond declared Y translation')
        before=previous['config'].copy();now=pickup_meta['config'].copy();before.pop('seed',None);now.pop('seed',None)
        if json.dumps(before,sort_keys=True)!=json.dumps(now,sort_keys=True):raise ValueError('Pickup task config mismatch')
        (out/'pickup_evaluation_overrides.json').write_text(json.dumps({
            'purpose':'Evaluate pickup in the original calibrated pick/place scene',
            'training_block_initial_position':previous.get('block_initial_position'),
            'evaluation_block_initial_position':pickup_meta['block_initial_position'],
            'rack_and_tool_translation_xyz_m':[0,a.rack_offset_y,0],
            'training_rack_parts':old_parts,'evaluation_rack_parts':new_parts,
            'evaluation_physics':physics,
            'compatibility':'Task, source signatures, rack geometry/rotation and HOME checked; only declared rack Y translation allowed; block follows original demo; physics shift recorded'},indent=2))
    snap=json.loads(snapshot_path.read_text());pose=scene.pose_config
    baseline=demo_joint_baseline(scene,layout,snap)
    home_R=Rotation.from_euler('z',layout['tool_yaw_deg'],degrees=True).as_matrix()@Rotation.from_quat(np.roll(snap['rigid_body_states_13']['handle'][3:7],-1)).as_matrix()
    place_cfg=TaskConfig(decimation=8,episode_seconds=a.place_seconds,
        initial_tip_clearance=layout.get('initial_tip_clearance_m',.075),reset_xy_range=0,
        closure_bias=.7,closure_scale=.5,closure_min=-.1,closure_max=1.2,
        reset_snapshot=str(snapshot_path),require_web_up=pose.get('task_orientation',{}).get('require_web_up',True),
        max_palm_up_component=pose.get('task_orientation',{}).get('max_palm_up_component',.25),
        require_face_pinch=True,independent_support_actions=True,index_support_action_rad=.008,
        middle_support_action_rad=.025,command_space_arm=True,
        tool_down_angle_range_deg=tuple(pose.get('task_orientation',{}).get('tool_down_angle_range_deg',[20,70])),
        seed=a.seed,work_xy=tuple(layout['home_xy_m']),reset_tool_quaternion_wxyz=tuple(np.roll(Rotation.from_matrix(home_R).as_quat(),1)))
    place_options=PickPlaceConfig(xy_lower=(-.08,.48),xy_upper=(.22,.76),retreat_closure=.8,
        squeeze_min_jaw_force_N=float(pose.get('squeeze_min_jaw_force_N',0)),
        squeeze_force_hold_seconds=float(pose.get('squeeze_force_hold_seconds',.25)),
        release_middle_joint_deltas=pose.get('release_middle_joint_deltas'),release_track_tip=bool(pose.get('release_track_tip',False)))
    results=[]
    state_keys=['joint_pos','joint_vel','handle_root_state_w','left_root_state_w','right_root_state_w','block_root_state_w']
    for episode in range(a.episodes):
        ep=out/f'episode_{episode:03d}';ep.mkdir()
        obs=pickup.reset()
        if a.enable_cameras:
            from calibrated_demo_render import CalibratedDemoRenderer
            renderer=CalibratedDemoRenderer(scene,a.calibrated_rig,a.world_base,ep,control_dt=pickup.dt)
            renderer.frame(observation_index=0,control_step=-1)
        if a.preview_only:
            renderer.close();renderer=None
            state=scene.get_state()
            (out/'scene_preview.json').write_text(json.dumps({
                'scope':'Initialized scene preview only; no task execution or success evaluation',
                'rack_root_xyz_m':scene_cfg['rack_tool_root_xyz_m'],
                'initial_joint_targets':pickup.initial_q[0].cpu().tolist(),
                'tool_root_state_w':state['tool_root_state_w'][0].cpu().tolist(),
                'block_root_state_w':state['block_root_state_w'][0].cpu().tolist()},indent=2))
            break
        traces={k:[] for k in ['stage','phase','q','target','tool','block','action','reward','done']}
        row={'episode':episode,'pickup_success':False,'place_success':False,'success':False,'handoff':None}
        total_step=0
        with (ep/'steps.jsonl').open('w') as log,torch.no_grad():
            def record(stage,phase,action,reward,done,env,info):
                state=scene.get_state()
                padded=torch.zeros((1,9),device=scene.device);padded[:,:action.shape[-1]]=action
                for key,value in [('stage',np.array(stage)),('phase',phase.cpu().numpy()),('q',state['joint_pos'].cpu().numpy()),
                    ('target',env.q_target.cpu().numpy()),('tool',state['tool_root_state_w'].cpu().numpy()),
                    ('block',state['block_root_state_w'].cpu().numpy()),('action',padded.cpu().numpy()),
                    ('reward',reward.cpu().numpy()),('done',done.cpu().numpy())]:traces[key].append(value.copy())
                m=info['measurements']
                log.write(json.dumps({'step':total_step,'stage':stage,'phase':int(phase[0]),'reward':float(reward[0]),
                    'tool_rack_force_N':float(pickup.contacts.tool[0]),'robot_rack_force_N':float(pickup.contacts.robot[0]),
                    'goal_error_m':float(info['goal_error'][0]) if 'goal_error' in info else None,
                    'block_lift_m':float(state['block_root_state_w'][0,2]-scene.env_origins[0,2]-scene.table_height-.0061),
                    'block_jaw_forces_N':state['block_jaw_contact_forces_w'][0].norm(dim=-1).tolist(),
                    'tool_grip_error_m':float(m['grip_error'][0]),
                    'bilateral_block_contact':bool(m['both'][0]) if stage==1 else None,
                    'valid_hand_tool_orientation':bool(m['orientation_ok'][0]) if stage==1 else None,
                    'face_pinch_ok':bool(m['face_pinch_ok'][0]) if stage==1 else bool(m['face'][0]),
                    'tip_target_error_m':float((place.target_tip-m['tip']).norm(dim=-1)[0]) if stage==1 else None})+'\n')
            for step in range(pickup.max_steps):
                phase=pickup.phase.clone()
                action=torch.zeros((1,9),device=scene.device) if pickup_policy is None else pickup_policy.act(obs)
                obs,reward,done,info=pickup.step(action,auto_reset=False)
                record(0,phase,action,reward,done,pickup,info);total_step+=1
                if renderer and total_step%2==0:renderer.frame(observation_index=total_step,control_step=total_step-1)
                if bool(done[0]):break
            row['pickup_success']=bool(info['success'][0]);row['pickup_duration_s']=(step+1)*pickup.dt
            row['pickup_max_phase']=int(pickup.max_phase[0])
            row['pickup_failure_causes']=[k for k,v in info['failure_causes'].items() if bool(v[0])]
            row['pickup_timeout']=bool(info['phase_timeout'][0] or info['time_outs'][0])
            if row['pickup_success']:
                before={k:scene.get_state()[k].clone() for k in state_keys}
                place=ContinuousPickPlaceTask(scene,place_cfg,place_options,pickup.q_target,goal_xy,baseline)
                after=scene.get_state()
                diffs={k:float((before[k]-after[k]).abs().max()) for k in state_keys}
                if any(v!=0 for v in diffs.values()):raise RuntimeError('Physics changed during handoff initialization')
                row['handoff']={'state_max_abs_differences':diffs,'continuous':True,'snapshot_restored':False}
                (ep/'handoff.json').write_text(json.dumps({'continuity':row['handoff'],
                    'states':{k:v.cpu().tolist() for k,v in before.items()},'joint_targets':pickup.q_target.cpu().tolist()},indent=2))
                (ep/'place_task.json').write_text(json.dumps(place.metadata(),indent=2))
                _,place_policy=load_policy(out/'checkpoints/place.pt',place.observation_size,place.action_size)
                obs=place.observe()
                for step in range(place.max_steps):
                    phase=place.phase.clone();action=place_policy.act(obs);pickup.contacts.clear()
                    obs,reward,done,info=place.step(action,auto_reset=False)
                    collision=bool(pickup.contacts.robot[0]>pickup.cfg.collision_failure_N)
                    if collision:done=done.clone();done[0]=True
                    record(1,phase,action,reward,done,place,info);total_step+=1
                    if renderer and total_step%2==0:renderer.frame(observation_index=total_step,control_step=total_step-1)
                    if bool(done[0]):break
                row.update(place_success=bool(place.episode_success[0]) and not collision,
                    place_duration_s=(step+1)*place.dt,place_max_phase=int(place.max_phase[0]),
                    place_failure=bool(info['failure'][0]),place_phase_timeout=bool(info['phase_timeout'][0]),
                    place_timeout=bool(info['time_outs'][0]),robot_rack_collision=collision,
                    goal_error_m=float(info['goal_error'][0]),block_grasped_once=bool(place.grasped_once[0]))
                row['success']=row['place_success']
        np.savez_compressed(ep/'trajectory.npz',**{k:np.stack(v) for k,v in traces.items()},dt=pickup.dt)
        if renderer:
            if total_step%2:renderer.frame(observation_index=total_step,control_step=total_step-1)
            renderer.close();renderer=None
        results.append(row)
        (out/'evaluation.json').write_text(json.dumps({'episodes':results,'successes':sum(r['success'] for r in results),
            'count':len(results),'pickup_phases':PICKUP_PHASES,'place_phases':PLACE_PHASES,
            'physics':'Original calibrated demo settings throughout both stages; passive rack added; no physics reset at handoff',
            'start_xy_m':start_xy,'goal_xy_m':goal_xy,'camera_mode':'calibrated_wrist_global',
            'trajectory_action_layout':'9 columns: pickup all 9; place first 6 and last 3 zero'},indent=2))
        print('TWEEZER_PIPELINE_RESULT',json.dumps(row),flush=True)
    drift=[path for path,digest in hashes.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest]
    (out/'source_check.json').write_text(json.dumps({'unchanged':not drift,'changed_files':drift},indent=2))
    if drift:raise RuntimeError('Source changed during pipeline evaluation')
finally:
    if renderer:renderer.close()
    if scene:scene.close()
    app.close(wait_for_replicator=False)
