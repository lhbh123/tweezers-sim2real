"""Train/evaluate/render physical, randomized pick-and-place with residual PPO."""
import argparse,json,hashlib,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def record_execution_source_provenance(output_dir, tracked_paths, launch_hashes, phase):
    """Check source files at execution boundaries without discarding run data.

    These are disk checks, not a claim that Python's already-loaded code objects
    equal a later disk read. Imports use live src; the launch copies are evidence.
    """
    report_path=output_dir/'execution_source_provenance.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {
        'schema_version':1,'import_mode':'live_project_sources',
        'launch_sha256':launch_hashes,'checks':[],
        'limitations':[
            'Hashes describe disk contents at the observed boundaries, not loaded Python bytecode.',
            'An edit and restoration between checks may be missed; a detected change does not identify the exact code executed.',
            'Results are retained even when source drift or a missing source file is detected.']}
    current={};archived={};errors={};tracked={}
    for source_path in tracked_paths:
        path=source_path.resolve();key=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        tracked[path]=key
        for label,candidate,destination in [('live',path,current),('archive',output_dir/'source'/path.name,archived)]:
            try:destination[key]=hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError as exc:destination[key]=None;errors[label+':'+key]=str(exc)
    loaded={}
    for name,module in list(sys.modules.items()):
        module_path=getattr(module,'__file__',None)
        if module_path:
            path=Path(module_path).resolve()
            if path in tracked:loaded[name]={'path':str(path),'tracked_key':tracked[path]}
    previous=report['checks'][-1]['live_sha256'] if report['checks'] else launch_hashes
    check={'phase':phase,'live_sha256':current,'archive_sha256':archived,
        'live_differs_from_launch':[key for key,value in current.items() if value!=launch_hashes.get(key)],
        'archive_differs_from_launch':[key for key,value in archived.items() if value!=launch_hashes.get(key)],
        'live_differs_from_previous_boundary':[key for key,value in current.items() if value!=previous.get(key)],
        'loaded_tracked_modules':loaded,'read_errors':errors}
    report['checks'].append(check)
    drift=any(row['live_differs_from_launch'] or row['archive_differs_from_launch'] or row['read_errors'] for row in report['checks'])
    report['status']='source_drift_detected' if drift else ('consistent_at_observed_boundaries' if phase=='before_shutdown' else 'initial_boundary_consistent')
    report['complete']=phase=='before_shutdown'
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if drift:print('EXECUTION_SOURCE_PROVENANCE',report['status'],phase,flush=True)
    return report

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',required=True);p.add_argument('--num-envs',type=int,default=32);p.add_argument('--seed',type=int,default=17)
p.add_argument('--evaluate',help='zero for nominal geometric controller, otherwise residual PPO checkpoint')
p.add_argument('--iterations',type=int,default=75);p.add_argument('--resume');p.add_argument('--initialize')
p.add_argument('--learning-rate',type=float,default=3e-5);p.add_argument('--std',type=float,default=.08)
p.add_argument('--episode-seconds',type=float,default=36.);p.add_argument('--speed',type=float,default=.06)
p.add_argument('--xy-lower',type=float,nargs=2,default=[.48,-.22]);p.add_argument('--xy-upper',type=float,nargs=2,default=[.76,.08])
p.add_argument('--retreat-closure',type=float,default=.7)
p.add_argument('--open-closure',type=float,default=.7);p.add_argument('--closed-closure',type=float,default=1.05)
p.add_argument('--release-extra-opening-max',type=float,default=0.)
p.add_argument('--release-contact-clear-seconds',type=float,default=0.)
p.add_argument('--withdraw-initial-speed',type=float)
p.add_argument('--cases',help='JSON with explicit start and goal XY lists for edge-case tests')
p.add_argument('--pose',help='Hand grasp/robot asset config, defaults to original XHand v15')
p.add_argument('--snapshot',help='Dynamically stable pre-held tool state matching selected hand')
p.add_argument('--tactile',action='store_true',help='Record W01 contact-force maps, single environment')
p.add_argument('--contact-shapes',action='store_true',help='Record native PhysX robot/tool contact collider identities after each physics step')
p.add_argument('--collision-geometry',action='store_true',help='Export current native PhysX convex collision representation after reset without changing physics settings')
p.add_argument('--tacmap',action='store_true',help='Run the official SharpaTacmap geometric sensor, single environment')
p.add_argument('--tacmap-resolution-step',type=int,choices=[1,3],default=1)
p.add_argument('--tacmap-diagnostics',action='store_true',help='Record the original GPU CPD and preblur tensors without repeating sensor work')
p.add_argument('--tacmap-diagnostic-rays',action='store_true',help='Also store exact GPU world rays for every pixel; implies diagnostics and uses more archive space')
p.add_argument('--hand-stiffness',type=float,default=10.);p.add_argument('--hand-damping',type=float,default=.3)
p.add_argument('--finger-friction',type=float,default=1.4)
p.add_argument('--spring',type=float,default=5.)
p.add_argument('--camera-config');p.add_argument('--record-substeps',action='store_true')
p.add_argument('--calibrated-rig',help='Verified camera_rig_isaac.json; records rectified wrist/global RGB')
p.add_argument('--world-base',help='Verified world_base_isaac.json for the selected scene')
p.add_argument('--scene-layout',help='JSON with table dimensions/centre, tool spawn, HOME XY and reset tool yaw')
# Lens plugins must be registered before AppLauncher/omni/pxr import USD.
camera_args,_=p.parse_known_args()
if camera_args.calibrated_rig:
    calibrated_examples=Path(camera_args.calibrated_rig).resolve().parents[1]/'examples'
    sys.path.insert(0,str(calibrated_examples))
    from isaac_camera_bootstrap import register_lens_schemas
    register_lens_schemas()
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.multi_gpu=False;a.enable_cameras=bool(a.camera_config or a.calibrated_rig)
if a.calibrated_rig:
    if not a.world_base or a.camera_config or a.num_envs!=1 or not a.evaluate:
        p.error('--calibrated-rig requires --world-base, --num-envs 1 and --evaluate; omit --camera-config')
    a.kit_args+=' --enable omni.usd.schema.omni_lens_distortion'
elif a.world_base:
    p.error('--world-base requires --calibrated-rig')
if (a.tacmap_diagnostics or a.tacmap_diagnostic_rays) and not a.tacmap:
    p.error('--tacmap-diagnostics/--tacmap-diagnostic-rays require --tacmap')
out=ROOT/a.out;out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
pose=Path(a.pose).resolve() if a.pose else ROOT/'configs/human_grasp_middle_backup_v14.json';snapshot=Path(a.snapshot).resolve() if a.snapshot else ROOT.parent/'models/snapshots/xhand_stable_grasp.json'
pose_config=json.loads(pose.read_text())
files=[ROOT/'src'/n for n in ['rl_task.py','pick_place_task.py','ppo.py','rl_scene.py','grasp_contacts.py','arm_command.py','arm_clearance.py','kinematics.py','reset_reference.py','probe_render.py']]+[pose,snapshot,Path(__file__)]
files += [Path(value).resolve() for value in [a.camera_config,a.cases,a.scene_layout] if value]
if a.calibrated_rig:
    files += [ROOT/'src/calibrated_demo_render.py',Path(a.calibrated_rig).resolve(),Path(a.world_base).resolve(),
              calibrated_examples/'isaac_camera_rig.py',calibrated_examples/'isaac_camera_bootstrap.py']
if a.tactile or a.tacmap:files.extend([ROOT/'src/sharpa_tactile.py',ROOT/'src/sharpa_materials.py'])
if a.tacmap:files.extend([ROOT/'src/sharpa_tacmap.py',ROOT/'src/sharpa_tacmap_compat.py'])
if a.contact_shapes:files.append(ROOT/'src/sharpa_contact_shapes.py')
if a.collision_geometry:files.append(ROOT/'src/sharpa_collision_geometry.py')
hashes={}
for f in files:hashes[str(f.resolve().relative_to(ROOT)) if f.resolve().is_relative_to(ROOT) else str(f.resolve())]=hashlib.sha256(f.read_bytes()).hexdigest();shutil.copy2(f,source/f.name)
model_hashes={}
if pose_config.get('robot_asset_dir'):
    model_root=Path(pose_config['robot_asset_dir'])
    if not model_root.is_absolute():model_root=(ROOT/model_root).resolve()
    for model_file in sorted(model_root.rglob('*')):
        if model_file.is_file() and model_file.suffix in ('.urdf','.usd','.usda','.json'):
            model_hashes[str(model_file.resolve())]=hashlib.sha256(model_file.read_bytes()).hexdigest()
(out/'robot_asset_sha256.json').write_text(json.dumps(model_hashes,indent=2))
(out/'launch.json').write_text(json.dumps(dict(arguments=vars(a),source_sha256=hashes),indent=2))
app=AppLauncher(a).app
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from rl_scene import BatchedTweezersScene
from rl_task import TaskConfig
from pick_place_task import PickPlaceTask,PickPlaceConfig,PHASES
from ppo import PPO,PPOConfig,ActorCritic
np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
snap=json.loads(snapshot.read_text());R=Rotation.from_quat(np.roll(snap['rigid_body_states_13']['handle'][3:7],-1)).as_matrix()
layout=json.loads(Path(a.scene_layout).read_text()) if a.scene_layout else {}
layout_yaw=float(layout.get('tool_yaw_deg',0.))
if not np.isfinite(layout_yaw):raise ValueError('Layout tool_yaw_deg must be finite')
R=Rotation.from_euler('z',layout_yaw,degrees=True).as_matrix() @ R
reset_orientation=tuple(np.roll(Rotation.from_matrix(R).as_quat(),1)) if layout_yaw else None
scene=BatchedTweezersScene(num_envs=a.num_envs,device=a.device,dt=1/480,pose_json=pose,tool_world_R=R,
    tool_position=tuple(layout.get('tool_spawn_xyz_m',[.48,0,.95])),
    table_size=tuple(layout.get('table_size_m',[1.10,.80,.04])),table_center_xy=tuple(layout.get('table_center_xy_m',[.35,0.])),
    spring=a.spring,spring_damping=.02,hand_stiffness=a.hand_stiffness,hand_damping=a.hand_damping,finger_friction=a.finger_friction,solver_iterations=64,solver_velocity_iterations=8,
    contact_offset=.0002,simplify_arm_collisions=False,enable_grasp_support=True,enable_face_metrics=True)
cfg=TaskConfig(decimation=8,episode_seconds=a.episode_seconds,initial_tip_clearance=.075,reset_xy_range=0,
    closure_bias=a.open_closure,closure_scale=max(.5,abs(a.closed_closure-a.open_closure)+.05),closure_min=-.1,closure_max=max(1.2,a.closed_closure+.1),reset_snapshot=str(snapshot),
    require_web_up=pose_config.get('task_orientation',{}).get('require_web_up',True),max_palm_up_component=pose_config.get('task_orientation',{}).get('max_palm_up_component',.25),require_face_pinch=True,independent_support_actions=True,
    index_support_action_rad=.008,middle_support_action_rad=.025,command_space_arm=True,tool_down_angle_range_deg=tuple(pose_config.get('task_orientation',{}).get('tool_down_angle_range_deg',[20,70])),seed=a.seed,
    work_xy=tuple(layout.get('home_xy_m',[.48,0.])),reset_tool_quaternion_wxyz=reset_orientation)
env=PickPlaceTask(scene,cfg,PickPlaceConfig(xy_lower=tuple(a.xy_lower),xy_upper=tuple(a.xy_upper),max_speed_m_s=a.speed,retreat_closure=a.retreat_closure,open_closure=a.open_closure,closed_closure=a.closed_closure,
    squeeze_min_jaw_force_N=float(pose_config.get('squeeze_min_jaw_force_N',0.)),
    squeeze_force_hold_seconds=float(pose_config.get('squeeze_force_hold_seconds',.25)),
    release_extra_opening_max=a.release_extra_opening_max,release_contact_clear_seconds=a.release_contact_clear_seconds,withdraw_initial_speed_m_s=a.withdraw_initial_speed,
    release_middle_joint_deltas=pose_config.get('release_middle_joint_deltas'),
    release_track_tip=bool(pose_config.get('release_track_tip',False))))
if a.cases:
    env.reset_override=json.loads(Path(a.cases).read_text());assert len(env.reset_override['start'])==a.num_envs
obs=env.reset()
(out/'scene_layout.json').write_text(json.dumps(dict(table_size_m=scene.table_size.tolist(),table_center_xy_m=scene.table_center_xy.tolist(),
    table_height_m=scene.table_height,home_xy_m=list(cfg.work_xy),tool_yaw_deg=layout_yaw,
    start_xy_m=(env.start[:,:2]-scene.env_origins[:,:2]).cpu().tolist(),goal_xy_m=(env.goal[:,:2]-scene.env_origins[:,:2]).cpu().tolist()),indent=2)+'\n')
if a.collision_geometry:
    from sharpa_collision_geometry import capture_collision_geometry
    capture_collision_geometry(scene,out/'collision_geometry',phase='after_task_reset_before_observation_callbacks')
contact_shape_reader=None
if a.contact_shapes:
    from sharpa_contact_shapes import ContactShapeRecorder
    contact_shape_reader=ContactShapeRecorder(scene,out/'contact_shapes.jsonl')
    scene.physics_step_callbacks.append(contact_shape_reader.record_step)
(out/'task.json').write_text(json.dumps(env.metadata(),indent=2));print('PICK_PLACE_READY',env.num_envs,env.observation_size,flush=True)
if a.tactile or a.tacmap:
    from sharpa_materials import capture_scene_materials
    try:capture_scene_materials(scene,out/'runtime_material_readback.json')
    except Exception as exc:
        (out/'runtime_material_readback.json').write_text(json.dumps({
            'status':'runtime_readback_unavailable','read_only':True,'error':str(exc),
            'notes':['No USD fallback value is labelled as a live runtime material value.']},indent=2)+'\n')
if not a.evaluate:
    trainer=PPO(env,out,PPOConfig(rollout_steps=96,learning_rate=a.learning_rate,initial_std=a.std,target_kl=.02,entropy_coef=.001,gamma=.995),seed=a.seed,resume=a.resume,initialize=a.initialize)
    record_execution_source_provenance(out,files,hashes,'after_imports_and_setup')
    trainer.train(a.iterations,save_every=5)
else:
    policy=None
    if a.evaluate!='zero':
        ckpt=torch.load(a.evaluate,map_location=env.device,weights_only=False);policy=ActorCritic(ckpt['observation_size'],ckpt['action_size']).to(env.device);policy.load_state_dict(ckpt['policy']);policy.eval()
    renderer=None
    if a.camera_config or a.calibrated_rig:
        # Goal is four thin VISUAL bars. No collision, body or joint API is applied.
        from pxr import UsdGeom,Gf,UsdShade,Sdf
        material=UsdShade.Material.Define(scene.stage,'/World/GoalMaterial');shader=UsdShade.Shader.Define(scene.stage,'/World/GoalMaterial/Surface');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.015,.55,.06));shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0,.06,.005));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        for i in range(env.num_envs):
            g=env.goal[i].cpu().tolist()
            for j,(offset,size) in enumerate([((-.013,0),(.002,.028)),((.013,0),(.002,.028)),((0,-.013),(.028,.002)),((0,.013),(.028,.002))]):
                cube=UsdGeom.Cube.Define(scene.stage,f'/World/Goal_{i}_{j}');cube.CreateSizeAttr(1.)
                xf=UsdGeom.Xformable(cube);xf.AddTranslateOp().Set(Gf.Vec3d(g[0]+offset[0],g[1]+offset[1],scene.table_height+float(scene.env_origins[i,2])+.0005));xf.AddScaleOp().Set(Gf.Vec3f(size[0],size[1],.0004));cube.CreateDisplayColorAttr([Gf.Vec3f(.1,.95,.3)]);UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        if a.calibrated_rig:
            from calibrated_demo_render import CalibratedDemoRenderer
            renderer=CalibratedDemoRenderer(scene,a.calibrated_rig,a.world_base,out,control_dt=env.dt)
        else:
            from probe_render import ProbeRenderer
            renderer=ProbeRenderer(scene,a.camera_config,out)
    tactile_reader=None
    if a.tactile:
        if a.num_envs!=1:raise ValueError('Tactile video recording currently requires one environment')
        from sharpa_tactile import SharpaTactileReader
        tactile_reader=SharpaTactileReader.from_scene(scene)
        tactile_time=[0.]
        def record_tactile(dt):
            tactile_time[0]+=dt
            tactile_reader.record(tactile_time[0],dt)
        scene.physics_step_callbacks.append(record_tactile)
    tacmap_reader=None
    if a.tacmap:
        if a.num_envs!=1:raise ValueError('Official TacMap recording currently requires one environment')
        from sharpa_tacmap import OfficialTacmapRecorder
        tacmap_reader=OfficialTacmapRecorder(scene,resolution_step=a.tacmap_resolution_step,
            diagnostics=a.tacmap_diagnostics,diagnostic_rays=a.tacmap_diagnostic_rays)
        if tactile_reader is not None:
            for force_surface,sensor in zip(tactile_reader.maps,tacmap_reader.rows):
                force_hashes=[force_surface.metadata['point_sha256'],force_surface.metadata['normal_sha256']]
                if force_hashes != [item['sha256'] for item in sensor['surface_assets']]:
                    raise RuntimeError('Force-map and TacMap source surfaces differ for '+sensor['finger'])
        print('OFFICIAL_TACMAP_READY',json.dumps({'sensors':len(tacmap_reader.sensors),'resolution':tacmap_reader.resolution}),flush=True)
    record_execution_source_provenance(out,files,hashes,'after_imports_and_setup')
    scene.capture_grasp_substeps=a.record_substeps
    live=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);success=torch.zeros_like(live)
    final_error=torch.full((env.num_envs,),float('nan'),device=env.device);duration=torch.zeros_like(final_error);max_phase=env.phase.clone();max_lift=env.episode_max_lift.clone()
    starts=env.start.cpu().numpy()-scene.env_origins.cpu().numpy();goals=env.goal.cpu().numpy()-scene.env_origins.cpu().numpy()
    arrays={k:[] for k in ['block','tip','phase','applied','residual','goal_error','lift','both','retained','face','success','q','tool','left','right','active','normal_angle','forces','orientation_ok','orientation_error','palm_normal','web_axis','release_extra_opening','release_middle_blend','squeeze_force_hold_time_s','squeeze_force_gate_ready','squeeze_min_jaw_force_N']}
    render_control_steps=[]
    raw=(out/'grasp_substeps.jsonl').open('w') if a.record_substeps else None
    with torch.no_grad():
      for step in range(env.max_steps):
        action=torch.zeros((env.num_envs,env.action_size),device=env.device) if policy is None else policy.act(obs)
        active=live.clone();action[~live]=0
        obs,reward,done,info=env.step(action,auto_reset=False);m=info['measurements'];s=m['state']
        success|=env.episode_success&active;max_phase=torch.maximum(max_phase,torch.where(active,env.phase,max_phase));max_lift=torch.maximum(max_lift,torch.where(active,m['lift'],max_lift));duration+=active*env.dt
        for k,v in [('block',s['block_root_state_w']),('tip',m['tip']),('phase',env.phase),('applied',info['applied_action']),('residual',action),('goal_error',info['goal_error']),('lift',m['lift']),('both',m['both']),('retained',m['grip_error']),('face',m['face_pinch_ok']),('success',env.episode_success),('q',s['joint_pos']),('tool',s['tool_root_state_w']),('left',s['left_root_state_w']),('right',s['right_root_state_w']),('active',active),('normal_angle',m['tool_down_angle_deg']),('forces',m['forces']),('orientation_ok',m['orientation_ok']),('orientation_error',m['orientation_error']),('palm_normal',m['palm_normal_w']),('web_axis',m['web_side_axis_w']),('release_extra_opening',env.release_extra_opening)]:arrays[k].append(v.cpu().numpy().copy())
        arrays['release_middle_blend'].append(env.release_middle_blend.cpu().numpy().copy())
        for key in ('squeeze_force_hold_time_s','squeeze_force_gate_ready','squeeze_min_jaw_force_N'):
            arrays[key].append(m[key].cpu().numpy().copy())
        if raw:raw.write(json.dumps(dict(time=(step+1)*env.dt,substeps=scene.grasp_contact_substeps))+'\n')
        just=live&done;final_error[just]=info['goal_error'][just];live&=~done
        if renderer and (step%2==0 or not bool(live.any())):
            renderer.frame();render_control_steps.append(step)
        if tacmap_reader is not None and (step%2==0 or not bool(live.any())):
            tacmap_reader.record((step+1)*env.dt,step)
        if step%120==119:print('PICK_PLACE_PROGRESS',round((step+1)*env.dt,1),'phase',torch.bincount(env.phase,minlength=10).tolist(),'success',int(success.sum()),'live',int(live.sum()),'tip_error_max_mm',float((env.target_tip-m['tip']).norm(dim=-1)[live].max()*1000) if bool(live.any()) else 0,flush=True)
        if step%120==119:
            print('PICK_PLACE_DIAGNOSTIC',json.dumps(dict(tool_angle_deg=m['tool_down_angle_deg'][:4].tolist(),face=m['face_pinch_ok'][:4].tolist(),orientation_ok=m['orientation_ok'][:4].tolist(),grip_error_m=m['grip_error'][:4].tolist(),jaw_forces_N=m['forces'][:4].tolist(),lift_m=m['lift'][:4].tolist())),flush=True)
        if not bool(live.any()):break
        # Completed evaluation rows are frozen in accounting; no reset affects scored traces.
    if raw:raw.close()
    if renderer:renderer.close()
    if tactile_reader is not None:
        frame_times = ((np.asarray(render_control_steps)+1)*env.dt if render_control_steps
                       else np.asarray(tacmap_reader.times) if tacmap_reader is not None else None)
        tactile_reader.save(out/'tactile.npz',frame_times_s=frame_times)
    if tacmap_reader is not None:tacmap_reader.save(out/'tacmap.npz')
    np.savez_compressed(out/'trajectory.npz',**{k:np.stack(v) for k,v in arrays.items()},start=starts,goal=goals,dt=env.dt,origins=scene.env_origins.cpu().numpy(),render_control_steps=np.array(render_control_steps,dtype=np.int64))
    report=dict(algorithm='nominal_geometric_controller' if policy is None else 'goal_conditioned_residual_PPO_with_geometric_phase_controller',checkpoint=a.evaluate,checkpoint_sha256=hashlib.sha256(Path(a.evaluate).read_bytes()).hexdigest() if policy is not None else None,checkpoint_iterations=ckpt.get('iteration') if policy is not None else None,checkpoint_transitions=ckpt.get('samples') if policy is not None else None,seed=a.seed,episodes=env.num_envs,successes=int(success.sum()),success_rate=float(success.float().mean()),success_per_env=success.tolist(),start=starts.tolist(),goal=goals.tolist(),final_xy_error_m=final_error.tolist(),duration_s=duration.tolist(),max_phase=max_phase.tolist(),max_lift_m=max_lift.tolist(),task=env.metadata())
    report['execution_source_provenance_file']='execution_source_provenance.json'
    (out/'evaluation.json').write_text(json.dumps(report,indent=2));scene.stage.Export(str(out/'final_scene.usda'));print('PICK_PLACE_RESULT',json.dumps({k:report[k] for k in ['successes','episodes','max_phase','final_xy_error_m']}),flush=True)
if contact_shape_reader is not None:contact_shape_reader.close()
record_execution_source_provenance(out,files,hashes,'before_shutdown')
scene.close();app.close()
