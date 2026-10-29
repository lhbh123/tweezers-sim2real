"""Train/evaluate an independent rack-pickup PPO; original policy stays untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser(description=__doc__)
# Isaac Lab pre-parses argv here with help disabled. Register its arguments
# before our required --out argument so --help can reach the final parser.
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
p.add_argument('--out',required=True)
p.add_argument('--task-config',default=str(ROOT/'configs/rack_pickup_train.json'))
p.add_argument('--scene-config',default=str(ROOT/'configs/tweezer_rack_scene.json'))
p.add_argument('--num-envs',type=int,default=8)
p.add_argument('--iterations',type=int,default=1000)
p.add_argument('--rollout-steps',type=int,default=256)
p.add_argument('--save-every',type=int,default=25)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--learning-rate',type=float,default=1e-4)
p.add_argument('--std',type=float,default=.15)
p.add_argument('--resume',help='Resume only a checkpoint from this rack-pickup task')
p.add_argument('--evaluate',help='zero for nominal controller, otherwise a rack-pickup checkpoint')
p.add_argument('--video',action='store_true',help='Single-env evaluation only')
p.add_argument('--eval-episodes',type=int,default=1)
a=p.parse_args(); a.headless=True; a.enable_cameras=a.video; a.multi_gpu=False
if a.num_envs<1 or a.iterations<1 or a.rollout_steps*a.num_envs<4 or a.save_every<1 or a.eval_episodes<1:
    p.error('Positive counts required and rollout batch must contain at least four samples')
if a.video and (not a.evaluate or a.num_envs!=1):p.error('--video requires --evaluate and --num-envs 1')
if a.resume and a.evaluate:p.error('--resume and --evaluate are mutually exclusive')
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
scene_cfg=json.loads(Path(a.scene_config).read_text())
task_cfg=json.loads(Path(a.task_config).read_text());task_cfg['seed']=a.seed
layout_path=ROOT/'configs/calibrated_demo_y_table.json'
snapshot=ROOT.parent/'models/snapshots/xhand_stable_grasp.json'
pose_path=ROOT/'configs/xhand_local.json'
tracked=[Path(__file__),ROOT/'src/rack_pickup_task.py',ROOT/'src/tweezer_rack_scene.py',ROOT/'src/rl_scene.py',
         ROOT/'src/ppo.py',ROOT/'src/reset_reference.py',ROOT/'src/arm_command.py',ROOT/'src/grasp_contacts.py',
         ROOT/'src/kinematics.py',ROOT/'src/arm_clearance.py',ROOT/'src/rl_task.py',
         Path(a.task_config),Path(a.scene_config),layout_path,snapshot,pose_path]
(out/'source').mkdir()
hashes={}
for i,path in enumerate(tracked):
    dest=f'{i:02d}_{path.name}';shutil.copy2(path,out/'source'/dest)
    hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
(out/'launch.json').write_text(json.dumps(dict(args=vars(a),source_sha256=hashes),indent=2))
app=AppLauncher(a).app
scene=None;trainer=None;renderer=None
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from tweezer_rack_scene import TweezerRackScene
    from rack_pickup_task import RackPickupTask,RackPickupConfig,PHASES
    from ppo import PPO,PPOConfig,ActorCritic
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    R=Rotation.from_euler('z',scene_cfg['tool_yaw_deg'],degrees=True).as_matrix()@np.diag([1.,-1.,-1.])
    scene=TweezerRackScene(rack_root_xyz=scene_cfg['rack_tool_root_xyz_m'],
        rack_yaw_deg=scene_cfg['rack_yaw_deg'],rack_guides=scene_cfg['rack_guides'],
        block_xy=scene_cfg.get('block_xy_m',[0.,.65]),
        num_envs=a.num_envs,device=a.device,dt=1/480,
        pose_json=pose_path,tool_world_R=R,tool_position=scene_cfg['rack_tool_root_xyz_m'],
        table_height=scene_cfg['table_height_m'],table_size=scene_cfg['table_size_m'],
        table_center_xy=scene_cfg['table_center_xy_m'],spring=5.,enable_face_metrics=True)
    env=RackPickupTask(scene,RackPickupConfig(**task_cfg),snapshot,json.loads(layout_path.read_text()))
    (out/'task.json').write_text(json.dumps(env.metadata(),indent=2))
    checkpoint=a.resume or (a.evaluate if a.evaluate!='zero' else None)
    saved=None
    if checkpoint:
        saved=torch.load(checkpoint,map_location=env.device,weights_only=False)
        if saved.get('env_config',{}).get('task_kind')!='rack_pickup_fixed_start_v2':
            raise ValueError('Expected a fixed-start v2 rack-pickup checkpoint; train a new policy for this task')
        if (saved['observation_size'],saved['action_size'])!=(env.observation_size,env.action_size):
            raise ValueError('Checkpoint dimensions do not match')
        current_meta=env.metadata()
        for key in ['implementation_signature','rack_parts','block_initial_position','home_tool_position','home_tool_quaternion']:
            if saved['env_config'].get(key)!=current_meta[key]:
                raise ValueError('Checkpoint environment mismatch: '+key)
        previous=saved['env_config']['config'].copy();current=current_meta['config'].copy()
        previous.pop('seed',None);current.pop('seed',None)
        if json.dumps(previous,sort_keys=True)!=json.dumps(current,sort_keys=True):
            raise ValueError('Task config must match the checkpoint (except seed)')
    if not a.evaluate:
        trainer=PPO(env,out,PPOConfig(rollout_steps=a.rollout_steps,learning_rate=a.learning_rate,
            initial_std=a.std,epochs=4,minibatches=4,gamma=.995,gae_lambda=.95,target_kl=.02,entropy_coef=.001),
            seed=a.seed,resume=a.resume)
        trainer.train(a.iterations,save_every=a.save_every)
    else:
        policy=None
        if saved:
            policy=ActorCritic(env.observation_size,env.action_size).to(env.device)
            policy.load_state_dict(saved['policy']);policy.eval()
        if a.video:
            from probe_render import ProbeRenderer
            (out/'views.json').write_text(json.dumps(scene_cfg['views']))
            renderer=ProbeRenderer(scene,out/'views.json',out)
        results=[]
        for episode in range(a.eval_episodes):
            obs=env.reset();live=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            rows=[]; records=(out/f'episode_{episode:03d}.jsonl').open('w')
            traces={k:[] for k in ['q','target','tool','block','action','reward','phase','done']}
            with torch.no_grad():
                for step in range(env.max_steps):
                    phase=env.phase.clone()
                    action=torch.zeros((env.num_envs,env.action_size),device=env.device) if policy is None else policy.act(obs)
                    obs,reward,done,info=env.step(action,auto_reset=False)
                    m=info['measurements']
                    records.write(json.dumps(dict(step=step,phase=phase.tolist(),active=live.tolist(),reward=reward.tolist(),
                        target_distance_m=m['target_distance'].tolist(),target_angle_error=m['target_angle'].tolist(),
                        grip_error_m=m['grip_error'].tolist(),grip_angle_rad=m['grip_angle'].tolist(),
                        home_error_m=m['home_error'].tolist(),home_angle_rad=m['home_angle'].tolist(),
                        hand_error_rad=m['hand_error'].tolist(),relative_speed_m_s=m['relative_speed'].tolist(),
                        relative_angular_speed_rad_s=m['relative_angular_speed'].tolist(),
                        aperture_response_m=(env.gap_open-env.gap_closed).tolist(),
                        face=m['face'].tolist(),lift_m=m['lift'].tolist(),
                        tool_rack_force_N=env.contacts.tool.tolist(),robot_rack_force_N=env.contacts.robot.tolist(),
                        reward_terms={k:v.tolist() for k,v in info['reward_terms'].items()}))+'\n')
                    for key,value in [('q',m['state']['joint_pos']),('target',env.q_target),('tool',m['state']['tool_root_state_w']),
                                      ('block',m['state']['block_root_state_w']),
                                      ('action',action),('reward',reward),('phase',phase),('done',done)]:
                        traces[key].append(value.cpu().numpy().copy())
                    if renderer and step%2==0:renderer.frame()
                    for i in (done&live).nonzero(as_tuple=False).flatten().tolist():
                        row=dict(episode=episode,env=i,success=bool(info['success'][i]),max_phase=int(env.max_phase[i]),
                                 duration_s=(step+1)*env.dt,return_sum=float(env.episode_return[i]),
                                 failure_causes=[k for k,v in info['failure_causes'].items() if bool(v[i])],
                                 phase_timeout=bool(info['phase_timeout'][i]),time_out=bool(info['time_outs'][i]))
                        rows.append(row)
                        # Actual continuous state only: no restore of the reference snapshot.
                        if row['success']:
                            state=scene.get_state()
                            handoff=dict(joint_names=scene.joint_names,q_actual=state['joint_pos'][i].tolist(),
                                q_target=env.q_target[i].tolist(),q_velocity=state['joint_vel'][i].tolist(),
                                rigid_body_states_13={name:state[name+'_root_state_w'][i].tolist() for name in ['handle','left','right','block']},
                                env_origin=scene.env_origins[i].tolist(),candidate_only=True,
                                downstream_ppo_validated=False)
                            (out/f'handoff_{episode:03d}_{i:03d}.json').write_text(json.dumps(handoff,indent=2))
                    live&=~done
                    if not bool(live.any()):break
            records.close();results.extend(rows)
            np.savez_compressed(out/f'episode_{episode:03d}.npz',**{k:np.stack(v) for k,v in traces.items()},dt=env.dt)
        (out/'evaluation.json').write_text(json.dumps(dict(episodes=results,successes=sum(r['success'] for r in results),
            count=len(results),phases=PHASES,downstream_ppo_validated=False),indent=2))
        print('RACK_PICKUP_EVALUATION',json.dumps(results),flush=True)
    drift=[path for path,digest in hashes.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest]
    (out/'source_check.json').write_text(json.dumps(dict(unchanged=not drift,changed_files=drift),indent=2))
    if drift:raise RuntimeError('Tracked source changed during run')
finally:
    if renderer:renderer.close()
    if trainer:trainer.log.close()
    if scene:scene.close()
    app.close(wait_for_replicator=False)
