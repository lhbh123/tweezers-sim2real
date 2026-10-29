"""Frozen-policy diagnosis of each handoff predicate under controlled action noise."""
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
if a.resume or not a.evaluate or a.evaluate=='zero' or a.video or a.num_envs!=5:
    p.error('Diagnosis requires --evaluate CHECKPOINT --num-envs 5, without resume/video')
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
    import math
    policy=ActorCritic(env.observation_size,env.action_size).to(env.device)
    policy.load_state_dict(saved['policy']);policy.eval()
    modes=['deterministic','noise_only_handoff','stochastic','noise_off_handoff','zero']
    obs=env.reset();live=torch.ones(5,dtype=torch.bool,device=env.device)
    records=[[] for _ in modes];results=[None]*len(modes)
    cfg=env.cfg
    with (out/'steps.jsonl').open('w',buffering=1) as log,torch.no_grad():
        for step in range(env.max_steps):
            phase=env.phase.clone()
            dist=policy.distribution(policy.moments(obs))
            noise=torch.randn_like(dist.mean)
            noise[3]=noise[2]  # paired stochastic prefixes before handoff
            sampled=(dist.mean+dist.stddev*noise).tanh()
            action=dist.mean.tanh()
            if int(phase[1])==5:action[1]=sampled[1]
            action[2]=sampled[2]
            if int(phase[3])!=5:action[3]=sampled[3]
            action[4]=0
            obs,reward,done,info=env.step(action,auto_reset=False)
            m=info['measurements'];t=env.phase_steps*env.dt
            flags={
                'probe_finished':t>1.,
                'home_position':m['home_error']<cfg.handoff_home_m,
                'home_angle':m['home_angle']<math.radians(cfg.handoff_angle_deg),
                'face_contact':m['face'],
                'grip_position':m['grip_error']<cfg.handoff_position_m,
                'grip_angle':m['grip_angle']<math.radians(cfg.handoff_angle_deg),
                'relative_speed':m['relative_speed']<cfg.relative_speed_m_s,
                'relative_angular_speed':m['relative_angular_speed']<cfg.relative_angular_speed_rad_s,
                'hand_error':m['hand_error']<cfg.handoff_hand_rms_rad,
                'rack_clear':env.contacts.tool<cfg.rack_clear_force_N,
                'ever_detached':env.ever_detached,
                'aperture_response':env.gap_open-env.gap_closed>cfg.aperture_response_m}
            values={k:m[k] for k in ['home_error','home_angle','grip_error','grip_angle','relative_speed','relative_angular_speed','hand_error']}
            values.update(aperture_response=env.gap_open-env.gap_closed,tool_rack_force=env.contacts.tool)
            for i,mode in enumerate(modes):
                if not bool(live[i]):continue
                if int(phase[i])==5:
                    row={'mode':mode,'step':step,'phase_time_s':float(t[i]),
                         'ready_steps':int(env.ready_steps[i]),'reward':float(reward[i]),
                         'flags':{k:bool(v[i]) for k,v in flags.items()},
                         'values':{k:float(v[i]) for k,v in values.items()},'action':action[i].tolist()}
                    expected=all(row['flags'].values())
                    assert expected==(row['ready_steps']>0),row
                    records[i].append(row);log.write(json.dumps(row)+'\n')
                if bool(done[i]):
                    results[i]={'mode':mode,'success':bool(info['success'][i]),'max_phase':int(env.max_phase[i]),
                        'duration_s':(step+1)*env.dt,'phase_timeout':bool(info['phase_timeout'][i]),
                        'time_out':bool(info['time_outs'][i]),
                        'failure_causes':[k for k,v in info['failure_causes'].items() if bool(v[i])]}
                    live[i]=False
                    print('DIAGNOSIS_DONE',json.dumps(results[i]),flush=True)
            if step%300==0:print('DIAGNOSIS_PROGRESS',step,env.phase.tolist(),flush=True)
            if not bool(live.any()):break
    def longest(bits):
        best=current=0
        for bit in bits:
            current=current+1 if bit else 0;best=max(best,current)
        return best
    for i,result in enumerate(results):
        if result is None:result={'mode':modes[i],'success':False,'incomplete':True};results[i]=result
        rows=records[i];eligible=[r for r in rows if r['flags']['probe_finished']]
        result['handoff_samples']=len(rows);result['post_probe_samples']=len(eligible)
        if eligible:
            result['max_ready_steps']=max(r['ready_steps'] for r in rows)
            result['required_ready_steps']=math.ceil(cfg.stable_s/env.dt)
            result['criteria']={k:{'pass_fraction':sum(r['flags'][k] for r in eligible)/len(eligible),
                'longest_pass_steps':longest(r['flags'][k] for r in eligible),
                'longest_all_others_pass_steps':longest(all(v for key,v in r['flags'].items() if key!=k) for r in eligible)} for k in flags}
            result['measurements']={k:{'min':float(np.min([r['values'][k] for r in eligible])),
                'median':float(np.median([r['values'][k] for r in eligible])),
                'p95':float(np.percentile([r['values'][k] for r in eligible],95)),
                'max':float(np.max([r['values'][k] for r in eligible]))} for k in values}
    report={'modes':results,'seed':a.seed,'policy_iteration':saved['iteration'],'policy_samples':saved['samples'],
        'policy_sha256':hashlib.sha256(Path(a.evaluate).read_bytes()).hexdigest(),
        'latent_std':policy.log_std.clamp(-5,1).exp().tolist(),
        'scope':'Frozen final policy; normalization frozen; controlled evaluation, not exact replay of on-policy training'}
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    drift=[path for path,digest in hashes.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest]
    (out/'source_check.json').write_text(json.dumps({'unchanged':not drift,'changed_files':drift},indent=2))
    if drift:raise RuntimeError('Source changed during diagnosis')
finally:
    if scene:scene.close()
    app.close(wait_for_replicator=False)
