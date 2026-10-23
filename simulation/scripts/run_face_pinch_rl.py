"""Run the revised straight-tail, broad-face task with optional backup support."""
import argparse
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode',choices=['train','evaluate','render'])
p.add_argument('--out',required=True)
p.add_argument('--pose',required=True)
p.add_argument('--snapshot',required=True)
p.add_argument('--device',default='cuda:4')
p.add_argument('--num-envs',type=int,default=64)
p.add_argument('--seed',type=int,default=47)
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--checkpoint')
p.add_argument('--initializer')
p.add_argument('--initializer-std',type=float)
p.add_argument('--camera-config')
p.add_argument('--spring',type=float,default=5.)
p.add_argument('--spring-damping',type=float,default=.02)
p.add_argument('--closure-scale',type=float,default=1.2)
p.add_argument('--closure-bias',type=float,default=0.)
p.add_argument('--closure-max',type=float,default=1.2)
p.add_argument('--clearance',type=float,default=.018)
p.add_argument('--initial-std',type=float,default=.15)
p.add_argument('--learning-rate',type=float,default=.0001)
p.add_argument('--tool-angle-max',type=float,default=60.)
p.add_argument('--support-reward',type=float,default=1.5)
a=p.parse_args()
if (ROOT/a.out).exists():p.error('Use a fresh output directory to preserve previous evidence')
if a.mode!='train' and not a.checkpoint:p.error('Evaluation/render requires an explicit checkpoint')
cmd=['/cpfs/user/lvzhaoyang/apps/bin/isaac-python','scripts/train_rl.py',
    '--out',a.out,'--pose',a.pose,'--reset-snapshot',a.snapshot,'--device',a.device,
    '--headless','--num-envs',str(a.num_envs),'--seed',str(a.seed),
    '--dt',str(1/480),'--decimation','8','--spring',str(a.spring),
    '--spring-damping',str(a.spring_damping),'--solver-iterations','64',
    '--solver-velocity-iterations','8','--contact-offset','.0002','--no-simplify-arm-collisions',
    '--clearance',str(a.clearance),'--max-wrist-descent',str(a.clearance+.003),
    '--closure-bias',str(a.closure_bias),'--closure-scale',str(a.closure_scale),
    '--closure-min','-.1','--closure-max',str(a.closure_max),
    '--mode','lift','--episode-seconds','5','--require-web-up','--max-palm-up','.25',
    '--require-face-pinch','--tool-angle-range','20',str(a.tool_angle_max),
    '--independent-support-actions','--command-space-arm','--index-action-scale','.008',
    '--middle-action-scale','.025','--support-reward',str(a.support_reward)]
if a.mode=='train':
    cmd+=['--iterations',str(a.iterations),'--initial-std',str(a.initial_std),
          '--learning-rate',str(a.learning_rate),'--target-kl','.015']
    if a.initializer:cmd+=['--initialize-policy',a.initializer,'--freeze-observation-normalizer']
    if a.initializer_std is not None:cmd+=['--initialize-std',str(a.initializer_std)]
else:
    cmd+=['--iterations','0','--evaluate',a.checkpoint]
    if a.mode=='render':
        cmd+=['--render-video']
        if a.camera_config:cmd+=['--camera-config',a.camera_config]
raise SystemExit(subprocess.run(cmd,cwd=ROOT).returncode)
