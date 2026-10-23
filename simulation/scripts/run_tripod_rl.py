"""Reproduce the current thumb/index/middle/web-supported tweezer task."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('mode', choices=['train', 'evaluate', 'render'])
p.add_argument('--out', required=True)
p.add_argument('--device', default='cuda:0')
p.add_argument('--seed', type=int)
p.add_argument('--num-envs', type=int)
p.add_argument('--iterations', type=int, default=25)
p.add_argument('--checkpoint', default='runs/rl_tripod_stow_bc_s47/iteration_00025.pt')
p.add_argument('--initializer', default='runs/tripod_bc_demo05_v1/initializer.pt')
a = p.parse_args()
out = ROOT / a.out
if out.exists():
    p.error('Use a new output directory; existing evidence is never overwritten.')
if a.num_envs is not None and a.num_envs < 1:
    p.error('--num-envs must be positive')
common = [
    '/cpfs/user/lvzhaoyang/apps/bin/isaac-python', 'scripts/train_rl.py',
    '--out', a.out, '--device', a.device, '--headless',
    '--pose', 'configs/human_grasp_tripod_stow_pick_v1.json',
    '--reset-snapshot', 'runs/tripod_stow_open16/closure0_copy0_snapshot.json',
    '--dt', str(1/480), '--decimation', '8',
    '--spring', '5', '--spring-damping', '.02',
    '--solver-iterations', '64', '--solver-velocity-iterations', '8',
    '--contact-offset', '.0002', '--no-simplify-arm-collisions',
    '--clearance', '.018', '--max-wrist-descent', '.019',
    '--closure-bias', '0', '--closure-scale', '1.2',
    '--closure-min', '-.1', '--closure-max', '1.2',
    '--mode', 'lift', '--episode-seconds', '5',
    '--require-web-up', '--max-palm-up', '.25', '--require-tripod-support',
    '--tool-angle-range', '20', '60', '--independent-support-actions',
    '--command-space-arm', '--index-action-scale', '.008',
    '--middle-action-scale', '.025', '--support-reward', '1.5',
]
if a.mode == 'train':
    common += ['--num-envs', str(a.num_envs or 128), '--seed', str(47 if a.seed is None else a.seed),
               '--iterations', str(a.iterations), '--initialize-policy', a.initializer,
               '--freeze-observation-normalizer', '--initial-std', '.15',
               '--learning-rate', '.0001', '--target-kl', '.015']
else:
    common += ['--num-envs', str(a.num_envs or 64), '--evaluate', a.checkpoint,
               '--iterations', '0', '--seed', str((20261016 if a.mode == 'render' else 20261017) if a.seed is None else a.seed)]
    if a.mode == 'render':
        common += ['--render-video', '--camera-config', 'analysis/tripod_stow_cameras_wide.json']
raise SystemExit(subprocess.run(common, cwd=ROOT).returncode)
