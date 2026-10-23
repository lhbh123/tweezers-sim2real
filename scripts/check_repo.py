#!/usr/bin/env python3
"""Check relocatable runtime inputs and frozen evidence without simulator imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check():
    required = [
        'assets/ur5e_xhand_right/robot.urdf', 'assets/ur5e_xhand_right/robot.usd',
        'models/policies/xhand_pick_place_ppo.pt', 'models/snapshots/xhand_stable_grasp.json',
        'calibration/results/camera_rig_isaac.json', 'calibration/results/world_base_isaac.json',
        'calibration/configs/link_to_tcp.confirmed.json',
        'calibration/examples/isaac_camera_bootstrap.py', 'calibration/examples/isaac_camera_rig.py',
        'calibration/results/wrist_intrinsics_v3.json', 'calibration/results/global_intrinsics_v3.json',
        'calibration/results/handeye.json', 'calibration/results/global_extrinsics.json',
        'simulation/configs/xhand_local.json', 'simulation/configs/calibrated_demo_y_table.json',
        'simulation/configs/calibrated_demo_y_case.json', 'scripts/run_demo.sh', 'env/setup_sim.sh',
    ]
    for name in required:
        if not (ROOT / name).is_file():
            raise ValueError('Missing runtime input: ' + name)
    model_hash = sha256(ROOT / 'models/policies/xhand_pick_place_ppo.pt')
    if model_hash != '91523ab6e677e2d8ce85e655f0f29c6aad88d288ba2a710c703f749c48431fc6':
        raise ValueError('Frozen PPO checkpoint hash differs')
    pose = json.loads((ROOT / 'simulation/configs/xhand_local.json').read_text())
    asset = (ROOT / 'simulation' / pose['robot_asset_dir']).resolve()
    if asset != ROOT / 'assets/ur5e_xhand_right':
        raise ValueError('Active pose must load the merged repository asset')
    rig = json.loads((ROOT / 'calibration/results/camera_rig_isaac.json').read_text())
    link = json.loads((ROOT / 'calibration/configs/link_to_tcp.confirmed.json').read_text())
    if not rig['ready_for_isaac_wrist'] or not link['verified']:
        raise ValueError('Verified link/TCP configuration required')
    if rig['cameras']['wrist']['reference_link'] != 'wrist_3_link':
        raise ValueError('Unexpected wrist link')
    for name in ['wrist', 'global']:
        if rig['cameras'][name]['resolution'] != [640, 480]:
            raise ValueError('Unexpected calibrated image resolution')
    layout = json.loads((ROOT / 'simulation/configs/calibrated_demo_y_table.json').read_text())
    cases = json.loads((ROOT / 'simulation/configs/calibrated_demo_y_case.json').read_text())
    if layout['table_size_m'][1] <= layout['table_size_m'][0]:
        raise ValueError('Expected Y-axis long table')
    for xy in cases['start'] + cases['goal'] + [layout['home_xy_m']]:
        if not all(abs(x - c) < size / 2 - .02 for x, c, size in
                   zip(xy, layout['table_center_xy_m'], layout['table_size_m'])):
            raise ValueError('Task point outside table margin')
    for folder in ['calibration/src', 'calibration/examples', 'simulation/src']:
        for path in (ROOT / folder).glob('**/*.py'):
            ast.parse(path.read_text(), filename=str(path))
    ast.parse((ROOT / 'simulation/scripts/run_pick_place.py').read_text())
    archive = ROOT / 'archives/calibration_confirmed_20261023'
    archive_count = 0
    if (archive / 'MANIFEST.sha256.json').exists():
        manifest = json.loads((archive / 'MANIFEST.sha256.json').read_text())
        for name, expected in manifest['files'].items():
            if sha256(archive / name) != expected:
                raise ValueError('Frozen calibration archive changed: ' + name)
            archive_count += 1
    return dict(status='passed', root=str(ROOT), runtime_inputs=len(required),
                policy_sha256=model_hash, asset_dir=str(asset),
                calibration_archive_files_checked=archive_count,
                active_rig='calibration/results/camera_rig_isaac.json',
                scope='Static runtime paths, frozen checkpoint, verified rig, layout bounds, source syntax and optional frozen archive hashes')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    try:
        result = check()
        payload = json.dumps(result, indent=2) + '\n'
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('x') as out:
                out.write(payload)
        print(payload, end='')
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
