#!/usr/bin/env python3
"""Independently validate contact-supported lift evidence from a recorded run.

Exit 0: all checks pass. Exit 1: evidence fails a requirement. Exit 2: unreadable
or incomplete evidence. Reported PhysX velocities and position differences are
reported separately; neither is silently substituted for the other.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


USD_AUDIT = r'''
import json, sys
sys.path.insert(0, sys.argv[2])
from pxr import Usd, UsdPhysics, UsdGeom
stage = Usd.Stage.Open(sys.argv[1])
block = stage.GetPrimAtPath('/World/Block')
if not block: raise RuntimeError('No /World/Block in scene')
rigid = UsdPhysics.RigidBodyAPI(block)
constraints = []
fixtures = []
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        joint = UsdPhysics.Joint(prim)
        targets = [str(p) for p in joint.GetBody0Rel().GetTargets()] + [str(p) for p in joint.GetBody1Rel().GetTargets()]
        if any(p == '/World/Block' or p.startswith('/World/Block/') for p in targets):
            constraints.append(str(prim.GetPath()))
        if 'PreheldHandle' in str(prim.GetPath()):
            fixtures.append({'path': str(prim.GetPath()), 'enabled': bool(joint.GetJointEnabledAttr().Get())})
shape = UsdGeom.Cube(stage.GetPrimAtPath('/World/Block/Shape'))
scale = stage.GetPrimAtPath('/World/Block/Shape').GetAttribute('xformOp:scale').Get()
height = float(shape.GetSizeAttr().Get()) * abs(float(scale[2]))
print('USD_AUDIT_JSON=' + json.dumps({
    'rigid_body': bool(rigid), 'rigid_body_enabled': bool(rigid.GetRigidBodyEnabledAttr().Get()),
    'kinematic': bool(rigid.GetKinematicEnabledAttr().Get()),
    'mass_kg': UsdPhysics.MassAPI(block).GetMassAttr().Get(),
    'block_constraints': constraints, 'block_height_m': height,
    'fixture_joints': fixtures,
    'gravity_disabled': bool(block.GetAttribute('physxRigidBody:disableGravity').Get()),
}))
'''


def audit_scene(scene):
    sim = Path(os.environ.get('ISAAC_SIM_PATH', '/cpfs/user/lvzhaoyang/apps/isaac-sim-5.0.0'))
    usd = next(iter(sorted((sim / 'extscache').glob('omni.usd.libs-*'))), None)
    if usd is None:
        raise RuntimeError('Isaac USD Python libraries unavailable; scene cannot be independently audited')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(usd / 'bin') + ':' + env.get('LD_LIBRARY_PATH', '')
    command = [str(sim / 'python.sh'), '-c', USD_AUDIT, str(scene), str(usd)]
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
    for line in run.stdout.splitlines():
        if line.startswith('USD_AUDIT_JSON='):
            return json.loads(line.split('=', 1)[1])
    raise RuntimeError(f'USD scene audit failed: {run.stderr[-1000:]}')


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'rms': float(np.sqrt(np.mean(values ** 2))),
            'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(values.max())}


def validate(path, options):
    if options.min_height <= 0 or options.min_duration <= 0 or options.max_range <= 0:
        raise ValueError('Height, duration and position-range thresholds must be positive')
    if not 0 <= options.min_contact_fraction <= 1 or options.contact_threshold < 0:
        raise ValueError('Contact fraction must be in [0, 1], and force threshold nonnegative')
    rows = json.loads((path / 'telemetry.json').read_text())
    metadata = json.loads((path / 'metadata.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    scene_path = path / 'scene.usda'
    scene = audit_scene(scene_path)
    final_path = path / 'final_scene.usda'
    final_scene = audit_scene(final_path) if final_path.exists() else None
    t = np.array([r['t'] for r in rows], dtype=float)
    positions = np.array([r['block'] for r in rows], dtype=float)
    velocity = np.array([r['block_velocity'] for r in rows], dtype=float)
    forces = np.array([r['force'] for r in rows], dtype=float).reshape(len(rows), -1, 3)
    fixture = np.array([r['fixture_active'] for r in rows], dtype=bool)
    if len(t) < 2 or forces.shape[1] != 2:
        raise ValueError('Need at least two samples and exactly two jaw contact filters')
    if not all(np.isfinite(x).all() for x in [t, positions, velocity, forces]):
        raise ValueError('Non-finite telemetry values')
    dt = float(metadata['args']['dt'])
    if dt <= 0 or not np.allclose(np.diff(t), dt, atol=max(1e-9, dt * 1e-4)):
        raise ValueError('Telemetry timestamps do not form the claimed uniform physics timestep')
    count = max(2, int(math.ceil(options.min_duration / dt - 1e-9)))
    tail = slice(max(0, len(rows) - count), len(rows))
    duration = min(count, len(rows)) * dt
    base = float(metadata['table_height']) + scene['block_height_m'] / 2
    lift = positions[:, 2] - base
    contact = np.linalg.norm(forces, axis=-1)
    bilateral = (contact > options.contact_threshold).all(axis=-1)
    spans = np.ptp(positions[tail], axis=0)
    diagonal_range = float(np.linalg.norm(spans))
    reported_speed = np.linalg.norm(velocity[tail], axis=-1)
    fd_velocity = np.diff(positions[tail], axis=0) / np.diff(t[tail])[:, None]
    finite_speed = np.linalg.norm(fd_velocity, axis=-1)
    tail_positions = positions[tail]
    lag = min(len(tail_positions) - 1, max(1, round(.1 / dt)))
    smooth_speed = np.linalg.norm((tail_positions[lag:] - tail_positions[:-lag]) / (lag * dt), axis=-1)
    preload = float(metadata['args'].get('preload', 0))
    post_reset = t >= preload - dt * 1e-6
    log_path = path.parent / (path.name + '.log')
    release_event = 'RESET_FIXTURE_RELEASED' in log_path.read_text(errors='replace') if log_path.exists() else False
    release_samples = np.flatnonzero(~fixture)
    all_dynamic_scenes = [scene] + ([final_scene] if final_scene else [])
    checks = {
        'enough_continuous_samples': duration >= options.min_duration - 1e-9,
        'continuous_final_lift_above_threshold': bool(np.min(lift[tail]) > options.min_height),
        'bilateral_jaw_contact_fraction': bool(bilateral[tail].mean() >= options.min_contact_fraction),
        'final_block_position_range': diagonal_range < options.max_range,
        'no_fixture_during_final_window': bool(not fixture[tail].any()),
        'no_fixture_after_reset_phase': bool(not fixture[post_reset].any()),
        'not_declared_fixed_handle': not bool(metadata.get('handle_fixed_to_hand', True) or result.get('handle_fixed', True)),
        'fixture_release_event_recorded': preload == 0 or release_event,
        'block_is_dynamic_gravity_body': all(s['rigid_body'] and s['rigid_body_enabled'] and not s['kinematic'] and not s['gravity_disabled'] for s in all_dynamic_scenes),
        'block_has_no_joint_constraints': all(not s['block_constraints'] for s in all_dynamic_scenes) and metadata.get('block_constraints') == [],
        'final_usd_fixture_disabled_if_available': final_scene is None or all(not j['enabled'] for j in final_scene['fixture_joints']),
        'physics_clock_matches_telemetry': abs(float(result['world_time']) - len(rows) * dt) <= 2.1 * dt,
    }
    if any('actual_fixture_joint_enabled' in r for r in rows):
        enabled = np.array([r['actual_fixture_joint_enabled'] for r in rows], dtype=bool)
        checks['usd_attribute_fixture_disabled_after_reset'] = bool(not enabled[post_reset].any())
    longest = current = 0
    for ok in (lift > options.min_height) & ~fixture:
        current = current + 1 if ok else 0
        longest = max(longest, current)
    artifacts = [path / x for x in ['telemetry.json', 'trajectory.json', 'metadata.json', 'result.json', 'scene.usda']]
    if final_scene: artifacts.append(final_path)
    report = {
        'success': all(checks.values()), 'checks': checks, 'run': str(path.resolve()),
        'criteria': dict(vars(options)), 'window_start_s': float(t[tail][0]),
        'window_duration_s': duration, 'min_lift_m': float(lift[tail].min()),
        'longest_continuous_lift_without_fixture_s': longest * dt,
        'bilateral_contact_fraction': float(bilateral[tail].mean()),
        'block_xyz_range_m': spans.tolist(), 'block_bounding_box_diagonal_m': diagonal_range,
        'per_jaw_normal_force_n': [stats(contact[tail, i]) for i in range(2)],
        'physics_reported_speed_m_s': stats(reported_speed),
        'position_finite_difference_speed_m_s': stats(finite_speed),
        'position_100ms_difference_speed_m_s': stats(smooth_speed),
        'first_fixture_inactive_s': float(t[release_samples[0]]) if len(release_samples) else None,
        'startup_scene_audit': scene, 'final_scene_audit': final_scene,
        'evidence_limits': ([] if final_scene else ['Initial scene snapshot precedes fixture release; runtime release is supported by telemetry and release log, not a saved final USD state.']) +
                           ['Contact/pose evidence validates this simulated run, not hardware realism or arbitrary source-code behavior.'],
        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
    }
    report['criteria'].pop('run', None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--min-height', type=float, default=.075, help='Required lift above initial block center, metres')
    parser.add_argument('--min-duration', type=float, default=2.)
    parser.add_argument('--min-contact-fraction', type=float, default=.95)
    parser.add_argument('--contact-threshold', type=float, default=.001, help='Minimum per-jaw normal force, N')
    parser.add_argument('--max-range', type=float, default=.003, help='Maximum final position bounding-box diagonal, metres')
    options = parser.parse_args()
    path = Path(options.run)
    try:
        report = validate(path, options)
    except Exception as error:
        print(json.dumps({'success': False, 'evidence_error': str(error)}, indent=2))
        return 2
    (path / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
