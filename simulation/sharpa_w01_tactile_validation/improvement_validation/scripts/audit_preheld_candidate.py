"""CPU-only independent limit and snapshot audit for preheld probe inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_limits(pose, names):
    urdf = Path(pose['robot_asset_dir']) / 'robot.urdf'
    nodes = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
    limits = []
    for name in names:
        node = nodes[name]
        lim = node.find('limit')
        limits.append([float(lim.get('lower', -np.pi)), float(lim.get('upper', np.pi))])
    return urdf, np.asarray(limits)


def limit_audit(q, bounds, names):
    q = np.asarray(q)
    violation = np.maximum(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]), 0.)
    return {'max_violation_rad': float(violation.max()),
            'violating_joints_gt_1e_6': [names[i] for i in np.where(violation.max(axis=0) > 1e-6)[0]],
            'minimum_limit_margin_rad': float(np.minimum(q - bounds[:, 0], bounds[:, 1] - q).min())}


def audit(pose_path, snapshot_path, variants_path):
    pose = json.loads(Path(pose_path).read_text())
    snap = json.loads(Path(snapshot_path).read_text())
    source = snap.get('zero_velocity_variant', snap)
    variants = json.loads(Path(variants_path).read_text())
    names = snap['joint_names']
    urdf, bounds = load_limits(pose, names)
    target = np.asarray(source['q_target'], float)
    actual = np.asarray(source['q_actual'], float)
    delta = np.asarray([pose['closure_delta'].get(n, 0.) for n in names])
    interval = [-float('inf'), float('inf')]
    limiting = [None, None]
    for i, d in enumerate(delta):
        if abs(d) < 1e-15:
            continue
        lo, hi = sorted((bounds[i] - target[i]) / d)
        if lo > interval[0]: interval[0], limiting[0] = float(lo), names[i]
        if hi < interval[1]: interval[1], limiting[1] = float(hi), names[i]
    inferred_closures = {name: float((target[i] - pose['q_reference'][name]) / delta[i])
                         for i, name in enumerate(names) if abs(delta[i]) > 1e-15}
    candidates = []
    for variant in variants:
        base = target.copy()
        q0 = actual.copy()
        for name, value in variant.get('target_delta', {}).items(): base[names.index(name)] += value
        for name, value in variant.get('initial_delta', {}).items(): q0[names.index(name)] += value
        knots = [variant.get('closure_start', 0.), variant.get('closure_end', 0.)]
        knots += [float(k[1]) for k in variant.get('closure_schedule', [])]
        targets = base[None] + np.asarray(knots)[:, None] * delta
        candidates.append({'name': variant['name'], 'all_closure_knots_checked': knots,
                           'target_limits': limit_audit(targets, bounds, names),
                           'initial_actual_limits': limit_audit(q0[None], bounds, names)})
    bodies = source['rigid_body_states_13']
    result = {
        'files_sha256': {str(Path(p).resolve()): digest(p) for p in (pose_path, snapshot_path, variants_path, urdf)},
        'closure_semantics': 'With --snapshot, closure is added to snapshot q_target, not pose q_reference.',
        'joint_names': names, 'q_target_limits': limit_audit(target[None], bounds, names),
        'q_actual_limits': limit_audit(actual[None], bounds, names),
        'snapshot_target_minus_actual_rad': dict(zip(names, (target - actual).tolist())),
        'inferred_absolute_closure_by_joint': inferred_closures,
        'feasible_extra_closure_without_target_delta': interval,
        'limiting_joints_at_interval_ends': limiting,
        'q_velocity_count': len(source.get('q_velocity', [])),
        'q_velocity_matches_joint_count': len(source.get('q_velocity', [])) == len(names),
        'zero_velocity_variant_used': 'zero_velocity_variant' in snap,
        'zero_velocity_variant_body_pose_matches_top_level': all(
            np.array_equal(np.asarray(value[:7]), np.asarray(snap['rigid_body_states_13'][name][:7]))
            for name, value in bodies.items()),
        'body_quaternion_norms': {name: float(np.linalg.norm(value[3:7])) for name, value in bodies.items()},
        'body_velocities_max_abs': {name: float(np.abs(value[7:13]).max()) for name, value in bodies.items()},
        'candidates': candidates,
        'notes': [
            'Actual initial pose and PD target are different; preserve the target bias when restoring.',
            'Do not silently clamp joint targets. A schedule can exceed limits between valid start/end values.',
            'New compliant snapshots require independent cold reset, not just a stable end frame.',
            'Snapshot output existence is insufficient: the old probe exports every stable_hold, including failures of broad_face_hold.',
            'Do not rotate to preferred_tool_world_R without rerunning gravity stability validation.'
        ]}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--variants', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = audit(args.pose, args.snapshot, args.variants)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({'output': str(out), 'extra_closure_range': result['feasible_extra_closure_without_target_delta'],
                      'candidates': result['candidates']}, ensure_ascii=False))


if __name__ == '__main__': main()
