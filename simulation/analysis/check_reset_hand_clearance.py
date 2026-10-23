#!/usr/bin/env python3
"""Independent CPU regression for source-mesh table clearance at stable resets.

This reads every URDF collision, applies mesh scale before collision origin,
and compares full base-root FK with the implementation's wrist-root test.
It does not modify the shared implementation, snapshots, or pose configs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import ASSET, Kinematics
from arm_clearance import ArmTableClearance
from reset_reference import StableResetReference

ARM = {'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link',
       'wrist_2_link', 'wrist_3_link'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collision_vertices(root):
    """Own binary-STL parser and complete URDF collision transformation."""
    result, inventory = {}, []
    for link in root.findall('link'):
        name = link.get('name')
        if name not in ARM and not name.startswith('right_hand_'):
            continue
        components = []
        for index, collision in enumerate(link.findall('collision')):
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                raise ValueError(f'Unsupported nonmesh collision {name}/{index}')
            path = ASSET / mesh.get('filename')
            data = path.read_bytes()
            count = int.from_bytes(data[80:84], 'little')
            if len(data) != 84 + 50 * count:
                raise ValueError(f'Expected complete binary STL: {path}')
            packed = np.ndarray((count, 12), dtype='<f4', buffer=data,
                                offset=84, strides=(50, 4))
            vertices = packed[:, 3:12].reshape(-1, 3).astype(float)
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            assert scale.shape == (3,) and np.isfinite(scale).all()
            origin = collision.find('origin')
            attrs = {} if origin is None else origin.attrib
            xyz = np.fromstring(attrs.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(attrs.get('rpy', '0 0 0'), sep=' ')
            vertices = (vertices * scale) @ Rotation.from_euler('xyz', rpy).as_matrix().T + xyz
            components.append(vertices)
            inventory.append(dict(link=name, collision_index=index,
                                  mesh=str(path), mesh_sha256=sha(path),
                                  triangle_count=count, scale=scale.tolist(),
                                  origin_xyz=xyz.tolist(), origin_rpy=rpy.tolist(),
                                  origin_explicit=origin is not None))
        if components:
            result[name] = np.concatenate(components)
    return result, inventory


def minima(vertices, transforms, shift):
    return {name: float((v @ transforms[name][:3, :3].T +
                         transforms[name][:3, 3] + shift)[:, 2].min())
            for name, v in vertices.items()}


def make_scene(config, kin):
    geo = config['tool_geometry']
    length = float(geo['length_m'])
    tip_length = float(geo.get('tip_length_m', .015))
    # Exactly the geometry expressions in RLScene; never the legacy .1325 fallback.
    tip_z = length - tip_length / 2
    assert abs(tip_z - .1525) < 1e-12
    return SimpleNamespace(kin=kin, joint_names=kin.active, device='cpu',
        env_origins=np.zeros((1, 3)), table_height=.72, pose_config=config,
        tip_local_z=tip_z,
        tip_half_size=(float(geo.get('strip_thickness_m', .001))/2,
                       min(.005, float(geo['strip_width_m']))/2, tip_length/2))


def check_case(name, snapshot_path, config_path, tip_z, should_reject, vertices):
    kin = Kinematics()
    config = json.loads(config_path.read_text())
    scene = make_scene(config, kin)
    ref = StableResetReference(snapshot_path, scene)
    q = dict(zip(ref.joint_names, ref.q_actual))
    hand_q = {n: q[n] for n in kin.hand}
    tip = np.array([.48, 0., tip_z])
    translation = tip - ref.tip_position
    # Full base-root FK, original actual q and full snapshot base translation.
    oracle = minima({n: v for n, v in vertices.items() if n.startswith('right_hand_')},
                    kin.fk(q), ref.snapshot_base + translation - [0., 0., .72])
    wrist = ref.wrist_transform.copy()
    wrist[:3, 3] += translation - [0., 0., .72]
    implementation = ref.arm_clearance.hand_clearance(wrist, hand_q)
    error = max(abs(oracle[n] - implementation[n]) for n in oracle)
    assert error < 1e-10, (name, error)
    build_error, built = None, None
    origins = np.array([[0., 0., 0.], [1.8, -.9, .04], [-.9, 1.8, -.03]])
    try:
        built = ref.build(tip, env_origins=origins, position_frame='local')
    except ValueError as exc:
        build_error = str(exc)
    rejected = build_error is not None
    assert rejected == should_reject, (name, build_error, should_reject)
    result = dict(name=name, snapshot=str(snapshot_path), snapshot_sha256=sha(snapshot_path),
                  config=str(config_path), config_sha256=sha(config_path),
                  scene_tip_local_z_m=scene.tip_local_z,
                  desired_tip_world_m=tip.tolist(), source_tip_world_m=ref.tip_position.tolist(),
                  source_base_world_m=ref.snapshot_base.tolist(),
                  independent_hand_clearance_by_link_m=oracle,
                  implementation_oracle_max_abs_error_m=error,
                  lowest_link=min(oracle, key=oracle.get), minimum_m=min(oracle.values()),
                  expected_rejected=should_reject, rejected=rejected, exception=build_error,
                  min_hand_clearance_setting_m=config.get('hand_reset_table_clearance_m', .001),
                  passed=True)
    if rejected:
        assert 'Hand source meshes lack table clearance' in build_error
        if name == 'old_horizontal_low_approach':
            assert abs(oracle['right_hand_pinky_link2'] - (-.007676728280321236)) < 1e-9
    else:
        hand_after, arm_after, tip_errors = [], [], []
        for row, origin in enumerate(origins):
            q_after = dict(zip(ref.joint_names, built['q_actual'][row].numpy().astype(float)))
            hand_after.append(minima({n: v for n, v in vertices.items() if n.startswith('right_hand_')},
                                     kin.fk(q_after), np.zeros(3)))
            arm_after.append(minima({n: v for n, v in vertices.items() if n in ARM},
                                    kin.fk(q_after), np.zeros(3)))
            poses = built['object_states']
            actual_tips = []
            for body in ('left', 'right'):
                pose = poses[body][row].numpy().astype(float)
                R = Rotation.from_quat(np.r_[pose[4:7], pose[3]]).as_matrix()
                actual_tips.append(pose[:3] + R @ np.array([0., 0., .1525]))
            tip_errors.append(float(np.linalg.norm(np.mean(actual_tips, axis=0) - tip - origin)))
        assert max(tip_errors) < 3e-7
        post_error = max(abs(row[n] - oracle[n]) for row in hand_after for n in oracle)
        assert post_error < 2e-7, (name, post_error)
        assert min(min(r.values()) for r in arm_after) >= config['arm_reset_table_clearance_m'] - 2e-7
        result.update(returned_float32_hand_clearance_by_env_m=hand_after,
                      returned_arm_clearance_by_env_m=arm_after,
                      actual_q_vs_snapshot_translation_error_m=post_error,
                      returned_actual_tip_position_errors_m=tip_errors,
                      tested_environment_origins_m=origins.tolist(),
                      ik_cache_entries=len(ref._ik_cache),
                      cached_three_environment_local_pose_equivalence=len(ref._ik_cache) == 1)
        assert len(ref._ik_cache) == 1
    return result


def synthetic_loader_audit():
    """Expose generic input gaps without mutating the original URDF or source."""
    result = []
    for name in ['multiple_collision', 'nonunit_scale', 'omitted_collision_origin', 'nonzero_collision_origin']:
        kin = Kinematics()
        link = next(e for e in kin.root.findall('link') if e.get('name') == 'right_hand_pinky_link2')
        collision = link.find('collision')
        if name == 'multiple_collision':
            second = copy.deepcopy(collision)
            R = kin.fk({}, root='wrist_3_link')['right_hand_pinky_link2'][:3, :3]
            local_shift = R.T @ np.array([0., 0., -.25])
            second.find('origin').set('xyz', ' '.join(str(x) for x in local_shift))
            link.append(second)
        elif name == 'nonunit_scale':
            collision.find('geometry/mesh').set('scale', '1.3 0.8 1.7')
        elif name == 'omitted_collision_origin':
            collision.remove(collision.find('origin'))
        else:
            collision.find('origin').set('xyz', '0.007 -0.009 -0.02')
            collision.find('origin').set('rpy', '0.3 -0.2 0.1')
        meshes, _ = collision_vertices(kin.root)
        relative = kin.fk({}, root='wrist_3_link')
        expected = minima({n: v for n, v in meshes.items() if n.startswith('right_hand_')}, relative, np.zeros(3))
        error, exception = None, None
        try:
            loader = ArmTableClearance(kin)
            actual = loader.hand_clearance(np.eye(4), {})
            error = max(abs(expected[n] - actual[n]) for n in expected)
        except Exception as exc:
            exception = f'{type(exc).__name__}: {exc}'
        result.append(dict(case=name, supported=exception is None and error < 1e-10,
                           max_clearance_error_m=error, exception=exception,
                           affects_current_asset=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'reports/reset_hand_clearance_regression.json')
    args = parser.parse_args()
    source_hashes = {str(ROOT/p): sha(ROOT/p) for p in ['src/arm_clearance.py', 'src/reset_reference.py', 'src/rl_scene.py', 'src/kinematics.py']}
    vertices, inventory = collision_vertices(Kinematics().root)
    impl = ArmTableClearance(Kinematics())
    hand_names = {n for n in vertices if n.startswith('right_hand_')}
    assert hand_names == set(impl.hand_vertices)
    assert ARM == set(impl.vertices)
    cases = [
        ('old_horizontal_low_approach', 'runs/tripod_horizontal_hold12/closure0_copy0_snapshot.json', 'configs/human_grasp_tripod_horizontal_settled_v1.json', .7458, True),
        ('old_render07_hold', 'runs/tripod_task_wrist_settle02/constant_saved_joint_target_copy0_snapshot.json', 'configs/human_grasp_tripod_webtail_neartable_settled_v1.json', .7458, False),
    ]
    for variant in ['open', 'closed035', 'closed045']:
        for stage, z in [('approach', .7458), ('pinch_geometry_only', .7276)]:
            cases.append((f'stow_{variant}_{stage}', f'analysis/tripod_stow_v1/{variant}_geometry_snapshot.json',
                          f'configs/human_grasp_tripod_horizontal_stow_{variant}_v1.json', z, False))
    results = []
    for name, snapshot, config, z, reject in cases:
        if not (ROOT/snapshot).exists():
            continue
        result = check_case(name, ROOT/snapshot, ROOT/config, z, reject, vertices)
        results.append(result)
        print(json.dumps({k: result[k] for k in ['name', 'minimum_m', 'lowest_link', 'rejected', 'passed']}), flush=True)
    synthetic = synthetic_loader_audit()
    # Quantify the legacy standalone fallback as a counterexample, not as a valid test.
    legacy = StableResetReference(ROOT/'runs/tripod_horizontal_hold12/closure0_copy0_snapshot.json')
    legacy_wrist = legacy.wrist_transform.copy()
    legacy_wrist[:3, 3] += np.array([.48, 0., .7458]) - legacy.tip_position - [0., 0., .72]
    legacy_values = impl.hand_clearance(legacy_wrist, dict(zip(legacy.joint_names, legacy.q_actual)))
    legacy_hazard = dict(incorrect_standalone_tip_local_m=legacy.tip_local.tolist(),
                         incorrectly_computed_min_hand_clearance_m=min(legacy_values.values()),
                         correct_scene_tip_local_z_m=.1525,
                         correct_min_hand_clearance_m=results[0]['minimum_m'],
                         warning='The fallback describes a shorter tool and is invalid for this 160mm scene.')
    hand_only_config = json.loads((ROOT/'configs/human_grasp_tripod_horizontal_settled_v1.json').read_text())
    hand_only_config.pop('arm_reset_table_clearance_m')
    hand_only_config['hand_reset_table_clearance_m'] = .001
    hand_only = StableResetReference(ROOT/'runs/tripod_horizontal_hold12/closure0_copy0_snapshot.json',
                                     make_scene(hand_only_config, Kinematics()))
    activation = dict(hand_only_setting_enables_guard=hand_only.arm_clearance is not None,
                      current_final_configs_include_arm_setting=True)
    unchanged = all(sha(path) == value for path, value in source_hashes.items())
    report = dict(scope='Independent CPU source-URDF/STL reset-time table clearance; no physical stability or motion guarantee.',
                  source_hashes=source_hashes, source_unchanged_during_test=unchanged,
                  urdf=str(ASSET/'robot.urdf'), urdf_sha256=sha(ASSET/'robot.urdf'),
                  hand_collision_link_count=len(hand_names), arm_collision_link_count=len(ARM),
                  source_collision_inventory=inventory, cases=results,
                  all_current_asset_regressions_pass=all(r['passed'] for r in results),
                  standalone_legacy_tip_fallback_counterexample=legacy_hazard,
                  hand_guard_activation=activation,
                  synthetic_generic_urdf_loader_cases=synthetic,
                  findings=[
                      'Current supplied asset has one mesh collision per physical link, explicit origin, and no nonunit scales; all 13 hand meshes and 6 arm meshes are included.',
                      'Source-mesh minimum world z is exact at triangle vertices; oracle uses complete collision geometry and independent full base-root FK.',
                      'Current loader only reads the first collision and ignores mesh scale, and omitted collision origin raises AttributeError. These generic limitations do not change the current source-asset results.',
                      'StableResetReference activates the hand test only when scene.pose_config.arm_reset_table_clearance_m exists; a hand-only threshold does not enable it.',
                      'Standalone StableResetReference without scene or explicit tip_local retains legacy .1325 m fallback; all tests use scene-derived .1525 m.',
                      'The guard checks actual snapshot hand q at reset, not commanded q_target settling, every future action, self collisions, contact offsets, or PhysX convex cooking.',
                      'Stow pinch z .7276 is geometric path-end inspection only. Current TaskConfig stable-snapshot reset requires >=15mm initial_tip_clearance, so it is not a claim this low point can be requested as a task reset.',
                  ],
                  suggested_changes=['Iterate all collision elements; apply mesh scale; use identity for omitted origin; explicitly reject unsupported primitive geometry.',
                                     'Keep the guard enabled for final scene configs; consider independent hand-clearance activation and require explicit tool geometry for standalone resets.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'report': str(args.output), 'cases': len(results), 'all_current_asset_pass': report['all_current_asset_regressions_pass'], 'source_unchanged': unchanged}), flush=True)


if __name__ == '__main__':
    main()
