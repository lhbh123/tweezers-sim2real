"""Offline source-vertex/table and joint-limit audit of all saved probe states.

This reads probe data only. It does not modify a scene, controller or success gate.
Self/tool collision and measured support are separate audits.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import Kinematics
from arm_clearance import ArmTableClearance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    run = args.run
    report = json.loads((run / 'report.json').read_text())
    pose_path = run / 'source' / Path(report['arguments']['pose']).name
    pose = json.loads(pose_path.read_text())
    kin = Kinematics(Path(pose['robot_asset_dir']) / 'robot.urdf',
                     palm_link=pose['palm_link'], hand_root_link=pose['hand_root_link'])
    clear = ArmTableClearance(kin)
    meshes = {**clear.vertices, **clear.hand_vertices}
    names = report['joint_names']
    lower = np.array([kin.by_name[n]['lower'] for n in names])
    upper = np.array([kin.by_name[n]['upper'] for n in names])
    with np.load(run / 'timeseries.npz', allow_pickle=False) as data:
        q, target, bodies, times = (data[k] for k in ['q', 'q_target', 'bodies', 'time'])
        finite = all(np.isfinite(data[k]).all() for k in data.files)
    origins = np.asarray(report['env_origins'])
    rows = []
    geometry = pose['tool_geometry']
    for env in range(q.shape[1]):
        row = {'environment': env, 'candidate': report['candidates'][env]['name'],
               'samples': len(times), 'source_mesh_table_below_tolerance_samples': 0,
               'min_robot_table_clearance_by_link_m': {name: float('inf') for name in meshes}}
        for label, values in [('actual', q[:, env]), ('target', target[:, env])]:
            margins = np.minimum(values - lower, upper - values)
            row[label + '_joint_limit_violation_samples'] = int((margins < -1e-4).sum())
            row[label + '_minimum_limit_margin_rad'] = float(margins.min())
            ix = np.unravel_index(margins.argmin(), margins.shape)
            row[label + '_nearest_limit'] = {'time_s': float(times[ix[0]]), 'joint': names[ix[1]],
                                            'value_rad': float(values[ix]),
                                            'limits_rad': [float(lower[ix[1]]), float(upper[ix[1]])]}
        for step, values in enumerate(q[:, env]):
            fk = kin.fk(dict(zip(names, values)))
            for name, vertices in meshes.items():
                transform = fk[name]
                # Both UR5e base and tabletop are 0.72 m above the environment origin.
                z = float((vertices @ transform[2, :3] + transform[2, 3]).min())
                row['min_robot_table_clearance_by_link_m'][name] = min(
                    row['min_robot_table_clearance_by_link_m'][name], z)
                row['source_mesh_table_below_tolerance_samples'] += int(z < -1e-4)
        row['min_robot_table_clearance_m'] = min(row['min_robot_table_clearance_by_link_m'].values())
        row['tool_table_below_tolerance_samples'] = 0
        row['min_tool_table_clearance_by_body_m'] = {}
        for body, name in enumerate(['handle', 'left', 'right']):
            state = bodies[:, env, body]
            if body == 0:
                dims = [(np.array([2 * geometry['hinge_offset_m'] + geometry['strip_thickness_m'],
                                   geometry['handle_width_m'], geometry['handle_length_m']]),
                         np.array([0., 0., -geometry['handle_length_m'] / 2]))]
            else:
                length = geometry['length_m']; tip_length = geometry.get('tip_length_m', .015)
                dims = [(np.array([geometry['strip_thickness_m'], geometry['strip_width_m'], length - tip_length]),
                         np.array([0., 0., (length - tip_length) / 2])),
                        (np.array([geometry['strip_thickness_m'], min(.005, geometry['strip_width_m']), tip_length]),
                         np.array([0., 0., length - tip_length / 2]))]
            rot = Rotation.from_quat(np.roll(state[:, 3:7], -1, axis=1)).as_matrix()
            minimum = np.full(len(times), np.inf)
            for size, center in dims:
                vertices = np.array(list(itertools.product([-1, 1], repeat=3))) * size / 2 + center
                z = np.einsum('ni,vi->nv', rot[:, 2, :], vertices) + state[:, 2, None] - origins[env, 2] - .72
                minimum = np.minimum(minimum, z.min(axis=1))
            row['min_tool_table_clearance_by_body_m'][name] = float(minimum.min())
            row['tool_table_below_tolerance_samples'] += int((minimum < -1e-4).sum())
        rows.append(row)
        print(json.dumps({'environment': env, 'candidate': row['candidate'],
                          'min_robot_table_clearance_m': row['min_robot_table_clearance_m'],
                          'min_tool_table_clearance_m': min(row['min_tool_table_clearance_by_body_m'].values()),
                          'actual_joint_limit_violations': row['actual_joint_limit_violation_samples']}), flush=True)
    launch = json.loads((run / 'launch.json').read_text())
    hashes_match = all(sha(run / 'source' / Path(name).name) == digest
                       for name, digest in launch['source_sha256'].items())
    result = {'schema_version': 1, 'run': str(run), 'all_arrays_finite': bool(finite),
              'source_archive_hashes_match': hashes_match, 'fixture_present': report['fixture_present'],
              'runtime_object_state_writes': report['runtime_object_state_writes'],
              'all_saved_physics_states_checked': True, 'physics_dt_s': report['physics_dt_s'],
              'penetration_and_joint_limit_tolerance': 1e-4, 'rows': rows,
              'source_sha256': {str(run / 'timeseries.npz'): sha(run / 'timeseries.npz'),
                                str(pose_path): sha(pose_path), str(kin.path): sha(kin.path),
                                str(Path(__file__)): sha(__file__)},
              'limitations': ['Source vertices and authored boxes versus infinite tabletop plane only; not self/tool collision.',
                              'Recorded joint FK does not capture all solver compliance between rigid links.',
                              'Retained and physical-support criteria are evaluated in separate reports.']}
    out = args.out or run / 'independent_bounds_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('BOUNDS_AUDIT_READY', out, flush=True)


if __name__ == '__main__':
    main()
