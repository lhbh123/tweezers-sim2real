"""Measure actual hand joint margins and kinematic smoothness across saved runs.

All runs are resampled at the same rate before finite differences. Reset fixtures
and post-termination frames are excluded. Metrics describe robot motion and joint
geometry; lower velocity is not evidence of a more human-like grasp.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import ASSET, Kinematics


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(path, kin):
    frames = read(path / 'trajectory.json')
    meta, task, launch = read(path / 'metadata.json'), read(path / 'task.json'), read(path / 'launch.json')
    names = task.get('joint_names', meta.get('robot_joints'))
    if names is None:
        raise ValueError(f'{path}: no recorded joint-name mapping')
    first_done = next((i for i, frame in enumerate(frames) if frame.get('done', False)), len(frames)-1)
    valid_start = float(meta.get('reset_fixture_release_s', 0.))
    selected = [f for f in frames[:first_done+1] if float(f.get('time', f.get('t'))) >= valid_start]
    times = np.array([f.get('time', f.get('t')) for f in selected], dtype=float)
    if len(times) < 5 or not np.all(np.diff(times) > 0):
        raise ValueError(f'{path}: insufficient/invalid timestamps')
    positions = []
    for frame in selected:
        q = frame.get('joint_pos', frame.get('joints'))
        q = q if isinstance(q, dict) else dict(zip(names, q))
        positions.append([q[n] for n in kin.hand])
    positions = np.asarray(positions)
    if not np.isfinite(positions).all():
        raise ValueError(f'{path}: nonfinite actual hand state')
    return dict(path=path, times=times, q=positions, names=names,
                source=dict(trajectory=str(path / 'trajectory.json'), sha256=digest(path / 'trajectory.json'),
                            recorded_frames=len(frames), sampled_native_frames=len(selected),
                            native_median_dt_s=float(np.median(np.diff(times))),
                            reset_fixture_excluded_before_s=valid_start, first_done_index=first_done,
                            effective_start_s=float(times[0]), effective_end_s=float(times[-1]),
                            full_run_duration_s=meta.get('args', {}).get('seconds', task.get('episode_seconds')),
                            self_collisions_enabled=task.get('robot_self_collisions_enabled'),
                            control=launch.get('arguments', {}).get('evaluate', 'scripted state-feedback controller')))


def metrics(run, kin, prior, hz, duration=None):
    end = run['times'][-1] if duration is None else min(run['times'][-1], run['times'][0]+duration)
    times = run['times'][0] + np.arange(int(np.floor((end-run['times'][0])*hz+1e-8))+1)/hz
    q = np.stack([np.interp(times, run['times'], run['q'][:, i]) for i in range(len(kin.hand))], axis=1)
    lo = np.array([kin.by_name[n]['lower'] for n in kin.hand])
    hi = np.array([kin.by_name[n]['upper'] for n in kin.hand])
    margin = np.minimum(q-lo, hi-q)
    soft_lo = np.array([prior['soft_bounds'][n][0] for n in kin.hand])
    soft_hi = np.array([prior['soft_bounds'][n][1] for n in kin.hand])
    soft_margin = np.minimum(q-soft_lo, soft_hi-q)
    ref = np.array([prior['q_reference'][n] for n in kin.hand])
    velocity, acceleration, jerk = [np.diff(q, n=i, axis=0)*hz**i for i in (1, 2, 3)]
    per_joint = []
    for i, name in enumerate(kin.hand):
        per_joint.append(dict(joint=name, minimum_angle_deg=float(np.rad2deg(q[:, i].min())),
                              maximum_angle_deg=float(np.rad2deg(q[:, i].max())),
                              mean_angle_deg=float(np.rad2deg(q[:, i].mean())),
                              minimum_hard_margin_deg=float(np.rad2deg(margin[:, i].min())),
                              minimum_authored_soft_margin_deg=float(np.rad2deg(soft_margin[:, i].min())),
                              rms_velocity_rad_s=float(np.sqrt(np.mean(velocity[:, i]**2))),
                              rms_acceleration_rad_s2=float(np.sqrt(np.mean(acceleration[:, i]**2))),
                              rms_jerk_rad_s3=float(np.sqrt(np.mean(jerk[:, i]**2))),
                              total_absolute_travel_deg=float(np.rad2deg(np.abs(np.diff(q[:, i])).sum()))))
    return dict(start_s=float(times[0]), end_s=float(times[-1]), duration_s=float(times[-1]-times[0]),
                sample_rate_hz=hz, samples=len(q),
                minimum_hard_joint_margin_deg=float(np.rad2deg(margin.min())),
                p05_hard_joint_margin_deg=float(np.rad2deg(np.percentile(margin, 5))),
                minimum_normalized_hard_margin=float((margin/(hi-lo)).min()),
                joint_sample_fraction_within_3deg_hard_limit=float((margin < np.deg2rad(3)).mean()),
                joint_sample_fraction_outside_hard_limits_0p01deg=float((margin < -np.deg2rad(.01)).mean()),
                authored_soft_limit_violation_joint_sample_fraction=float((soft_margin < 0).mean()),
                rms_distance_to_common_robot_prior_deg=float(np.rad2deg(np.sqrt(np.mean((q-ref)**2)))),
                rms_hand_velocity_rad_s=float(np.sqrt(np.mean(velocity**2))),
                p95_absolute_hand_velocity_rad_s=float(np.percentile(np.abs(velocity),95)),
                rms_hand_acceleration_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
                rms_hand_jerk_rad_s3=float(np.sqrt(np.mean(jerk**2))),
                mean_joint_total_absolute_travel_deg=float(np.rad2deg(np.abs(np.diff(q,axis=0)).sum(axis=0).mean())),
                per_joint=per_joint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--prior', type=Path, default=ROOT/'configs/human_grasp_proximal.json')
    parser.add_argument('--hz', type=float, default=30)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/hand_motion_comparison.json')
    args=parser.parse_args()
    kin, prior = Kinematics(), read(args.prior)
    runs=[load_run(p, kin) for p in args.runs]
    common_duration=min(r['times'][-1]-r['times'][0] for r in runs)
    report=dict(method='actual hand positions; linear resampling at common Hz; unfiltered forward finite differences',
                urdf_sha256=digest(ASSET/'robot.urdf'), common_authored_prior=str(args.prior),
                common_authored_prior_sha256=digest(args.prior), common_window_duration_s=float(common_duration),
                limitations=[
                    'Hard margins use URDF mechanical limits. Soft bounds and reference angles are authored robot priors, not human anatomical truth.',
                    'A smaller reference-angle error can reflect the chosen robot prior; it is not a measured 3D human-pose error.',
                    'Lower motion speed/acceleration/jerk does not establish natural grasp shape, especially with different trajectories.',
                    'The old and new runs differ in duration, initialization, controller, task motion, and physics assumptions; these are descriptive comparisons, not a controlled causal benchmark.',
                    'No filtering is used. Finite-difference acceleration/jerk depend on the declared common sampling rate.',
                    'The common window starts after each run becomes fixture-free; motion phases are not aligned by dynamic time warping.'],
                runs=[dict(name=r['path'].name, source=r['source'],
                           full_effective_task=metrics(r,kin,prior,args.hz),
                           common_duration_window=metrics(r,kin,prior,args.hz,common_duration)) for r in runs])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(args.output),'runs':[dict(name=r['name'],**{k:v for k,v in r['full_effective_task'].items() if k!='per_joint'}) for r in report['runs']]},indent=2))


if __name__=='__main__':
    main()
