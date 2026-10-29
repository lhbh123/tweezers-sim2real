#!/usr/bin/env python3
"""Small reproducible pilot: one nominal case and seeded local random starts."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def save_summary(out, cases, results, stop_reason):
    interrupted_path = out / 'interruptions.json'
    interruptions = json.loads(interrupted_path.read_text()) if interrupted_path.exists() else []
    summary = dict(manifest='pilot_manifest.json', attempted=len(results)+len(interruptions),
                   completed=len(results), planned=len(cases), complete=len(results)==len(cases),
                   stop_reason=stop_reason, interrupted_attempts=interruptions,
                   data_valid=sum(bool(r.get('data_valid')) for r in results),
                   task_successes=sum(bool(r.get('task_success')) for r in results),
                   training_candidates=sum(bool(r.get('training_candidate')) for r in results), episodes=results)
    (out / 'pilot_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'pilot_summary.csv').open('w', newline='') as stream:
        keys = ['episode_id', 'condition_type', 'simulation_seed', 'start_xy_m', 'goal_xy_m',
                'data_valid', 'task_success', 'training_candidate', 'transition_count', 'frame_count',
                'duration_s', 'final_xy_error_m', 'max_phase', 'wall_seconds', 'error']
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'simulation/runs' / ('training_pilot_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--episodes', type=int)
    p.add_argument('--sampling-seed', type=int)
    p.add_argument('--start-radius-m', type=float)
    p.add_argument('--resume', action='store_true', help='Use the existing manifest and skip completed episode records')
    p.add_argument('--max-new-episodes', type=int, help='Stop normally after this many new episodes; 0 only checks the plan')
    a = p.parse_args()
    if a.max_new_episodes is not None and a.max_new_episodes < 0:
        p.error('--max-new-episodes must be nonnegative')
    out = (a.out if a.out.is_absolute() else ROOT / a.out).resolve()
    if a.resume:
        manifest = json.loads((out / 'pilot_manifest.json').read_text())
        stored = [manifest['episodes'], manifest['sampling_seed'], manifest['randomization']['block_start_xy']['radius_m']]
        supplied = [a.episodes, a.sampling_seed, a.start_radius_m]
        if any(x is not None and x != y for x, y in zip(supplied, stored)):
            p.error('Resume uses the original manifest; do not change sampling parameters')
        a.episodes, a.sampling_seed, a.start_radius_m = stored
    else:
        a.episodes = 10 if a.episodes is None else a.episodes
        a.sampling_seed = 20261027 if a.sampling_seed is None else a.sampling_seed
        a.start_radius_m = .015 if a.start_radius_m is None else a.start_radius_m
    if a.episodes < 2 or not 0 <= a.start_radius_m <= .03:
        p.error('Use >=2 episodes and a start radius between 0 and 0.03 m')
    if not a.resume:
        out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=a.resume)
    rng = random.Random(a.sampling_seed)
    cases = []
    for i in range(a.episodes):
        seed = a.sampling_seed + i
        start = [.22, .48] if i == 0 else [round(.22 + rng.uniform(-a.start_radius_m, a.start_radius_m), 7),
                                                  round(.48 + rng.uniform(-a.start_radius_m, a.start_radius_m), 7)]
        row = dict(episode_id=f'episode_{i:03d}', simulation_seed=seed,
                   condition_type='nominal_reference' if i == 0 else 'seeded_uniform_start_xy',
                   start_xy_m=start, goal_xy_m=[-.08, .76])
        cases.append(row)
        case_path = inputs / (row['episode_id'] + '.json')
        expected = {'start': [start], 'goal': [row['goal_xy_m']]}
        if a.resume:
            if json.loads(case_path.read_text()) != expected:
                p.error('Saved case differs from original sampling plan: ' + row['episode_id'])
        else:
            case_path.write_text(json.dumps(expected, indent=2) + '\n')
    manifest = dict(schema_version=1, sampling_seed=a.sampling_seed, episodes=a.episodes,
                    randomization={'block_start_xy': {'distribution': 'independent uniform per axis for episodes 1..N-1',
                                      'centre_m': [.22, .48], 'radius_m': a.start_radius_m},
                                   'goal': 'fixed [-0.08, 0.76] m',
                                   'fixed': ['robot HOME and joint snapshot', 'tool pose/yaw and pregrasp', 'block size/orientation',
                                             'camera intrinsics/extrinsics', 'table', 'lighting/materials', 'mass/friction', 'controller/checkpoint/speed'],
                                   'note': 'Simulation seeds vary, PPO mean deterministic; changing seed alone does not randomize fixed overrides; same seeds do not promise bitwise GPU determinism'},
                    cases=cases)
    if a.resume:
        if json.loads((out / 'pilot_manifest.json').read_text()) != manifest:
            p.error('Existing manifest differs from reconstructed plan')
    else:
        (out / 'pilot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary_path = out / 'pilot_summary.json'
    results = json.loads(summary_path.read_text())['episodes'] if a.resume and summary_path.exists() else []
    completed = {r['episode_id'] for r in results}
    for row in cases:
        directory = out / row['episode_id']
        if row['episode_id'] in completed:
            if not directory.is_dir():
                p.error('Completed episode directory missing: ' + str(directory))
        elif directory.exists():
            p.error('Unindexed/incomplete episode exists; preserve it under interrupted/ before retrying: ' + str(directory))
    from validate_training_episode import validate
    new_count = 0
    for row in cases:
        if row['episode_id'] in completed:
            print('SKIP_COMPLETED', row['episode_id'], flush=True)
            continue
        if a.max_new_episodes is not None and new_count >= a.max_new_episodes:
            break
        episode = out / row['episode_id']
        cmd = ['bash', str(ROOT / 'scripts/run_demo.sh'), str(episode), '--seed', str(row['simulation_seed']),
               '--cases', str(inputs / (row['episode_id'] + '.json')), '--training-data', '--skip-scene-export']
        began = time.monotonic()
        print('START', row['episode_id'], 'start=', row['start_xy_m'], 'goal=', row['goal_xy_m'], flush=True)
        result = {**row, 'command': cmd}
        try:
            with (out / (row['episode_id'] + '.log')).open('w') as log:
                with subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                    try:
                        code = process.wait(timeout=1800)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise
            result['exit_code'] = code
            if code:
                raise RuntimeError(f'Simulation exited {code}; see episode log')
            # Save requested AND actual initial conditions, not just the RNG seed.
            initial = json.loads((episode / 'scene_layout.json').read_text())
            (episode / 'initial_conditions.json').write_text(json.dumps({**row, 'actual_layout': initial,
                'actual_initial_robot_tool_block_states': 'training_transitions.npz observation_*[0]',
                'randomization': manifest['randomization']}, indent=2) + '\n')
            if any(abs(x-y) > 1e-6 for x,y in zip(initial['start_xy_m'][0],row['start_xy_m'])):
                raise ValueError('Actual start differs from requested case')
            if any(abs(x-y) > 1e-6 for x,y in zip(initial['goal_xy_m'][0],row['goal_xy_m'])):
                raise ValueError('Actual goal differs from requested case')
            result.update(validate(episode))
        except Exception as exc:
            result.update(data_valid=False, training_candidate=False, error=str(exc))
        result['wall_seconds'] = round(time.monotonic() - began, 2)
        results.append(result)
        new_count += 1
        save_summary(out, cases, results, 'in_progress')
        print('RESULT', row['episode_id'], json.dumps({k:result.get(k) for k in ['data_valid','task_success','training_candidate','error','wall_seconds']}), flush=True)
    reason = 'all_planned_episodes_finished' if len(results) == len(cases) else 'invocation_limit'
    save_summary(out, cases, results, reason)
    print('PILOT_COMPLETE' if len(results) == len(cases) else 'PILOT_PAUSED', out, flush=True)
    return 0 if all(r.get('data_valid') for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
