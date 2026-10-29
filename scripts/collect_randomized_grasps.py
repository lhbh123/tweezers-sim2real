#!/usr/bin/env python3
"""Plan/run five seeded preheld-grasp episodes with bounded scene randomization."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'simulation/src'))
from episode_randomization import sample_episode


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def write_summary(out, manifest, results, reason):
    data = dict(planned=len(manifest['episodes']), completed=len(results),
                complete=len(results)==len(manifest['episodes']), stop_reason=reason,
                data_valid=sum(bool(r.get('data_valid')) for r in results),
                task_successes=sum(bool(r.get('task_success')) for r in results),
                training_candidates=sum(bool(r.get('training_candidate')) for r in results), episodes=results)
    write_json(out/'collection_summary.json', data)
    fields=['episode_id','seed','start_xy_m','goal_xy_m','home_offset_xy_m','home_height_offset_m',
            'tool_yaw_offset_deg','data_valid','task_success','training_candidate','final_xy_error_m',
            'transition_count','frame_count','wall_seconds','error']
    with (out/'collection_summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader();writer.writerows(results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('simulation/runs')/('randomized_grasp_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--episodes', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--profile', type=Path, default=ROOT/'simulation/configs/randomized_collection_v1.json')
    p.add_argument('--episode-seconds', type=float)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true', help='Write/check all inputs without starting Isaac Sim')
    p.add_argument('--max-new-episodes', type=int)
    a=p.parse_args()
    if a.max_new_episodes is not None and a.max_new_episodes<0:p.error('Invalid episode limit')
    out=(a.out if a.out.is_absolute() else ROOT/a.out).resolve()
    if a.resume:
        manifest=json.loads((out/'collection_manifest.json').read_text())
        for supplied,stored in [(a.episodes,len(manifest['episodes'])),(a.seed,manifest['sampling_seed']),
                                (a.episode_seconds,manifest['episode_seconds'])]:
            if supplied is not None and supplied!=stored:p.error('Resume cannot change the saved sampling plan')
    else:
        count=5 if a.episodes is None else a.episodes
        seed=20261027 if a.seed is None else a.seed
        duration=36. if a.episode_seconds is None else a.episode_seconds
        if count<1 or seed<0 or not 0<duration<=60:p.error('Invalid episode count, seed or duration')
        profile_path=a.profile if a.profile.is_absolute() else ROOT/a.profile
        profile=json.loads(profile_path.read_text())
        layout=json.loads((ROOT/'simulation/configs/calibrated_demo_y_table.json').read_text())
        hand_names=list(json.loads((ROOT/'simulation/configs/xhand_local.json').read_text())['q_reference'])
        rows=[]
        # Sample every episode, including 000; no hidden fixed reference case.
        for i in range(count):
            conditions=sample_episode(profile,layout,hand_names,seed+i)
            rows.append(dict(episode_id=f'episode_{i:03d}',conditions=conditions))
        out.mkdir(parents=True,exist_ok=False)
        inputs=out/'inputs';inputs.mkdir()
        hashes={}
        for row in rows:
            c=row['conditions'];directory=inputs/row['episode_id'];directory.mkdir()
            for name,data in [('conditions.json',c),('layout.json',c['layout']),
                              ('case.json',{'start':[c['start_xy_m']],'goal':[c['goal_xy_m']]})]:
                path=directory/name;write_json(path,data)
                hashes[str(path.relative_to(out))]=digest(path)
        runtime=['scripts/collect_randomized_grasps.py','scripts/run_randomized_collection.sh',
                 'scripts/run_demo.sh','simulation/scripts/run_calibrated_demo.sh',
                 'simulation/scripts/run_pick_place.py','simulation/src/episode_randomization.py',
                 'simulation/src/reset_reference.py','simulation/src/pick_place_task.py',
                 'simulation/src/rl_task.py','simulation/src/rl_scene.py','simulation/src/training_record.py',
                 'simulation/src/calibrated_demo_render.py','scripts/validate_training_episode.py',
                 'simulation/configs/xhand_local.json','models/snapshots/xhand_stable_grasp.json',
                 'models/policies/xhand_pick_place_ppo.pt','calibration/results/camera_rig_isaac.json',
                 'calibration/results/world_base_isaac.json']
        manifest=dict(schema_version=1,sampling_seed=seed,episode_seconds=duration,profile=profile,
                      task='preheld tweezers pick and place; no table-to-hand tool pickup',
                      randomization='Every episode samples start, visible goal, HOME XY/height/yaw, bounded hand commands and both local camera mounts',
                      episodes=rows,input_sha256=hashes,runtime_sha256={name:digest(ROOT/name) for name in runtime})
        write_json(out/'collection_manifest.json',manifest)
    for name,h in manifest['input_sha256'].items():
        if digest(out/name)!=h:raise ValueError('Sampled input changed: '+name)
    for name,h in manifest['runtime_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Runtime changed since planning; create a new batch: '+name)
    results_path=out/'collection_summary.json'
    results=json.loads(results_path.read_text())['episodes'] if a.resume and results_path.exists() else []
    completed={r['episode_id'] for r in results}
    for row in manifest['episodes']:
        directory=out/row['episode_id']
        if directory.exists() and row['episode_id'] not in completed:
            p.error('Unindexed episode directory exists; preserve under interrupted/ before retrying: '+str(directory))
        if row['episode_id'] in completed and not directory.is_dir():p.error('Completed episode directory missing')
    if a.plan_only:
        if not results_path.exists():write_summary(out,manifest,results,'plan_only')
        print('PLAN_READY',out,flush=True)
        print('Planned episodes:',len(manifest['episodes']),'; no simulation started',flush=True)
        return 0
    from validate_training_episode import validate
    new_count=0
    for row in manifest['episodes']:
        name=row['episode_id'];c=row['conditions']
        if name in completed:
            print('SKIP_COMPLETED',name,flush=True);continue
        if a.max_new_episodes is not None and new_count>=a.max_new_episodes:break
        episode=out/name;directory=out/'inputs'/name
        cmd=['bash',str(ROOT/'scripts/run_demo.sh'),str(episode),'--seed',str(c['seed']),
             '--cases',str(directory/'case.json'),'--scene-layout',str(directory/'layout.json'),
             '--episode-randomization',str(directory/'conditions.json'),'--training-data','--skip-scene-export',
             '--episode-seconds',str(manifest['episode_seconds'])]
        result={k:c[k] for k in ['seed','start_xy_m','goal_xy_m','home_offset_xy_m','home_height_offset_m','tool_yaw_offset_deg']}
        result.update(episode_id=name,command=cmd)
        print('START',name,'start=',c['start_xy_m'],'goal=',c['goal_xy_m'],flush=True)
        began=time.monotonic()
        try:
            with (out/(name+'.log')).open('w') as log:
                with subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as process:
                    try:
                        while True:
                            try:
                                code=process.wait(timeout=30);break
                            except subprocess.TimeoutExpired:
                                if time.monotonic()-began>1800:raise
                                print('RUNNING',name,round(time.monotonic()-began),'s; log:',log.name,flush=True)
                    except (subprocess.TimeoutExpired,KeyboardInterrupt):
                        os.killpg(process.pid,signal.SIGKILL);process.wait();raise
            result['exit_code']=code
            if code:raise RuntimeError(f'Simulation exited {code}; see log and initial_conditions.json')
            result.update(validate(episode))
            initial=json.loads((episode/'initial_conditions.json').read_text())
            if not initial['valid']:raise ValueError('Initial grasp invalid')
            result['initialization_valid']=True
        except Exception as exc:
            result.update(data_valid=False,training_candidate=False,error=str(exc))
        result['wall_seconds']=round(time.monotonic()-began,2)
        results.append(result);new_count+=1
        write_summary(out,manifest,results,'in_progress')
        print('RESULT',name,json.dumps({k:result.get(k) for k in ['data_valid','task_success','training_candidate','error']}),flush=True)
    reason='all_planned_episodes_finished' if len(results)==len(manifest['episodes']) else 'invocation_limit'
    write_summary(out,manifest,results,reason)
    print('COLLECTION_FINISHED' if reason.startswith('all_') else 'COLLECTION_PAUSED',out,flush=True)
    return 0 if all(r.get('data_valid') for r in results) else 1


if __name__=='__main__':
    sys.exit(main())
