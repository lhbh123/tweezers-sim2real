#!/usr/bin/env python3
"""Snapshot multiple RL runs without pooling different experiment stages.

Example:
  python scripts/summarize_rl_runs.py 'runs/rl_*'
  python scripts/summarize_rl_runs.py runs/rl_distal30_10_s5 runs/rl_eval_approach350

Produces structured JSON, training-only curves, and a separate independent-eval
figure. It can be rerun while JSONL logs grow. Checkpoint files are hashed without
deserializing them; hashes computed now are distinguished from launch-time hashes.
"""
import argparse
import ast
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_snapshot(path):
    data = path.read_bytes()
    return data, dict(path=str(path), sha256=hash_bytes(data), bytes=len(data))


def read_json(path, warnings):
    if not path.is_file():
        return {}, None
    data, source = file_snapshot(path)
    try:
        return json.loads(data), source
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        warnings.append(f'{path}: JSON unavailable/incomplete: {error}')
        return {}, source


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def archived_scene_options(run, launch):
    """Read literal defaults and launch keywords; never execute archived code."""
    source, trainer = run/'rl_scene.py', run/'train_rl.py'
    options, evidence = {}, {}
    if not source.exists():
        return options, evidence
    tree = ast.parse(source.read_text())
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != 'BatchedTweezersScene':
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                for arg, default in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
                    options[arg.arg] = literal(default)
                    evidence[arg.arg] = 'archived rl_scene.py constructor default'
    if trainer.exists():
        for call in ast.walk(ast.parse(trainer.read_text())):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'BatchedTweezersScene':
                for kw in call.keywords:
                    if not kw.arg:
                        continue
                    value = literal(kw.value)
                    if isinstance(kw.value, ast.Attribute) and isinstance(kw.value.value, ast.Name) and kw.value.value.id == 'args':
                        value = launch.get('arguments', {}).get(kw.value.attr)
                    if value is not None:
                        options[kw.arg] = value
                        evidence[kw.arg] = 'archived train_rl.py call + recorded launch arguments'
    return options, evidence


def filter_configuration(run, task, options, evidence):
    enabled = task.get('filtered_mechanical_self_pairs', options.get('filter_mechanical_self_pairs'))
    provenance = 'task metadata' if 'filtered_mechanical_self_pairs' in task else evidence.get('filter_mechanical_self_pairs')
    actual = task.get('filtered_mechanical_pairs')
    pairs, count, pair_source = None, None, None
    if isinstance(actual, list):
        pairs = sorted({tuple(sorted(Path(x).name for x in pair)) for pair in actual})
        count, pair_source = len(actual), 'recorded runtime pair list, normalized across env replicas'
    elif (run/'rl_scene.py').exists():
        source = (run/'rl_scene.py').read_text()
        if 'FilteredPairsAPI' not in source:
            enabled, pairs, count = False, [], 0
            provenance = pair_source = 'no explicit pair filtering in archived scene implementation'
        elif enabled is True:
            # The archived implementation lists mechanical target links literally.
            # Treat this as source inference, not an observed runtime relation list.
            targets = []
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'link':
                    value = literal(node.iter)
                    if isinstance(value, (tuple, list)) and all(isinstance(x,str) and x.startswith('right_hand_') for x in value):
                        targets.extend(value)
            if targets and '/Robot/root_joint/wrist_3_link' in source:
                pairs = sorted({tuple(sorted(('wrist_3_link', x))) for x in targets})
                pair_source = 'inferred from literal archived scene filter loop; runtime pair list was not recorded'
    return dict(enabled=enabled, enabled_source=provenance,
                unique_link_pairs=[list(p) for p in pairs] if pairs is not None else None,
                unique_pair_count_per_env=len(pairs) if pairs is not None else None,
                recorded_total_relations=count, pair_source=pair_source)


def conditions(run, launch, task):
    args = launch.get('arguments', {})
    options, evidence = archived_scene_options(run, launch)
    self_coll = task.get('robot_self_collisions_enabled', options.get('self_collisions'))
    pose = args.get('pose')
    archived_pose = run/Path(pose).name if pose else None
    runtime_pose = run/'runtime_pose.json'
    runtime_data = json.loads(runtime_pose.read_text()) if runtime_pose.exists() else {}
    dt = task.get('physics_dt', args.get('dt'))
    decimation = task.get('decimation', args.get('decimation'))
    result = dict(pose=pose, pose_recorded_sha256=launch.get('source_sha256', {}).get(pose),
                  pose_archive_sha256=hash_bytes(archived_pose.read_bytes()) if archived_pose and archived_pose.exists() else None,
                  runtime_pose_sha256=hash_bytes(runtime_pose.read_bytes()) if runtime_pose.exists() else None,
                  runtime_geometry_override=runtime_data.get('runtime_geometry_override'),
                  initial_tip_clearance_m=task.get('initial_tip_clearance', args.get('clearance')),
                  reset_xy_range_m=task.get('reset_xy_range', args.get('xy_range')),
                  reset_snapshot=task.get('reset_snapshot', args.get('reset_snapshot')),
                  reset_snapshot_recorded_sha256=task.get('reset_snapshot_sha256'),
                  aperture_reward_weight=task.get('aperture_reward_weight', args.get('aperture_reward')),
                  seed=task.get('seed', args.get('seed')), physics_dt_s=dt, decimation=decimation,
                  control_dt_s=dt*decimation if dt is not None and decimation is not None else None,
                  episode_seconds=task.get('episode_seconds', args.get('episode_seconds')),
                  target_lift_m=task.get('target_lift'), closure_bias=task.get('closure_bias'),
                  closure_scale=task.get('closure_scale'), mode=task.get('mode'),
                  num_envs=task.get('num_envs', args.get('num_envs')),
                  fixture_present=task.get('fixture_present'),
                  self_collision_enabled=self_coll,
                  self_collision_source='task metadata' if 'robot_self_collisions_enabled' in task else evidence.get('self_collisions'),
                  mechanical_pair_filter=filter_configuration(run, task, options, evidence),
                  physical_parameters={k: args.get(k, options.get(k)) for k in ['tilt','friction','spring','spring_damping','solver_iterations','solver_velocity_iterations','hand_stiffness','hand_damping','tool_width','bridge_width','contact_offset']},
                  source_code_sha256=launch.get('source_sha256', {}))
    result['condition_id'] = hash_bytes(json.dumps(clean(result),sort_keys=True).encode())[:16]
    return result


def resolve_checkpoint(value, run):
    if not value or value == 'zero':
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [ROOT/path, run/path, run/path.name]
    return next((p.resolve() for p in candidates if p.is_file()), None)


def checkpoint_provenance(value, run, recorded_hash=None):
    result = dict(reference=value, recorded_run_time_sha256=recorded_hash,
                  computed_summary_time_sha256=None, resolved_path=None)
    if value == 'zero':
        result['kind'] = 'constant zero action baseline; no checkpoint'
        return result
    path = resolve_checkpoint(value,run)
    if path is None:
        result['kind'] = 'checkpoint path unavailable' if value else 'fresh initialization'
        return result
    before = path.stat()
    checksum = hash_bytes(path.read_bytes())
    after = path.stat()
    stable = (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    result.update(kind='checkpoint', resolved_path=str(path), bytes=after.st_size,
                  computed_summary_time_sha256=checksum if stable else None,
                  stable_during_hash=stable,
                  sha256_matches_recorded=checksum==recorded_hash if stable and recorded_hash else None,
                  hash_caveat='Computed at summary time; no run-time checkpoint digest was recorded.' if not recorded_hash else None)
    warnings=[]
    launch,_=read_json(path.parent/'launch.json',warnings)
    task,_=read_json(path.parent/'task.json',warnings)
    if launch or task:
        result['source_training_run'] = str(path.parent)
        result['source_training_conditions'] = conditions(path.parent,launch,task)
        result['source_training_resume_checkpoint'] = launch.get('arguments',{}).get('resume')
    match=re.fullmatch(r'iteration_(\d+)\.pt',path.name)
    result['checkpoint_iteration']=int(match.group(1)) if match else (0 if path.name=='untrained.pt' else None)
    return result


def read_training(path,warnings):
    if not path.is_file():
        return None
    data, source=file_snapshot(path)
    rows,parse_errors=[],[]
    lines=data.splitlines()
    for number,line in enumerate(lines,1):
        if not line.strip():
            continue
        try:
            row=json.loads(line)
            if not isinstance(row,dict):
                raise ValueError('row is not an object')
            row['source_line']=number
            rows.append(clean(row))
        except (ValueError,UnicodeDecodeError) as error:
            parse_errors.append(dict(line=number,last_line=number==len(lines),error=str(error)))
    if parse_errors:
        warnings.append(f'{path}: {len(parse_errors)} malformed/incomplete JSONL lines omitted and recorded')
    # A restart in the same directory must not be connected into one curve.
    segment=0
    for i,row in enumerate(rows):
        if i and row.get('iteration',0)<=rows[i-1].get('iteration',0):
            segment+=1
        row['log_segment']=segment
    return dict(source=source, records=rows, record_count=len(rows), parse_errors=parse_errors,
                distinct_log_segments=segment+1 if rows else 0,
                success_semantics='rollout-averaged rolling success over up to 1000 completed training episodes; not an independent evaluation',
                episode_count_semantics='episodes metric is averaged across rollout steps; it is not an exact integer evaluation denominator',
                first_iteration=rows[0].get('iteration') if rows else None,
                last_iteration=rows[-1].get('iteration') if rows else None,
                latest_metrics=rows[-1] if rows else None)


def wilson(k,n,z=1.959963984540054):
    if not n:return None
    p=k/n; den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,centre-radius),min(1.,centre+radius)]


def scan(run):
    warnings=[]
    launch, launch_source=read_json(run/'launch.json',warnings)
    task, task_source=read_json(run/'task.json',warnings)
    evaluation,eval_source=read_json(run/'evaluation.json',warnings)
    train=read_training(run/'training.jsonl',warnings)
    result=dict(run_name=run.name,run_dir=str(run.resolve()),warnings=warnings,
                launch_source=launch_source,task_source=task_source,
                training_conditions=conditions(run,launch,task),training=train,
                resume_checkpoint=checkpoint_provenance(launch.get('arguments',{}).get('resume'),run),
                evaluation=None)
    if train is not None:
        initial = launch.get('arguments',{}).get('resume')
        if not initial and (run/'untrained.pt').is_file():
            initial = str((run/'untrained.pt').resolve())
        result['training_initialization_checkpoint'] = checkpoint_provenance(initial,run)
    if evaluation:
        n,k=evaluation.get('episodes'),evaluation.get('successes')
        valid=isinstance(n,int) and isinstance(k,int) and n>0 and 0<=k<=n
        if not valid:
            warnings.append('Independent evaluation lacks valid integer successes/episodes; no derived rate is calculated')
        recorded=evaluation.get('checkpoint_sha256',launch.get('checkpoint_sha256'))
        result['evaluation']=dict(source=eval_source,kind='single visualization rollout' if n==1 else 'independent evaluation batch',
                                  episodes=n,successes=k,success_rate=k/n if valid else None,
                                  reported_success_rate=evaluation.get('success_rate'),
                                  wilson_95_interval=wilson(k,n) if valid else None,
                                  conditions=conditions(run,launch,evaluation.get('task',task)),
                                  checkpoint=checkpoint_provenance(evaluation.get('checkpoint'),run,recorded),
                                  per_episode={key:evaluation[key] for key in ['max_lift_m','contact_fraction','retention_fraction','done_step'] if key in evaluation},
                                  inference_caveat='Wilson interval assumes Bernoulli trials; one deterministic vector batch is not evidence across independent training seeds.')
        if valid and evaluation.get('success_rate') is not None and abs(k/n-evaluation['success_rate'])>1e-6:
            warnings.append('Reported evaluation success rate disagrees with exact successes/episodes')
    return result


def setup_label(c):
    pose=Path(c.get('pose') or 'unknown_pose').stem.replace('human_grasp_','')
    sc={True:'on',False:'off',None:'?'}[c.get('self_collision_enabled')]
    filt=c['mechanical_pair_filter']['unique_pair_count_per_env']
    clearance=c.get('initial_tip_clearance_m');dt=c.get('physics_dt_s')
    return ' | '.join([
        pose, f'clear={1000*clearance:g}mm' if clearance is not None else 'clear=?',
        f'SC={sc}/filter={filt if filt is not None else "?"}',f'seed={c.get("seed")}',
        f'dt={1000*dt:.3g}ms' if dt is not None else 'dt=?'])


def plot_training(runs,path):
    training=[r for r in runs if r['training'] and r['training']['records']]
    height=9+max(1,len(training))*.32
    fig=plt.figure(figsize=(16,height),layout='constrained')
    grid=fig.add_gridspec(3,2,height_ratios=[1,1,max(.45,len(training)*.11)])
    axes=[fig.add_subplot(grid[i//2,i%2]) for i in range(4)]
    specs=[('success_rate_1000','Training rolling success (not independent eval)',1.),
           ('episode_return_mean','Rolling mean episode return',1.),
           ('episode_max_lift_mean','Rolling mean episode max lift (mm)',1000.),
           ('bilateral_contact','Rollout bilateral-contact fraction',1.)]
    for ri,run in enumerate(training):
        rows=run['training']['records'];color=plt.get_cmap('tab10')(ri%10)
        for segment in sorted({x['log_segment'] for x in rows}):
            selected=[r for r in rows if r['log_segment']==segment]
            x=[r.get('iteration',math.nan) for r in selected]
            for ax,(key,_,scale) in zip(axes,specs):
                y=[scale*r[key] if isinstance(r.get(key),(float,int)) and (key!='success_rate_1000' or (isinstance(r.get('episodes'),(float,int)) and r['episodes']>0)) else math.nan for r in selected]
                ax.plot(x,y,color=color,lw=1.3,label=f'T{ri+1}: {run["run_name"]}' if segment==0 else None)
    for ax,(_,title,_) in zip(axes,specs):
        ax.set_title(title,fontsize=11,loc='left');ax.set_xlabel('Optimizer iteration (resume retains inherited iteration)');ax.grid(alpha=.2)
    axes[0].set_ylim(-.03,1.03);axes[3].set_ylim(-.03,1.03)
    if training:axes[0].legend(fontsize=8,loc='best')
    tableax=fig.add_subplot(grid[2,:]);tableax.axis('off')
    lines=[f'T{i+1}  {r["run_name"]}: {setup_label(r["training_conditions"])}'+(f' | resumed from {Path(r["resume_checkpoint"]["reference"]).name}' if r['resume_checkpoint']['reference'] else ' | fresh') for i,r in enumerate(training)]
    tableax.text(0,1,'EXPERIMENT CONDITIONS (curves remain separate)\n'+'\n'.join(lines or ['No training records present']),va='top',fontsize=9,family='monospace',linespacing=1.65)
    fig.suptitle('PPO training snapshots — rolling training metrics only',fontsize=17)
    fig.savefig(path,dpi=150);plt.close(fig)


def plot_evaluation(runs,path):
    evaluations=[r for r in runs if r['evaluation'] and r['evaluation']['success_rate'] is not None]
    fig,ax=plt.subplots(figsize=(16,max(4,1.3*len(evaluations)+2)),layout='constrained')
    labels=[]
    for i,run in enumerate(evaluations):
        ev=run['evaluation'];p=ev['success_rate'];lo,hi=ev['wilson_95_interval']
        color='#8c8c8c' if ev['episodes']==1 else '#1f77b4'
        ax.errorbar(p,i,xerr=np.array([[p-lo],[hi-p]]),fmt='o',color=color,capsize=5,ms=7)
        ax.annotate(f'{ev["successes"]}/{ev["episodes"]} = {p:.1%}',(p,i),xytext=(0,13),textcoords='offset points',ha='center',fontsize=10)
        cp=Path(ev['checkpoint']['reference'] or 'unknown').name
        labels.append(f'{run["run_name"]}  [{cp}]\n{setup_label(ev["conditions"])}'+('\nSINGLE RENDER ROLLOUT — not a robustness estimate' if ev['episodes']==1 else ''))
    ax.set_yticks(range(len(labels)),labels,fontsize=9);ax.invert_yaxis();ax.set_xlim(-.035,1.035)
    ax.set_xlabel('Independent batch success fraction; bars: 95% Wilson interval')
    ax.set_title('Independent evaluation batches — no pooling across conditions or stages',loc='left',fontsize=15)
    ax.grid(axis='x',alpha=.2)
    if not labels:ax.text(.5,.5,'No independent evaluation files yet',transform=ax.transAxes,ha='center')
    fig.savefig(path,dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='*',default=['runs/rl_*'],help='Run directories, metadata files, or glob patterns')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/rl_experiments.json')
    parser.add_argument('--plot',type=Path,default=ROOT/'reports/rl_training_curves.png')
    parser.add_argument('--eval-plot',type=Path,default=ROOT/'reports/rl_independent_evaluations.png')
    args=parser.parse_args()
    paths,unmatched=set(),[]
    for pattern in args.inputs:
        matches=glob.glob(pattern)
        if not matches:unmatched.append(pattern)
        for value in matches:
            path=Path(value)
            if path.is_file():path=path.parent
            if path.is_dir():paths.add(path.resolve())
    runs,skipped=[],[]
    for path in sorted(paths):
        if not any((path/n).exists() for n in ['training.jsonl','evaluation.json']):
            skipped.append(dict(run=str(path),reason='no training/evaluation metric file yet'))
            continue
        runs.append(scan(path))
    report=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),inputs=args.inputs,
                metric_policy='Training rolling metrics and independent evaluation batches are separate; no total or pooled success rate is computed.',
                checkpoint_policy='Archived metadata/launch digests are preserved. Checkpoint bytes are hashed at summary time without loading pickle.',
                plot_files=dict(training=str(args.plot),independent_evaluation=str(args.eval_plot)),
                unmatched_inputs=unmatched,skipped_runs=skipped,runs=runs,
                cautions=['Success changes across clearance, pose, self-collision mode, filtered pairs, dt, seed, or reward/source-code revisions are not a controlled learning curve.',
                          'Resume lineage preserves the source checkpoint and original training conditions; training and evaluation may use different collision models.',
                          'An incomplete JSONL tail can be omitted during live logging; the source byte snapshot and parse errors are recorded.',
                          'No process liveness or training-completion claim is inferred from file contents.'])
    for path in [args.output,args.plot,args.eval_plot]:path.parent.mkdir(parents=True,exist_ok=True)
    plot_training(runs,args.plot);plot_evaluation(runs,args.eval_plot)
    tmp=args.output.with_suffix(args.output.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(report),indent=2,allow_nan=False));tmp.replace(args.output)
    print(json.dumps({'report':str(args.output),'training_plot':str(args.plot),'evaluation_plot':str(args.eval_plot),
                      'training_runs':sum(r['training'] is not None for r in runs),
                      'independent_batches':[dict(run=r['run_name'],successes=r['evaluation']['successes'],episodes=r['evaluation']['episodes']) for r in runs if r['evaluation']],
                      'warnings':{r['run_name']:r['warnings'] for r in runs if r['warnings']}},indent=2))


if __name__=='__main__':main()
