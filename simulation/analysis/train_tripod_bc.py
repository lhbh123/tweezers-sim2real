#!/usr/bin/env python3
"""Offline behavior cloning initializer from successful physical episodes.

Actor-only regression of native observations to bounded task actions. Episodes,
including exact duplicates, stay wholly on one side of the train/validation
split. This is not PPO and does not establish closed-loop task success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ppo import ActorCritic


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def as_bool(array, shape, name):
    if array.shape != shape or array.dtype.kind not in 'bifu':
        raise ValueError(f'{name} must be binary with shape {shape}')
    if not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError(f'{name} contains nonbinary/nonfinite values')
    return array.astype(bool)


def load_episodes(paths, observation_size=86, action_size=6, require_face_pinch=False):
    """Load only successful episodes; source arrays are never changed."""
    episodes, sources, seen = [], [], set()
    for value in paths:
        path = Path(value).resolve()
        digest = sha(path)
        if digest in seen:
            raise ValueError('Duplicate dataset bytes; provide each file only once')
        seen.add(digest)
        with np.load(path, allow_pickle=False) as data:
            required = {'observations', 'actions', 'success'}
            if not required <= set(data.files):
                raise ValueError(f'{path} missing {required-set(data.files)}')
            obs = data['observations']
            actions = data['actions']
            if obs.ndim != 3 or obs.shape[-1] != observation_size or min(obs.shape[:2]) < 1:
                raise ValueError(f'Expected nonempty observations[T,N,{observation_size}]')
            T, N, _ = obs.shape
            if actions.shape != (T, N, action_size):
                raise ValueError(f'Expected actions[T,N,{action_size}] matching observations')
            if obs.dtype.kind not in 'fiu' or actions.dtype.kind not in 'fiu':
                raise ValueError('Observations/actions must be numeric')
            # Reject bad datasets even if corrupt frames happen to be failed episodes.
            if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                raise ValueError('Nonfinite observations/actions; fix the source collector')
            if np.max(np.abs(actions)) > 1 + 1e-6:
                raise ValueError('Teacher actions exceed the task tanh action range [-1,1]')
            success = as_bool(data['success'], (N,), 'success')
            valid = np.ones((T, N), bool)
            if 'valid' in data.files:
                valid = as_bool(data['valid'], (T, N), 'valid')
            elif 'valid_mask' in data.files:
                valid = as_bool(data['valid_mask'], (T, N), 'valid_mask')
            if 'dones' in data.files:
                dones = as_bool(data['dones'], (T, N), 'dones')
                for env in range(N):
                    hits = np.flatnonzero(dones[:, env])
                    if len(hits):
                        valid[hits[0]+1:, env] = False
            if 'episode_id' in data.files:
                ids = data['episode_id']
                if ids.shape != (N,) or ids.dtype.kind not in 'iuUS':
                    raise ValueError('episode_id must be integer/string [N]; repeated episodes per column unsupported')
                ids = [str(x) for x in ids.tolist()]
            else:
                ids = [str(x) for x in range(N)]
            source = dict(path=str(path), sha256=digest, shape=list(obs.shape),
                          total_episodes=N, successful_env_indices=np.flatnonzero(success).tolist(),
                          failed_env_indices=np.flatnonzero(~success).tolist(),
                          ignored_npz_keys=sorted(set(data.files)-required-{'valid','valid_mask','dones','episode_id'}),
                          sources={}, metadata={})
            for name in ['task.json', 'report.json', 'launch.json', 'metadata.json']:
                p = path.parent/name
                if p.exists():
                    source['sources'][str(p)] = sha(p)
                    source['metadata'][name] = json.loads(p.read_text())
            if require_face_pinch:
                task = source['metadata'].get('task.json', {})
                if task.get('require_face_pinch') is not True or task.get('require_tripod_support') is not False:
                    raise ValueError('Face-pinch data require task.json with require_face_pinch=true and require_tripod_support=false')
                if observation_size != 90 or action_size != 6:
                    raise ValueError('Current face-pinch schema is exactly 90 native observations and 6 actions')
                if task.get('middle_web_support_requirement') != 'optional_backup_contacts_not_rewarded_or_required':
                    raise ValueError('Face-pinch task must explicitly declare middle/web support optional')
            archive = path.parent/'source'
            if archive.exists():
                for p in sorted(archive.rglob('*')):
                    if p.is_file():
                        source['sources'][str(p)] = sha(p)
            for env in np.flatnonzero(success):
                rows = np.flatnonzero(valid[:, env])
                if not len(rows):
                    raise ValueError(f'Successful episode {env} has no valid frames')
                x = np.ascontiguousarray(obs[rows, env], dtype=np.float32)
                y = np.ascontiguousarray(actions[rows, env], dtype=np.float32)
                y = np.clip(y, -1, 1)  # only roundoff <= 1e-6, checked above
                fingerprint = hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
                episodes.append(dict(source_sha256=digest, source_path=str(path), env=int(env),
                                     episode_id=ids[env], rows=rows.tolist(), observations=x, actions=y,
                                     content_sha256=fingerprint))
            sources.append(source)
    if len(episodes) < 2:
        raise ValueError('Need at least two successful episodes for an episode-disjoint holdout')
    return episodes, sources


def split_episodes(episodes, seed, heldout_fraction):
    if not 0 < heldout_fraction < 1:
        raise ValueError('heldout_fraction must be in (0,1)')
    # Union identical trajectories and repeated explicit episode IDs. This also
    # prevents copies in separate dataset files entering both train and heldout.
    parent = list(range(len(episodes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen_id, seen_content = {}, {}
    for i, ep in enumerate(episodes):
        key = (ep['source_sha256'], ep['episode_id'])
        for table, item in [(seen_id, key), (seen_content, ep['content_sha256'])]:
            if item in table:
                parent[find(i)] = find(table[item])
            table[item] = i
    groups = {}
    for i in range(len(episodes)):
        groups.setdefault(find(i), []).append(i)
    group_list = list(groups.values())
    if len(group_list) < 2:
        raise ValueError('Need >=2 distinct episode groups; duplicate trajectories cannot form a holdout')
    order = np.random.default_rng(seed).permutation(len(group_list))
    n_valid = min(len(order)-1, max(1, int(math.ceil(len(order)*heldout_fraction))))
    valid = sorted(i for g in order[:n_valid] for i in group_list[g])
    train = sorted(i for g in order[n_valid:] for i in group_list[g])
    assert not set(train) & set(valid)
    assert not ({episodes[i]['content_sha256'] for i in train} & {episodes[i]['content_sha256'] for i in valid})
    return dict(train=train, heldout=valid, groups=group_list,
                seed=int(seed), heldout_fraction=float(heldout_fraction))


def episode_manifest(ep):
    return {key: value for key, value in ep.items() if key not in ('observations', 'actions')}


def dataset_tensors(episodes, indices, device):
    return (torch.as_tensor(np.concatenate([episodes[i]['observations'] for i in indices]), device=device),
            torch.as_tensor(np.concatenate([episodes[i]['actions'] for i in indices]), device=device))


@torch.no_grad()
def metrics(policy, data):
    obs, target = data
    prediction = policy.act(obs)
    residual = prediction-target
    normalized = policy.moments(obs)
    return dict(mse=float(residual.square().mean()), mae=float(residual.abs().mean()),
                rmse_by_action=residual.square().mean(0).sqrt().cpu().tolist(),
                maximum_abs_error_by_action=residual.abs().amax(0).cpu().tolist(),
                p95_abs_error_by_action=torch.quantile(residual.abs(), .95, dim=0).cpu().tolist(),
                normalized_clip_fraction=float((normalized.abs() >= 9.999).float().mean()),
                frames=len(obs))


def fit(args):
    if args.epochs < 1 or args.batch_size < 1 or args.evaluate_every < 1:
        raise ValueError('epochs/batch_size/evaluate_every must be positive')
    if not 0 < args.initial_std <= 1 or args.std_floor <= 0:
        raise ValueError('initial_std must be in (0,1] and std_floor positive')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output directory must be new or empty; do not overwrite prior evidence')
    observation_size = int(getattr(args, 'observation_size', 86))
    action_size = int(getattr(args, 'action_size', 6))
    task_kind = getattr(args, 'task_kind', 'tripod-support')
    if observation_size < 1 or action_size < 1:
        raise ValueError('Observation/action dimensions must be positive')
    episodes, sources = load_episodes(args.datasets, observation_size, action_size,
                                     require_face_pinch=task_kind == 'face-pinch')
    split = split_episodes(episodes, args.seed, args.heldout_fraction)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    datasets = {name: dataset_tensors(episodes, split[name], device) for name in ['train', 'heldout']}
    policy = ActorCritic(observation_size, action_size, args.initial_std).to(device)
    # Only train episodes affect normalizer buffers. Holdout is not seen here.
    policy.moments.update(datasets['train'][0])
    raw_var = policy.moments.var.detach().clone()
    policy.moments.var.clamp_(min=args.std_floor**2)
    fixed_moments = {n: t.detach().clone() for n, t in policy.moments.state_dict().items()}
    for p in policy.critic.parameters():
        p.requires_grad_(False)
    policy.log_std.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.learning_rate)
    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = args.out/'source'
    source_dir.mkdir()
    shutil.copy2(__file__, source_dir/Path(__file__).name)
    shutil.copy2(ROOT/'src/ppo.py', source_dir/'ppo.py')
    entrypoint = getattr(args, 'entrypoint_file', None)
    if entrypoint and Path(entrypoint).resolve() != Path(__file__).resolve():
        shutil.copy2(entrypoint, source_dir/Path(entrypoint).name)
    manifest = dict(algorithm='behavior_cloning_initializer', observation_size=observation_size, action_size=action_size,
                    task_kind=task_kind, face_pinch_metadata_verified=task_kind == 'face-pinch',
                    source_datasets=sources, episodes=[episode_manifest(x) for x in episodes], split=split,
                    source_hashes={str(Path(__file__).resolve()):sha(__file__), str(ROOT/'src/ppo.py'):sha(ROOT/'src/ppo.py')},
                    native_observations_only=True, time_or_phase_appended=False,
                    policy_runtime_teacher=False, failed_episodes_used=False,
                    heldout_used_for_model_selection=True, independent_task_success_test=False,
                    criterion='Strict source success[N]==true; no relaxing recorded physical success criteria.',
                    arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    manifest['arguments']['datasets'] = [str(x) for x in args.datasets]
    if entrypoint:
        manifest['source_hashes'][str(Path(entrypoint).resolve())] = sha(entrypoint)
    data_manifest = args.out/'data_manifest.json'
    data_manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    initial = {name:metrics(policy,data) for name,data in datasets.items()}
    best_state, best_epoch, best_score = None, 0, math.inf
    begin = time.monotonic()
    obs, target = datasets['train']
    normalization = dict(fit_split='train successful episodes only',
                         mean=policy.moments.mean.cpu().tolist(), variance=policy.moments.var.cpu().tolist(),
                         original_running_variance=raw_var.cpu().tolist(),
                         count=float(policy.moments.count), standard_deviation_floor=args.std_floor,
                         floored_dimensions=(raw_var < args.std_floor**2).nonzero().flatten().cpu().tolist(),
                         train_constant_dimensions=(obs.amax(0)-obs.amin(0) < 1e-8).nonzero().flatten().cpu().tolist())
    with (args.out/'training.jsonl').open('w', buffering=1) as log:
        for epoch in range(1,args.epochs+1):
            order = torch.randperm(len(obs), device=device)
            total, count = 0., 0
            for idx in order.split(args.batch_size):
                prediction = policy.actor(policy.moments(obs[idx])).tanh()
                loss = F.mse_loss(prediction,target[idx]) if args.loss=='mse' else F.smooth_l1_loss(prediction,target[idx],beta=.1)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite supervised loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())*len(idx)
                count += len(idx)
            row = dict(epoch=epoch, training_objective=total/count, elapsed_s=time.monotonic()-begin)
            if epoch%args.evaluate_every==0 or epoch in (1,args.epochs):
                row.update({name:metrics(policy,data) for name,data in datasets.items()})
                score = row['heldout']['mse']
                if score < best_score:
                    best_score, best_epoch = score, epoch
                    best_state = {n:t.detach().cpu().clone() for n,t in policy.state_dict().items()}
                print(json.dumps(row),flush=True)
            log.write(json.dumps(row)+'\n')
    policy.load_state_dict(best_state)
    assert all(torch.equal(t,policy.moments.state_dict()[name]) for name,t in fixed_moments.items())
    final = {name:metrics(policy,data) for name,data in datasets.items()}
    per_episode = []
    with torch.no_grad():
        for name in ['train','heldout']:
            for index in split[name]:
                ep = episodes[index]
                per_episode.append(dict(split=name, source_path=ep['source_path'], env=ep['env'],
                                        episode_id=ep['episode_id'], metrics=metrics(policy,dataset_tensors(episodes,[index],device))))
    validation = dict(initial=initial, selected=final, per_episode=per_episode,
                      selection='minimum heldout action MSE; heldout is validation, not independent test',
                      epochs_trained=args.epochs, selected_epoch=best_epoch)
    metadata = dict(algorithm='behavior_cloning_initializer', source_hashes=manifest['source_hashes'],
                    dataset_sha256=[s['sha256'] for s in sources],
                    data_manifest=str(data_manifest.resolve()), data_manifest_sha256=sha(data_manifest),
                    source_metadata_hashes={p:h for s in sources for p,h in s['sources'].items()},
                    criteria=manifest['criterion'], split=split, epoch=best_epoch,
                    validation=validation, normalization=normalization,
                    initial_action_std=args.initial_std, critic_trained=False,
                    no_PPO_optimizer_or_sample_counters=True, runtime_teacher=False,
                    observation_size=observation_size, action_size=action_size,
                    task_kind=task_kind, face_pinch_metadata_verified=task_kind == 'face-pinch')
    checkpoint = args.out/'initializer.pt'
    torch.save(dict(policy=best_state, observation_size=observation_size, action_size=action_size,
                    algorithm='behavior_cloning_initializer', metadata=metadata), checkpoint)
    # Strict loader/inference round-trip using the same interface as PPO initialize.
    loaded = torch.load(checkpoint,map_location=device,weights_only=False)
    clone = ActorCritic(observation_size,action_size).to(device)
    clone.load_state_dict(loaded['policy'],strict=True)
    assert torch.equal(clone.act(datasets['heldout'][0]),policy.act(datasets['heldout'][0]))
    assert not ({'optimizer','samples','iteration'} & set(loaded))
    report = dict(metadata=metadata, checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),
                  compatibility_roundtrip_pass=True, successful_episode_count=len(episodes),
                  train_episode_count=len(split['train']), heldout_episode_count=len(split['heldout']),
                  closed_loop_task_success_measured=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(checkpoint=str(checkpoint),selected_epoch=best_epoch,
                         heldout_action_mse=final['heldout']['mse'],train_action_mse=final['train']['mse'],
                         closed_loop_task_success_measured=False)),flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('datasets',nargs='+',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--heldout-fraction',type=float,default=.2)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--std-floor',type=float,default=.01)
    p.add_argument('--initial-std',type=float,default=.15)
    p.add_argument('--loss',choices=['mse','smooth_l1'],default='mse')
    p.add_argument('--evaluate-every',type=int,default=10)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--observation-size',type=int,default=86)
    p.add_argument('--action-size',type=int,default=6)
    p.add_argument('--task-kind',choices=['tripod-support','face-pinch'],default='tripod-support')
    return p


if __name__=='__main__':
    fit(parser().parse_args())
