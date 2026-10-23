"""Behavior cloning of a four-action physics teacher, with whole-env splits.

Inputs are exactly the native 69 observations. Time and phase never enter the
network; phase is used only for balanced sampling and reported error strata.
This does not perform PPO, an online evaluation, or claim policy task success.
"""
import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ppo import ActorCritic


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    obs, act, phase = data['observations'], data['actions'], data['phases']
    if obs.ndim != 3 or obs.shape[-1] != 69 or act.shape != (*obs.shape[:2], 4):
        raise ValueError('Expected observations[T,E,69] and actions[T,E,4]')
    if phase.shape != obs.shape[:2] or data['dones'].shape != obs.shape[:2]:
        raise ValueError('Phase and done arrays must match [T,E]')
    for name in ['observations', 'actions', 'next_observations', 'rewards', 'times']:
        if not np.isfinite(data[name]).all():
            raise ValueError(f'Nonfinite {name}')
    if np.abs(act).max() > 1.00001:
        raise ValueError('Actions are not bounded to [-1,1]')
    # Include the terminal transition, exclude every sample after termination.
    live = np.ones(obs.shape[:2], dtype=bool)
    live[1:] = np.cumsum(data['dones'][:-1], axis=0) == 0
    return data, live


def split_environments(env_ids, rng, val_fraction, test_fraction):
    if len(env_ids) < 4:
        raise ValueError('At least four selected environments are required for separate train/validation/test')
    ids = rng.permutation(env_ids)
    n_val, n_test = max(1, round(len(ids)*val_fraction)), max(1, round(len(ids)*test_fraction))
    if n_val+n_test >= len(ids)-1:
        raise ValueError('Splits leave fewer than two training environments')
    splits = dict(train=ids[n_val+n_test:], validation=ids[:n_val], test=ids[n_val:n_val+n_test])
    sets = [set(x.tolist()) for x in splits.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    return splits


def flatten_split(data, live, env_ids, device):
    t, e = np.where(live[:, env_ids])
    e = np.asarray(env_ids)[e]
    return dict(observations=torch.as_tensor(data['observations'][t,e], device=device),
                actions=torch.as_tensor(data['actions'][t,e], device=device),
                phases=torch.as_tensor(data['phases'][t,e], device=device),
                sample_indices=np.c_[t,e].tolist())


@torch.no_grad()
def evaluate(policy, split):
    predictions = []
    for obs in split['observations'].split(2048):
        predictions.append(policy.act(obs))
    pred = torch.cat(predictions)
    errors = pred-split['actions']
    phases = {}
    for phase in torch.unique(split['phases']).tolist():
        err = errors[split['phases'] == phase]
        phases[str(phase)] = dict(samples=len(err), mse=float(err.square().mean()),
                                  mae_by_action=err.abs().mean(0).cpu().tolist())
    return dict(samples=len(pred), mse=float(errors.square().mean()),
                smooth_l1=float(F.smooth_l1_loss(pred, split['actions'], beta=.1)),
                phase_macro_mse=float(np.mean([v['mse'] for v in phases.values()])),
                mae_by_action=errors.abs().mean(0).cpu().tolist(),
                rmse_by_action=errors.square().mean(0).sqrt().cpu().tolist(),
                p95_abs_error_by_action=torch.quantile(errors.abs(), .95, dim=0).cpu().tolist(),
                max_abs_error_by_action=errors.abs().amax(0).cpu().tolist(),
                phases=phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--validation-fraction', type=float, default=.125)
    parser.add_argument('--test-fraction', type=float, default=.125)
    parser.add_argument('--std-floor', type=float, default=.01)
    parser.add_argument('--initial-std', type=float, default=.15)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--include-unsuccessful', action='store_true')
    parser.add_argument('--loss', choices=['smooth_l1', 'mse'], default='smooth_l1')
    args = parser.parse_args()
    if args.std_floor <= 0 or args.steps < 1 or args.eval_every < 1:
        raise ValueError('Positive steps/evaluation interval/std floor required')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    data, live = load_dataset(args.dataset)
    source_report_path = args.dataset.parent/'report.json'
    source_report = json.loads(source_report_path.read_text())
    task = source_report['task']
    if not np.isclose(task['closure_bias'], .6) or not np.isclose(task['closure_scale'], 1.4):
        raise ValueError('This initializer requires the teacher action encoding bias=.6, scale=1.4')
    closure = task['closure_bias'] + task['closure_scale']*data['actions'][...,3]
    if closure.min() < .6-1e-5 or closure.max() > 1.8+1e-5:
        raise ValueError('Teacher closures are incompatible with the requested [.6,1.8] runtime limits')
    env_ids = np.arange(data['observations'].shape[1])
    if not args.include_unsuccessful:
        env_ids = env_ids[data['episode_success'].astype(bool)]
    splits = split_environments(env_ids, rng, args.validation_fraction, args.test_fraction)
    datasets = {k:flatten_split(data,live,ids,device) for k,ids in splits.items()}
    policy = ActorCritic(69, 4, args.initial_std).to(device)
    # BC trains only the actor. Start the untrained PPO value head at zero.
    with torch.no_grad():
        policy.critic[-1].weight.zero_(); policy.critic[-1].bias.zero_()
        policy.moments.update(datasets['train']['observations'])
        observed_variance = policy.moments.var.clone()
        floored = observed_variance < args.std_floor**2
        constant = datasets['train']['observations'].var(0,unbiased=False) < 1e-8
        policy.moments.var.clamp_(min=args.std_floor**2)
    # Verify statistics against training examples alone, not the held-out data.
    train_mean = datasets['train']['observations'].mean(0)
    torch.testing.assert_close(policy.moments.mean, train_mean, rtol=1e-5, atol=2e-6)
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.learning_rate)
    train = datasets['train']
    phase_groups = [(train['phases']==p).nonzero().flatten() for p in torch.unique(train['phases'])]
    phase_groups = [x for x in phase_groups if len(x)]
    generator = torch.Generator(device=device).manual_seed(args.seed+1)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__,args.out/'pretrain_orientation_policy.py')
    initial = {name:evaluate(policy,datasets[name]) for name in ['train','validation']}
    best_score, best_state, best_step = math.inf, None, 0
    history=[]; start=time.monotonic()
    for step in range(1,args.steps+1):
        phase_choice = torch.randint(len(phase_groups),(args.batch_size,),device=device,generator=generator)
        sample_ids = torch.empty_like(phase_choice)
        for index, group in enumerate(phase_groups):
            selected = phase_choice == index
            sample_ids[selected] = group[torch.randint(len(group),(int(selected.sum()),),device=device,generator=generator)]
        obs = train['observations'][sample_ids]
        action = train['actions'][sample_ids]
        prediction = policy.actor(policy.moments(obs)).tanh()
        loss = F.smooth_l1_loss(prediction,action,beta=.1) if args.loss=='smooth_l1' else F.mse_loss(prediction,action)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
        optimizer.step()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            validation = evaluate(policy,datasets['validation'])
            row=dict(step=step, minibatch_loss=float(loss),validation_mse=validation['mse'],
                     validation_phase_macro_mse=validation['phase_macro_mse'],elapsed_s=time.monotonic()-start)
            history.append(row)
            if validation['phase_macro_mse'] < best_score:
                best_score,best_step=validation['phase_macro_mse'],step
                best_state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
            print('BC_PROGRESS',json.dumps(row),flush=True)
    policy.load_state_dict(best_state)
    final = {name:evaluate(policy,split) for name,split in datasets.items()}
    # Test environments are evaluated only now, after validation model selection.
    source_hashes = dict(dataset=sha256(args.dataset), teacher_report=sha256(source_report_path),
                         actor_implementation=sha256(ROOT/'src/ppo.py'), pretraining_script=sha256(__file__))
    normalization = dict(fit_split='train only',count=float(policy.moments.count),
                         requested_standard_deviation_floor=args.std_floor,
                         effective_min_standard_deviation=float(math.sqrt(args.std_floor**2+1e-5)),
                         variance_floor_indices=floored.nonzero().flatten().cpu().tolist(),
                         near_constant_train_indices=constant.nonzero().flatten().cpu().tolist(),
                         mean=policy.moments.mean.cpu().tolist(),
                         variance_before_floor=observed_variance.cpu().tolist(),
                         variance_used=policy.moments.var.cpu().tolist(),
                         max_train_mean_difference=float((policy.moments.mean-train_mean).abs().max()),
                         validation_or_test_in_statistics=False)
    metadata=dict(algorithm='BC-only',label='OFFLINE BEHAVIOR CLONING INITIALIZER — NOT PPO OR TASK SUCCESS',
                  captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_dataset=str(args.dataset),source_sha256=source_hashes,
                  teacher_source_sha256=source_report.get('source_sha256'),
                  actual_scene_geometry=source_report.get('actual_scene_geometry'),
                  snapshot_sha256=source_report.get('snapshot_sha256'),
                  observation_size=69,action_size=4,policy_inputs='native observations array only',
                  phase_and_time_in_policy_inputs=False,
                  phases_used_for='uniform phase selection followed by uniform frame selection within training phase; error strata only',
                  split_unit='whole environment/episode; no environment appears in more than one split',
                  split_env_ids={k:v.tolist() for k,v in splits.items()},
                  selected_successful_only=not args.include_unsuccessful,
                  source_env_count=data['observations'].shape[1], source_teacher_success_count=int(data['episode_success'].sum()),
                  selected_env_count=len(env_ids),post_termination_samples_excluded=int((~live).sum()),
                  action_encoding=dict(closure_bias=.6,closure_scale=1.4,
                                       teacher_limits=[task['closure_min'],task['closure_max']],
                                       intended_runtime_limits=[.6,1.8],
                                       observed_closure_range=[float(closure.min()),float(closure.max())],
                                       action_labels=task['action_semantics']),
                  normalization=normalization,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  best_step=best_step,selection_metric='validation phase-macro action MSE',
                  initial=initial,final=final,history=history,
                  critic='untrained; final linear layer set to zero',policy_log_std_initial=float(math.log(args.initial_std)),
                  optimizer_for_export='none; PPO must create a fresh optimizer',ppo_samples=0,ppo_updates=0,
                  closed_loop_evaluation_performed=False,
                  limitations=['Held-out losses use teacher-forced recorded observations and do not measure closed-loop task success.',
                               'Test environments share the same scene/teacher family; this is not cross-domain generalization.',
                               'The native previous_action features remain present; rollout drift must be checked independently.'])
    checkpoint=dict(policy=best_state,iteration=0,samples=0,observation_size=69,action_size=4,
                    algorithm='BC-only',env_config=task,metadata=metadata)
    checkpoint_path=args.out/'bc_policy.pt'
    torch.save(checkpoint,checkpoint_path)
    # Confirm exported policy reproduces held-out predictions and unchanged std.
    loaded=torch.load(checkpoint_path,map_location=device,weights_only=False)
    probe=ActorCritic(69,4,args.initial_std).to(device);probe.load_state_dict(loaded['policy'])
    with torch.no_grad():
        torch.testing.assert_close(probe.act(datasets['test']['observations']),policy.act(datasets['test']['observations']))
        torch.testing.assert_close(probe.log_std.exp(),torch.full((4,),args.initial_std,device=device))
        assert torch.isfinite(probe.act(datasets['test']['observations'])).all()
    metadata['checkpoint_sha256']=sha256(checkpoint_path)
    metadata['checks']=dict(environment_split_disjoint=True,train_only_normalization=True,
                            native_69d_only=True,checkpoint_reload_predictions_identical=True,
                            initial_std_verified=True,all_report_metrics_finite=True)
    (args.out/'bc_report.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    (args.out/'split_indices.json').write_text(json.dumps({name:split['sample_indices'] for name,split in datasets.items()}))
    print('BC_COMPLETE',json.dumps(dict(checkpoint=str(checkpoint_path),best_step=best_step,
                                      validation=final['validation']['mse'],test=final['test']['mse'],
                                      algorithm='BC-only',closed_loop_evaluation_performed=False)),flush=True)


if __name__=='__main__':
    main()
