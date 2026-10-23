"""Small, from-scratch PPO implementation used by the tweezer experiments.

This module performs on-policy reinforcement-learning updates. Optional policy
initialization is recorded separately; no teacher is invoked during rollouts.
Rollouts retain the normalized observations and pre-tanh samples so likelihood
ratios use exactly the same inputs and distribution as the sampling policy.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import hashlib
import math
import time
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    rollout_steps: int = 48
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    value_coef: float = .5
    entropy_coef: float = .003
    max_grad_norm: float = 1.
    target_kl: float = .035
    initial_std: float = .35
    initialize_std_override: float | None = None
    freeze_observation_normalizer: bool = False


class RunningMoments(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('var', torch.ones(size))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        n = x.shape[0]
        delta = x.mean(0) - self.mean
        total = self.count + n
        variance_sum = self.var * self.count + x.var(0, unbiased=False) * n
        variance_sum += delta.square() * self.count * n / total
        self.mean.add_(delta * n / total)
        self.var.copy_(variance_sum / total)
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-5)).clamp(-10, 10)


class ActorCritic(nn.Module):
    def __init__(self, observations, actions, initial_std=.35):
        super().__init__()
        self.observations, self.actions = observations, actions
        self.moments = RunningMoments(observations)

        def net(outputs, final_gain):
            dims = [observations, 256, 256, 128, outputs]
            layers = []
            for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
                layer = nn.Linear(a, b)
                nn.init.orthogonal_(layer.weight, final_gain if i == 3 else math.sqrt(2))
                nn.init.zeros_(layer.bias)
                layers.append(layer)
                if i != 3:
                    layers.append(nn.ELU())
            return nn.Sequential(*layers)

        self.actor = net(actions, .01)
        self.critic = net(1, 1.)
        self.log_std = nn.Parameter(torch.full((actions,), math.log(initial_std)))

    def distribution(self, normalized_obs):
        return Normal(self.actor(normalized_obs), self.log_std.clamp(-5, 1).exp())

    @staticmethod
    def log_prob(dist, latent):
        # Numerically stable log(1 - tanh(u)^2), including saturated samples.
        correction = 2 * (math.log(2) - latent - nn.functional.softplus(-2 * latent))
        return (dist.log_prob(latent) - correction).sum(-1)

    def sample(self, normalized_obs):
        dist = self.distribution(normalized_obs)
        latent = dist.sample()
        return latent.tanh(), latent, self.log_prob(dist, latent), self.critic(normalized_obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        x = self.moments(obs)
        dist = self.distribution(x)
        return (dist.mean if deterministic else dist.sample()).tanh()


def generalized_advantage(rewards, dones, values, final_value, gamma, lam):
    """Timeout rewards must already include gamma * V(terminal_observation)."""
    adv = torch.empty_like(rewards)
    last_adv = torch.zeros_like(final_value)
    for t in reversed(range(rewards.shape[0])):
        next_value = final_value if t == rewards.shape[0] - 1 else values[t + 1]
        mask = (~dones[t]).float()
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        adv[t] = last_adv
    return adv, adv + values


class PPO:
    def __init__(self, env, out, cfg=None, seed=0, resume=None, initialize=None):
        self.env, self.out = env, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(seed)
        self.device = torch.device(env.device)
        self.policy = ActorCritic(env.observation_size, env.action_size, self.cfg.initial_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)
        self.iteration = 0
        self.samples = 0
        self.initialization = None
        if resume and initialize:
            raise ValueError('Use resume or initialize, not both')
        if initialize:
            initial_path = Path(initialize)
            saved = torch.load(initial_path, map_location=self.device, weights_only=False)
            if (saved['observation_size'],saved['action_size']) != (env.observation_size,env.action_size):
                raise ValueError('Initialized policy dimensions do not match this task')
            self.policy.load_state_dict(saved['policy'])
            if self.cfg.initialize_std_override is not None:
                if not 0 < self.cfg.initialize_std_override <= 1:
                    raise ValueError('Initializer exploration std must be in (0,1]')
                with torch.no_grad():self.policy.log_std.fill_(math.log(self.cfg.initialize_std_override))
            self.initialization = {'checkpoint':str(initial_path),
                'sha256':hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                'method':saved.get('algorithm',saved.get('method','external policy initialization')),
                'source_metadata':saved.get('metadata'),
                'optimizer_reset':True,'PPO_samples_start_at_zero':True,
                'exploration_std_override':self.cfg.initialize_std_override}
        if resume:
            saved = torch.load(resume, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(saved['policy'])
            self.optimizer.load_state_dict(saved['optimizer'])
            self.iteration = saved['iteration']
            self.samples = saved['samples']
            self.initialization = saved.get('initialization')
        self.obs = env.reset()
        self.start = time.monotonic()
        self.log = (self.out / 'training.jsonl').open('a', buffering=1)
        (self.out / 'ppo_config.json').write_text(json.dumps({**asdict(self.cfg), 'seed': seed}, indent=2))

    def save(self, name='latest.pt'):
        path = self.out / name
        tmp = path.with_suffix('.tmp')
        torch.save(dict(policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict(),
                        iteration=self.iteration, samples=self.samples,
                        observation_size=self.env.observation_size, action_size=self.env.action_size,
                        initialization=self.initialization,
                        config=asdict(self.cfg), env_config=self.env.metadata()), tmp)
        tmp.replace(path)

    @torch.no_grad()
    def collect(self):
        n, t = self.env.num_envs, self.cfg.rollout_steps
        observations = torch.empty((t, n, self.env.observation_size), device=self.device)
        latents = torch.empty((t, n, self.env.action_size), device=self.device)
        old_logp = torch.empty((t, n), device=self.device)
        values, rewards = torch.empty_like(old_logp), torch.empty_like(old_logp)
        dones = torch.empty((t, n), device=self.device, dtype=torch.bool)
        metrics = {}
        raw_reward = 0.
        # Normalization stays fixed for this rollout and its PPO updates.
        if not self.cfg.freeze_observation_normalizer:
            self.policy.moments.update(self.obs)
        for step in range(t):
            x = self.policy.moments(self.obs)
            action, latent, logp, value = self.policy.sample(x)
            next_obs, reward, done, info = self.env.step(action)
            observations[step], latents[step], old_logp[step], values[step] = x, latent, logp, value
            rewards[step], dones[step] = reward, done
            raw_reward += float(reward.mean())
            if 'time_outs' in info:
                # True terminations can contain invalid simulator state. Do not
                # evaluate them and then mask: NaN * 0 would poison the rollout.
                timeout_ids = info['time_outs'].nonzero(as_tuple=False).flatten()
                if len(timeout_ids):
                    terminal_obs = info['terminal_observation'][timeout_ids]
                    terminal_value = self.policy.critic(self.policy.moments(terminal_obs)).squeeze(-1)
                    rewards[step, timeout_ids] += self.cfg.gamma * terminal_value
            for key, val in info.get('metrics', {}).items():
                metrics[key] = metrics.get(key, 0.) + float(val)
            self.obs = next_obs
        final_value = self.policy.critic(self.policy.moments(self.obs)).squeeze(-1)
        advantages, returns = generalized_advantage(rewards, dones, values, final_value,
                                                    self.cfg.gamma, self.cfg.gae_lambda)
        metrics = {key: value / t for key, value in metrics.items()}
        metrics['reward_per_step'] = raw_reward / t
        return (observations.flatten(0, 1), latents.flatten(0, 1), old_logp.flatten(),
                values.flatten(), advantages.flatten(), returns.flatten()), metrics

    def update(self, batch):
        obs, latents, old_logp, old_values, advantages, returns = batch
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        count = len(obs)
        totals = dict(policy_loss=0., value_loss=0., latent_entropy=0., approx_kl=0., clip_fraction=0.)
        updates, stopped = 0, False
        for epoch in range(self.cfg.epochs):
            order = torch.randperm(count, device=self.device)
            for indices in order.tensor_split(self.cfg.minibatches):
                dist = self.policy.distribution(obs[indices])
                logp = self.policy.log_prob(dist, latents[indices])
                ratio = (logp - old_logp[indices]).exp()
                unclipped = -advantages[indices] * ratio
                clipped = -advantages[indices] * ratio.clamp(1 - self.cfg.clip, 1 + self.cfg.clip)
                policy_loss = torch.maximum(unclipped, clipped).mean()
                value = self.policy.critic(obs[indices]).squeeze(-1)
                value_loss = .5 * (returns[indices] - value).square().mean()
                # Gaussian entropy encourages exploration before the tanh transform.
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    log_ratio = logp - old_logp[indices]
                    kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                    cf = ((ratio - 1).abs() > self.cfg.clip).float().mean()
                for key, value in zip(totals, (policy_loss, value_loss, entropy, kl, cf)):
                    totals[key] += float(value.detach())
                updates += 1
                if float(kl) > self.cfg.target_kl:
                    stopped = True
                    break
            if stopped:
                break
        totals = {key: value / max(updates, 1) for key, value in totals.items()}
        totals['optimizer_updates'] = updates
        return totals

    def train(self, iterations, save_every=25):
        self.save(('initialized.pt' if self.initialization else 'untrained.pt') if self.iteration == 0 else 'resumed.pt')
        for _ in range(iterations):
            begin = time.monotonic()
            batch, metrics = self.collect()
            metrics.update(self.update(batch))
            self.iteration += 1
            self.samples += self.env.num_envs * self.cfg.rollout_steps
            metrics.update(iteration=self.iteration, samples=self.samples,
                           elapsed_s=time.monotonic() - self.start,
                           samples_per_second=self.env.num_envs * self.cfg.rollout_steps / (time.monotonic() - begin))
            self.log.write(json.dumps(metrics) + '\n')
            print(json.dumps(metrics), flush=True)
            if self.iteration % save_every == 0:
                self.save()
                self.save(f'iteration_{self.iteration:05d}.pt')
        self.save()

