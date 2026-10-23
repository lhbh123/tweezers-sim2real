"""Numerical PPO checks, independent of Isaac runtime and scene behavior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from ppo import ActorCritic, RunningMoments, generalized_advantage

torch.set_num_threads(2)
torch.manual_seed(7)
policy = ActorCritic(5, 3)
x = torch.randn(11, 5)
a, u, lp, v = policy.sample(x)
dist = policy.distribution(x)
expected = TransformedDistribution(dist, [TanhTransform(cache_size=1)]).log_prob(a).sum(-1)
torch.testing.assert_close(lp, expected, atol=2e-5, rtol=2e-5)
torch.testing.assert_close((policy.log_prob(dist, u) - lp).exp(), torch.ones(11))
assert torch.isfinite(policy.log_prob(dist, torch.full_like(u, 20))).all()

# One termination (no bootstrap) and one time limit (explicit terminal bootstrap).
r = torch.tensor([[1., 1.], [2., 2.]])
d = torch.tensor([[False, False], [True, True]])
v = torch.tensor([[.5, .5], [.7, .7]])
r[-1, 1] += .9 * 4.
adv, ret = generalized_advantage(r, d, v, torch.tensor([99., 99.]), .9, 1.)
torch.testing.assert_close(ret, torch.tensor([[2.8, 6.04], [2., 5.6]]))

mom = RunningMoments(5)
data = torch.randn(1000, 5) * 3 + 2
mom.update(data[:330]); mom.update(data[330:])
torch.testing.assert_close(mom.mean, data.mean(0), atol=1e-5, rtol=1e-5)
torch.testing.assert_close(mom.var, data.var(0, unbiased=False), atol=1e-5, rtol=1e-5)

# Policy-gradient sanity check: known one-step quadratic reward, no supervision.
small = ActorCritic(2, 1, .2)
opt = torch.optim.Adam(small.parameters(), lr=.0003)
obs = torch.zeros(512, 2)
initial = small.act(obs).mean().item()
for step in range(200):
    action, latent, logp, _ = small.sample(obs)
    reward = -(action.squeeze(-1) - .65).square()
    advantage = (reward - reward.mean()).detach()
    loss = -(logp * advantage).mean()
    opt.zero_grad(); loss.backward(); opt.step()
final = small.act(obs).mean().item()
assert abs(final - .65) < .2, (initial, final)
print({'squashed_likelihood': 'pass', 'timeout_bootstrap': 'pass',
       'running_moments': 'pass', 'policy_gradient_initial_action': initial,
       'policy_gradient_final_action': final})
