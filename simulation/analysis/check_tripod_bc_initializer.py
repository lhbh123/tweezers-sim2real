#!/usr/bin/env python3
"""CPU data-split, normalizer, and PPO-loader checks using explicit synthetic fixtures."""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
sys.path.insert(0,str(ROOT/'src'))
from train_tripod_bc import load_episodes,split_episodes,fit,parser,sha
from ppo import PPO,ActorCritic


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='tripod-bc-check-') as temp:
        temp=Path(temp);rng=np.random.default_rng(930)
        obs=rng.normal(0,.3,(20,6,86)).astype(np.float32)
        obs[:,:,8]=.5
        obs[:,5,:]=999  # failed episode must not enter split/statistics
        actions=np.tanh(obs[:,:,:6]*.2).astype(np.float32)
        success=np.array([1,1,1,1,1,0],bool)
        path=temp/'fixture.npz'
        def save(p,**kw):
            np.savez(p,observations=kw.get('observations',obs),actions=kw.get('actions',actions),
                     success=kw.get('success',success),episode_id=np.arange(6),
                     phases=np.full((20,6),734),times=np.arange(20))
        save(path)
        eps,src=load_episodes([path]);split=split_episodes(eps,17,.2)
        assert [e['env'] for e in eps]==[0,1,2,3,4]
        assert not set(split['train'])&set(split['heldout'])
        assert len(split['train'])==4 and len(split['heldout'])==1
        assert all(e['observations'].shape[-1]==86 for e in eps)
        assert src[0]['ignored_npz_keys']==['phases','times']
        checks.append('Failed episodes excluded; whole episodes split; phase/time ignored.')
        duplicate=dict(eps[0]);duplicate['env']=20;duplicate['episode_id']='newid'
        copies=eps+[duplicate];split2=split_episodes(copies,17,.3)
        assert (0 in split2['train'])==(5 in split2['train'])
        checks.append('Exact duplicate trajectories remain in the same split group.')
        for label,kwargs in [
            ('observation shape',{'observations':obs[:,:,:85]}),
            ('action shape',{'actions':actions[:,:,:5]}),
            ('out of range actions',{'actions':actions+2}),
            ('nonbinary success',{'success':np.full(6,2)}),
            ('no successful episodes',{'success':np.zeros(6)}),
            ('nonfinite observations',{'observations':np.where(np.arange(86)==0,np.nan,obs)}),
        ]:
            invalid=temp/'invalid.npz';save(invalid,**kwargs)
            try:load_episodes([invalid])
            except ValueError:pass
            else:raise AssertionError(f'Accepted invalid {label}')
        checks.append('Six malformed/nonfinite/out-of-range/empty-success fixtures rejected.')
        args=parser().parse_args([str(path),'--out',str(temp/'fit'),'--epochs','30','--evaluate-every','30','--batch-size','64','--cpu-threads','2'])
        report=fit(args);saved=torch.load(report['checkpoint'],weights_only=False)
        train=np.concatenate([eps[i]['observations'] for i in split['train']])
        mean=train.mean(0)*len(train)/(len(train)+1e-4)
        np.testing.assert_allclose(saved['policy']['moments.mean'].numpy(),mean,atol=2e-7,rtol=2e-6)
        assert float(saved['policy']['moments.var'][8])>=args.std_floor**2-1e-10
        assert max(abs(saved['policy']['moments.mean'].numpy()))<1
        checks.append('Normalizer fit exactly train-only (including failed-env exclusion); constant variance floored.')
        torch.manual_seed(args.seed);initial=ActorCritic(86,6,args.initial_std)
        for key,value in saved['policy'].items():
            if key.startswith('critic.') or key=='log_std':
                assert torch.equal(value,initial.state_dict()[key])
        checks.append('Critic and exploration std remain untrained; only actor is fitted.')
        class Dummy:
            device='cpu';observation_size=86;action_size=6
            def reset(self):return torch.zeros(2,86)
        ppo=PPO(Dummy(),temp/'ppo_init',initialize=report['checkpoint'])
        assert ppo.samples==0 and ppo.iteration==0 and not ppo.optimizer.state
        assert ppo.initialization['method']=='behavior_cloning_initializer'
        assert ppo.policy.log_std.requires_grad
        assert all(p.requires_grad for p in ppo.policy.critic.parameters())
        ppo.log.close()
        assert not {'optimizer','samples','iteration'}&saved.keys()
        checks.append('Actual PPO.initialize accepts checkpoint, resets optimizer/sample count, and can train critic/std.')
        outcome=dict(passed=True,synthetic_fixture_only=True,checks=checks,
                     source_hashes={str(ROOT/'analysis/train_tripod_bc.py'):sha(ROOT/'analysis/train_tripod_bc.py'),
                                    str(ROOT/'src/ppo.py'):sha(ROOT/'src/ppo.py')},
                     actual_task_success_claimed=False)
        (ROOT/'reports/tripod_bc_initializer_checks.json').write_text(json.dumps(outcome,indent=2)+'\n')
        print(json.dumps(outcome),flush=True)


if __name__=='__main__':main()
