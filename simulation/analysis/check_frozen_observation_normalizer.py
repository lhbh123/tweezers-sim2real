#!/usr/bin/env python3
"""Exercise actual PPO.collect/update with frozen and live BC normalization."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ppo import PPO,PPOConfig


class FakeEnv:
    device='cpu';num_envs=4;observation_size=86;action_size=6
    def reset(self):
        self.step_count=0
        self.state=torch.arange(self.num_envs*86,dtype=torch.float32).reshape(self.num_envs,86)/100
        return self.state.clone()
    def step(self,action):
        self.step_count+=1
        self.state=self.state+.07
        self.state[:,:6]+=action*.01
        done=torch.zeros(4,dtype=torch.bool)
        timeouts=torch.zeros(4,dtype=torch.bool)
        terminal=self.state.clone()
        if self.step_count%3==0:
            done[:2]=True;timeouts[1]=True
            terminal[0]=float('nan')  # true termination must not be bootstrapped
            self.state[:2]=0.05
        return self.state.clone(),-action.square().sum(-1),done,dict(time_outs=timeouts,terminal_observation=terminal,metrics={})
    def metadata(self):return {'synthetic_test_environment':True}


def clone_buffers(policy):
    return {n:v.clone() for n,v in policy.moments.named_buffers()}


def main():
    torch.set_num_threads(2)
    checkpoint=ROOT/'runs/tripod_bc_demo05_v1/initializer.pt'
    results=[]
    with tempfile.TemporaryDirectory(prefix='frozen-normalizer-check-') as temp:
        for freeze in [True,False]:
            cfg=PPOConfig(rollout_steps=8,epochs=1,minibatches=2,
                          freeze_observation_normalizer=freeze)
            trainer=PPO(FakeEnv(),Path(temp)/str(freeze),cfg,seed=41,initialize=checkpoint)
            initial=clone_buffers(trainer.policy);rollouts=[]
            for i in range(2):
                before=clone_buffers(trainer.policy)
                batch,_=trainer.collect();after=clone_buffers(trainer.policy)
                assert all(torch.isfinite(t).all() for t in batch)
                unchanged={n:torch.equal(before[n],after[n]) for n in before}
                delta=float(after['count']-before['count'])
                if freeze:assert all(unchanged.values())
                else:assert delta==4 and not unchanged['mean'] and not unchanged['var']
                trainer.update(batch)
                assert all(torch.equal(v,dict(trainer.policy.moments.named_buffers())[n]) for n,v in after.items())
                rollouts.append(dict(rollout=i,buffer_exact_equality=unchanged,count_delta=delta,
                                     update_preserves_buffers=True,all_rollout_tensors_finite=True))
            final=clone_buffers(trainer.policy)
            if freeze:assert all(torch.equal(initial[n],final[n]) for n in initial)
            else:assert float(final['count']-initial['count'])==8
            trainer.save()
            saved=torch.load(trainer.out/'latest.pt',weights_only=False)
            assert saved['config']['freeze_observation_normalizer']==freeze
            assert all(torch.equal(final[n],saved['policy']['moments.'+n]) for n in final)
            trainer.log.close()
            results.append(dict(freeze_observation_normalizer=freeze,rollouts=rollouts,
                                count_before=float(initial['count']),count_after=float(final['count']),
                                serialized_flag_and_buffers_match=True))
    report=dict(passed=True,synthetic_fake_env=True,actual_collect_and_update=True,
                initialized_from=str(checkpoint),initializer_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_ppo_sha256=hashlib.sha256((ROOT/'src/ppo.py').read_bytes()).hexdigest(),
                test_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=results,
                note='False updates statistics once from current obs at start of collect, so count increases by num_envs per rollout, not rollout_steps*num_envs.')
    out=ROOT/'reports/frozen_observation_normalizer_checks.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
