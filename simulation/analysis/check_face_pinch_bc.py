#!/usr/bin/env python3
"""Small synthetic CPU regression: native90 schema, provenance, split and PPO init."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
sys.path.insert(0,str(ROOT/'src'))
from train_face_pinch_bc import face_parser
from train_tripod_bc import load_episodes,split_episodes,fit,sha
from ppo import PPO


def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='face-pinch-bc-check-') as temp:
        temp=Path(temp)
        rng=np.random.default_rng(481)
        obs=rng.normal(0,.3,(20,6,90)).astype(np.float32)
        obs[:,:,8]=.5
        obs[:,5,:]=999
        actions=np.tanh(obs[:,:,:6]*.2).astype(np.float32)
        success=np.array([1,1,1,1,1,0],bool)
        path=temp/'fixture.npz'
        task=dict(require_face_pinch=True,require_tripod_support=False,
                  middle_web_support_requirement='optional_backup_contacts_not_rewarded_or_required')
        def save(observation=obs):
            np.savez(path,observations=observation,actions=actions,success=success,
                     episode_id=np.arange(6),phases=np.full((20,6),734),times=np.arange(20))
        def rejected(label):
            try:load_episodes([path],90,6,True)
            except ValueError:return
            raise AssertionError('Accepted invalid '+label)
        save();rejected('missing task metadata')
        (temp/'task.json').write_text(json.dumps({**task,'require_face_pinch':False}))
        rejected('nonface task')
        (temp/'task.json').write_text(json.dumps({**task,'require_tripod_support':True}))
        rejected('old mandatory support')
        (temp/'task.json').write_text(json.dumps(task))
        save(obs[:,:,:86]);rejected('old86 observation data')
        save()
        checks.append('Missing/nonface/continuous-support metadata and old86 data rejected.')
        episodes,sources=load_episodes([path],90,6,True)
        split=split_episodes(episodes,17,.2)
        assert len(episodes)==5 and all(ep['observations'].shape[-1]==90 for ep in episodes)
        assert sources[0]['ignored_npz_keys']==['phases','times']
        copies=episodes+[dict(episodes[0],env=20,episode_id='duplicate')]
        grouped=split_episodes(copies,17,.3)
        assert (0 in grouped['train'])==(5 in grouped['train'])
        checks.append('Only success episodes used; phase/time ignored; duplicate whole trajectories grouped.')
        args=face_parser().parse_args([str(path),'--out',str(temp/'fit'),'--epochs','20',
                                      '--evaluate-every','20','--batch-size','64','--cpu-threads','2'])
        args.entrypoint_file=ROOT/'analysis/train_face_pinch_bc.py'
        report=fit(args)
        saved=torch.load(report['checkpoint'],weights_only=False)
        assert saved['observation_size']==90 and saved['action_size']==6
        assert saved['metadata']['face_pinch_metadata_verified'] is True
        assert saved['metadata']['task_kind']=='face-pinch'
        train=np.concatenate([episodes[i]['observations'] for i in split['train']])
        expected=train.mean(0)*len(train)/(len(train)+1e-4)
        np.testing.assert_allclose(saved['policy']['moments.mean'].numpy(),expected,atol=2e-7,rtol=2e-6)
        assert max(abs(saved['policy']['moments.mean'].numpy()))<1
        assert float(saved['policy']['moments.var'][8])>=args.std_floor**2-1e-10
        assert str(args.entrypoint_file) in saved['metadata']['source_hashes']
        checks.append('90→6 checkpoint, wrapper+core+task provenance; moments train-only; constant variance floor.')
        class Dummy:
            device='cpu';observation_size=90;action_size=6
            def reset(self):return torch.zeros(2,90)
        ppo=PPO(Dummy(),temp/'ppo',initialize=report['checkpoint'])
        assert ppo.samples==0 and ppo.iteration==0 and not ppo.optimizer.state
        assert ppo.initialization['method']=='behavior_cloning_initializer'
        assert ppo.policy.act(torch.zeros(2,90)).shape==(2,6)
        ppo.log.close()
        assert not {'optimizer','samples','iteration'}&saved.keys()
        checks.append('Actual PPO.initialize accepts90→6, resets counters/optimizer; BC has no fabricated PPO samples.')
        outcome=dict(passed=True,synthetic_fixture_only=True,checks=checks,
            source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'analysis/train_face_pinch_bc.py',ROOT/'analysis/train_tripod_bc.py',ROOT/'src/ppo.py']},
            actual_task_success_claimed=False)
        (ROOT/'reports/face_pinch_bc_initializer_checks.json').write_text(json.dumps(outcome,indent=2)+'\n')
        print(json.dumps(outcome))


if __name__=='__main__':main()
