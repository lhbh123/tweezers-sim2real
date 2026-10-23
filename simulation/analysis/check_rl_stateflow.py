"""Regression checks for PPO terminal bootstrapping and vector task reset ownership.

The scene below models in-place simulator buffers, not physical dynamics. These
checks deliberately do not claim to test recovery of an actual PhysX NaN state.
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ppo import PPO, PPOConfig
from rl_task import TaskConfig, TweezerTask


class IdentityMoments:
    def update(self, value):
        pass

    def __call__(self, value):
        return value


class ProbePolicy:
    moments = IdentityMoments()

    def critic(self, obs):
        assert torch.isfinite(obs).all(), 'Critic received an invalid non-timeout terminal observation'
        return obs[:, :1]

    def sample(self, obs):
        zeros = torch.zeros((len(obs), 1), device=obs.device)
        return zeros, zeros, zeros[:, 0], self.critic(obs)[:, 0]


class TerminalProbe:
    num_envs, observation_size, action_size = 3, 1, 1

    def __init__(self, device):
        self.device = device

    def step(self, action):
        kw = dict(device=self.device)
        # Reset states deliberately have values very different from terminal V.
        return (torch.tensor([[99.], [88.], [2.]], **kw),
                torch.tensor([-5., 1., 2.], **kw),
                torch.tensor([True, True, False], **kw),
                {'time_outs': torch.tensor([False, True, False], **kw),
                 'terminal_observation': torch.tensor([[float('nan')], [4.], [9.]], **kw)})


def check_bootstrap(device):
    ppo = PPO.__new__(PPO)
    ppo.device = device
    ppo.env = TerminalProbe(device)
    ppo.cfg = PPOConfig(rollout_steps=1, gamma=.9)
    ppo.policy = ProbePolicy()
    ppo.obs = torch.zeros((3, 1), device=device)
    batch, metrics = ppo.collect()
    expected = torch.tensor([-5., 4.6, 3.8], device=device)
    torch.testing.assert_close(batch[-1], expected)
    assert all(torch.isfinite(item).all() for item in batch)
    assert abs(metrics['reward_per_step'] - (-2 / 3)) < 1e-6


class BufferScene:
    """Isaac-style buffers returned live and overwritten in place on reset."""
    num_envs, dt, table_height = 3, .01, .72
    arm_joint_indices = list(range(6))
    hand_joint_indices = list(range(6, 18))

    def __init__(self, device):
        self.device = device
        self._all_ids = torch.arange(self.num_envs, device=device)
        self.kin = SimpleNamespace(hand=[f'hand_{i}' for i in range(12)])
        self.pose_config = dict(q_reference={n: .2 for n in self.kin.hand},
                                closure_delta={n: .1 for n in self.kin.hand},
                                soft_bounds={n: [0., 1.] for n in self.kin.hand})
        self.default_joint_pos = torch.zeros((self.num_envs, 18), device=device)
        self.env_origins = torch.zeros((self.num_envs, 3), device=device)
        self.state = dict(joint_pos=self.default_joint_pos.clone(),
                          joint_vel=self.default_joint_pos.clone(),
                          wrist_pos_w=torch.zeros((self.num_envs, 3), device=device),
                          wrist_quat_w=torch.tensor([1., 0., 0., 0.], device=device).repeat(self.num_envs, 1),
                          block_jaw_contact_forces_w=torch.zeros((self.num_envs, 2, 3), device=device))
        self.default_object_states = {}
        for name in ['tool', 'left', 'right', 'block']:
            state = torch.zeros((self.num_envs, 13), device=device)
            state[:, 3] = 1
            if name == 'block':
                state[:, 2] = self.table_height + .0061
            self.default_object_states[name] = state
            self.state[name + '_root_state_w'] = state.clone()

    def reset(self, ids, joint_pos, object_states):
        self.state['joint_pos'][ids] = joint_pos
        self.state['joint_vel'][ids] = 0
        for name, state in object_states.items():
            self.state[name + '_root_state_w'][ids] = state

    def get_state(self):
        return self.state

    def get_wrist_jacobian(self):
        return torch.eye(6, device=self.device).expand(self.num_envs, 6, 6)

    def step(self, targets, **kwargs):
        self.state['joint_pos'].copy_(targets)
        self.state['block_root_state_w'][:, 2] += .01


class BufferTask(TweezerTask):
    def _make_reset_bank(self):
        self.bank_q = self.scene.default_joint_pos[:1].clone()
        self.bank_q[:, self.hand] = self.ref + self.cfg.closure_bias * self.closure
        self.bank_delta = torch.zeros((1, 3), device=self.device)
        self.wrist_tool_offset = torch.zeros(3, device=self.device)
        self.wrist_tool_quat = torch.tensor([1., 0., 0., 0.], device=self.device)
        self.wrist_orientation = self.wrist_tool_quat.clone()
        self.initial_wrist_local = torch.zeros(3, device=self.device)


def check_task_buffers(device):
    scene = BufferScene(device)
    task = BufferTask(scene, TaskConfig(reset_xy_range=0))
    torch.testing.assert_close(task.q_target, task.bank_q.expand(scene.num_envs, -1))
    torch.testing.assert_close(task.hand_target, task.q_target[:, task.hand])
    assert task.q_target.data_ptr() != scene.state['joint_pos'].data_ptr()
    task.steps[0] = task.max_steps - 1
    action = torch.full((scene.num_envs, 4), .2, device=device)
    obs, reward, done, info = task.step(action)
    assert done.tolist() == [True, False, False]
    assert info['time_outs'].tolist() == [True, False, False]
    torch.testing.assert_close(info['terminal_observation'][:, -4:], action)
    torch.testing.assert_close(obs[0, -4:], torch.zeros(4, device=device))
    torch.testing.assert_close(obs[1:, -4:], action[1:])
    terminal_copy = info['terminal_observation'].clone()
    assert task.steps.tolist() == [0, 1, 1]
    torch.testing.assert_close(task.q_target[0], task.bank_q[0])
    untouched = task.q_target[1:].clone()
    task.reset(torch.tensor([0], device=device))
    torch.testing.assert_close(task.q_target[1:], untouched)
    torch.testing.assert_close(info['terminal_observation'], terminal_copy)
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    # Auto-reset must not publish mixed pre-reset derived / post-reset live state.
    assert 'measurements' not in info


def check_invalid_task_state(device):
    scene = BufferScene(device)
    task = BufferTask(scene, TaskConfig(reset_xy_range=0))
    original_step = scene.step

    def corrupt_after_step(targets, **kwargs):
        original_step(targets, **kwargs)
        scene.state['block_root_state_w'][0, 2] = float('nan')
        # This infinity is hidden by observe's velocity clamp; raw state must win.
        scene.state['joint_vel'][1, 0] = float('inf')

    scene.step = corrupt_after_step
    task.steps[:2] = task.max_steps - 1
    obs, reward, done, info = task.step(torch.zeros((3, 4), device=device))
    assert done.tolist() == [True, True, False]
    assert info['time_outs'].tolist() == [False, False, False]
    torch.testing.assert_close(reward[:2], torch.full((2,), -5., device=device))
    assert torch.isfinite(obs).all()
    assert not torch.isfinite(info['terminal_observation'][0]).all()
    assert all(math.isfinite(v) for v in info['metrics'].values())
    assert all(math.isfinite(v) for v in task.return_history + task.lift_history)
    assert task.steps.tolist() == [0, 0, 1]


def check_support_actions(device):
    scene=BufferScene(device)
    names=list(scene.kin.hand);names[0]='right_hand_index_joint1';names[1]='right_hand_mid_joint2'
    scene.kin.hand=names
    for field in ['q_reference','closure_delta','soft_bounds']:
        scene.pose_config[field]=dict(zip(names,scene.pose_config[field].values()))
    scene.state['palm_quat_w']=scene.state['wrist_quat_w']
    support={'role_contact_strength_N':torch.full((3,5),.02,device=device),
        'role_support_z_N':torch.full((3,5),.02,device=device),
        'middle_under_contact_count':torch.ones(3,device=device,dtype=torch.long),
        'web_contact_count':torch.ones(3,device=device,dtype=torch.long),
        'web_normal_force_w':torch.tensor([0.,0.,.02],device=device).repeat(3,1),
        'web_friction_force_w':torch.zeros((3,3),device=device)}
    scene.read_grasp_support=lambda state:support
    task=BufferTask(scene,TaskConfig(reset_xy_range=0,require_tripod_support=True,independent_support_actions=True))
    assert task.action_size==6 and task.observe().shape[-1]==80
    assert task.measurements()['grasp_support_ok'].all()
    support['role_contact_strength_N'][0,2]=0
    support['web_contact_count'][1]=0
    support['role_support_z_N'][2,2]=-.02
    assert not task.measurements()['grasp_support_ok'].any()
    action=torch.zeros((3,6),device=device);action[:,4]=1;action[:,5]=-1
    before=task.hand_target.clone();obs,reward,done,info=task.step(action,auto_reset=False)
    expected=before.clone();expected[:,0]+=.15*.03;expected[:,1]-=.15*.12
    torch.testing.assert_close(task.hand_target,expected)
    torch.testing.assert_close(obs[:,-15:-9],action)
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()


if __name__ == '__main__':
    torch.set_num_threads(2)
    devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
    for device in devices:
        check_bootstrap(device)
        check_task_buffers(device)
        check_invalid_task_state(device)
        check_support_actions(device)
    print(json.dumps({'devices': devices, 'invalid_terminal_isolation': 'pass',
                      'timeout_terminal_value_not_reset_value': 'pass',
                      'continuing_state_bootstrap': 'pass', 'task_reset_targets': 'pass',
                      'partial_reset_isolation': 'pass', 'terminal_observation_ownership': 'pass',
                      'invalid_state_precedes_timeout': 'pass',
                      'invalid_reward_and_history_sanitized': 'pass',
                      'autoreset_does_not_publish_live_measurements': 'pass',
                      'independent_support_actions_and_three_missing_support_rejections':'pass',
                      'actual_physx_nan_recovery_tested': False}))
