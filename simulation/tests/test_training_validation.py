"""Offline corruption tests for observation/action/video alignment."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from validate_training_episode import validate


class TrainingValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.d = {
            'observation_time_s': np.arange(4) / 60,
            'observation_qpos_rad': np.arange(4)[:, None] * np.ones((4, 7)) * .001,
            'observation_qvel_rad_s': np.ones((4, 7)) * .06,
            'observation_phase': np.ones(4, dtype=int),
            'observation_T_world_tcp': np.tile(np.eye(4), (4, 1, 1)),
            'observation_tool_state_w': np.zeros((4, 13)),
            'observation_block_state_w': np.zeros((4, 13)),
            'action_joint_position_target_rad': np.ones((3, 7)) * .01,
            'action_actuator_target_buffer_rad': np.ones((3, 7)) * .01,
            'action_nominal_control': np.ones((3, 6)) * .2,
            'action_policy_residual': np.ones((3, 6)) * .1,
            'action_applied_control': np.ones((3, 6)) * .3,
            'action_done': np.array([False, False, True]),
            'action_success': np.array([False, False, True]),
            'action_failure': np.zeros(3, dtype=bool),
            'action_phase_timeout': np.zeros(3, dtype=bool),
            'action_time_limit': np.zeros(3, dtype=bool),
            'frame_observation_index': np.array([0, 2, 3]),
        }
        for i, T in enumerate(self.d['observation_T_world_tcp']):
            T[0, 3] = i * .01
        names = ['joint_' + str(i) for i in range(7)]
        for name, data in {
            'training_schema.json': dict(joint_names=names, control_dt_s=1/60,
                                         residual_scale=[1]*6, joint_soft_limits_rad=[[-2,2]]*7),
            'evaluation.json': dict(successes=1, final_xy_error_m=[.001], max_phase=[9]),
            'execution_source_provenance.json': dict(complete=True, status='consistent_at_observed_boundaries'),
            'camera_recording.json': dict(frames=3),
            'camera_manifest.json': dict(joint_names=names, rig={'cameras': {'wrist': {'T_TCP_Cw': np.eye(4).tolist()}}}),
        }.items():
            (self.path / name).write_text(json.dumps(data))
        self.frames = []
        for f, oi in enumerate([0,2,3]):
            self.frames.append(dict(frame_index=f, observation_index=oi, control_step=oi-1,
                elapsed_simulation_time_s=oi/60, joint_positions_rad=self.d['observation_qpos_rad'][oi].tolist(),
                T_world_TCP=self.d['observation_T_world_tcp'][oi].tolist(),
                T_world_Cw=self.d['observation_T_world_tcp'][oi].tolist(), T_world_Cg=np.eye(4).tolist()))
        # This stays independent when the new recording is corrupted below.
        np.savez(self.path/'trajectory.npz', q=self.d['observation_qpos_rad'][1:,None],
                 tool=self.d['observation_tool_state_w'][1:,None], block=self.d['observation_block_state_w'][1:,None],
                 residual=self.d['action_policy_residual'][:,None], applied=self.d['action_applied_control'][:,None])

    def run_validation(self):
        np.savez(self.path/'training_transitions.npz', **self.d)
        (self.path/'camera_frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.frames))
        return validate(self.path, decode=False)

    def test_terminal_frame_has_no_future_action(self):
        self.assertTrue(self.run_validation()['training_candidate'])
        rows = [json.loads(s) for s in (self.path/'training_samples.jsonl').read_text().splitlines()]
        self.assertEqual([r['next_action_index'] for r in rows], [0,2,None])

    def test_reject_one_step_state_shift(self):
        self.d['observation_qpos_rad'] = np.roll(self.d['observation_qpos_rad'], 1, axis=0)
        with self.assertRaisesRegex(ValueError, 'legacy_q'):
            self.run_validation()

    def test_reject_measured_position_as_command(self):
        self.d['action_joint_position_target_rad'] = self.d['observation_qpos_rad'][1:].copy()
        with self.assertRaisesRegex(ValueError, 'command_buffer_exact'):
            self.run_validation()

    def test_reject_wrong_control_composition(self):
        self.d['action_nominal_control'][0,0] += .1
        with self.assertRaisesRegex(ValueError, 'nominal_plus_residual'):
            self.run_validation()

    def test_reject_shifted_frame_timestamp(self):
        self.frames[1]['elapsed_simulation_time_s'] += 1/60
        with self.assertRaisesRegex(ValueError, 'frame_1_time'):
            self.run_validation()

    def test_reject_missing_initial_frame(self):
        self.d['frame_observation_index'] = np.array([1,2,3])
        with self.assertRaisesRegex(ValueError, 'initial_regular_terminal_frame_indices'):
            self.run_validation()

    def test_failed_episode_is_not_training_candidate(self):
        self.d['action_success'][-1] = False
        self.d['action_failure'][-1] = True
        (self.path/'evaluation.json').write_text(json.dumps(dict(successes=0, final_xy_error_m=[.2], max_phase=[3])))
        result = self.run_validation()
        self.assertTrue(result['data_valid'])
        self.assertFalse(result['training_candidate'])


if __name__ == '__main__':
    unittest.main()
