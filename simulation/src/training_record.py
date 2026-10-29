"""Record explicit (observation[t], joint target[t], observation[t+1]) transitions.

Read-only recorder: does not reset, step, render or change controller commands.
"""
import json
from pathlib import Path

import numpy as np


def array(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.array(value, copy=True)


class TrainingRecorder:
    def __init__(self, env, out, T_link_tcp):
        if env.num_envs != 1:
            raise ValueError('Training recorder currently supports one environment')
        self.env, self.scene, self.out = env, env.scene, Path(out)
        self.T_link_tcp = np.asarray(T_link_tcp)
        self.time_origin = float(self.scene.sim.current_time)
        self.observations, self.actions = {}, {}
        self.frame_observations = []
        self.record_observation()

    @staticmethod
    def append(destination, row):
        for key, value in row.items():
            destination.setdefault(key, []).append(np.array(value, copy=True))

    def record_observation(self):
        from scipy.spatial.transform import Rotation

        data, state = self.scene.robot.data, self.scene.get_state()
        i = self.scene.robot.body_names.index('wrist_3_link')
        p = array(data.body_link_pos_w[0, i])
        q = array(data.body_link_quat_w[0, i])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        T[:3, 3] = p
        row = {
            'time_s': float(self.scene.sim.current_time) - self.time_origin,
            'qpos_rad': array(state['joint_pos'][0]),
            'qvel_rad_s': array(state['joint_vel'][0]),
            'T_world_tcp': T @ self.T_link_tcp,
            'tool_state_w': array(state['tool_root_state_w'][0]),
            'left_jaw_state_w': array(state['left_root_state_w'][0]),
            'right_jaw_state_w': array(state['right_root_state_w'][0]),
            'block_state_w': array(state['block_root_state_w'][0]),
            'phase': int(self.env.phase[0]),
        }
        self.append(self.observations, row)

    def record_transition(self, residual, reward, done, info):
        e, data = self.env, self.scene.robot.data
        # BatchedTweezersScene.step sends e.q_target to this exact API buffer.
        # Check the buffer rather than inferring targets from measured positions.
        target = array(e.q_target[0])
        api_target = array(data.joint_pos_target[0])
        if not np.array_equal(target, api_target):
            raise ValueError('Recorded joint command differs from actuator target buffer')
        m = info['measurements']
        row = {
            'joint_position_target_rad': target,
            'actuator_target_buffer_rad': api_target,
            'nominal_control': array(e.nominal_action[0]),
            'policy_residual': array(residual[0]),
            'applied_control': array(info['applied_action'][0]),
            'reward': float(reward[0]),
            'done': bool(done[0]),
            'success': bool(e.episode_success[0]),
            'failure': bool(info['failure'][0]),
            'phase_timeout': bool(info['phase_timeout'][0]),
            'time_limit': bool(info['time_outs'][0]),
            'jaw_force_N': array(m['forces'][0]),
            'retained_error_m': float(m['grip_error'][0]),
            'face_pinch_ok': bool(m['face_pinch_ok'][0]),
            'lift_m': float(m['lift'][0]),
        }
        self.append(self.actions, row)
        self.record_observation()

    def frame(self, observation_index):
        self.frame_observations.append(observation_index)

    def close(self):
        payload = {'observation_' + k: np.stack(v) for k, v in self.observations.items()}
        payload.update({'action_' + k: np.stack(v) for k, v in self.actions.items()})
        payload['frame_observation_index'] = np.asarray(self.frame_observations, dtype=np.int64)
        payload['control_dt_s'] = np.asarray(self.env.dt)
        np.savez_compressed(self.out / 'training_transitions.npz', **payload)
        e = self.env
        scale = [e.place_cfg.residual_xyz_m / e.cfg.max_wrist_step] * 3 + [
            e.place_cfg.residual_closure / e.cfg.closure_scale,
            e.place_cfg.residual_index_rad / e.cfg.index_support_action_rad,
            e.place_cfg.residual_middle_rad / e.cfg.middle_support_action_rad]
        metadata = {
            'schema_version': 1,
            'transition_convention': 'observation[t], action[t], observation[t+1]; T actions and T+1 observations',
            'action_type': 'absolute_joint_position_target',
            'action_key': 'action_joint_position_target_rad',
            'action_source': 'env.q_target checked against robot.data.joint_pos_target after scene.step',
            'joint_names': self.scene.robot.joint_names,
            'arm_indices': array(e.arm).tolist(), 'hand_indices': array(e.hand).tolist(),
            'joint_units': 'radians; velocities radians/second',
            'joint_soft_limits_rad': array(self.scene.robot.data.soft_joint_pos_limits[0]).tolist(),
            'control_dt_s': e.dt,
            'simulation_time_origin_s': self.time_origin,
            'root_state_order': 'xyz, quaternion_wxyz, linear_velocity_xyz, angular_velocity_xyz; world frame',
            'transform_convention': 'column vectors; metres; T_A_B maps B to A',
            'residual_scale': scale,
            'applied_control_rule': 'clip(nominal + scale * residual, -1, 1); xyz forced zero during phase 0',
            'control_channels': ['wrist_dx', 'wrist_dy', 'wrist_dz', 'closure', 'index_support', 'middle_support'],
            'frame_alignment': 'video frame f is observation[frame_observation_index[f]]; following action begins at that index if index < T',
            'video_timing': 'initial observation included; then every two control steps and terminal observation; exact times in camera_frames.jsonl, final gap can be one control step',
            'phases': e.metadata()['phases'],
            'privileged_diagnostics': ['block_state_w', 'phase', 'jaw_force_N', 'retained_error_m', 'face_pinch_ok', 'lift_m'],
            'recommended_observation': ['wrist_rgb', 'global_rgb', 'qpos_rad', 'qvel_rad_s'],
            'task_instruction': 'Use the preheld tweezers to pick up the block and place it in the green target region.',
            'task_scope': 'Fixed HOME, preheld tweezers, fixed green visible goal marker; not a trained visual policy or real robot execution',
        }
        if hasattr(e,'episode_randomization'):
            metadata['episode_randomization_file']='initial_conditions.json'
            metadata['task_scope']='Perturbed preheld HOME/hand commands and camera mounts; randomized block/visible goal; not a trained visual policy or real execution'
        (self.out / 'training_schema.json').write_text(json.dumps(metadata, indent=2) + '\n')
        manifest_path = self.out / 'camera_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['timestamp_note'] = metadata['video_timing'] + '; see training_transitions.npz frame_observation_index'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
