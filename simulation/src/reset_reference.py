"""Translate an observed stable grasp without rebuilding its contact geometry.

This module does not import Isaac Sim. CPU IK builds a small reset bank once;
the returned float32 tensors can then be sampled/reset entirely on the GPU.
Quaternion convention is wxyz throughout. No function advances the simulator.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kinematics import Kinematics, transform


def _numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _matrix(pose):
    pose = _numpy(pose)
    q = pose[3:7]
    return transform(pose[:3], Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix())


def _quaternion(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return np.r_[q[3], q[:3]]


class StableResetReference:
    """Observed robot/tool reset reference, with scene-order joint mapping.

    ``snapshot`` is a JSON filename or its parsed dictionary. ``scene`` supplies
    kinematics, joint order, device, environment origins, and robot base height;
    these can also be supplied explicitly for standalone use.

    ``extra_closure`` is added ONCE to the snapshot's existing q_target using
    ``closure_delta``. It never changes q_actual or the human reference pose.
    A snapshot that already includes a margin should normally use zero here.

    ``build`` preserves all snapshot body/wrist rotations by default. Its
    optional orientation is a desired handle/tip-frame orientation; supplying
    it rotates the ENTIRE observed grasp coherently about the actual tip centre.
    A new gravity orientation requires its own dynamic validation.
    """

    def __init__(self, snapshot, scene=None, *, kin=None, joint_names=None,
                 device=None, env_origins=None, base_position=None,
                 extra_closure=0., closure_delta=None, tip_local=None):
        self.snapshot_path = str(snapshot) if isinstance(snapshot,(str,Path)) else None
        if self.snapshot_path:
            raw = Path(self.snapshot_path).read_bytes()
            self.snapshot = json.loads(raw)
            self.snapshot_sha256 = hashlib.sha256(raw).hexdigest()
        else:
            self.snapshot = snapshot
            self.snapshot_sha256 = None
        self.scene = scene
        clearance = scene.pose_config.get('arm_reset_table_clearance_m') if scene is not None else None
        self.arm_clearance_min = clearance
        self.arm_reset_seed = scene.pose_config.get('arm_reset_seed') if scene is not None else None
        self.kin = kin or (scene.kin if scene is not None else Kinematics())
        self.joint_names = list(joint_names or (scene.joint_names if scene is not None else self.kin.active))
        self.device = str(device or (scene.device if scene is not None else 'cpu'))
        self.default_origins = env_origins if env_origins is not None else (
            scene.env_origins if scene is not None else np.zeros((1,3)))
        self.snapshot_base = _numpy(self.snapshot.get('arm_base_world_xyz',(0.,0.,.72)))
        self.base_position = _numpy(base_position if base_position is not None else (
            (0.,0.,scene.table_height) if scene is not None else self.snapshot_base))
        source = self.snapshot.get('zero_velocity_variant',self.snapshot)
        source_names = self.snapshot['joint_names']
        if len(set(self.joint_names)) != len(self.joint_names) or set(self.joint_names) != set(source_names):
            raise ValueError('Scene joint_names must contain each snapshot joint exactly once')
        order = [source_names.index(n) for n in self.joint_names]
        self.q_actual = _numpy(source['q_actual'])[order].copy()
        self.q_target = _numpy(source['q_target'])[order].copy()
        self.extra_closure = float(extra_closure)
        if closure_delta is None and scene is not None:
            closure_delta = scene.pose_config.get('closure_delta')
        if extra_closure and closure_delta is None:
            raise ValueError('An extra closure requires an explicit closure_delta or scene.pose_config')
        if closure_delta:
            for name, delta in closure_delta.items():
                self.q_target[self.joint_names.index(name)] += extra_closure*float(delta)
        self.arm_indices = [self.joint_names.index(n) for n in self.kin.arm]
        q_actual_by_name = dict(zip(self.joint_names,self.q_actual))
        self.wrist_transform = self.kin.fk(q_actual_by_name)['wrist_3_link'].copy()
        self.wrist_transform[:3,3] += self.snapshot_base
        self.body_transforms = {name:_matrix(source['rigid_body_states_13'][name][:7])
                                for name in ('handle','left','right')}
        self.wrist_to_tool = np.linalg.inv(self.wrist_transform) @ self.body_transforms['handle']
        if tip_local is None:
            tip_local = (0., 0., getattr(scene, 'tip_local_z', .1325))
        point = np.r_[_numpy(tip_local),1.]
        self.tip_positions = {name:(self.body_transforms[name]@point)[:3] for name in ('left','right')}
        self.tip_position = .5*(self.tip_positions['left']+self.tip_positions['right'])
        separation = self.tip_positions['right']-self.tip_positions['left']
        self.tip_gap_center_m = float(np.linalg.norm(separation))
        axis = separation/max(self.tip_gap_center_m,1e-12)
        tip_half_size = np.array(getattr(scene, 'tip_half_size', (.0005,.0025,.0075)))
        projected_half_width = sum(float(np.abs(self.body_transforms[n][:3,:3].T@axis)@tip_half_size)
                                   for n in ('left','right'))
        self.tip_gap_projected_inner_m = self.tip_gap_center_m-projected_half_width
        self.tip_orientation = _quaternion(self.body_transforms['handle'][:3,:3])
        self.tip_local = _numpy(tip_local)
        self._ik_cache = {}
        self.arm_clearance = None
        self.arm_ik_reports = []
        if clearance is not None:
            from arm_clearance import ArmTableClearance
            self.arm_clearance = ArmTableClearance(self.kin)
        self.snapshot_fk_position_error = None
        if 'world_wrist_pose_7' in self.snapshot:
            self.snapshot_fk_position_error = float(np.linalg.norm(
                self.wrist_transform[:3,3]-_numpy(self.snapshot['world_wrist_pose_7'])[:3]))

    def build(self, desired_tip_position=None, *, env_origins=None, env_ids=None,
              position_frame='world', desired_tip_orientation=None, device=None):
        """Return q_actual/q_target (K,18), object_states (K,13), and audit poses.

        ``desired_tip_position`` may be (3,) or (K,3). With ``position_frame``
        ``'world'`` (default), positions ALREADY include each environment origin.
        With ``'local'``, origins are added here. None reproduces the original
        snapshot position independently in each environment. ``env_ids`` selects
        rows from the scene/default origin array; explicit ``env_origins`` must
        already contain only the selected rows.

        Rotations are preserved unless ``desired_tip_orientation`` is supplied
        as a wxyz quaternion (4,), (K,4), matrix (3,3), or (K,3,3).
        IK is computed for wrist_3_link, retaining the actual snapshot hand
        joints and the original small arm target-minus-actual tracking bias.
        """
        if position_frame not in ('world','local'):
            raise ValueError("position_frame must be 'world' or 'local'")
        origins = _numpy(self.default_origins if env_origins is None else env_origins)
        origins = origins.reshape(-1,3)
        if env_ids is not None and env_origins is None:
            origins = origins[_numpy(env_ids).astype(int)]
        count = len(origins)
        if desired_tip_position is None:
            positions = self.tip_position[None]+origins
        else:
            positions = np.broadcast_to(_numpy(desired_tip_position),(count,3)).copy()
            if position_frame=='local': positions += origins
        if desired_tip_orientation is None:
            desired_rotations = np.broadcast_to(self.body_transforms['handle'][:3,:3],(count,3,3))
        else:
            value = _numpy(desired_tip_orientation)
            if value.shape in ((3,3),(count,3,3)):
                desired_rotations = np.broadcast_to(value,(count,3,3))
            else:
                quats = np.broadcast_to(value,(count,4))
                desired_rotations = Rotation.from_quat(np.c_[quats[:,1:],quats[:,0]]).as_matrix()
        if not np.isfinite(positions).all() or not np.isfinite(desired_rotations).all():
            raise ValueError('Reset position and orientation must be finite')
        if count and (not np.allclose(desired_rotations@desired_rotations.transpose(0,2,1),np.eye(3),atol=1e-5)
                      or not np.allclose(np.linalg.det(desired_rotations),1.,atol=1e-5)):
            raise ValueError('Desired orientation matrices must be proper rotations')
        actual = np.tile(self.q_actual,(count,1))
        target = np.tile(self.q_target,(count,1))
        states = {name:np.zeros((count,13)) for name in self.body_transforms}
        wrist_poses = np.zeros((count,7))
        ik_errors = np.zeros(count)
        target_bias = self.q_target[self.arm_indices]-self.q_actual[self.arm_indices]
        for row in range(count):
            rotation = desired_rotations[row]@self.body_transforms['handle'][:3,:3].T
            wrist = self.wrist_transform.copy()
            wrist[:3,:3] = rotation@wrist[:3,:3]
            wrist[:3,3] = positions[row]+rotation@(self.wrist_transform[:3,3]-self.tip_position)
            local_wrist = wrist.copy()
            local_wrist[:3,3] -= origins[row]+self.base_position
            key = tuple(np.round(local_wrist[:3].ravel(),8))
            if key not in self._ik_cache:
                if self.arm_clearance is not None:
                    hand_positions={n:self.q_actual[self.joint_names.index(n)] for n in self.kin.hand}
                    solution = self.arm_clearance.inverse(local_wrist,self.q_actual[self.arm_indices],self.arm_clearance_min,self.arm_reset_seed,
                        hand_joint_positions=hand_positions,min_hand_clearance=self.scene.pose_config.get('hand_reset_table_clearance_m',.001))
                    self.arm_ik_reports.append(solution)
                else:
                    solution = self.kin.inverse_arm(local_wrist,self.q_actual[self.arm_indices],frame='wrist_3_link')
                if not solution['success'] or solution['error']>1e-4:
                    raise ValueError(f'Unreachable stable reset wrist pose at row {row}: {solution}')
                self._ik_cache[key] = (np.array([solution['q'][n] for n in self.kin.arm]),solution['error'])
            arm,ik_errors[row] = self._ik_cache[key]
            actual[row,self.arm_indices] = arm
            target[row,self.arm_indices] = arm+target_bias
            wrist_poses[row] = np.r_[wrist[:3,3],_quaternion(wrist[:3,:3])]
            for name,source in self.body_transforms.items():
                states[name][row,:3] = positions[row]+rotation@(source[:3,3]-self.tip_position)
                states[name][row,3:7] = _quaternion(rotation@source[:3,:3])
        output_device = device or self.device
        tensor = lambda x: torch.as_tensor(x,dtype=torch.float32,device=output_device)
        return {'q_actual':tensor(actual),'q_target':tensor(target),'q_velocity':tensor(np.zeros_like(actual)),
                'object_states':{name:tensor(value) for name,value in states.items()},
                'wrist_pose_w':tensor(wrist_poses),'tip_position_w':tensor(positions),
                'env_origins':tensor(origins),'ik_error':tensor(ik_errors),
                'tip_gap_center_m':tensor(np.full(count,self.tip_gap_center_m)),
                'tip_gap_projected_inner_m':tensor(np.full(count,self.tip_gap_projected_inner_m))}


def build_stable_reset(snapshot, scene, desired_tip_position=None, *, extra_closure=0.,
                       position_frame='world', env_origins=None, env_ids=None,
                       desired_tip_orientation=None):
    """One-shot convenience function; reuse StableResetReference for reset banks."""
    return StableResetReference(snapshot,scene,extra_closure=extra_closure).build(
        desired_tip_position,position_frame=position_frame,env_origins=env_origins,
        env_ids=env_ids,desired_tip_orientation=desired_tip_orientation)
