"""Initialize the frozen pick/place controller from a live pickup handoff.

No robot or object state is written by this adapter. The original reset bank
is read-only reference geometry; reset() initializes controller memory only.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from kinematics import transform
from arm_clearance import ArmTableClearance
from pick_place_task import PickPlaceTask


def demo_joint_baseline(scene, layout, snapshot):
    """Reproduce the original demo's observation zero without moving the robot."""
    cfg=scene.pose_config
    rotation=Rotation.from_euler('z',layout['tool_yaw_deg'],degrees=True).as_matrix()
    rotation=rotation@Rotation.from_quat(np.roll(snapshot['rigid_body_states_13']['handle'][3:7],-1)).as_matrix()
    hand=transform(layout['tool_spawn_xyz_m'],rotation)@np.linalg.inv(np.asarray(cfg['T_hand_tool']))
    hand[:3,3]-=[0,0,scene.table_height]
    ik=scene.kin.inverse_arm(hand)
    joints=dict(cfg['q_reference'])
    if cfg.get('arm_reset_table_clearance_m') is not None:
        wrist_to_hand=scene.kin.fk(joints,root='wrist_3_link')[scene.palm_link]
        ik=ArmTableClearance(scene.kin).inverse(hand@np.linalg.inv(wrist_to_hand),
            [ik['q'][n] for n in scene.kin.arm],
            min_clearance=float(cfg['arm_reset_table_clearance_m']),
            preferred_seed=cfg.get('arm_reset_seed'),hand_joint_positions=joints)
    if ik['error']>1e-4:raise ValueError('Cannot reconstruct original demo observation baseline')
    joints.update(ik['q'])
    return torch.tensor([[joints[n] for n in scene.joint_names]],device=scene.device,dtype=torch.float32)


class HandoffSceneView:
    """Share live physics, with the old policy's original observation baseline."""
    def __init__(self, scene, baseline):
        self._scene=scene
        self.default_joint_pos=baseline.expand(scene.num_envs,-1).clone()

    def __getattr__(self,name):
        return getattr(self._scene,name)


class ContinuousPickPlaceTask(PickPlaceTask):
    def __init__(self,scene,cfg,place_cfg,joint_target,goal_xy,baseline):
        self.handoff_target=joint_target.clone()
        self.handoff_goal_xy=torch.as_tensor(goal_xy,device=scene.device,dtype=joint_target.dtype)
        super().__init__(HandoffSceneView(scene,baseline),cfg,place_cfg)

    def reset(self,ids=None):
        """Attach controller state to current physics; never restore a snapshot."""
        ids=self.scene._all_ids if ids is None else ids
        state=self.scene.get_state()
        self.q_target[ids]=self.handoff_target[ids]
        self.hand_target[ids]=self.q_target[ids][:,self.hand]
        self.arm_command_reference[ids]=self.arm_command.fk_jacobian(self.q_target[ids][:,self.arm])[0]
        self.start[ids]=state['block_root_state_w'][ids,:3]
        self.goal[ids]=self.start[ids]
        self.goal[ids,:2]=self.handoff_goal_xy+self.scene.env_origins[ids,:2]
        for v in [self.previous_action,self.previous_previous_action,self.last_delta,
                  self.residual_action,self.nominal_action,self.steps,self.phase,self.phase_steps,
                  self.phase_ready,self.success_steps,self.episode_success,self.episode_return,
                  self.episode_max_lift,self.grasped_once,self.max_phase,self.lift_streak,
                  self.release_extra_opening,self.release_middle_blend,self.squeeze_force_steps,
                  self.squeeze_force_hold_time_s,self.squeeze_force_gate_ready]:
            v[ids]=0
        self.block_tip_offset[ids]=torch.tensor([0,0,-.0015],device=self.device)
        self.target_tip[ids]=self.measurements()['tip'][ids]
        return self.observe()

    def metadata(self):
        return {**super().metadata(),'initial_tool':'Actual continuous rack-pickup result; no snapshot restore',
                'randomization':'None; measured block start and explicit goal',
                'fixture_present':True,'fixture_description':'Passive rack remains in the scene',
                'observation_joint_baseline':'Reconstructed original demo spawn joint reference',
                'handoff_robot_or_object_state_writes':False}
