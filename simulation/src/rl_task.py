"""Human-pose-prior tweezer task: state-feedback policy, no phase controller.

Four policy outputs command Cartesian wrist increments and a hand closure
synergy around the video-informed reference. A damped Jacobian inverse handles
UR5e kinematics; it supplies no reaching/squeezing/lifting schedule. Tool and
block states are written only at episode reset.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def quat_mul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((w, v), -1)


def quat_inv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_apply(q, v):
    uv = 2 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1)


@dataclass
class TaskConfig:
    decimation: int = 4
    episode_seconds: float = 5.
    initial_tip_clearance: float = .002
    reset_xy_range: float = .002
    target_lift: float = .06
    max_wrist_step: float = .002
    max_wrist_descent: float = .08
    closure_bias: float = .4
    closure_scale: float = 1.1
    closure_min: float = -.2
    closure_max: float = 2.
    hand_smoothing: float = .15
    natural_weight: float = .03
    action_smooth_weight: float = .02
    seed: int = 0
    mode: str = 'lift'
    reset_snapshot: str | None = None
    work_xy: tuple = (.48,0.)
    aperture_reward_weight: float = 0.
    reset_tool_quaternion_wxyz: tuple | None = None
    require_palm_down: bool = False
    require_web_up: bool = False
    min_web_up_component: float = .35
    max_palm_up_component: float | None = None
    require_tripod_support: bool = False
    monitor_grasp_contacts: bool = False
    require_face_pinch: bool = False
    grasp_contact_threshold_N: float = .003
    support_vertical_threshold_N: float = .003
    support_reward_weight: float = .5
    independent_support_actions: bool = False
    index_support_action_rad: float = .03
    middle_support_action_rad: float = .12
    command_space_arm: bool = False
    tool_down_angle_range_deg: tuple | None = None


class TweezerTask:
    action_size = 4

    def __init__(self, scene, cfg=None):
        self.scene, self.cfg = scene, cfg or TaskConfig()
        if self.cfg.require_tripod_support and self.cfg.require_face_pinch:
            raise ValueError('Continuous tripod and optional-support face pinch are separate task definitions')
        self.monitor_grasp = self.cfg.monitor_grasp_contacts or self.cfg.require_tripod_support or self.cfg.require_face_pinch
        self.face_assignment = scene.pose_config.get('face_pinch_assignment', {'thumb':'right','index':'left'})
        if set(self.face_assignment) != {'thumb','index'} or set(self.face_assignment.values()) != {'left','right'}:
            raise ValueError('Face pinch requires thumb/index assigned to opposite jaw bodies')
        self.device, self.num_envs = scene.device, scene.num_envs
        self.dt = scene.dt * self.cfg.decimation
        self.max_steps = round(self.cfg.episode_seconds / self.dt)
        self.arm = scene.arm_joint_indices
        self.hand = scene.hand_joint_indices
        self.residual_joints = scene.pose_config.get('residual_joints', {'index':'right_hand_index_joint1','middle':'right_hand_mid_joint2'})
        self.ref = torch.tensor([scene.pose_config['q_reference'][n] for n in scene.kin.hand], device=self.device)
        self.closure = torch.tensor([scene.pose_config['closure_delta'][n] for n in scene.kin.hand], device=self.device)
        bounds = scene.pose_config['soft_bounds']
        self.lo = torch.tensor([bounds[n][0] for n in scene.kin.hand], device=self.device)
        self.hi = torch.tensor([bounds[n][1] for n in scene.kin.hand], device=self.device)
        self.hand_target = self.ref.repeat(self.num_envs, 1)
        self.action_size = 6 if self.cfg.independent_support_actions else 4
        self.previous_action = torch.zeros((self.num_envs, self.action_size), device=self.device)
        self.previous_previous_action = self.previous_action.clone()
        self.steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_steps = self.steps.clone()
        self.episode_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_lift = torch.zeros(self.num_envs, device=self.device)
        self.return_history, self.success_history = [], []
        self.lift_history = []
        self.total_episodes = 0
        self.rng = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        self.bank_q_target = None
        self.bank_object_states = None
        self._make_reset_bank()
        self.q_target = scene.default_joint_pos.clone()
        self.arm_command = None
        if self.cfg.command_space_arm:
            from arm_command import ArmCommandKinematics
            self.arm_command = ArmCommandKinematics(scene.kin, self.device)
            self.arm_command_reference = torch.eye(4,device=self.device).repeat(self.num_envs,1,1)
        self.reset()
        self.observation_size = self.observe().shape[-1]

    def metadata(self):
        semantics=['wrist_dx','wrist_dy','wrist_dz','hand_closure_synergy']
        if self.cfg.independent_support_actions:semantics+=['index_MCP_residual','middle_PIP_residual']
        return {**asdict(self.cfg), 'action_semantics': semantics,
                'joint_names': list(self.scene.joint_names),
                'robot_model': self.scene.pose_config.get('model','UR5e + XHand'),
                'robot_asset_dir': str(self.scene.kin.asset_dir),
                'palm_link': self.scene.kin.palm_link,
                'hand_root_link': self.scene.kin.hand_root_link,
                'arm_joint_names': list(self.scene.kin.arm),
                'finger_torsional_patch_settings': getattr(self.scene,'finger_torsional_patch_settings',None),
                'robot_self_collisions_enabled': self.scene.self_collisions,
                'filtered_mechanical_self_pairs': self.scene.filter_mechanical_self_pairs,
                'filtered_mechanical_pairs': list(self.scene.filtered_mechanical_pairs),
                'tool_geometry': self.scene.pose_config['tool_geometry'],
                'grasp_topology': self.scene.pose_config.get('grasp_topology'),
                'backup_support_geometry': self.scene.pose_config.get('backup_support_geometry'),
                'palm_collision': self.scene.pose_config.get('palm_collision'),
                'arm_reset_table_clearance_m':self.scene.pose_config.get('arm_reset_table_clearance_m'),
                'arm_reset_IK_evidence':getattr(getattr(self,'reset_reference',None),'arm_ik_reports',None),
                'web_contact_region': self.scene.pose_config.get('web_contact_region'),
                'grasp_support_temporal_aggregation':'mean_over_every_physics_substep_in_control_interval' if self.monitor_grasp else None,
                'face_pinch_assignment':self.face_assignment if self.cfg.require_face_pinch else None,
                'face_pinch_criteria':getattr(getattr(self.scene,'grasp_contact_reader',None),'face_criteria',None) if self.cfg.require_face_pinch else None,
                'middle_web_support_requirement':'optional_backup_contacts_not_rewarded_or_required' if self.cfg.require_face_pinch else ('continuous_tripod' if self.cfg.require_tripod_support else 'none'),
                'robot_effort_limits_Nm': dict(self.scene.effort_limits),
                'human_reference_method': self.scene.pose_config['method'],
                'human_video_sha256': self.scene.pose_config.get('human_evidence', {}).get('video_sha256'),
                'no_task_time_tool_or_block_state_writes': True,
                'fixture_present': False, 'physics_dt': self.scene.dt,
                'arm_controller': 'command_space_URDF_IK_exact_zero_hold' if self.cfg.command_space_arm else 'actual_state_differential_IK',
                'reset_snapshot_sha256': getattr(getattr(self, 'reset_reference', None), 'snapshot_sha256', None),
                'num_envs': self.num_envs}

    def _make_reset_bank(self):
        """A small CPU IK bank is computed once; sampling/reset are GPU operations."""
        scene = self.scene
        self.bank_object_states = None
        self.bank_q_target = None
        if self.cfg.reset_snapshot:
            if self.cfg.initial_tip_clearance < .015:
                raise ValueError('Stable snapshot resets require at least 15 mm tip clearance to prevent initial cube intersection.')
            from reset_reference import StableResetReference
            reference = StableResetReference(self.cfg.reset_snapshot, scene)
            # Express the saved target in this same human-reference synergy.
            hand_target = torch.tensor(reference.q_target[self.hand], device=self.device)
            saved_closure = float(((hand_target - self.ref) * self.closure).sum() / self.closure.square().sum().clamp_min(1e-9))
            self.reset_reference = StableResetReference(self.cfg.reset_snapshot, scene,
                extra_closure=self.cfg.closure_bias - saved_closure)
            points = np.array([[self.cfg.work_xy[0]+dx, self.cfg.work_xy[1]+dy, scene.table_height+.0078+self.cfg.initial_tip_clearance]
                               for dx,dy in [(0.,0.),(-.002,0.),(.002,0.),(0.,-.002),(0.,.002)]])
            built = self.reset_reference.build(points, env_origins=np.zeros((5,3)), position_frame='local',
                                               desired_tip_orientation=self.cfg.reset_tool_quaternion_wxyz)
            self.bank_q = built['q_actual']
            self.bank_q_target = built['q_target']
            self.bank_object_states = built['object_states']
            self.bank_delta = torch.zeros((5,3), device=self.device)
            H = self.reset_reference.wrist_to_tool
            self.wrist_tool_offset = torch.tensor(H[:3,3], device=self.device, dtype=torch.float32)
            q = Rotation.from_matrix(H[:3,:3]).as_quat()
            self.wrist_tool_quat = torch.tensor(np.r_[q[3],q[:3]], device=self.device, dtype=torch.float32)
            self.wrist_orientation = built['wrist_pose_w'][0,3:7].clone()
            self.initial_wrist_local = built['wrist_pose_w'][0,:3].clone()
            return
        # Tip centre and lowest edge are geometrically known, independent of object state.
        tiplocal = np.array([0., 0., getattr(scene, 'tip_local_z', .1325), 1.])
        tip0 = .5 * (scene._initial_tool_T['left'] @ tiplocal + scene._initial_tool_T['right'] @ tiplocal)
        offset = np.array([*self.cfg.work_xy, scene.table_height + .0078 + self.cfg.initial_tip_clearance]) - tip0[:3]
        bank_q, bank_delta = [], []
        initial_arm = [scene.initial_joint_positions[n] for n in scene.kin.arm]
        hand_to_tool = np.asarray(scene.pose_config['T_hand_tool'])
        for dx, dy in [(0., 0.), (-.002, 0.), (.002, 0.), (0., -.002), (0., .002)]:
            delta = offset + [dx, dy, 0.]
            tool = scene.tool_initial_T.copy()
            tool[:3, 3] += delta
            palm = tool @ np.linalg.inv(hand_to_tool)
            palm[:3, 3] -= [0., 0., scene.table_height]
            ik = scene.kin.inverse_arm(palm, initial_arm)
            if ik['error'] > 1e-4:
                raise RuntimeError(f'Reset bank IK failed: {ik}')
            q = dict(scene.initial_joint_positions)
            q.update(ik['q'])
            for name in scene.kin.hand:
                q[name] = scene.pose_config['q_reference'][name] + self.cfg.closure_bias * scene.pose_config['closure_delta'][name]
            bank_q.append([q[n] for n in scene.joint_names])
            bank_delta.append(delta)
        self.bank_q = torch.tensor(np.array(bank_q), device=self.device, dtype=torch.float32)
        self.bank_delta = torch.tensor(np.array(bank_delta), device=self.device, dtype=torch.float32)
        # Wrist-tool transform supplies a retention measurement, not a constraint.
        q0 = dict(zip(scene.joint_names, bank_q[0]))
        wrist = scene.kin.fk(q0)['wrist_3_link']
        wrist[:3, 3] += [0., 0., scene.table_height]
        tool = scene.tool_initial_T.copy()
        tool[:3, 3] += bank_delta[0]
        H = np.linalg.inv(wrist) @ tool
        self.wrist_tool_offset = torch.tensor(H[:3, 3], device=self.device, dtype=torch.float32)
        q = Rotation.from_matrix(H[:3, :3]).as_quat()
        self.wrist_tool_quat = torch.tensor(np.r_[q[3], q[:3]], device=self.device, dtype=torch.float32)
        q = Rotation.from_matrix(wrist[:3, :3]).as_quat()
        self.wrist_orientation = torch.tensor(np.r_[q[3], q[:3]], device=self.device, dtype=torch.float32)
        self.initial_wrist_local = torch.tensor(wrist[:3, 3], device=self.device, dtype=torch.float32)

    def reset(self, ids=None):
        ids = self.scene._all_ids if ids is None else ids
        count = len(ids)
        choice = torch.randint(len(self.bank_q), (count,), device=self.device, generator=self.rng)
        q, delta = self.bank_q[choice], self.bank_delta[choice]
        target_q = q if self.bank_q_target is None else self.bank_q_target[choice]
        objects = {}
        for name, state in self.scene.default_object_states.items():
            s = state[ids].clone()
            if name != 'block':
                if self.bank_object_states is None:
                    s[:, :3] += delta
                else:
                    s = self.bank_object_states[name][choice].clone()
                    s[:, :3] += self.scene.env_origins[ids]
            else:
                s[:, :2] += (torch.rand((count, 2), generator=self.rng, device=self.device) * 2 - 1) * self.cfg.reset_xy_range
            objects[name] = s
        self.scene.reset(ids, joint_pos=q, object_states=objects)
        self.q_target[ids] = target_q
        if self.arm_command is not None:
            self.arm_command_reference[ids] = self.arm_command.fk_jacobian(target_q[:,self.arm])[0]
        self.hand_target[ids] = target_q[:, self.hand]
        if self.bank_q_target is not None:
            self.scene.robot.set_joint_position_target(target_q, env_ids=ids)
            self.scene.robot.write_data_to_sim()
        self.previous_action[ids] = 0
        self.previous_previous_action[ids] = 0
        self.steps[ids] = 0
        self.success_steps[ids] = 0
        self.episode_success[ids] = False
        self.episode_return[ids] = 0
        self.episode_max_lift[ids] = 0
        return self.observe()

    def measurements(self):
        s = self.scene.get_state()
        tool, left, right, block = [s[n + '_root_state_w'] for n in ['tool', 'left', 'right', 'block']]
        tip_offset = torch.tensor([0., 0., getattr(self.scene, 'tip_local_z', .1325)], device=self.device).expand(self.num_envs, 3)
        ltip = left[:, :3] + quat_apply(left[:, 3:7], tip_offset)
        rtip = right[:, :3] + quat_apply(right[:, 3:7], tip_offset)
        centre = .5 * (ltip + rtip)
        expected_tool = s['wrist_pos_w'] + quat_apply(s['wrist_quat_w'], self.wrist_tool_offset.expand(self.num_envs, 3))
        expected_orientation = quat_mul(s['wrist_quat_w'], self.wrist_tool_quat.expand(self.num_envs, 4))
        grip_error = (expected_tool - tool[:, :3]).norm(dim=-1)
        orientation_error = 1 - (expected_orientation * tool[:, 3:7]).sum(-1).square().clamp(0, 1)
        forces = s['block_jaw_contact_forces_w'].norm(dim=-1)
        both = (forces > .003).all(-1)
        lift = block[:, 2] - self.scene.env_origins[:, 2] - (self.scene.table_height + .0061)
        tool_axis = quat_apply(tool[:, 3:7], torch.tensor([0.,0.,1.], device=self.device).expand(self.num_envs,3))
        tool_down_angle = torch.rad2deg(torch.atan2(-tool_axis[:,2], tool_axis[:,:2].norm(dim=-1)))
        root_to_tip = centre-tool[:,:3]
        root_to_tip = root_to_tip/root_to_tip.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        root_to_tip_angle = torch.rad2deg(torch.atan2(-root_to_tip[:,2],root_to_tip[:,:2].norm(dim=-1)))
        palm_normal = None
        web_side = None
        if 'palm_quat_w' in s:
            palm_normal = quat_apply(s['palm_quat_w'], torch.tensor(self.scene.pose_config.get('grasp_topology', {}).get('palm_outward_axis_local',[0.,1.,0.]),device=self.device,dtype=s['palm_quat_w'].dtype).expand(self.num_envs,3))
            radial = self.scene.pose_config.get('grasp_topology', {}).get('web_edge_axis_local', [1.,0.,0.])
            radial = torch.tensor(radial,device=self.device,dtype=s['palm_quat_w'].dtype)
            radial = radial/radial.norm().clamp_min(1e-9)
            web_side = quat_apply(s['palm_quat_w'],radial.expand(self.num_envs,3))
        orientation_ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.require_palm_down:
            if palm_normal is None: raise RuntimeError('Palm pose is required for orientation validation')
            orientation_ok &= palm_normal[:,2] < -.05
        if self.cfg.require_web_up:
            if web_side is None: raise RuntimeError('Palm frame is required for radial/web-side orientation validation')
            orientation_ok &= web_side[:,2] >= self.cfg.min_web_up_component
        if self.cfg.max_palm_up_component is not None:
            if palm_normal is None: raise RuntimeError('Palm frame is required for upward-palm limit')
            orientation_ok &= palm_normal[:,2] <= self.cfg.max_palm_up_component
        if self.cfg.tool_down_angle_range_deg is not None:
            lo,hi = self.cfg.tool_down_angle_range_deg
            orientation_ok &= (tool_down_angle>=lo)&(tool_down_angle<=hi)
            orientation_ok &= (root_to_tip_angle>=lo)&(root_to_tip_angle<=hi)
        support = None
        support_ok = torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        face_ok = torch.ones_like(support_ok)
        face_forces = torch.zeros((self.num_envs,2),device=self.device)
        if self.monitor_grasp:
            support = self.scene.read_grasp_support(s)
            if support is None: raise RuntimeError('Grasp constraints require actual tool-hand contact monitoring')
        if self.cfg.require_tripod_support:
            contact = support['role_contact_strength_N'][:,:3] > self.cfg.grasp_contact_threshold_N
            web_force = support['web_normal_force_w']+support['web_friction_force_w']
            support_ok = contact.all(-1)
            support_ok &= support['middle_under_contact_count'] > 0
            support_ok &= support['role_support_z_N'][:,2] > self.cfg.support_vertical_threshold_N
            support_ok &= support['web_contact_count'] > 0
            support_ok &= support['web_normal_force_w'].norm(dim=-1) > self.cfg.grasp_contact_threshold_N
            support_ok &= web_force[:,2] > self.cfg.support_vertical_threshold_N
        if self.cfg.require_face_pinch:
            raw = support['broad_face_inward_force_N']
            counts = support['broad_face_contact_count']
            indices = [('left','right').index(self.face_assignment[role]) for role in ['thumb','index']]
            face_forces = torch.stack([raw[:,jaw,role] for role,jaw in enumerate(indices)],-1)
            face_counts = torch.stack([counts[:,jaw,role] for role,jaw in enumerate(indices)],-1)
            face_ok = (face_forces>self.cfg.grasp_contact_threshold_N).all(-1) & (face_counts>0).all(-1)
        return dict(state=s, tip=centre, ltip=ltip, rtip=rtip, gap=(ltip - rtip).norm(dim=-1),
                    tool_axis_w=tool_axis, tool_down_angle_deg=tool_down_angle,
                    root_to_tip_axis_w=root_to_tip, root_to_tip_down_angle_deg=root_to_tip_angle,
                    palm_normal_w=palm_normal, web_side_axis_w=web_side, orientation_ok=orientation_ok,
                    grasp_support=support, grasp_support_ok=support_ok,
                    face_pinch_ok=face_ok, face_pinch_inward_force_N=face_forces,
                    target_error=block[:, :3] - centre, grip_error=grip_error,
                    orientation_error=orientation_error, forces=forces, both=both, lift=lift)

    def observe(self, m=None):
        m = self.measurements() if m is None else m
        s = m['state']
        target_lift = torch.full((self.num_envs, 1), self.cfg.target_lift, device=self.device)
        observation = torch.cat((
            (s['joint_pos'] - self.scene.default_joint_pos) * .5,
            s['joint_vel'].clamp(-10, 10) * .1,
            m['target_error'] * 10,
            (s['tool_root_state_w'][:, :3] - s['wrist_pos_w']) * 5,
            s['tool_root_state_w'][:, 3:7],
            s['tool_root_state_w'][:, 7:13].clamp(-5, 5) * .1,
            s['block_root_state_w'][:, 7:13].clamp(-5, 5) * .1,
            m['gap'][:, None] * 30,
            m['forces'].clamp(0, 3),
            m['lift'][:, None] * 10,
            m['grip_error'][:, None] * 20,
            m['orientation_error'][:, None],
            target_lift * 10,
            self.previous_action), -1)
        if self.monitor_grasp:
            support=m['grasp_support']
            extra=torch.cat((support['role_contact_strength_N'].clamp(0,2),
                support['role_support_z_N'][:,2:3].clamp(-1,1),
                support['web_normal_force_w'].norm(dim=-1,keepdim=True).clamp(0,2),
                (support['web_normal_force_w']+support['web_friction_force_w'])[:,2:3].clamp(-1,1),
                m['web_side_axis_w'][:,2:3]),-1)
            observation=torch.cat((observation,extra),-1)
        if self.cfg.require_face_pinch:
            observation=torch.cat((observation,m['grasp_support']['broad_face_inward_force_N'].flatten(1).clamp(0,2)),-1)
        if self.cfg.command_space_arm:
            observation=torch.cat((observation,(self.q_target[:,self.arm]-s['joint_pos'][:,self.arm])*5),-1)
        return observation.clamp(-20, 20)

    def step(self, action, auto_reset=True, render=False):
        action = action.clamp(-1, 1)
        s = self.scene.get_state()
        delta = action[:, :3] * self.cfg.max_wrist_step
        # Workspace and speed limits apply identically during training and evaluation.
        local = s['wrist_pos_w'] - self.scene.env_origins
        desired = local + delta
        desired = torch.maximum(torch.minimum(desired, self.initial_wrist_local + torch.tensor([.10, .10, .18], device=self.device)),
                                self.initial_wrist_local - torch.tensor([.10, .10, self.cfg.max_wrist_descent], device=self.device))
        if self.arm_command is not None:
            origin = self.arm_command_reference[:,:3,3]
            self.q_target[:,self.arm] = self.arm_command.increment(self.q_target[:,self.arm],delta,
                self.arm_command_reference,origin-torch.tensor([.10,.10,self.cfg.max_wrist_descent],device=self.device),
                origin+torch.tensor([.10,.10,.18],device=self.device))
        else:
            qerr = quat_mul(self.wrist_orientation.expand(self.num_envs, 4), quat_inv(s['wrist_quat_w']))
            rot_error = (2 * qerr[:, 1:] * torch.sign(qerr[:, :1])).clamp(-.05, .05)
            error = torch.cat((desired - local, rot_error), -1)
            jac = self.scene.get_wrist_jacobian()
            reg = .02**2 * torch.eye(6, device=self.device).expand(self.num_envs, 6, 6)
            dq = (jac.transpose(-1, -2) @ torch.linalg.solve(jac @ jac.transpose(-1, -2) + reg, error.unsqueeze(-1))).squeeze(-1)
            self.q_target[:, self.arm] = s['joint_pos'][:, self.arm] + dq.clamp(-.025, .025)
        c = (self.cfg.closure_bias + action[:, 3] * self.cfg.closure_scale).clamp(self.cfg.closure_min, self.cfg.closure_max)
        target = self.ref + c[:, None] * self.closure
        if self.cfg.independent_support_actions:
            target[:,self.scene.kin.hand.index(self.residual_joints['index'])] += action[:,4]*self.cfg.index_support_action_rad
            target[:,self.scene.kin.hand.index(self.residual_joints['middle'])] += action[:,5]*self.cfg.middle_support_action_rad
        target = target.clamp(self.lo, self.hi)
        self.hand_target += self.cfg.hand_smoothing * (target - self.hand_target)
        self.q_target[:, self.hand] = self.hand_target
        self.scene.step(self.q_target, decimation=self.cfg.decimation, render=render)
        self.steps += 1
        m = self.measurements()
        distance = m['target_error'].norm(dim=-1)
        retained = (m['grip_error'] < .035) & (m['orientation_error'] < .3)
        natural = ((m['state']['joint_pos'][:, self.hand] - self.ref) / .15).square().mean(-1)
        smooth = (action - self.previous_action).square().mean(-1)
        jerk = (action - 2 * self.previous_action + self.previous_previous_action).square().mean(-1)
        reach = torch.exp(-distance / .035)
        grasp = m['both'].float() * retained * m['orientation_ok'] * m['grasp_support_ok'] * m['face_pinch_ok']
        lifted = (m['lift'] / self.cfg.target_lift).clamp(0, 1.5) * grasp
        reward = .2 * retained + 1.2 * reach + 1.8 * grasp + 6 * lifted
        above_block = m['tip'][:,2] - m['state']['block_root_state_w'][:,2]
        desired_gap = .0125 + .002 * torch.sigmoid((above_block - .008) / .003)
        aperture_reward = torch.exp(-((m['gap']-desired_gap)/.003).square()) * retained
        reward += self.cfg.aperture_reward_weight * aperture_reward
        if self.cfg.mode == 'hold':
            reward = 2 * torch.exp(-m['grip_error'] / .01) + torch.exp(-m['orientation_error'] / .1)
        if self.cfg.require_tripod_support:
            support=m['grasp_support']
            digits=(support['role_contact_strength_N'][:,:3]>self.cfg.grasp_contact_threshold_N).float().mean(-1)
            middle=((support['middle_under_contact_count']>0)&(support['role_support_z_N'][:,2]>self.cfg.support_vertical_threshold_N)).float()
            web=((support['web_contact_count']>0)&((support['web_normal_force_w']+support['web_friction_force_w'])[:,2]>self.cfg.support_vertical_threshold_N)).float()
            support_score=.5*m['grasp_support_ok'].float()+.25*digits+.125*middle+.125*web
            reward += self.cfg.support_reward_weight*support_score*retained
        if self.cfg.require_face_pinch:
            # Thumb/index press the two broad faces; middle and web are free to
            # touch or separate. No reward or success gate requires their contact.
            face_score=(m['face_pinch_inward_force_N']/.03).clamp(0,1).mean(-1)
            reward += self.cfg.support_reward_weight*face_score*retained
        reward -= self.cfg.natural_weight * natural + self.cfg.action_smooth_weight * smooth + .005 * jerk
        ready = self.steps > round(.3 / self.dt)
        tool_lost = ((m['grip_error'] > .055) | (m['orientation_error'] > .6)) & ready
        block_fallen = m['lift'] < -.03
        # Check raw simulator state before observation clipping can hide infinity.
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for value in m['state'].values():
            if torch.is_floating_point(value) and value.shape[0] == self.num_envs:
                invalid |= ~torch.isfinite(value).flatten(1).all(-1)
        terminal = tool_lost | block_fallen | invalid
        reward -= 5 * terminal
        reward = torch.where(invalid, torch.full_like(reward, -5.), reward)
        reward = torch.nan_to_num(reward, nan=-5., posinf=-5., neginf=-5.)
        successful_now = (m['lift'] >= self.cfg.target_lift * .9) & m['both'] & retained & m['orientation_ok'] & m['grasp_support_ok'] & m['face_pinch_ok']
        self.success_steps = torch.where(successful_now, self.success_steps + 1, torch.zeros_like(self.success_steps))
        self.episode_success |= self.success_steps >= round(.5 / self.dt)
        finite_lift = torch.nan_to_num(m['lift'], nan=0., posinf=0., neginf=0.)
        self.episode_max_lift = torch.maximum(self.episode_max_lift, finite_lift)
        self.episode_return += reward
        time_out = (self.steps >= self.max_steps) & ~terminal
        done = terminal | time_out
        self.previous_previous_action.copy_(self.previous_action)
        self.previous_action.copy_(action)
        terminal_obs = self.observe(m)
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.total_episodes += len(ids)
            self.return_history.extend(self.episode_return[ids].tolist())
            self.success_history.extend(self.episode_success[ids].float().tolist())
            self.lift_history.extend(self.episode_max_lift[ids].tolist())
            self.return_history = self.return_history[-1000:]
            self.success_history = self.success_history[-1000:]
            self.lift_history = self.lift_history[-1000:]
        metrics = dict(reach=float(reach.nan_to_num(0.).mean()), bilateral_contact=float(m['both'].float().mean()),
                       retained=float(retained.float().mean()), cube_lift_m=float(finite_lift.mean()),
                       natural_orientation_fraction=float(m['orientation_ok'].float().mean()),
                       tool_down_angle_mean_deg=float(m['tool_down_angle_deg'].nan_to_num().mean()),
                       natural_pose_penalty=float(natural.nan_to_num(0., posinf=0., neginf=0.).mean()), episodes=self.total_episodes,
                       success_rate_1000=float(np.mean(self.success_history)) if self.success_history else 0.,
                       episode_max_lift_mean=float(np.mean(self.lift_history)) if self.lift_history else 0.,
                       episode_return_mean=float(np.mean(self.return_history)) if self.return_history else 0.)
        if self.cfg.require_tripod_support:
            metrics['tripod_support_fraction']=float(m['grasp_support_ok'].float().mean())
        if self.cfg.require_face_pinch:
            metrics['broad_face_pinch_fraction']=float(m['face_pinch_ok'].float().mean())
        if len(ids) and auto_reset:
            self.reset(ids)
        info = dict(time_outs=time_out, terminal_observation=terminal_obs, metrics=metrics)
        if not auto_reset:
            info['measurements'] = m
        return self.observe(), reward, done, info

