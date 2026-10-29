"""Independent nine-action residual PPO task for physically picking up rack tweezers.

Only reset writes tool states. No attachment, artificial wrench or old PPO weights.
"""
from dataclasses import dataclass, asdict
import math
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from rl_task import quat_apply, quat_inv, quat_mul
from reset_reference import StableResetReference, _matrix
from arm_command import ArmCommandKinematics

PHASES = ['pregrasp', 'approach', 'grasp', 'lift_clear', 'transfer_home', 'handoff_check']

@dataclass
class RackPickupConfig:
    decimation: int = 8
    episode_seconds: float = 45.
    phase_timeout_s: float = 12.
    pregrasp_timeout_s: float = 18.
    # Fixed base-relative standby configuration, in scene.kin.arm order.
    initial_arm_q_rad: tuple = (.883515161, -2.023809282, 2.443558301,
                                -.320460430, 1.931510518, 1.020332944)
    pregrasp_distance_scale_m: float = .15
    pregrasp_angle_scale_rad: float = .5
    pregrasp_position_progress_weight: float = 8.
    pregrasp_angle_progress_weight: float = 2.
    pregrasp_arm_speed_rad_s: float = .12
    pregrasp_hand_error_rad: float = .08
    settle_s: float = .5
    approach_offset_m: tuple = (-.065, 0., 0.)
    lift_m: float = .035
    clear_offset_m: tuple = (0., -.09, .035)
    speed_m_s: float = .04
    lift_speed_m_s: float = .015
    angular_speed_rad_s: float = .35
    position_residual_m: float = .004
    rotation_residual_rad: float = .06
    closure_residual: float = .12
    support_residual_rad: float = .035
    hand_open_fraction: float = .35
    grasp_ramp_s: float = 1.5
    face_force_min_N: float = .003
    face_force_max_N: float = 2.
    rack_clear_force_N: float = .003
    collision_failure_N: float = 15.
    handoff_position_m: float = .015
    handoff_angle_deg: float = 8.
    handoff_home_m: float = .012
    handoff_hand_rms_rad: float = .15
    relative_speed_m_s: float = .03
    relative_angular_speed_rad_s: float = .3
    stable_s: float = .6
    aperture_probe_closure: float = .04
    aperture_response_m: float = .0002
    reset_arm_noise_rad: float = 0.
    seed: int = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            values = value if isinstance(value, (tuple, list)) else [value]
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError(name+' must be finite')
        for name in ['episode_seconds','phase_timeout_s','speed_m_s','lift_speed_m_s',
                     'angular_speed_rad_s','grasp_ramp_s','stable_s','handoff_position_m',
                     'handoff_angle_deg','handoff_home_m','relative_speed_m_s','relative_angular_speed_rad_s',
                     'pregrasp_timeout_s','pregrasp_distance_scale_m','pregrasp_angle_scale_rad',
                     'pregrasp_arm_speed_rad_s','pregrasp_hand_error_rad']:
            if getattr(self,name)<=0: raise ValueError(name+' must be positive')
        if not isinstance(self.decimation,int) or self.decimation<1:
            raise ValueError('decimation must be a positive integer')
        if len(self.approach_offset_m)!=3 or len(self.clear_offset_m)!=3:
            raise ValueError('Offsets must be XYZ triples')
        if len(self.initial_arm_q_rad)!=6:
            raise ValueError('initial_arm_q_rad must contain six arm joint angles')
        if self.pregrasp_position_progress_weight<0 or self.pregrasp_angle_progress_weight<0:
            raise ValueError('Pregrasp progress weights must be nonnegative')
        if self.reset_arm_noise_rad<0 or not 0<=self.hand_open_fraction<1:
            raise ValueError('Invalid reset noise or hand opening fraction')
        if self.face_force_min_N<=0 or self.face_force_max_N<=self.face_force_min_N:
            raise ValueError('Invalid contact force interval')
        if self.aperture_response_m<=0 or self.aperture_probe_closure<=0:
            raise ValueError('Aperture check must require a positive response')


def dropped_tool(phase, lift, tool_height, table_height):
    """Regression guard: returning below rack height at HOME is not a drop."""
    return ((phase<=3)&(lift<-.06))|(tool_height<table_height+.04)


def quat_error(target, actual):
    q = quat_mul(target, quat_inv(actual))
    return 2*q[:, 1:]*torch.where(q[:, :1] >= 0, 1., -1.)


def pregrasp_reward_terms(distance_before, distance_after, angle_before, angle_after, cfg):
    """Signed progress is per-transition; error costs are per-second rates.

    Both errors use the same nominal target before phase advancement. Signed
    differences telescope over a closed excursion; stationary states earn no
    progress and all error rates are nonpositive.
    """
    position_progress = cfg.pregrasp_position_progress_weight * (
        distance_before-distance_after)/cfg.pregrasp_distance_scale_m
    angle_progress = cfg.pregrasp_angle_progress_weight * (
        angle_before-angle_after)/cfg.pregrasp_angle_scale_rad
    error_rate = -2*(1-torch.exp(-distance_after/cfg.pregrasp_distance_scale_m))
    error_rate -= 1-torch.exp(-angle_after/cfg.pregrasp_angle_scale_rad)
    return position_progress, angle_progress, error_rate


class RackContacts:
    """Actual PhysX normal resultant magnitudes, max over control substeps."""
    def __init__(self, scene):
        from isaacsim.core.prims import RigidPrim
        from pxr import Usd, UsdPhysics
        self.scene = scene
        self.views = []
        filters = [scene.env_regex+'/Rack/'+p['name'] for p in scene.rack_parts]
        paths = [scene.env_paths[0]+'/Tweezers/'+n for n in ['Handle', 'Left', 'Right']]
        paths += [str(p.GetPath()) for p in Usd.PrimRange(scene.stage.GetPrimAtPath(scene.env_paths[0]+'/Robot'))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        for i, path in enumerate(paths):
            view = RigidPrim(path.replace('/env_0/', '/env_.*/'), name='rack_contact_'+str(i),
                reset_xform_properties=False, track_contact_forces=True,
                contact_filter_prim_paths_expr=filters, max_contact_count=max(1024, 64*scene.num_envs))
            view.initialize()
            ids = torch.tensor([int(p.split('/env_')[1].split('/')[0]) for p in view.prim_paths], device=scene.device)
            if sorted(ids.tolist()) != list(range(scene.num_envs)):
                raise RuntimeError('Invalid rack contact environment ordering')
            self.views.append((view, ids, i < 3))
        self.tool = torch.zeros(scene.num_envs, device=scene.device)
        self.robot = torch.zeros_like(self.tool)
        scene.physics_step_callbacks.append(self.sample)

    def clear(self):
        self.tool.zero_(); self.robot.zero_()

    def sample(self, dt):
        for view, ids, is_tool in self.views:
            force = view.get_contact_force_matrix(dt=dt).norm(dim=-1).sum(-1)
            target = self.tool if is_tool else self.robot
            target[ids] = torch.maximum(target[ids], force)


class RackPickupTask:
    action_size = 9

    def __init__(self, scene, cfg, snapshot, layout):
        self.scene, self.cfg = scene, cfg
        self.device, self.num_envs = scene.device, scene.num_envs
        self.dt = scene.dt*cfg.decimation
        self.max_steps = round(cfg.episode_seconds/self.dt)
        self.rng = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.arm, self.hand = scene.arm_joint_indices, scene.hand_joint_indices
        self.kin = ArmCommandKinematics(scene.kin, self.device)
        self.reference = StableResetReference(snapshot, scene)
        ref = self.reference
        source_hand = ref.wrist_transform.copy()
        palm_fixed = scene.kin.fk(dict(zip(scene.joint_names, ref.q_actual)), root='wrist_3_link')[scene.palm_link]
        source_hand = source_hand @ palm_fixed
        relative = np.linalg.inv(source_hand) @ ref.body_transforms['handle']
        self.relative_position = self.tensor(relative[:3,3])
        self.relative_quat = self.quat(relative[:3,:3])
        rack_R = scene.tool_initial_T[:3,:3]
        root = np.asarray(scene.rack_root_xyz)
        grasp_tool = np.eye(4); grasp_tool[:3,:3] = rack_R; grasp_tool[:3,3] = root
        grasp_wrist = grasp_tool @ np.linalg.inv(ref.wrist_to_tool)
        home_R = Rotation.from_euler('z', layout['tool_yaw_deg'], degrees=True).as_matrix() @ ref.body_transforms['handle'][:3,:3]
        home_tip = [*layout['home_xy_m'], scene.table_height+.0078+layout.get('initial_tip_clearance_m', .075)]
        home = ref.build(home_tip, env_origins=np.zeros((1,3)), position_frame='local', desired_tip_orientation=home_R)
        self.home_tool_position = home['object_states']['handle'][0,:3]
        self.home_tool_quat = home['object_states']['handle'][0,3:7]
        self.home_wrist_position = home['wrist_pose_w'][0,:3]
        self.home_wrist_quat = home['wrist_pose_w'][0,3:7]
        self.grasp_position = self.tensor(grasp_wrist[:3,3])
        self.grasp_quat = self.quat(grasp_wrist[:3,:3])
        self.pre_position = self.grasp_position+self.tensor(cfg.approach_offset_m)
        self.lift_position = self.grasp_position+self.tensor([0,0,cfg.lift_m])
        self.clear_position = self.grasp_position+self.tensor(cfg.clear_offset_m)
        # Match the original demo's nominal open closure (.7), not arbitrary q_reference.
        ref_hand = self.tensor([scene.pose_config['q_reference'][n] for n in scene.kin.hand])
        self.closure = self.tensor([scene.pose_config['closure_delta'][n] for n in scene.kin.hand])
        self.hold_hand = ref_hand+.7*self.closure
        limits = scene.robot.data.soft_joint_pos_limits[0,self.hand]
        self.hand_low, self.hand_high = limits[:,0], limits[:,1]
        self.open_hand = (self.hold_hand*cfg.hand_open_fraction).clamp(self.hand_low,self.hand_high)
        fixed_arm = self.tensor(cfg.initial_arm_q_rad)
        if bool(((fixed_arm<self.kin.lo)|(fixed_arm>self.kin.hi)).any()):
            raise ValueError('Fixed initial arm angles exceed URDF joint limits')
        pre = grasp_wrist.copy(); pre[:3,3] = self.pre_position.cpu().numpy()-[0,0,scene.table_height]
        from arm_clearance import ArmTableClearance
        # Reachability check only: never use target IK to choose the reset pose.
        fixed_seed = fixed_arm.cpu().numpy()
        ik = ArmTableClearance(scene.kin).inverse(pre, fixed_seed,
            min_clearance=.015, preferred_seed=fixed_seed,
            hand_joint_positions=dict(zip(scene.kin.hand,self.open_hand.cpu().tolist())))
        if ik['error'] > 1e-4:
            raise RuntimeError('Pregrasp IK infeasible: '+str(ik))
        self.initial_q = scene.default_joint_pos.clone()
        self.initial_q[:,self.arm] = fixed_arm
        self.initial_q[:,self.hand] = self.open_hand
        self.q_target = self.initial_q.clone()
        self.contacts = RackContacts(scene)
        n, d = self.num_envs, self.device
        self.phase = torch.zeros(n,dtype=torch.long,device=d)
        self.steps = self.phase.clone(); self.phase_steps = self.phase.clone(); self.ready_steps = self.phase.clone()
        self.max_phase = self.phase.clone()
        self.clear_reached = torch.zeros(n,dtype=torch.bool,device=d)
        self.episode_success = self.clear_reached.clone(); self.ever_detached = self.clear_reached.clone()
        self.episode_return = torch.zeros(n,device=d)
        self.previous_action = torch.zeros((n,9),device=d)
        self.last_relative_pos = torch.zeros((n,3),device=d)
        self.last_relative_quat = torch.zeros((n,4),device=d)
        self.gap_open = torch.zeros(n,device=d); self.gap_closed = torch.zeros(n,device=d)
        self.previous_probe_error = torch.zeros(n,device=d)
        self.success_history=[]; self.return_history=[]; self.total_episodes=0
        self.reset()
        self.observation_size = self.observe().shape[-1]

    def tensor(self, x):
        return torch.as_tensor(x,device=self.device,dtype=torch.float32)

    def quat(self, R):
        return self.tensor(np.roll(Rotation.from_matrix(R).as_quat(),1))

    def metadata(self):
        sources=['rack_pickup_task.py','tweezer_rack_scene.py','rl_scene.py','reset_reference.py','grasp_contacts.py','arm_command.py','kinematics.py','arm_clearance.py','rl_task.py']
        signature={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in sources}
        signature['scene_pose']=hashlib.sha256(json.dumps(self.scene.pose_config,sort_keys=True).encode()).hexdigest()
        initial_T, _ = self.kin.fk_jacobian(self.initial_q[:1,self.arm])
        return dict(task_kind='rack_pickup_fixed_start_v2', implementation_signature=signature,
            initial_arm_joint_names=list(self.scene.kin.arm),
            initial_joint_targets=self.initial_q[0].tolist(),
            initial_wrist_position=(initial_T[0,:3,3]+self.tensor([0,0,self.scene.table_height])).tolist(),
            pregrasp_wrist_position=self.pre_position.tolist(),
            pregrasp_wrist_quaternion=self.grasp_quat.tolist(),
            rack_parts=self.scene.rack_parts,
            block_initial_position=self.scene.block_initial_T[:3,3].tolist(),
            config=asdict(self.cfg), phases=PHASES,
            physics_dt=self.scene.dt, control_dt=self.dt, joint_names=self.scene.joint_names,
            action_semantics=['wrist_target_dx','wrist_target_dy','wrist_target_dz',
                'wrist_target_rx','wrist_target_ry','wrist_target_rz','closure','index_support','middle_support'],
            home_tool_position=self.home_tool_position.tolist(), home_tool_quaternion=self.home_tool_quat.tolist(),
            reference_hand_tool_position=self.relative_position.tolist(), reference_hand_tool_quaternion=self.relative_quat.tolist(),
            no_task_time_object_state_writes=True, pretrained_policy_used=False,
            handoff_scope='Candidate state only; original PPO is not executed; tolerances are provisional',
            reward_semantics='Continuous terms are rates multiplied by control_dt; signed pregrasp error reductions are per-transition; event terms occur once')

    def reset(self, ids=None):
        ids = self.scene._all_ids if ids is None else ids
        q = self.initial_q[ids].clone()
        q[:,self.arm] += (torch.rand((len(ids),6),device=self.device,generator=self.rng)*2-1)*self.cfg.reset_arm_noise_rad
        objects = {name: state[ids].clone() for name,state in self.scene.default_object_states.items()}
        # Constructor spawned all three tool bodies at the actual rack pose.
        self.scene.reset(ids,joint_pos=q,object_states=objects)
        self.q_target[ids] = q
        for value in [self.phase,self.steps,self.phase_steps,self.ready_steps,self.max_phase,self.clear_reached,
                      self.episode_success,self.ever_detached,self.episode_return,self.previous_action,
                      self.gap_open,self.gap_closed,self.previous_probe_error]:
            value[ids]=0
        self.contacts.tool[ids]=0; self.contacts.robot[ids]=0
        m=self.measurements()
        self.last_relative_pos[ids]=m['rel_pos'][ids]; self.last_relative_quat[ids]=m['rel_quat'][ids]
        return self.observe(m)

    def targets(self):
        n=self.num_envs
        p=self.pre_position.expand(n,3).clone(); q=self.grasp_quat.expand(n,4).clone()
        p[(self.phase==1)|(self.phase==2)]=self.grasp_position
        p[self.phase==3]=self.lift_position
        moving=self.phase==4
        p[moving]=self.clear_position
        home=(moving&self.clear_reached)|(self.phase==5)
        p[home]=self.home_wrist_position; q[home]=self.home_wrist_quat
        return p+self.scene.env_origins,q

    def measurements(self):
        s=self.scene.get_state(); tool=s['tool_root_state_w']
        inv=quat_inv(s['palm_quat_w'])
        rp=quat_apply(inv,tool[:,:3]-s['palm_pos_w'])
        rq=quat_mul(inv,tool[:,3:7])
        pos=(rp-self.relative_position).norm(dim=-1)
        angle=2*torch.acos((rq*self.relative_quat).sum(-1).abs().clamp(0,1))
        support=self.scene.read_grasp_support(s)
        assignment=self.scene.pose_config.get('face_pinch_assignment',{'thumb':'right','index':'left'})
        raw=support['broad_face_inward_force_N']
        forces=torch.stack([raw[:,['left','right'].index(assignment[role]),i] for i,role in enumerate(['thumb','index'])],-1)
        face=(forces>self.cfg.face_force_min_N).all(-1)
        tips=[]
        for name in ['left','right']:
            body=s[name+'_root_state_w']
            tips.append(body[:,:3]+quat_apply(body[:,3:7],self.tensor([0,0,self.scene.tip_local_z]).expand(self.num_envs,3)))
        gap=(tips[0]-tips[1]).norm(dim=-1)
        local=tool[:,:3]-self.scene.env_origins
        target_p,target_q=self.targets()
        rel_speed=(rp-self.last_relative_pos).norm(dim=-1)/self.dt
        rel_omega=quat_error(rq,self.last_relative_quat).norm(dim=-1)/self.dt
        return dict(state=s,rel_pos=rp,rel_quat=rq,grip_error=pos,grip_angle=angle,forces=forces,face=face,gap=gap,
            relative_speed=rel_speed,relative_angular_speed=rel_omega,
            tool_local=local,lift=local[:,2]-self.scene.rack_root_xyz[2],
            target_distance=(target_p-s['wrist_pos_w']).norm(dim=-1),
            target_angle=quat_error(target_q,s['wrist_quat_w']).norm(dim=-1),
            home_error=(local-self.home_tool_position).norm(dim=-1),
            home_angle=quat_error(self.home_tool_quat.expand(self.num_envs,4),tool[:,3:7]).norm(dim=-1),
            hand_error=(s['joint_pos'][:,self.hand]-self.hold_hand).square().mean(-1).sqrt())

    def observe(self,m=None):
        m=self.measurements() if m is None else m; s=m['state']; p,q=self.targets()
        obs=torch.cat((s['joint_pos']*.5,s['joint_vel']*.1,
            (p-s['wrist_pos_w'])*10,quat_error(q,s['wrist_quat_w']),
            m['rel_pos']*10,m['rel_quat'],m['tool_local'],s['tool_root_state_w'][:,7:13]*.1,
            m['forces'].clamp(0,5),self.contacts.tool[:,None].clamp(0,5),self.contacts.robot[:,None].clamp(0,5),
            m['gap'][:,None]*30,m['lift'][:,None]*10,
            torch.nn.functional.one_hot(self.phase,6).float(),
            (self.phase_steps*self.dt)[:,None]*.1,self.clear_reached[:,None].float(),
            self.ever_detached[:,None].float(),self.gap_open[:,None]*30,self.gap_closed[:,None]*30,
            self.previous_action),-1)
        return obs.nan_to_num(0,posinf=20,neginf=-20).clamp(-20,20)

    def step(self,action,auto_reset=True,render=False):
        c=self.cfg; action=action.clamp(-1,1); old=self.phase.clone()
        m0=self.measurements(); s=m0['state']; p,q=self.targets()
        p=p+action[:,:3]*c.position_residual_m
        rot=action[:,3:6]*c.rotation_residual_rad
        theta=rot.norm(dim=-1,keepdim=True)
        dq=torch.cat((torch.cos(theta/2),rot*torch.sinc(theta/(2*math.pi))*.5),-1)
        q=quat_mul(dq,q)
        from isaaclab.utils.math import quat_from_matrix
        command_T,J=self.kin.fk_jacobian(self.q_target[:,self.arm])
        command_p=command_T[:,:3,3]+self.scene.env_origins+self.tensor([0,0,self.scene.table_height])
        command_q=quat_from_matrix(command_T[:,:3,:3])
        linear=(p-command_p)*.5
        speed=torch.full((self.num_envs,),c.speed_m_s,device=self.device)
        speed[self.phase==3]=c.lift_speed_m_s
        linear*=torch.minimum(torch.ones_like(speed),speed*self.dt/linear.norm(dim=-1).clamp_min(1e-8))[:,None]
        angular=quat_error(q,command_q)*.5
        angular*=torch.minimum(torch.ones_like(speed),c.angular_speed_rad_s*self.dt/angular.norm(dim=-1).clamp_min(1e-8))[:,None]
        error=torch.cat((linear,angular),-1)
        delta=(J.transpose(-1,-2)@torch.linalg.solve(J@J.transpose(-1,-2)+.02**2*torch.eye(6,device=self.device),error[:,:,None])).squeeze(-1)
        self.q_target[:,self.arm]=(self.q_target[:,self.arm]+delta.clamp(-.025,.025)).clamp(self.kin.lo,self.kin.hi)
        t=self.phase_steps*self.dt
        blend=torch.where(self.phase<2,0.,torch.where(self.phase==2,(t/c.grasp_ramp_s).clamp(0,1),1.))
        hand=self.open_hand+(self.hold_hand-self.open_hand)*blend[:,None]
        probe=torch.zeros_like(t)
        probe[(self.phase==5)&(t<.5)]=-c.aperture_probe_closure
        probe[(self.phase==5)&(t>=.5)&(t<1.)]=c.aperture_probe_closure
        hand_action=action[:,6:].clone()
        hand_action[(self.phase==5)&(t<1.)]=0  # calibrated probe cannot be cancelled by residuals
        hand+=(hand_action[:,0]*c.closure_residual+probe)[:,None]*self.closure
        joints=self.scene.pose_config.get('residual_joints',{'index':'right_hand_index_joint1','middle':'right_hand_mid_joint2'})
        for i,key in enumerate(['index','middle']):
            hand[:,self.scene.kin.hand.index(joints[key])]+=hand_action[:,1+i]*c.support_residual_rad
        self.q_target[:,self.hand]+=.15*(hand.clamp(self.hand_low,self.hand_high)-self.q_target[:,self.hand])
        settling=(self.phase==0)&(t<c.settle_s)
        self.q_target[settling]=self.initial_q[settling]
        self.contacts.clear()
        self.scene.step(self.q_target,decimation=c.decimation,render=render)
        self.steps+=1; self.phase_steps+=1
        m=self.measurements(); t=self.phase_steps*self.dt
        self.clear_reached|=(self.phase==4)&((m['state']['wrist_pos_w']-self.scene.env_origins-self.clear_position).norm(dim=-1)<.008)
        grip=m['face']&(m['grip_error']<c.handoff_position_m)&(m['grip_angle']<math.radians(c.handoff_angle_deg))
        steady=(m['relative_speed']<c.relative_speed_m_s)&(m['relative_angular_speed']<c.relative_angular_speed_rad_s)
        detached=(m['lift']>.02)&(self.contacts.tool<c.rack_clear_force_N)&grip
        first_detached=detached&~self.ever_detached&(self.phase>=3)
        self.ever_detached|=first_detached
        probe_open=(self.phase==5)&(t>.3)&(t<=.5)
        probe_closed=(self.phase==5)&(t>.8)&(t<=1.)
        self.gap_open[probe_open]=m['gap'][probe_open]
        self.gap_closed[probe_closed]=m['gap'][probe_closed]
        aperture_ok=(self.gap_open-self.gap_closed)>c.aperture_response_m
        home=(m['home_error']<c.handoff_home_m)&(m['home_angle']<math.radians(c.handoff_angle_deg))
        handoff=home&grip&steady&(m['hand_error']<c.handoff_hand_rms_rad)&(self.contacts.tool<c.rack_clear_force_N)&self.ever_detached
        ready=torch.zeros_like(grip)
        arm_speed=m['state']['joint_vel'][:,self.arm].square().mean(-1).sqrt()
        open_error=(m['state']['joint_pos'][:,self.hand]-self.open_hand).square().mean(-1).sqrt()
        ready|=(self.phase==0)&(t>c.settle_s)&(m['target_distance']<.008)&(m['target_angle']<.1)&(arm_speed<c.pregrasp_arm_speed_rad_s)&(open_error<c.pregrasp_hand_error_rad)
        ready|=(self.phase==1)&(m['target_distance']<.006)&(m['target_angle']<.1)
        ready|=(self.phase==2)&(t>c.grasp_ramp_s)&grip&steady
        ready|=(self.phase==3)&detached
        ready|=(self.phase==4)&self.clear_reached&home&grip
        ready|=(self.phase==5)&(t>1.)&handoff&aperture_ok
        self.ready_steps=torch.where(ready,self.ready_steps+1,torch.zeros_like(self.ready_steps))
        needed=torch.where(self.phase==5,math.ceil(c.stable_s/self.dt),math.ceil(.15/self.dt))
        advance=(self.ready_steps>=needed)&(self.phase<5)
        success=(self.ready_steps>=needed)&(self.phase==5)
        physical=torch.cat([m['state'][k] for k in ['joint_pos','joint_vel','tool_root_state_w','left_root_state_w','right_root_state_w']],-1)
        invalid=~torch.isfinite(physical).all(-1)
        dropped=dropped_tool(self.phase,m['lift'],m['tool_local'][:,2],self.scene.table_height)
        displaced=(self.phase<3)&((m['tool_local'][:,:2]-self.tensor(self.scene.rack_root_xyz[:2])).norm(dim=-1)>.04)
        lost=(self.phase>=3)&(m['grip_error']>.06)
        collision=self.contacts.robot>c.collision_failure_N
        # Robot body origins under the table are a conservative emergency stop,
        # not a substitute for full mesh clearance validation.
        below=(m['state']['body_pos_w'][:,:,2]-self.scene.env_origins[:,None,2]<self.scene.table_height-.01).any(-1)
        phase_timeout=t>torch.where(old==0,c.pregrasp_timeout_s,c.phase_timeout_s)
        failure=invalid|dropped|displaced|lost|collision|below
        advance&=~failure&~phase_timeout
        success&=~failure&~phase_timeout
        timeout=(self.steps>=self.max_steps)&~failure&~success&~phase_timeout
        done=failure|success|phase_timeout|timeout
        quality=torch.exp(-(m['grip_error']/.012).square()-(m['grip_angle']/.2).square())
        stable_score=torch.exp(-(m['relative_speed']/.04).square()-(m['relative_angular_speed']/.5).square())
        proximity=torch.exp(-(m['target_distance']/.03).square())
        alignment=torch.exp(-(m['target_angle']/.25).square())
        contact=(m['forces']/ .03).clamp(0,1).mean(-1)
        phase_rate=torch.where(old<2,2*proximity+alignment,
            torch.where(old==2,2*contact+2*quality+stable_score,
            torch.where(old==3,3*(m['lift']/c.lift_m).clamp(0,1)*grip.float()+2*quality+stable_score,
            torch.where(old==4,2*proximity+2*quality+stable_score,2*quality+2*stable_score+aperture_ok.float()))))
        pos_progress,rot_progress,pre_error_rate=pregrasp_reward_terms(
            m0['target_distance'],m['target_distance'],m0['target_angle'],m['target_angle'],c)
        pre_active=(old==0)&~settling
        pos_progress=torch.where(pre_active,pos_progress,0.)
        rot_progress=torch.where(pre_active,rot_progress,0.)
        # Keep fingers open in transit and slow down near the arrival region.
        open_rate=.5*(open_error/c.pregrasp_hand_error_rad).square().clamp(max=4)
        arrival_speed_rate=.25*torch.exp(-(m['target_distance']/.03).square())*(
            arm_speed/c.pregrasp_arm_speed_rad_s).square().clamp(max=4)
        pre_rate=pre_error_rate-open_rate-arrival_speed_rate
        phase_rate=torch.where(old==0,pre_rate,phase_rate)
        push=2*((m['tool_local']-self.tensor(self.scene.rack_root_xyz)).norm(dim=-1)/.02).clamp(0,5)*(old<2)
        force_penalty=(m['forces']-c.face_force_max_N).clamp_min(0).sum(-1)
        rack_penalty=self.contacts.robot.clamp(0,20)+((old>=3)&(t>1.)).float()*self.contacts.tool.clamp(0,5)
        action_penalty=.02*action.square().mean(-1)+.05*(action-self.previous_action).square().mean(-1)
        continuous=(phase_rate-push-force_penalty-rack_penalty-action_penalty-.2)*self.dt
        events=5*advance.float()+10*first_detached.float()+30*success.float()-15*failure.float()-5*(phase_timeout|timeout).float()
        reward=(continuous+events+pos_progress+rot_progress).nan_to_num(-15,posinf=-15,neginf=-15)
        self.phase[advance]+=1; self.phase_steps[advance]=0; self.ready_steps[advance]=0
        self.max_phase=torch.maximum(self.max_phase,self.phase)
        self.episode_success|=success; self.episode_return+=reward
        self.previous_action.copy_(action); self.last_relative_pos.copy_(m['rel_pos']); self.last_relative_quat.copy_(m['rel_quat'])
        terminal_obs=self.observe(m)
        info=dict(time_outs=timeout, terminal_observation=terminal_obs, success=success.clone(),
            failure=failure,phase_timeout=phase_timeout,measurements=m,
            failure_causes=dict(invalid=invalid,dropped=dropped,displaced=displaced,lost=lost,collision=collision,below_table=below),
            reward_terms=dict(phase_rate=phase_rate,continuous=continuous,events=events,collision_rate=rack_penalty,
                              pregrasp_position_progress=pos_progress,pregrasp_angle_progress=rot_progress,
                              pregrasp_error_rate=torch.where(old==0,pre_error_rate,0.),
                              pregrasp_open_rate=torch.where(old==0,open_rate,0.),
                              pregrasp_arrival_speed_rate=torch.where(old==0,arrival_speed_rate,0.)),
            metrics=dict(phase_mean=float(self.phase.float().mean()),grip_error_m=float(m['grip_error'].mean()),
                         target_distance_m=float(m['target_distance'].mean()),target_angle_error=float(m['target_angle'].mean()),
                         face_contact_fraction=float(m['face'].float().mean()),rack_robot_force_N=float(self.contacts.robot.mean())))
        ids=done.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.total_episodes+=len(ids)
            self.success_history.extend(success[ids].float().tolist());self.return_history.extend(self.episode_return[ids].tolist())
            self.success_history=self.success_history[-100:];self.return_history=self.return_history[-100:]
        info['metrics'].update(episodes=self.total_episodes,success_rate_100=float(np.mean(self.success_history)) if self.success_history else 0,
                               episode_return_mean=float(np.mean(self.return_history)) if self.return_history else 0)
        for key,value in info['reward_terms'].items():
            info['metrics']['reward_'+key]=float(value.mean())
        for key,value in info['failure_causes'].items():
            info['metrics']['failure_fraction_'+key]=float(value.float().mean())
        for i,name in enumerate(PHASES):
            info['metrics']['phase_fraction_'+name]=float((old==i).float().mean())
        if auto_reset and len(ids): self.reset(ids)
        return self.observe(),reward,done,info
