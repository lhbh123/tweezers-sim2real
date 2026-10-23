"""Goal-conditioned residual-RL pick/transfer/place task.

An explicit geometric phase controller supplies nominal robot commands. PPO
learns bounded corrections from physical rollouts. This is intentionally a
hybrid controller, not claimed to be end-to-end learned. Rigid-body states are
written only inside reset(). No attachments or virtual grasp forces are used.
"""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from rl_task import TweezerTask,TaskConfig,quat_apply

PHASES=['settle','approach','descend','squeeze','lift','transfer','lower','release','withdraw','settle_at_goal']

def squeeze_readiness(forces,phase,phase_time_s,previous_steps,dt,
                       minimum_jaw_force_N=0.,force_hold_seconds=.25):
    """Update a measured bilateral-force streak and the squeeze exit gate.

    Forces are the actual block/jaw normal-resultant magnitudes from
    measurements()['forces']. Zero threshold preserves the old timed gate.
    This gate does not alter joint targets or increase grip force.
    """
    timed=(phase==3)&(phase_time_s>=1.)
    if minimum_jaw_force_N<=0.:
        return torch.zeros_like(previous_steps),timed
    sufficient=(phase==3)&torch.isfinite(forces).all(-1)&(forces>=minimum_jaw_force_N).all(-1)
    steps=torch.where(sufficient,previous_steps+1,torch.zeros_like(previous_steps))
    return steps,timed&sufficient&(steps*dt>=force_hold_seconds)

@dataclass
class PickPlaceConfig:
    xy_lower:tuple=(.48,-.22)
    xy_upper:tuple=(.76,.08)
    min_transfer_m:float=.12
    travel_height_m:float=.080
    goal_tolerance_m:float=.010
    stable_seconds:float=.75
    max_speed_m_s:float=.060
    residual_xyz_m:float=.00012
    residual_closure:float=.035
    residual_index_rad:float=.002
    residual_middle_rad:float=.006
    retreat_closure:float=.7
    open_closure:float=.7
    closed_closure:float=1.05
    squeeze_min_jaw_force_N:float=0.
    squeeze_force_hold_seconds:float=.25
    release_extra_opening_max:float=0.
    release_extra_opening_rate:float=.3
    release_contact_clear_seconds:float=0.
    withdraw_initial_speed_m_s:float|None=None
    release_middle_joint_deltas:dict|None=None
    release_middle_ramp_seconds:float=.5
    release_middle_delay_seconds:float=.8
    release_track_tip:bool=False

class PickPlaceTask(TweezerTask):
    def __init__(self,scene,cfg,place_cfg=None):
        self.place_cfg=place_cfg or PickPlaceConfig()
        for name in ('squeeze_min_jaw_force_N','squeeze_force_hold_seconds'):
            value=float(getattr(self.place_cfg,name))
            if not np.isfinite(value) or value<0.:
                raise ValueError(name+' must be finite and nonnegative')
        d=scene.device;n=scene.num_envs
        self.goal=torch.zeros((n,3),device=d)
        self.start=self.goal.clone();self.phase=torch.zeros(n,dtype=torch.long,device=d)
        self.phase_steps=self.phase.clone();self.phase_ready=self.phase.clone()
        self.grasped_once=torch.zeros(n,dtype=torch.bool,device=d)
        self.max_phase=self.phase.clone();self.lift_streak=self.phase.clone()
        self.squeeze_force_steps=self.phase.clone()
        self.squeeze_force_hold_time_s=torch.zeros(n,device=d)
        self.squeeze_force_gate_ready=torch.zeros(n,dtype=torch.bool,device=d)
        self.last_delta=torch.zeros((n,3),device=d)
        self.residual_action=torch.zeros((n,6),device=d)
        self.nominal_action=self.residual_action.clone()
        self.target_tip=self.goal.clone()
        self.block_tip_offset=torch.zeros((n,3),device=d)
        self.block_tip_offset[:,2]=-.0015
        self.reset_override=None
        self.release_extra_opening=torch.zeros(n,device=d)
        self.release_middle_blend=torch.zeros(n,device=d)
        middle_deltas=self.place_cfg.release_middle_joint_deltas or {}
        if any(name not in scene.kin.hand or 'middle' not in name for name in middle_deltas):
            raise ValueError('Release support offsets must name existing middle-finger joints')
        if self.place_cfg.release_middle_ramp_seconds<=0:
            raise ValueError('Middle release ramp duration must be positive')
        if self.place_cfg.release_middle_delay_seconds<0:
            raise ValueError('Middle release delay cannot be negative')
        self.release_middle_offsets=torch.tensor(
            [middle_deltas.get(name,0.) for name in scene.kin.hand],device=d)
        super().__init__(scene,cfg)
        self.observation_size=self.observe().shape[-1]

    def reset(self,ids=None):
        ids=self.scene._all_ids if ids is None else ids;n=len(ids)
        choice=torch.zeros(n,device=self.device,dtype=torch.long) # fixed HOME, independent of start and goal
        q=self.bank_q[choice];target=self.bank_q_target[choice]
        low=torch.tensor(self.place_cfg.xy_lower,device=self.device);high=torch.tensor(self.place_cfg.xy_upper,device=self.device)
        start=low+(high-low)*torch.rand((n,2),generator=self.rng,device=self.device)
        goal=low+(high-low)*torch.rand((n,2),generator=self.rng,device=self.device)
        for _ in range(100):
            bad=(goal-start).norm(dim=-1)<self.place_cfg.min_transfer_m
            if not bool(bad.any()):break
            goal[bad]=low+(high-low)*torch.rand((int(bad.sum()),2),generator=self.rng,device=self.device)
        if bool(((goal-start).norm(dim=-1)<self.place_cfg.min_transfer_m).any()):raise RuntimeError('Failed distinct goal sampling')
        if self.reset_override is not None:
            start=torch.as_tensor(self.reset_override['start'],device=self.device,dtype=q.dtype)[ids]
            goal=torch.as_tensor(self.reset_override['goal'],device=self.device,dtype=q.dtype)[ids]
        objects={}
        for name,state in self.scene.default_object_states.items():
            if name=='block':
                s=state[ids].clone();s[:,:2]=start+self.scene.env_origins[ids,:2]
            else:
                s=self.bank_object_states[name][choice].clone();s[:,:3]+=self.scene.env_origins[ids]
            objects[name]=s
        self.scene.reset(ids,joint_pos=q,object_states=objects)
        self.q_target[ids]=target;self.hand_target[ids]=target[:,self.hand]
        self.arm_command_reference[ids]=self.arm_command.fk_jacobian(target[:,self.arm])[0]
        self.scene.robot.set_joint_position_target(target,env_ids=ids);self.scene.robot.write_data_to_sim()
        self.start[ids]=objects['block'][:,:3];self.goal[ids]=self.start[ids];self.goal[ids,:2]=goal+self.scene.env_origins[ids,:2]
        for v in [self.previous_action,self.previous_previous_action,self.last_delta,self.residual_action,self.nominal_action]:v[ids]=0
        for v in [self.steps,self.phase,self.phase_steps,self.phase_ready,self.success_steps,self.episode_success,self.episode_return,self.episode_max_lift,self.grasped_once,self.max_phase,self.lift_streak]:v[ids]=0
        self.block_tip_offset[ids]=torch.tensor([0,0,-.0015],device=self.device)
        self.release_extra_opening[ids]=0
        self.release_middle_blend[ids]=0
        self.squeeze_force_steps[ids]=0
        self.squeeze_force_hold_time_s[ids]=0
        self.squeeze_force_gate_ready[ids]=False
        self.target_tip[ids]=self.measurements()['tip'][ids]
        return self.observe()

    def metadata(self):
        return {**super().metadata(),'task_kind':'randomized_pick_transfer_place_v15','controller':'geometric_phase_controller_plus_PPO_residual',
            'place_config':asdict(self.place_cfg),'phases':PHASES,'initial_tool':'preheld at fixed HOME independent of randomized cube and goal',
            'randomization':'continuous independent uniform XY, rejection only for minimum transfer distance',
            'success_definition':{'lifted_with_bilateral_contact_m':.045,'continuous_bilateral_lift_s':.25,'final_xy_error_m':self.place_cfg.goal_tolerance_m,
            'table_height_error_m':.0015,'linear_speed_m_s':.01,'angular_speed_rad_s':.5,'tool_tip_above_block_m':.035,
            'no_jaw_contact_N_max':.003,'continuous_stability_s':self.place_cfg.stable_seconds,'tool_must_remain_retained':True},
            'action_semantics':['residual_wrist_dx','residual_wrist_dy','residual_wrist_dz','residual_closure','residual_index','residual_middle'],
            'object_runtime_writes':False,'middle_web_contact_required':False,
            'squeeze_gate_force_source':'Actual block_jaw_contact_forces_w vector norms, not tactile grayscale or finger/tool force.',
            'squeeze_gate_semantics':'Threshold 0 retains timed >=1 s gate. Positive threshold additionally requires both measured jaw forces >= threshold continuously for the configured duration; any deficient/nonfinite sample resets the streak. Existing three-ready-sample phase debounce and 12 s phase timeout remain.'}

    def observe(self,m=None):
        m=self.measurements() if m is None else m
        base=super().observe(m)
        block=m['state']['block_root_state_w']
        phase=torch.nn.functional.one_hot(self.phase,len(PHASES)).float()
        extra=torch.cat(((self.goal-block[:,:3])*5,(self.goal-m['tip'])*5,(self.start-self.scene.env_origins)*2,
             phase,(self.phase_steps*self.dt).clamp(0,10)[:,None]*.2,self.residual_action,self.grasped_once[:,None].float()),-1)
        return torch.cat((base,extra),-1).clamp(-20,20)

    def _nominal(self,m):
        p=self.phase;t=self.phase_steps*self.dt;block=m['state']['block_root_state_w'][:,:3]
        floor=self.scene.env_origins[:,2]+self.scene.table_height
        tip=m['tip'];target=tip.clone()
        opened=self.place_cfg.open_closure;closed=self.place_cfg.closed_closure
        # Only open farther when real jaw contact remains after the nominal
        # release. Keep that opening through the first part of withdrawal.
        touching=m['forces'].amax(-1)>=.003
        opening=(p==7)&(t>=1.)&touching
        self.release_extra_opening[opening]=(self.release_extra_opening[opening]
            +self.place_cfg.release_extra_opening_rate*self.dt).clamp(max=self.place_cfg.release_extra_opening_max)
        released=opened-self.release_extra_opening
        closure=torch.full_like(t,opened)
        travel=floor+self.place_cfg.travel_height_m+.0061
        # Stage-local closed-loop targets. Source position is measured continuously.
        mask=(p==1)|(p==2)|(p==3)|(p==4)
        target[mask,:2]=block[mask,:2]
        target[p==1,2]=travel[p==1]
        target[(p==2)|(p==3),2]=block[(p==2)|(p==3),2]+.0015
        target[p==4,2]=travel[p==4]
        closure[p==3]=opened+(closed-opened)*(t[p==3]/.65).clamp(0,1)
        closure[(p>=4)&(p<=6)]=closed
        target[p==5,:2]=self.goal[p==5,:2]-self.block_tip_offset[p==5,:2]
        target[p==5,2]=travel[p==5]
        target[(p==6)|(p==7),:2]=self.goal[(p==6)|(p==7),:2]-self.block_tip_offset[(p==6)|(p==7),:2]
        target[(p==6)|(p==7),2]=floor[(p==6)|(p==7)]+.0076
        closure[p==7]=closed-(closed-opened)*(t[p==7]/.8).clamp(0,1)-self.release_extra_opening[p==7]
        target[p>=8,:2]=self.goal[p>=8,:2]-self.block_tip_offset[p>=8,:2]
        target[p>=8,2]=travel[p>=8]
        # Re-establish a secure empty-tool grip only AFTER tips clear the placed cube.
        clearance=(tip[:,2]-block[:,2]-.020)/.035
        closure[p>=8]=released[p>=8]+(self.place_cfg.retreat_closure-released[p>=8])*clearance[p>=8].clamp(0,1)
        # Optional W01 support withdrawal: move the middle finger through its
        # actuators while the jaws release. Restore it only as the tips clear
        # the block, using the same measured clearance as the empty-tool grip.
        # An empty offset mapping preserves the previous controller behavior.
        self.release_middle_blend.zero_()
        if self.place_cfg.release_middle_joint_deltas:
            self.release_middle_blend[p==7]=((t[p==7]-self.place_cfg.release_middle_delay_seconds)
                /self.place_cfg.release_middle_ramp_seconds).clamp(0,1)
            self.release_middle_blend[p>=8]=1-clearance[p>=8].clamp(0,1)
        # Component and norm limits avoid a faster diagonal path. Smooth acceleration.
        error=target-tip
        speed=torch.full_like(t,self.place_cfg.max_speed_m_s)
        speed[(p==2)|(p==6)]=.025
        if self.place_cfg.withdraw_initial_speed_m_s is not None:
            speed[(p==8)&((tip[:,2]-block[:,2])<.020)]=self.place_cfg.withdraw_initial_speed_m_s
        delta=error*.5
        limit=speed*self.dt
        delta*=torch.minimum(torch.ones_like(limit),limit/delta.norm(dim=-1).clamp_min(1e-8))[:,None]
        delta=delta.maximum(self.last_delta-.00008).minimum(self.last_delta+.00008)
        stationary=(p==0)|(p==3)|(p==9)
        if not self.place_cfg.release_track_tip:
            stationary|=p==7
        delta[stationary]=0
        self.target_tip=target
        action=torch.zeros((self.num_envs,6),device=self.device)
        action[:,:3]=delta/self.cfg.max_wrist_step
        action[:,3]=(closure-self.cfg.closure_bias)/self.cfg.closure_scale
        action[:,5]=-.2*((p>=3)&(p<=7)).float()
        return action

    def step(self,action,auto_reset=True,render=False):
        action=action.clamp(-1,1);old_phase=self.phase.clone();m0=self.measurements()
        nominal=self._nominal(m0);self.nominal_action=nominal.clone();self.residual_action=action.clone()
        applied=nominal.clone()
        scale=torch.tensor([self.place_cfg.residual_xyz_m/self.cfg.max_wrist_step]*3+
            [self.place_cfg.residual_closure/self.cfg.closure_scale,self.place_cfg.residual_index_rad/self.cfg.index_support_action_rad,
             self.place_cfg.residual_middle_rad/self.cfg.middle_support_action_rad],device=self.device)
        applied=(applied+scale*action).clamp(-1,1)
        applied[self.phase==0,:3]=0
        delta=applied[:,:3]*self.cfg.max_wrist_step
        self.last_delta=delta
        origin=self.arm_command_reference[:,:3,3]
        self.q_target[:,self.arm]=self.arm_command.increment(self.q_target[:,self.arm],delta,self.arm_command_reference,
            origin-torch.tensor([.35,.35,.09],device=self.device),origin+torch.tensor([.35,.35,.10],device=self.device))
        c=(self.cfg.closure_bias+applied[:,3]*self.cfg.closure_scale).clamp(self.cfg.closure_min,self.cfg.closure_max)
        target=self.ref+c[:,None]*self.closure
        target[:,self.scene.kin.hand.index(self.residual_joints['index'])]+=applied[:,4]*self.cfg.index_support_action_rad
        target[:,self.scene.kin.hand.index(self.residual_joints['middle'])]+=applied[:,5]*self.cfg.middle_support_action_rad
        target+=self.release_middle_blend[:,None]*self.release_middle_offsets[None,:]
        self.hand_target+=self.cfg.hand_smoothing*(target.clamp(self.lo,self.hi)-self.hand_target)
        self.q_target[:,self.hand]=self.hand_target
        self.scene.step(self.q_target,decimation=self.cfg.decimation,render=render)
        self.steps+=1;self.phase_steps+=1
        m=self.measurements();s=m['state'];block=s['block_root_state_w'];tip=m['tip'];p=self.phase;t=self.phase_steps*self.dt
        retained=(m['grip_error']<.035)&(m['orientation_error']<.3)
        valid_grasp=retained&m['orientation_ok']&m['face_pinch_ok']
        lifting_now=(m['lift']>.045)&m['both']&valid_grasp
        self.lift_streak=torch.where(lifting_now,self.lift_streak+1,torch.zeros_like(self.lift_streak))
        self.grasped_once|=self.lift_streak>=round(.25/self.dt)
        distance=(self.target_tip-tip).norm(dim=-1)
        ready=torch.zeros_like(retained)
        ready|=(p==0)&(t>=.5)
        ready|=(p==1)&(distance<.0015)
        ready|=(p==2)&(distance<.0018)&(t>.5)
        self.squeeze_force_steps,self.squeeze_force_gate_ready=squeeze_readiness(
            m['forces'],p,t,self.squeeze_force_steps,self.dt,
            self.place_cfg.squeeze_min_jaw_force_N,self.place_cfg.squeeze_force_hold_seconds)
        self.squeeze_force_hold_time_s=self.squeeze_force_steps*self.dt
        ready|=self.squeeze_force_gate_ready
        ready|=(p==4)&(distance<.002)&(m['lift']>.045)&m['both']
        ready|=(p==5)&(distance<.0015)
        ready|=(p==6)&(distance<.003)&(m['lift']<.0015)&((block[:,:2]-self.goal[:,:2]).norm(dim=-1)<.004)
        release_clear=(m['forces'].amax(-1)<.003) if self.place_cfg.release_contact_clear_seconds>0 else torch.ones_like(retained)
        ready|=(p==7)&(t>=1.2)&release_clear
        ready|=(p==8)&(distance<.002)
        self.phase_ready=torch.where(ready,self.phase_ready+1,torch.zeros_like(self.phase_ready))
        required_ready=torch.where(p==7,max(3,round(self.place_cfg.release_contact_clear_seconds/self.dt)),3)
        change=(self.phase_ready>=required_ready)&(p<9)
        lifting=change&(p==4)
        self.block_tip_offset[lifting]=block[lifting,:3]-tip[lifting]
        self.phase[change]+=1;self.phase_steps[change]=0;self.phase_ready[change]=0
        self.max_phase=torch.maximum(self.max_phase,self.phase)
        xy_error=(block[:,:2]-self.goal[:,:2]).norm(dim=-1)
        stable=(self.phase==9)&self.grasped_once&(xy_error<self.place_cfg.goal_tolerance_m)&(m['lift'].abs()<.0015)
        stable&=(block[:,7:10].norm(dim=-1)<.01)&(block[:,10:13].norm(dim=-1)<.5)
        stable&=(m['forces'].max(-1).values<.003)&((tip[:,2]-block[:,2])>.035)&valid_grasp
        self.success_steps=torch.where(stable,self.success_steps+1,torch.zeros_like(self.success_steps))
        self.episode_success|=self.success_steps>=round(self.place_cfg.stable_seconds/self.dt)
        invalid=~torch.isfinite(torch.cat((s['joint_pos'],s['tool_root_state_w'],block),-1)).all(-1)
        failure=(m['grip_error']>.055)|(m['orientation_error']>.6)|(m['lift']<-.03)|invalid
        # Failed phase progress ends explicitly, rather than falsely counting lifted-over-goal as placement.
        phase_timeout=(self.phase_steps*self.dt>12.)&(self.phase<9)
        terminal=failure|phase_timeout|self.episode_success
        time_out=(self.steps>=self.max_steps)&~terminal
        done=terminal|time_out
        progress=(self.phase-old_phase).float()
        reward=.25*retained+.5*torch.exp(-distance/.025)+progress*15
        reward+=.5*(m['face_pinch_inward_force_N']/.03).clamp(0,1).mean(-1)
        reward+=((self.phase>=5)&(self.phase<=6)).float()*m['both']*torch.exp(-xy_error/.1)
        reward+=2*stable.float()+30*self.episode_success.float()
        reward-=.02*action.square().mean(-1)+.01*(applied-self.previous_action).square().mean(-1)+10*(failure|phase_timeout).float()
        reward=reward.nan_to_num(-10,posinf=-10,neginf=-10)
        self.episode_return+=reward;self.episode_max_lift=torch.maximum(self.episode_max_lift,m['lift'].nan_to_num())
        self.previous_previous_action.copy_(self.previous_action);self.previous_action.copy_(applied)
        terminal_obs=self.observe(m);ids=done.nonzero(as_tuple=False).flatten()
        if len(ids):
            self.total_episodes+=len(ids);self.return_history.extend(self.episode_return[ids].tolist());self.success_history.extend(self.episode_success[ids].float().tolist());self.lift_history.extend(self.episode_max_lift[ids].tolist())
            self.return_history=self.return_history[-1000:];self.success_history=self.success_history[-1000:];self.lift_history=self.lift_history[-1000:]
        metrics=dict(episodes=self.total_episodes,success_rate_1000=float(np.mean(self.success_history)) if self.success_history else 0,
            episode_return_mean=float(np.mean(self.return_history)) if self.return_history else 0,phase_mean=float(self.phase.float().mean()),
            goal_error_m=float(xy_error.mean()),retained=float(retained.float().mean()),bilateral_contact=float(m['both'].float().mean()))
        info=dict(time_outs=time_out,terminal_observation=terminal_obs,metrics=metrics)
        if not auto_reset:
            m.update(squeeze_force_hold_time_s=self.squeeze_force_hold_time_s,
                squeeze_force_gate_ready=self.squeeze_force_gate_ready,
                squeeze_min_jaw_force_N=torch.full_like(self.squeeze_force_hold_time_s,self.place_cfg.squeeze_min_jaw_force_N))
            info.update(measurements=m,applied_action=applied,goal_error=xy_error,failure=failure,phase_timeout=phase_timeout)
        if len(ids) and auto_reset:self.reset(ids)
        return self.observe(),reward,done,info
