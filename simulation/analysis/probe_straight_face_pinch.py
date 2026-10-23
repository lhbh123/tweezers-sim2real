"""Free physical straight-tweezer hold/closure probe, not a learned controller.

Only robot joint targets change after reset. Tool/cube states are never written
during the experiment. Middle and web contacts are diagnostics, not acceptance
conditions. All face forces and closure moments use real contact data.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose',required=True)
    parser.add_argument('--tactile',action='store_true')
    parser.add_argument('--out',required=True)
    parser.add_argument('--snapshot')
    parser.add_argument('--camera-config',help='Native cameras; optional env index in each view')
    parser.add_argument('--preferred-orientation',action='store_true',help='Rotate the entire reset grasp to pose preferred_tool_world_R; requires new physical validation')
    parser.add_argument('--variants',help='JSON list: name, closure_start/end, target_delta, initial_delta')
    parser.add_argument('--closures',type=float,nargs='+',default=[0.,.25,.5,1.])
    parser.add_argument('--copies',type=int,default=1)
    parser.add_argument('--hold-seconds',type=float,default=2.)
    parser.add_argument('--ramp-seconds',type=float,default=2.)
    parser.add_argument('--settle-seconds',type=float,default=2.)
    parser.add_argument('--tool-position',type=float,nargs=3,default=[.48,0.,.95])
    parser.add_argument('--spring',type=float,default=5.)
    parser.add_argument('--spring-damping',type=float,default=.02)
    parser.add_argument('--hand-stiffness',type=float,default=10.)
    parser.add_argument('--hand-damping',type=float,default=.3)
    parser.add_argument('--finger-friction',type=float,default=1.4)
    parser.add_argument('--dt',type=float,default=1/480)
    parser.add_argument('--audit-every',type=int,default=8)
    parser.add_argument('--audit-envs',type=int,nargs='+',default=[0])
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();args.headless=True;args.multi_gpu=False
    if args.camera_config:args.enable_cameras=True
    root=Path(__file__).resolve().parents[1];out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
    inputs=[Path(__file__),Path(args.pose)]+[root/'src'/n for n in
        ['rl_scene.py','grasp_contacts.py','kinematics.py','reset_reference.py','arm_clearance.py','probe_render.py']]
    if args.tactile:inputs.append(root/'src/sharpa_tactile.py')
    inputs += [Path(p) for p in [args.snapshot,args.variants,args.camera_config] if p]
    hashes={}
    for path in inputs:
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path,source/path.name)
    pose=json.loads((source/Path(args.pose).name).read_text())
    if pose.get('tool_geometry',{}).get('handle_extension'):
        raise ValueError('This straight-tail probe refuses a bent handle_extension asset')
    candidates=json.loads((source/Path(args.variants).name).read_text()) if args.variants else [
        {'name':f'c{c:g}_copy{i}','closure_start':0.,'closure_end':c} for c in args.closures for i in range(args.copies)]
    (out/'launch.json').write_text(json.dumps({'arguments':vars(args),'source_sha256':hashes},indent=2))
    app=AppLauncher(args).app
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from scipy.spatial.transform import Rotation
    from isaaclab.utils.math import quat_apply,quat_inv,quat_mul
    sys.path.insert(0,str(source))
    from rl_scene import BatchedTweezersScene
    R=np.asarray(pose.get('preferred_tool_world_R',np.diag([1.,-1.,-1.])))
    if args.snapshot and not args.preferred_orientation:
        snapshot=json.loads((source/Path(args.snapshot).name).read_text())
        R=Rotation.from_quat(np.roll(snapshot['rigid_body_states_13']['handle'][3:7],-1)).as_matrix()
    scene=BatchedTweezersScene(num_envs=len(candidates),device=args.device,dt=args.dt,
        pose_json=source/Path(args.pose).name,tool_world_R=R,tool_position=args.tool_position,
        spring=args.spring,spring_damping=args.spring_damping,hand_stiffness=args.hand_stiffness,
        hand_damping=args.hand_damping,finger_friction=args.finger_friction,
        solver_iterations=64,solver_velocity_iterations=8,contact_offset=.0002,
        simplify_arm_collisions=False,self_collisions=True,filter_mechanical_self_pairs=True,
        enable_face_metrics=True)
    n=scene.num_envs;dev=scene.device
    reader=scene.grasp_contact_reader
    assignment=pose.get('face_pinch_assignment',{'thumb':'right','index':'left'})
    pairs=[(['left','right'].index(assignment[role]),['thumb','index'].index(role)) for role in ['thumb','index']]
    cpu=lambda x:x.detach().cpu().numpy()
    q=scene.default_joint_pos.clone();base_target=q.clone()
    objects={k:v.clone() for k,v in scene.default_object_states.items()}
    if args.snapshot:
        from reset_reference import StableResetReference
        ref=StableResetReference(source/Path(args.snapshot).name,scene)
        tip=ref.tip_position+np.asarray(args.tool_position)-ref.body_transforms['handle'][:3,3]
        built=ref.build(tip,env_origins=scene.env_origins,position_frame='local',desired_tip_orientation=R)
        q[:],base_target[:]=built['q_actual'],built['q_target']
        for name,states in built['object_states'].items():objects[name][:]=states
    delta=torch.tensor([pose['closure_delta'].get(name,0.) for name in scene.joint_names],device=dev)
    start=torch.tensor([c.get('closure_start',0.) for c in candidates],device=dev)
    end=torch.tensor([c.get('closure_end',0.) for c in candidates],device=dev)
    for i,c in enumerate(candidates):
        for name,value in c.get('initial_delta',{}).items():q[i,scene.joint_names.index(name)]+=value
        for name,value in c.get('target_delta',{}).items():base_target[i,scene.joint_names.index(name)]+=value
    lo=torch.tensor([scene.kin.by_name[n]['lower'] for n in scene.joint_names],device=dev)
    hi=torch.tensor([scene.kin.by_name[n]['upper'] for n in scene.joint_names],device=dev)
    for value in [q,base_target+start[:,None]*delta,base_target+end[:,None]*delta]:
        if ((value<lo)|(value>hi)).any():raise ValueError('Probe target outside URDF; no silent clipping')
    objects['block'][:,:3]=scene.env_origins+torch.tensor([-.1,.3,scene.table_height+.0061],device=dev)
    scene.reset(q=q,object_states=objects)
    scene.stage.GetRootLayer().Export(str(out/'initial_scene.usda'))
    renderer=None
    if args.camera_config:
        from probe_render import ProbeRenderer
        renderer=ProbeRenderer(scene,args.camera_config,out)
    s=scene.get_state()
    relpos=quat_apply(quat_inv(s['wrist_quat_w']),s['tool_root_state_w'][:,:3]-s['wrist_pos_w'])
    relq=quat_mul(quat_inv(s['wrist_quat_w']),s['tool_root_state_w'][:,3:7])
    tip_local=torch.tensor([0.,0.,scene.tip_local_z],device=dev).expand(n,3)
    half_tip=torch.tensor(scene.tip_half_size,device=dev)
    records=[];raw=(out/'raw_contact_points.jsonl').open('w')
    total=args.hold_seconds+args.ramp_seconds+args.settle_seconds
    steps=round(total/args.dt);began=time.perf_counter()
    tactile_reader=None;frame_times=[]
    if args.tactile:
        from sharpa_tactile import SharpaTactileReader
        tactile_reader=SharpaTactileReader.from_scene(scene)
        tactile_time=[0.]
        def record_tactile(dt):
            tactile_time[0]+=dt;tactile_reader.record(tactile_time[0],dt)
        scene.physics_step_callbacks.append(record_tactile)
    max_compare_error=0.
    print('FACE_PROBE_READY',json.dumps({'num_envs':n,'face_definition':reader.face_metric_definition()}),flush=True)
    for step in range(steps):
        t=(step+1)*args.dt
        phase=min(1.,max(0.,(t-args.hold_seconds)/max(args.ramp_seconds,1e-9)))
        closure=start+(end-start)*phase
        for i,c in enumerate(candidates):
            if c.get('closure_schedule'):
                knots=np.asarray(c['closure_schedule']);closure[i]=float(np.interp(t,knots[:,0],knots[:,1]))
        target=base_target+closure[:,None]*delta
        s=scene.step(target);support=scene.read_grasp_support(s)
        tool=s['tool_root_state_w'];expected=s['wrist_pos_w']+quat_apply(s['wrist_quat_w'],relpos)
        expectedq=quat_mul(s['wrist_quat_w'],relq)
        error=(tool[:,:3]-expected).norm(dim=-1)
        orierror=1-(tool[:,3:7]*expectedq).sum(-1).square().clamp(0,1)
        tips=[s[name+'_root_state_w'][:,:3]+quat_apply(s[name+'_root_state_w'][:,3:7],tip_local) for name in ['left','right']]
        spread=tips[1]-tips[0];span=spread.norm(dim=-1);axis=spread/span[:,None].clamp_min(1e-9)
        thickness=torch.zeros(n,device=dev)
        for name in ['left','right']:
            thickness+=(quat_apply(quat_inv(s[name+'_root_state_w'][:,3:7]),axis).abs()*half_tip).sum(-1)
        toolaxis=quat_apply(tool[:,3:7],torch.tensor([0.,0.,1.],device=dev).expand(n,3))
        # A lower bound from a bounding sphere around each straight-jaw segment.
        # Exact source/table intersection remains a separate geometry audit.
        tipclear=torch.minimum(tips[0][:,2],tips[1][:,2])-scene.table_height-scene.env_origins[:,2]-half_tip.norm()
        row={'time':torch.tensor(t,device=dev),'closure':closure,'q':s['joint_pos'],
             'q_target':target,'bodies':torch.stack([s[k+'_root_state_w'] for k in ['handle','left','right']],dim=1),
             'grip_error_m':error,'grip_orientation_error':orierror,'tip_clearance_bound_m':tipclear,
             'tip_gap_m':span-thickness,'tip_center_w':.5*(tips[0]+tips[1]),'tip_spread_axis_w':axis,
             'tool_angle_deg':torch.rad2deg(torch.atan2(-toolaxis[:,2],toolaxis[:,:2].norm(dim=-1))),
             'palm_normal_w':quat_apply(s['palm_quat_w'],torch.tensor(pose.get('grasp_topology',{}).get('palm_outward_axis_local',[0.,1.,0.]),device=dev,dtype=tool.dtype).expand(n,3)),
             'web_axis_w':quat_apply(s['palm_quat_w'],torch.tensor(pose.get('web_up_axis_hand',[1.,0.,0.]),device=dev,dtype=tool.dtype).expand(n,3)),
             'cube_contact_force_N':s['block_jaw_contact_forces_w'].norm(dim=-1).amax(-1)}
        for key,value in support.items():
            if key.startswith('broad_face_') or key in ['role_contact_strength_N','role_support_z_N','role_normal_force_w','role_friction_force_w','web_contact_count','web_support_z_N','middle_under_contact_count']:
                row[key]=value
        records.append({k:v.detach().clone() for k,v in row.items()})
        if renderer and step%round(1/(30*args.dt))==0:
            renderer.frame();frame_times.append(t)
        if step==0 or (step+1)%args.audit_every==0 or step==steps-1:
            for env_id in args.audit_envs:
                if not 0<=env_id<n:continue
                points=reader.audit_env_points(s,args.dt,env_id)
                totals=np.zeros((2,2));torques=np.zeros((2,2));counts=np.zeros((2,2))
                for point in points['normal_contacts']:
                    if point['role'] not in ['thumb','index'] or not point.get('broad_face_qualified',False):continue
                    j=['left','right'].index(point['tool']);r=['thumb','index'].index(point['role'])
                    totals[j,r]+=point['inward_force_component_N'];torques[j,r]+=point['closing_contact_torque_Nm'];counts[j,r]+=1
                for key,ref in [('broad_face_inward_force_N',totals),('broad_face_closing_torque_Nm',torques),('broad_face_contact_count',counts)]:
                    max_compare_error=max(max_compare_error,float(np.abs(cpu(support[key][env_id])-ref).max()))
                raw.write(json.dumps({'time_s':t,'env':env_id,'points':points})+'\n')
        if (step+1)%round(1/args.dt)==0:
            print('FACE_PROBE_PROGRESS',json.dumps({'time_s':t,'retained':int(((error<.035)&(orierror<.3)).sum()),
                'assigned_inward_N':cpu(support['broad_face_inward_force_N'][:,[j for j,r in pairs],[r for j,r in pairs]]).tolist()}),flush=True)
    if renderer:renderer.close()
    if tactile_reader is not None:tactile_reader.save(out/'tactile.npz',frame_times_s=frame_times if frame_times else None)
    raw.close();data={k:cpu(torch.stack([r[k] for r in records])) for k in records[0]}
    np.savez_compressed(out/'timeseries.npz',**data)
    final_mask=data['time']>=total-min(1.,args.settle_seconds or total)
    hold_mask=data['time']<=args.hold_seconds
    assignment=pose.get('face_pinch_assignment',{'thumb':'right','index':'left'})
    pairs=[(['left','right'].index(assignment[role]),['thumb','index'].index(role)) for role in ['thumb','index']]
    for i,c in enumerate(candidates):
        position=data['bodies'][final_mask,i,0,:3];quats=data['bodies'][final_mask,i,0,3:7]
        angle=np.rad2deg(2*np.arccos(np.abs(quats@quats[-1]).clip(0,1)))
        retained=bool((data['grip_error_m'][:,i]<.035).all() and (data['grip_orientation_error'][:,i]<.3).all() and
            data['tip_clearance_bound_m'][:,i].min()>.005 and data['cube_contact_force_N'][:,i].max()<1e-6)
        stable=retained and np.ptp(position,axis=0).max()<.001 and angle.max()<1.
        inward=np.stack([data['broad_face_inward_force_N'][:,i,j,r] for j,r in pairs],axis=-1)
        broad=(inward>.003).all(-1)
        c.update(retained=retained,stable_hold=bool(stable),broad_face_hold=bool(stable and broad[final_mask].mean()>=.95),
            final_1s_both_assigned_broad_face_fraction=float(broad[final_mask].mean()),
            final_1s_inward_force_mean_N=inward[final_mask].mean(0).tolist(),
            final_1s_closing_torque_mean_Nm=[float(data['broad_face_closing_torque_Nm'][final_mask,i,j,r].mean()) for j,r in pairs],
            final_tool_angle_deg=float(data['tool_angle_deg'][-1,i]),
            final_palm_normal_w=data['palm_normal_w'][-1,i].tolist(),
            final_web_axis_w=data['web_axis_w'][-1,i].tolist(),
            final_1s_position_range_m=np.ptp(position,axis=0).tolist(),final_1s_rotation_range_deg=float(angle.max()),
            min_tip_clearance_bound_m=float(data['tip_clearance_bound_m'][:,i].min()),
            max_cube_contact_force_N=float(data['cube_contact_force_N'][:,i].max()),
            initial_hold_gap_mean_m=float(data['tip_gap_m'][hold_mask,i].mean()) if hold_mask.any() else None,
            final_gap_mean_m=float(data['tip_gap_m'][final_mask,i].mean()),
            final_1s_contact_fraction_by_role=(data['role_contact_strength_N'][final_mask,i]>.003).mean(0).tolist(),
            final_1s_middle_net_up_mean_N=float(data['role_support_z_N'][final_mask,i,2].mean()),
            final_1s_web_net_up_mean_N=float(data['web_support_z_N'][final_mask,i].mean()),
            final_1s_width_margin_min_m=data['broad_face_width_margin_min_m'][final_mask,i].min(0).tolist())
        if stable:
            bodies={k:cpu(s[k+'_root_state_w'][i]).copy() for k in ['handle','left','right']}
            for value in bodies.values():value[:3]-=cpu(scene.env_origins[i])
            snap={'schema_version':1,'source_run':str(out),'source_candidate':c['name'],'physics_time_s':total,
                'joint_names':scene.joint_names,'q_actual':cpu(s['joint_pos'][i]).tolist(),
                'q_target':cpu(target[i]).tolist(),'q_velocity':cpu(s['joint_vel'][i]).tolist(),
                'rigid_body_states_13':{k:v.tolist() for k,v in bodies.items()},
                'world_wrist_pose_7':np.r_[cpu(s['wrist_pos_w'][i]-scene.env_origins[i]),cpu(s['wrist_quat_w'][i])].tolist(),
                'world_hand_pose_7':np.r_[cpu(s['palm_pos_w'][i]-scene.env_origins[i]),cpu(s['palm_quat_w'][i])].tolist(),
                'arm_base_world_xyz':[0,0,scene.table_height],'fixture_active':False,'args':vars(args),'face_pinch_hold':c}
            snap['zero_velocity_variant']={'q_actual':snap['q_actual'],'q_target':snap['q_target'],'q_velocity':[0.]*len(scene.joint_names),
                'rigid_body_states_13':{k:v[:7]+[0.]*6 for k,v in snap['rigid_body_states_13'].items()}}
            (out/(c['name']+'_snapshot.json')).write_text(json.dumps(snap,indent=2))
    scene.stage.GetRootLayer().Export(str(out/'final_scene.usda'))
    report={'arguments':vars(args),'joint_names':scene.joint_names,'env_origins':cpu(scene.env_origins).tolist(),
        'initial_arm_ik_evidence':scene.initial_arm_ik_evidence,
        'candidates':candidates,'face_definition':reader.face_metric_definition(),
        'face_pinch_assignment':assignment,'stable_count':sum(c['stable_hold'] for c in candidates),
        'broad_face_hold_count':sum(c['broad_face_hold'] for c in candidates),
        'fixture_present':False,'runtime_object_state_writes':False,'teacher_or_learned_policy':False,
        'experiment':'Joint-target-only linear closure ramp following initial free hold; mechanical diagnostic.',
        'middle_web_requirement':'Optional backup contacts only; no contact/support gate or reward.',
        'GPU_vs_raw_point_face_totals_max_error':max_compare_error,'source_sha256':hashes,
        'physics_dt_s':args.dt,'all_timeseries_values_finite':all(np.isfinite(v).all() for v in data.values()),
        'wall_time_s':time.perf_counter()-began,
        'limitations':['Stable end snapshot requires independent cold-reset verification.',
            'Contact geometry uses the collider broad plane and measured contact point; source finger-mesh audit is separate.',
            'End-frame stability does not establish successful block pickup or learned-policy performance.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('FACE_PROBE_RESULT',json.dumps({'stable':report['stable_count'],'broad_face_hold':report['broad_face_hold_count'],
        'point_sum_error':max_compare_error}),flush=True)
    scene.close();app.close()


if __name__=='__main__':main()
