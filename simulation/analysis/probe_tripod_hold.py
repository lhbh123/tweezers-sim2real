"""Fixture-free tripod feasibility probe; constant robot targets, never task-time object writes.

Run with isaac-python. A pose config provides q_reference/closure_delta/T_hand_tool.
The normal/friction contact records distinguish merged wrist contact from a geometrically
defined web region. This is a mechanical diagnostic, not a learned-policy evaluation.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scene-source', default='src/rl_scene.py')
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--closures', type=float, nargs='+', default=[0., .2, .4, .6])
    parser.add_argument('--copies', type=int, default=1)
    parser.add_argument('--variants', help='JSON list: name, closure, initial_closure, q_delta, tool_offset_hand_m')
    parser.add_argument('--initial-closure', type=float, default=0.)
    parser.add_argument('--snapshot', help='Optional later cold-reset verification; default starts from q_reference')
    parser.add_argument('--tool-quaternion', type=float, nargs=4, help='World wxyz; otherwise config preferred_tool_world_R')
    parser.add_argument('--hand-quaternion', type=float, nargs=4, help='World wxyz; alternative to tool quaternion')
    parser.add_argument('--tool-position', type=float, nargs=3, default=[.48, 0., .95])
    parser.add_argument('--spring', type=float, default=5.)
    parser.add_argument('--spring-damping', type=float, default=.02)
    parser.add_argument('--hand-stiffness', type=float, default=10.)
    parser.add_argument('--hand-damping', type=float, default=.3)
    parser.add_argument('--finger-friction', type=float, default=1.4)
    parser.add_argument('--tool-width', type=float)
    parser.add_argument('--bridge-width', type=float)
    parser.add_argument('--dt', type=float, default=1/480)
    parser.add_argument('--record-every', type=int, default=8)
    parser.add_argument('--contact-capacity', type=int, default=32768)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.tool_quaternion and args.hand_quaternion:
        parser.error('Choose tool or hand quaternion, not both')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    source = out/'source'
    source.mkdir()
    hashes = {}
    input_paths=[Path(__file__), Path(args.scene_source), project/'src/kinematics.py', project/'src/reset_reference.py', project/'src/arm_clearance.py', project/'src/grasp_contacts.py', Path(args.pose)]
    input_paths += [Path(p) for p in [args.snapshot,args.variants] if p]
    for path in input_paths:
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, source/path.name)
    frozen_pose=source/Path(args.pose).name
    frozen_snapshot=source/Path(args.snapshot).name if args.snapshot else None
    (out/'launch.json').write_text(json.dumps({'arguments':vars(args),'source_sha256':hashes}, indent=2))
    app = AppLauncher(args).app
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from pxr import Usd, UsdPhysics, UsdGeom
    from isaacsim.core.prims import RigidPrim
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul
    sys.path.insert(0, str(source))
    from grasp_contacts import TripodContactReader
    spec = importlib.util.spec_from_file_location('tripod_scene', source/Path(args.scene_source).name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pose = json.loads(frozen_pose.read_text())
    R = np.diag([1., -1., -1.])
    if 'preferred_tool_world_R' in pose:
        R = np.asarray(pose['preferred_tool_world_R'])
    elif 'preferred_hand_world_R' in pose:
        R = np.asarray(pose['preferred_hand_world_R']) @ np.asarray(pose['T_hand_tool'])[:3,:3]
    if args.snapshot:
        raw = json.loads(frozen_snapshot.read_text())
        qh = raw['rigid_body_states_13']['handle'][3:7]
        R = Rotation.from_quat(np.roll(qh,-1)).as_matrix()
    if args.tool_quaternion:
        R = Rotation.from_quat(np.roll(args.tool_quaternion,-1)).as_matrix()
    if args.hand_quaternion:
        R = Rotation.from_quat(np.roll(args.hand_quaternion,-1)).as_matrix() @ np.asarray(pose['T_hand_tool'])[:3,:3]
    candidates = json.loads((source/Path(args.variants).name).read_text()) if args.variants else [
        {'name':f'closure{c:g}_copy{k}', 'closure':c, 'initial_closure':args.initial_closure}
        for c in args.closures for k in range(args.copies)]
    n = len(candidates)
    scene = module.BatchedTweezersScene(num_envs=n, device=args.device, dt=args.dt,
        pose_json=frozen_pose, tool_world_R=R, tool_position=args.tool_position,
        spring=args.spring, spring_damping=args.spring_damping,
        hand_stiffness=args.hand_stiffness, hand_damping=args.hand_damping,
        finger_friction=args.finger_friction, tool_width=args.tool_width, bridge_width=args.bridge_width,
        solver_iterations=64, solver_velocity_iterations=8, contact_offset=.0002,
        simplify_arm_collisions=False, self_collisions=True, filter_mechanical_self_pairs=True)
    dev = scene.device
    def cpu(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    q = scene.default_joint_pos.clone()
    target = q.clone()
    objects = {name:states.clone() for name,states in scene.default_object_states.items()}
    if args.snapshot:
        from reset_reference import StableResetReference
        ref = StableResetReference(frozen_snapshot, scene)
        tip = ref.tip_position + np.asarray(args.tool_position) - ref.body_transforms['handle'][:3,3]
        built = ref.build(tip, env_origins=scene.env_origins, position_frame='local',
                          desired_tip_orientation=R)
        q[:], target[:] = built['q_actual'], built['q_target']
        for name,states in built['object_states'].items(): objects[name][:] = states
    hand_R = R @ np.asarray(pose['T_hand_tool'])[:3,:3].T
    for i,c in enumerate(candidates):
        for name,delta in pose['closure_delta'].items():
            j = scene.joint_names.index(name)
            q[i,j] += float(c.get('initial_closure',args.initial_closure))*delta
            target[i,j] += float(c.get('closure',0.))*delta
        for name,delta in c.get('q_delta',{}).items():
            j = scene.joint_names.index(name)
            q[i,j] += delta; target[i,j] += delta
        for name,delta in c.get('target_delta',{}).items():
            target[i,scene.joint_names.index(name)] += delta
        translation = hand_R @ np.asarray(c.get('tool_offset_hand_m',[0.,0.,0.]))
        translation += np.asarray(c.get('tool_offset_world_m',[0.,0.,0.]))
        for name in ['handle','left','right']:
            objects[name][i,:3] += torch.tensor(translation,device=dev,dtype=torch.float32)
        c['reachable'] = True
    lo = torch.tensor([scene.kin.by_name[x]['lower'] for x in scene.joint_names],device=dev)
    hi = torch.tensor([scene.kin.by_name[x]['upper'] for x in scene.joint_names],device=dev)
    if ((q<lo)|(q>hi)|(target<lo)|(target>hi)).any():
        raise ValueError('Requested initial/target joints exceed URDF limits; refusing silently clipped probe')
    objects['block'][:,:3] = scene.env_origins + torch.tensor([-.1,.3,scene.table_height+.0061],device=dev)
    contact_reader=TripodContactReader(scene,args.contact_capacity,pose.get('web_contact_region'))
    robot_paths, names = contact_reader.paths, contact_reader.names
    groups = contact_reader.groups
    group = contact_reader.classify
    group_ids = np.array([groups.index(group(x)) for x in names])
    views=contact_reader.views
    # Geometric tool corners include any newly authored handle extension.
    box_points={}
    for body in ['Handle','Left','Right']:
        bp=[]
        for prim in Usd.PrimRange(scene.stage.GetPrimAtPath(scene.env_paths[0]+'/Tweezers/'+body)):
            if prim.IsA(UsdGeom.Cube):
                xf=UsdGeom.Xformable(prim).GetLocalTransformation()
                size=UsdGeom.Cube(prim).GetSizeAttr().Get()
                for x in [-.5,.5]:
                    for y in [-.5,.5]:
                        for z in [-.5,.5]:
                            from pxr import Gf
                            bp.append(list(xf.Transform(Gf.Vec3d(x*size,y*size,z*size))))
        box_points[body.lower()]=torch.tensor(bp,device=dev,dtype=torch.float32)
    scene.reset(q=q,object_states=objects)
    scene.stage.GetRootLayer().Export(str(out/'initial_scene.usda'))
    s=scene.get_state()
    relative_pos=quat_apply(quat_inv(s['wrist_quat_w']),s['tool_root_state_w'][:,:3]-s['wrist_pos_w'])
    relative_q=quat_mul(quat_inv(s['wrist_quat_w']),s['tool_root_state_w'][:,3:7])
    web_region=pose.get('web_contact_region')
    web_axis=np.asarray(pose.get('web_up_axis_hand',[1.,0.,0.]),dtype=float)
    web_axis/=np.linalg.norm(web_axis)
    min_clearance=torch.full((n,),float('inf'),device=dev)
    first_loss=torch.full((n,),-1.,device=dev)
    max_cube_force=torch.zeros(n,device=dev)
    records=[]; point_file=(out/'contact_points.jsonl').open('w'); early=[]
    friction_available=True; friction_errors=[]; mismatch_max=0.; buffer_highwater=0
    tensor_reader_max_error=0.
    start=time.perf_counter(); steps=round(args.seconds/args.dt)
    def web_inside(p):
        if not web_region: return False
        if 'bbox_min_m' in web_region:
            return bool(np.all(p>=web_region['bbox_min_m']) and np.all(p<=web_region['bbox_max_m']))
        return bool(np.linalg.norm(p-np.asarray(web_region['center_m'])) <= web_region['radius_m'])
    print('TRIPOD_START',json.dumps({'out':str(out),'num_envs':n,'robot_filters':names}),flush=True)
    for step in range(steps):
        s=scene.step(target)
        tool=s['tool_root_state_w']
        expected=s['wrist_pos_w']+quat_apply(s['wrist_quat_w'],relative_pos)
        expected_q=quat_mul(s['wrist_quat_w'],relative_q)
        error=(tool[:,:3]-expected).norm(dim=-1)
        orient=1-(tool[:,3:7]*expected_q).sum(-1).square().clamp(0,1)
        clearance=torch.full((n,),float('inf'),device=dev)
        for name,p in box_points.items():
            state=s[name+'_root_state_w']
            world=state[:,None,:3]+quat_apply(state[:,None,3:7].expand(-1,len(p),-1).reshape(-1,4),
                p[None].expand(n,-1,-1).reshape(-1,3)).reshape(n,-1,3)
            clearance=torch.minimum(clearance,(world[:,:,2]-scene.env_origins[:,None,2]-scene.table_height).amin(-1))
        min_clearance=torch.minimum(min_clearance,clearance)
        max_cube_force=torch.maximum(max_cube_force,s['block_jaw_contact_forces_w'].norm(dim=-1).amax(-1))
        lost=(error>.035)|(orient>.3)|(clearance<.005)
        first_loss=torch.where((first_loss<0)&lost,torch.full_like(first_loss,(step+1)*args.dt),first_loss)
        if step+1 in [round(t/args.dt) for t in [.1,.2,.5]]:
            early.append({'time_s':(step+1)*args.dt,'q':cpu(s['joint_pos']).tolist(),
                'objects':{k:cpu(s[k+'_root_state_w']).tolist() for k in ['handle','left','right']}})
        if (step+1)%args.record_every==0:
            normal=np.zeros((n,3,len(names),3));friction=np.zeros_like(normal)
            middle_under=np.zeros(n,dtype=bool);web_normal=np.zeros((n,3));web_hits=np.zeros(n,dtype=int)
            web_friction=np.zeros((n,3))
            palmR=Rotation.from_quat(cpu(s['palm_quat_w'])[:,[1,2,3,0]]).as_matrix()
            palmpos=cpu(s['palm_pos_w']);origins=cpu(scene.env_origins)
            body_states=np.stack([cpu(s[k+'_root_state_w']) for k in ['handle','left','right']],axis=1)
            points_record={'time_s':(step+1)*args.dt,'contacts':[],'friction_anchors':[]}
            for body_id,view in enumerate(views):
                matrix=cpu(view.get_contact_force_matrix(dt=args.dt))
                f,p,norm,sep,count,index=[cpu(x) for x in view.get_contact_force_data(dt=args.dt)]
                assert matrix.shape==(n,len(names),3),matrix.shape
                buffer_highwater=max(buffer_highwater,int((index+count).max()))
                friction_data=None
                if friction_available:
                    try: friction_data=[cpu(x) for x in view.get_friction_data(dt=args.dt)]
                    except Exception as exc:
                        friction_available=False;friction_errors.append(repr(exc))
                for row,path in enumerate(view.prim_paths):
                    env=int(path.split('/env_')[1].split('/')[0])
                    normal[env,body_id]=matrix[row]
                    bodyR=Rotation.from_quat(body_states[env,body_id,[4,5,6,3]]).as_matrix()
                    for col,name in enumerate(names):
                        begin=int(index[row,col]);end=begin+int(count[row,col])
                        vectors=f[begin:end]*norm[begin:end]
                        reconstructed=vectors.sum(0)
                        mismatch_max=max(mismatch_max,float(np.linalg.norm(reconstructed-matrix[row,col])))
                        for k in range(begin,end):
                            if abs(float(f[k,0]))<1e-8: continue
                            point=p[k];ph=(point-palmpos[env])@palmR[env]
                            pt=(point-body_states[env,body_id,:3])@bodyR
                            radial=bodyR@np.array([pt[0],pt[1],0.])
                            in_web=body_id==0 and name=='wrist_3_link' and web_inside(ph)
                            vector=f[k,0]*norm[k]
                            if in_web:web_normal[env]+=vector;web_hits[env]+=1
                            if group(name)=='middle' and radial[2]<-.0001:middle_under[env]=True
                            points_record['contacts'].append({'env':env,'tool':['handle','left','right'][body_id],
                                'robot':name,'group':group(name),'point_world':point.tolist(),
                                'point_env':(point-origins[env]).tolist(),'point_hand':ph.tolist(),'point_tool':pt.tolist(),
                                'normal_world':norm[k].tolist(),'signed_normal_force_N':float(f[k,0]),
                                'normal_force_vector_on_tool_N':vector.tolist(),'separation_m':float(sep[k,0]),
                                'radial_offset_world_z_m':float(radial[2]),'within_web_region':in_web})
                        if friction_data is not None:
                            ff,fp,fc,fi=friction_data; fb=int(fi[row,col]);fe=fb+int(fc[row,col])
                            friction[env,body_id,col]=ff[fb:fe].sum(0)
                            for k in range(fb,fe):
                                if np.linalg.norm(ff[k])<1e-8:continue
                                ph=(fp[k]-palmpos[env])@palmR[env]
                                in_web=body_id==0 and name=='wrist_3_link' and web_inside(ph)
                                if in_web:web_friction[env]+=ff[k]
                                points_record['friction_anchors'].append({'env':env,'tool':['handle','left','right'][body_id],
                                    'robot':name,'force_on_tool_N':ff[k].tolist(),'point_world':fp[k].tolist(),
                                    'point_hand':ph.tolist(),'within_web_region':in_web})
            point_file.write(json.dumps(points_record)+'\n')
            group_normal=np.stack([normal[:,:,group_ids==g].sum((1,2)) for g in range(len(groups))],axis=1)
            group_friction=np.stack([friction[:,:,group_ids==g].sum((1,2)) for g in range(len(groups))],axis=1)
            strength=np.stack([np.linalg.norm(normal[:,:,group_ids==g],axis=-1).sum((1,2)) for g in range(len(groups))],axis=1)
            if friction_available and (not records or step+1==steps):
                tensor_readout=contact_reader.read_support_tensors(s,args.dt)
                for key,reference in [('normal_force_w',normal),('friction_force_w',friction),
                    ('role_normal_force_w',group_normal),('role_friction_force_w',group_friction),
                    ('web_contact_count',web_hits),('web_normal_force_w',web_normal),
                    ('web_friction_force_w',web_friction)]:
                    tensor_reader_max_error=max(tensor_reader_max_error,float(np.abs(cpu(tensor_readout[key])-reference).max()))
                if not np.array_equal(cpu(tensor_readout['middle_under_contact_count'])>0,middle_under):
                    raise RuntimeError('GPU middle-underside classification disagrees with independent point loop')
            records.append({'time':(step+1)*args.dt,'q':cpu(s['joint_pos']),
                'objects':body_states,'error':cpu(error),'orient_error':cpu(orient),'clearance':cpu(clearance),
                'normal_force':normal,'friction_force':friction,'group_normal':group_normal,'group_friction':group_friction,
                'group_strength':strength,'middle_under':middle_under,'web_normal':web_normal,
                'web_friction':web_friction,'web_hits':web_hits,
                'web_up_axis_w':np.einsum('nij,j->ni',palmR,web_axis)})
        if (step+1)%round(2/args.dt)==0:
            print('TRIPOD_PROGRESS',json.dumps({'time_s':(step+1)*args.dt,'retained':int((first_loss<0).sum())}),flush=True)
    point_file.close()
    data={key:np.stack([r[key] for r in records]) for key in records[0]}
    np.savez_compressed(out/'support_timeseries.npz',**data)
    (out/'env0_contact_audit.json').write_text(json.dumps(contact_reader.audit_env_points(s,args.dt),indent=2))
    tail=data['time']>=args.seconds-1
    masses=np.array([float(UsdPhysics.MassAPI(scene.stage.GetPrimAtPath(scene.env_paths[0]+'/Tweezers/'+name)).GetMassAttr().Get()) for name in ['Handle','Left','Right']])
    tip_offset=torch.tensor([0.,0.,scene.tip_local_z],device=dev).expand(n,3)
    tip_points=[s[name+'_root_state_w'][:,:3]+quat_apply(s[name+'_root_state_w'][:,3:7],tip_offset) for name in ['left','right']]
    spread=tip_points[1]-tip_points[0];tip_gap=spread.norm(dim=-1);spread_axis=spread/tip_gap[:,None].clamp_min(1e-9)
    tip_span=torch.zeros(n,device=dev)
    for name in ['left','right']:
        local=quat_apply(quat_inv(s[name+'_root_state_w'][:,3:7]),spread_axis)
        tip_span+=(local.abs()*torch.tensor(scene.tip_half_size,device=dev)).sum(-1)
    palm_normal=quat_apply(s['palm_quat_w'],torch.tensor([0.,1.,0.],device=dev).expand(n,3))
    for i,c in enumerate(candidates):
        positions=data['objects'][tail,i,0,:3]
        quats=data['objects'][tail,i,0,3:7]
        angular=float(np.rad2deg(2*np.arccos(np.abs(quats@quats[-1]).clip(0,1))).max())
        contact=(data['group_strength'][tail,i]>.003).mean(0)
        combined=data['group_normal'][tail,i]+data['group_friction'][tail,i]
        up_fraction=float((combined[:,2,2]>.003).mean())
        retained=bool(first_loss[i]<0 and max_cube_force[i]<1e-6)
        stable=bool(retained and np.ptp(positions,axis=0).max()<.001 and angular<1.)
        tripod=bool(stable and np.min(contact[:3])>=.95 and up_fraction>=.95 and data['middle_under'][tail,i].mean()>=.95)
        web_up=data['web_normal'][tail,i,2]+data['web_friction'][tail,i,2]
        web_contact_fraction=float((data['web_hits'][tail,i]>0).mean())
        web_up_fraction=float((web_up>.003).mean())
        four_role=bool(tripod and web_contact_fraction>=.95 and web_up_fraction>=.95)
        tail_axis=Rotation.from_quat(data['objects'][tail,i,0][:,[4,5,6,3]]).apply([0.,0.,1.])
        tail_angles=np.rad2deg(np.arctan2(-tail_axis[:,2],np.linalg.norm(tail_axis[:,:2],axis=-1)))
        c.update({'retained':retained,'stable_hold':stable,'three_digit_support':tripod,
            'four_role_support':four_role,
            'first_loss_s':float(first_loss[i]),'min_tool_table_clearance_m':float(min_clearance[i]),
            'max_cube_contact_N':float(max_cube_force[i]),'last_1s_position_range_m':np.ptp(positions,axis=0).tolist(),
            'last_1s_rotation_range_deg':angular,'last_1s_contact_fraction_by_group':dict(zip(groups,contact.tolist())),
            'last_1s_normal_force_mean_by_group_N':dict(zip(groups,data['group_normal'][tail,i].mean(0).tolist())),
            'last_1s_friction_force_mean_by_group_N':dict(zip(groups,data['group_friction'][tail,i].mean(0).tolist())),
            'last_1s_total_upward_support_by_group_N':dict(zip(groups,combined[:,:,2].mean(0).tolist())),
            'last_1s_middle_upward_support_fraction_gt3mN':up_fraction,
            'last_1s_middle_contact_under_tool_axis_fraction':float(data['middle_under'][tail,i].mean()),
            'last_1s_web_region_contact_fraction':web_contact_fraction,
            'last_1s_web_upward_support_fraction_gt3mN':web_up_fraction,
            'last_1s_web_upward_support_mean_N':float(web_up.mean()),
            'last_1s_web_up_axis_z_min':float(data['web_up_axis_w'][tail,i,2].min()),
            'last_1s_tool_down_angle_deg_range':[float(tail_angles.min()),float(tail_angles.max())],
            'final_palm_normal_w':cpu(palm_normal[i]).tolist(),
            'final_tip_net_projected_gap_m':float(tip_gap[i]-tip_span[i]),
            'final_tip_spread_axis_w':cpu(spread_axis[i]).tolist(),
            'last_1s_total_support_over_tool_weight':float(combined[:,:,2].sum(-1).mean()/(masses.sum()*9.81)),
            'last_1s_reported_handle_speed_mean_mps':float(np.linalg.norm(data['objects'][tail,i,0,7:10],axis=-1).mean()),
            'last_1s_position_difference_speed_mean_mps':float(np.linalg.norm(np.diff(positions,axis=0)/(args.dt*args.record_every),axis=-1).mean())})
        if stable:
            body={name:(cpu(s[name+'_root_state_w'][i]).copy()) for name in ['handle','left','right']}
            for value in body.values():value[:3]-=cpu(scene.env_origins[i])
            snap={'schema_version':1,'source_run':str(out),'source_candidate':c['name'],'physics_time_s':args.seconds,
                'joint_names':scene.joint_names,'q_actual':cpu(s['joint_pos'][i]).tolist(),'q_target':cpu(target[i]).tolist(),
                'q_velocity':cpu(s['joint_vel'][i]).tolist(),'rigid_body_states_13':{k:v.tolist() for k,v in body.items()},
                'world_wrist_pose_7':np.r_[cpu(s['wrist_pos_w'][i]-scene.env_origins[i]),cpu(s['wrist_quat_w'][i])].tolist(),
                'world_hand_pose_7':np.r_[cpu(s['palm_pos_w'][i]-scene.env_origins[i]),cpu(s['palm_quat_w'][i])].tolist(),
                'arm_base_world_xyz':[0,0,scene.table_height],'fixture_active':False,'args':vars(args),'tripod_hold':c}
            snap['zero_velocity_variant']={'q_actual':snap['q_actual'],'q_target':snap['q_target'],'q_velocity':[0.]*18,
                'rigid_body_states_13':{k:v[:7]+[0.]*6 for k,v in snap['rigid_body_states_13'].items()}}
            (out/(c['name']+'_snapshot.json')).write_text(json.dumps(snap,indent=2))
    scene.stage.GetRootLayer().Export(str(out/'final_scene.usda'))
    report={'arguments':vars(args),'candidates':candidates,'stable_count':sum(c['stable_hold'] for c in candidates),
        'three_digit_support_count':sum(c['three_digit_support'] for c in candidates),
        'four_role_support_count':sum(c['four_role_support'] for c in candidates),'source_sha256':hashes,
        'fixture_present':False,'task_time_object_state_writes':False,'constant_joint_targets':True,
        'contact_source_rows':[v.prim_paths for v in views],'contact_filter_columns':names,'force_groups':groups,
        'friction_data_available':friction_available,'friction_errors':friction_errors,
        'normal_force_matrix_reconstruction_max_error_N':mismatch_max,'contact_buffer_highwater':buffer_highwater,
        'GPU_tensor_reader_vs_individual_point_loop_max_error':tensor_reader_max_error,
        'contact_buffer_capacity':args.contact_capacity,'web_contact_region':web_region,
        'web_contact_tool_body':'Handle',
        'web_contact_warning':'Merged wrist contact is not automatically web contact; only points inside the configured hand-frame ROI are labeled web. This is collision-body ROI contact, not proof of contact with the original visual palm STL; verify source surface distance separately.',
        'force_definition':'World-frame forces on TOOL source body; normal scalar times normal sums checked against matrix. Friction anchors summed separately. Upward world Z support includes normal plus reported friction.',
        'sampling_dt_s':args.dt*args.record_every,'tool_body_masses_kg':masses.tolist(),'wall_time_s':time.perf_counter()-start,
        'criteria':'Stable: no35mm/quaternion.3 loss, >5mm table clearance throughout, no cube contact, last1s <1mm translation and<1deg rotation. Three-digit: stable plus thumb/index/middle contact>3mN in95%, middle upward support>3mN in95%, and middle contact below local tool axis in95%. Four-role: three-digit plus Handle contact in web ROI in95% and web net upward support>3mN in95%; actual visual-source tail/web surface contact still needs independent geometry audit. Web-up axis reported separately.',
        'limitations':['Contact support is sampled, not a continuum certificate.','Force balance and friction API output require inspection; no force sign is inferred from an image.','Stable snapshot alone does not guarantee cold-reset stability.']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'early_states.json').write_text(json.dumps({'origins':cpu(scene.env_origins).tolist(),'joint_names':scene.joint_names,'samples':early}))
    print('TRIPOD_RESULT',json.dumps({'out':str(out),'stable':report['stable_count'],'tripod':report['three_digit_support_count'],'four_role':report['four_role_support_count'],'contact_reconstruction_error':mismatch_max,'friction_available':friction_available}),flush=True)
    scene.close();app.close()


if __name__=='__main__':
    main()
