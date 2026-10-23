#!/usr/bin/env python3
"""Independent audit of this project's recorded PPO tweezer demonstration.

Checks recorded lift/contact samples, URDF position limits, the final USD
physical structure, and saved source write paths separately. It does not replay
the policy or claim that a static USD captures the Fabric simulation trajectory.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ASSET = Path('/cpfs/user/lvzhaoyang/work/ai_rl/assets/ur5e_xhand_right')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'analysis'))
from kinematics import Kinematics
from audit_recorded_tripod_support import audit_support
from audit_recorded_face_pinch import audit_face_pinch

USD_AUDIT = r'''
import json,sys,math
sys.path.insert(0,sys.argv[2])
from pxr import Usd,UsdPhysics,UsdGeom
stage=Usd.Stage.Open(sys.argv[1])
if not stage: raise ValueError('USD stage could not be opened')
env='/World/envs/env_0'
def value(prim,name,default=None):
    attribute=prim.GetAttribute(name)
    result=attribute.Get() if attribute else None
    return default if result is None else result
def attr(prim,name,default=None):
    attribute=prim.GetAttribute(name)
    return {'value':value(prim,name,default),'authored':bool(attribute and attribute.HasAuthoredValueOpinion()),
            'time_samples':attribute.GetNumTimeSamples() if attribute else 0}
bodies={}
for name,suffix in [('handle','Tweezers/Handle'),('left','Tweezers/Left'),('right','Tweezers/Right'),('block','Block')]:
    path=env+'/'+suffix;prim=stage.GetPrimAtPath(path)
    if not prim: raise ValueError('Missing '+path)
    rigid=UsdPhysics.RigidBodyAPI(prim)
    bodies[name]={'path':path,'rigid_body_api':bool(rigid),
        'rigid_body_enabled':bool(rigid.GetRigidBodyEnabledAttr().Get()),
        'kinematic_enabled':bool(rigid.GetKinematicEnabledAttr().Get()),
        'disable_gravity':attr(prim,'physxRigidBody:disableGravity',False),
        'mass_kg':UsdPhysics.MassAPI(prim).GetMassAttr().Get(),
        'collision_shapes':[str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]}
paths={b['path'] for b in bodies.values()}
connected=[];roots=[];filters=[]
for prim in stage.Traverse():
    path=str(prim.GetPath())
    if prim.IsA(UsdPhysics.Joint):
        joint=UsdPhysics.Joint(prim)
        endpoints=[str(p) for p in joint.GetBody0Rel().GetTargets()]+[str(p) for p in joint.GetBody1Rel().GetTargets()]
        if any(p in paths for p in endpoints):
            angular={n:attr(prim,'drive:angular:physics:'+n) for n in ['type','stiffness','damping','targetPosition','targetVelocity','maxForce']}
            connected.append({'path':path,'type':prim.GetTypeName(),'body_paths':endpoints,
                'joint_enabled':bool(joint.GetJointEnabledAttr().Get()),'angular_drive':angular,
                'axis':value(prim,'physics:axis'),'lower_limit_deg':value(prim,'physics:lowerLimit'),
                'upper_limit_deg':value(prim,'physics:upperLimit')})
    if path.startswith(env+'/Robot') and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        roots.append({'path':path,'enabled_self_collisions':attr(prim,'physxArticulation:enabledSelfCollisions',False)})
    if path.startswith(env) and prim.HasAPI(UsdPhysics.FilteredPairsAPI):
        filters += [[path,str(p)] for p in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()]
table=stage.GetPrimAtPath(env+'/Table');block_shape=stage.GetPrimAtPath(env+'/Block/Shape')
env_t=value(stage.GetPrimAtPath(env),'xformOp:translate',(0.,0.,0.))
table_top=float(env_t[2])+float(value(table,'xformOp:translate')[2])+float(value(table,'xformOp:scale')[2])*float(value(table,'size',1.))/2
block_size=[float(x)*float(value(block_shape,'size',1.)) for x in value(block_shape,'xformOp:scale')]
robot_base=stage.GetPrimAtPath(env+'/Robot')
robot_matrix=UsdGeom.Xformable(robot_base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
tool_shapes={}
for suffix in ['Handle/Shape','Handle/TailExtension','Left/Blade','Left/Tip','Right/Blade','Right/Tip']:
    prim=stage.GetPrimAtPath(env+'/Tweezers/'+suffix)
    if not prim: continue
    rotation=value(prim,'xformOp:orient')
    tool_shapes[suffix]={'size_m':[float(x)*float(value(prim,'size',1.)) for x in value(prim,'xformOp:scale')],
                        'local_translation_m':[float(x) for x in value(prim,'xformOp:translate')],
                        'local_quaternion_wxyz':[float(rotation.GetReal()),*map(float,rotation.GetImaginary())] if rotation else [1.,0.,0.,0.]}
scene=next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
materials={}
for name in ['RLFinger','RLSteel','RLPlastic']:
    p=stage.GetPrimAtPath('/World/Materials/'+name)
    materials[name]={n:value(p,'physics:'+n) for n in ['staticFriction','dynamicFriction','restitution']}
    materials[name]['combine_mode']=value(p,'physxMaterial:frictionCombineMode')
print('RL_USD_AUDIT='+json.dumps({'bodies':bodies,'tool_block_connected_joints':connected,
    'robot_articulation_roots':roots,'filtered_pairs':filters,'table_top_world_z_m':table_top,
    'block_size_m':block_size,'gravity_magnitude':value(scene,'physics:gravityMagnitude'),
    'robot_base_world_matrix_gf_row_vectors':[[float(x) for x in row] for row in robot_matrix],
    'tool_collision_boxes':tool_shapes,
    'gravity_direction':list(value(scene,'physics:gravityDirection')),
    'gpu_dynamics':value(scene,'physxScene:enableGPUDynamics'),
    'solver_type':value(scene,'physxScene:solverType'),
    'physics_hz':value(scene,'physxScene:timeStepsPerSecond'),
    'materials':materials}))
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usd_audit(path, timeout=60):
    sim=Path(os.environ.get('ISAAC_SIM_PATH','/cpfs/user/lvzhaoyang/apps/isaac-sim-5.0.0'))
    usd=next(iter(sorted((sim/'extscache').glob('omni.usd.libs-*'))),None)
    if usd is None: raise RuntimeError('Isaac USD libraries are unavailable')
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(usd/'bin')+':'+env.get('LD_LIBRARY_PATH','')
    result=subprocess.run([str(sim/'python.sh'),'-c',USD_AUDIT,str(path),str(usd)],
                          env=env,capture_output=True,text=True,timeout=timeout)
    for line in result.stdout.splitlines():
        if line.startswith('RL_USD_AUDIT='): return json.loads(line.split('=',1)[1])
    raise RuntimeError('USD audit failed: '+result.stderr[-1600:])


def source_audit(run,launch):
    hashes={name:{'expected':expected,'actual':sha(run/Path(name).name),
                  'matches':sha(run/Path(name).name)==expected}
            for name,expected in launch['source_sha256'].items()}
    calls=[]
    class Calls(ast.NodeVisitor):
        def __init__(self,filename): self.filename=filename;self.stack=[]
        def visit_FunctionDef(self,node):
            self.stack.append(node.name);self.generic_visit(node);self.stack.pop()
        def visit_Call(self,node):
            calls.append({'file':self.filename,'function':self.stack[-1] if self.stack else '<module>',
                          'line':node.lineno,'call':ast.unparse(node.func)})
            self.generic_visit(node)
    trees={}
    for name in ['rl_scene.py','rl_task.py','train_rl.py','ppo.py','reset_reference.py','grasp_contacts.py','arm_command.py','arm_clearance.py']:
        if not (run/name).exists(): continue
        trees[name]=ast.parse((run/name).read_text());Calls(name).visit(trees[name])
    all_pose_writes=[c for c in calls if c['call'].endswith(('write_root_state_to_sim','write_root_pose_to_sim',
        'write_root_velocity_to_sim','write_joint_state_to_sim','set_world_pose','set_world_poses',
        'set_local_pose','set_local_poses','set_joint_positions','set_joint_velocities',
        'set_velocities','set_transforms'))]
    camera_writes=[c for c in all_pose_writes if c['call']=='camera.set_world_pose']
    writes=[c for c in all_pose_writes if c not in camera_writes]
    external_forces=[c for c in calls if any(term in c['call'].lower() for term in
        ['external_force','external_wrench','apply_force','apply_torque'])]
    steps=[]
    for filename in ['rl_scene.py','rl_task.py']:
        for node in ast.walk(trees[filename]):
            if isinstance(node,ast.FunctionDef) and node.name=='step':
                steps.append({'file':filename,'line':node.lineno,'end_line':node.end_lineno,
                              'calls':[ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)]})
    eval_step_calls=[];policy_calls=[]
    for node in ast.walk(trees['train_rl.py']):
        if not isinstance(node,ast.Call): continue
        name=ast.unparse(node.func)
        if name=='env.step':
            eval_step_calls.append({'line':node.lineno,'auto_reset':next((ast.literal_eval(k.value)
                for k in node.keywords if k.arg=='auto_reset'),None)})
        if name in ['policy.act','policy.load_state_dict','torch.load']:
            policy_calls.append({'line':node.lineno,'call':name})
    drive_writes=[c for c in calls if any(x in c['call'] for x in
        ['CreateTargetPositionAttr','CreateTargetVelocityAttr','CreateStiffnessAttr','CreateDampingAttr'])]
    checks={'saved_source_hashes_match_launch':all(h['matches'] for h in hashes.values()),
        'direct_state_writes_only_in_reset':bool(writes) and all(c['function']=='reset' for c in writes),
        'no_external_tool_or_block_force_calls':not external_forces,
        'evaluation_disables_task_auto_reset':bool(eval_step_calls) and all(c['auto_reset'] is False for c in eval_step_calls),
        'passive_drive_targets_written_only_at_construction':bool(drive_writes) and all(c['function']=='__init__' for c in drive_writes),
        'evaluation_loads_and_calls_policy':all(any(c['call']==n for c in policy_calls) for n in ['policy.act','policy.load_state_dict','torch.load'])}
    return {'checks':checks,'hashes':hashes,'direct_state_write_calls':writes,
            'reset_reference_build_calls':[c for c in calls if c['call']=='self.reset_reference.build'],
            'visual_only_camera_pose_calls':camera_writes,
            'external_force_calls':external_forces,'passive_drive_authoring_calls':drive_writes,
            'evaluation_step_calls':eval_step_calls,'policy_calls':policy_calls,'step_functions':steps,
            'scope':'Static inspection of the exact saved source, not a runtime instrumentation trace.'}


def longest_interval(mask,t):
    best=None;start=None
    for i,ok in enumerate(mask):
        if ok and start is None: start=i
        if start is not None and (not ok or i==len(mask)-1):
            end=i if ok else i-1
            if best is None or t[end]-t[start]>t[best[1]]-t[best[0]]: best=(start,end)
            start=None
    if not np.any(mask): return {'start_s':None,'end_s':None,'sample_count':0,'observed_span_s':0.}
    a,b=best
    return {'start_s':float(t[a]),'end_s':float(t[b]),'sample_count':b-a+1,'observed_span_s':float(t[b]-t[a])}


def rotations_from_wxyz(quaternions):
    q=np.asarray(quaternions,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all():
        raise ValueError('Invalid recorded orientation quaternion dimensions/values')
    norm=np.linalg.norm(q,axis=1)
    if np.any(np.abs(norm-1)>.001):
        raise ValueError('Recorded rigid-body quaternion is not unit length')
    w,x,y,z=(q/norm[:,None]).T
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=1).reshape(-1,3,3)


def orientation_audit(rows,task,usd,active):
    """Reconstruct from asset FK and raw body states, never a logged pass flag."""
    count=len(rows);q=np.asarray([r['joint_pos'] for r in rows],float)
    tool=np.asarray([r['tool'] for r in rows],float)
    names=task['joint_names'];kin=Kinematics(ASSET/'robot.urdf')
    if len(set(names))!=len(names) or set(names)!=set(kin.active):
        raise ValueError('Joint-name mapping does not match the source URDF')
    base=np.asarray(usd['robot_base_world_matrix_gf_row_vectors'],float).T
    if base.shape!=(4,4) or not np.allclose(base[:3,:3].T@base[:3,:3],np.eye(3),atol=1e-6):
        raise ValueError('Robot base USD transform is not rigid')
    hand_rotations=np.array([base[:3,:3]@kin.fk(dict(zip(names,values)))['right_hand_link'][:3,:3] for values in q])
    palm=hand_rotations[:,:,1]
    radial=np.asarray((task.get('grasp_topology') or {}).get('web_edge_axis_local',[1.,0.,0.]),float)
    if radial.shape!=(3,) or not np.isfinite(radial).all() or np.linalg.norm(radial)<1e-8:
        raise ValueError('Invalid hand-local radial/web-side axis')
    radial/=np.linalg.norm(radial)
    web=hand_rotations@radial
    handle_axis=rotations_from_wxyz(tool[:,3:7])[:,:,2]
    def angle(axis):return np.degrees(np.arctan2(-axis[:,2],np.linalg.norm(axis[:,:2],axis=1)))
    handle_angle=angle(handle_axis)
    geo=task['tool_geometry'];length=float(geo.get('length_m',.14));tip_length=float(geo.get('tip_length_m',.015))
    tip_z=length-tip_length/2
    tip_axis=tip_angle=None;body_states_present=all('left' in r and 'right' in r for r in rows)
    if body_states_present:
        centres=[]
        for name in ['left','right']:
            states=np.asarray([r[name] for r in rows],float)
            if states.shape!=(count,13) or not np.isfinite(states).all():
                raise ValueError(f'Invalid actual {name} jaw state')
            centres.append(states[:,:3]+rotations_from_wxyz(states[:,3:7])[:,:,2]*tip_z)
        delta=.5*(centres[0]+centres[1])-tool[:,:3]
        norm=np.linalg.norm(delta,axis=1)
        if np.any(norm<1e-5):raise ValueError('Degenerate actual root-to-jaw-tip axis')
        tip_axis=delta/norm[:,None];tip_angle=angle(tip_axis)
    require_down=bool(task.get('require_palm_down',False))
    require_web_up=bool(task.get('require_web_up',False))
    min_web=float(task.get('min_web_up_component',.35))
    max_palm=task.get('max_palm_up_component')
    if not np.isfinite(min_web) or not -1<=min_web<=1:
        raise ValueError('Invalid minimum radial/web-side vertical component')
    if max_palm is not None and (not math.isfinite(max_palm) or not -1<=max_palm<=1):
        raise ValueError('Invalid maximum upward palm component')
    bounds=task.get('tool_down_angle_range_deg')
    if bounds is not None:
        if len(bounds)!=2 or not np.isfinite(bounds).all() or not -90<=bounds[0]<=bounds[1]<=90:
            raise ValueError('Invalid orientation acceptance range')
    gated=require_down or require_web_up or max_palm is not None or bounds is not None
    fields=['palm_normal_w','tool_axis_w','tool_down_angle_deg','root_to_tip_axis_w',
            'root_to_tip_down_angle_deg','orientation_acceptable']
    if require_web_up or any('web_side_axis_w' in r for r in rows): fields.append('web_side_axis_w')
    has_new_fields=any(any(k in r for k in fields) for r in rows)
    required=gated or has_new_fields
    complete=body_states_present and all(all(k in r for k in fields) for r in rows)
    checks={};errors={};computed_ok=np.ones(count,dtype=bool)
    if require_down:computed_ok &= palm[:,2]<-.05
    if require_web_up:computed_ok &= web[:,2]>=min_web
    if max_palm is not None:computed_ok &= palm[:,2]<=max_palm
    if bounds is not None:
        computed_ok &= (handle_angle>=bounds[0])&(handle_angle<=bounds[1])
        computed_ok &= np.zeros(count,dtype=bool) if tip_angle is None else (tip_angle>=bounds[0])&(tip_angle<=bounds[1])
    if required:
        checks['orientation_telemetry_and_actual_jaw_states_complete']=complete
        if complete:
            expected={'palm_normal_w':palm,'tool_axis_w':handle_axis,'tool_down_angle_deg':handle_angle,
                      'root_to_tip_axis_w':tip_axis,'root_to_tip_down_angle_deg':tip_angle}
            if 'web_side_axis_w' in fields: expected['web_side_axis_w']=web
            for name,value in expected.items():
                logged=np.asarray([r[name] for r in rows],float)
                valid=logged.shape==value.shape and np.isfinite(logged).all()
                discrepancy=float(np.max(np.abs(logged-value))) if valid else None
                errors[name]=discrepancy
                tolerance=.001 if name.endswith('_deg') else .0005
                checks['independent_orientation_matches_'+name]=bool(valid and discrepancy<=tolerance)
            logged_ok=np.asarray([r['orientation_acceptable'] for r in rows],bool)
            checks['orientation_pass_flag_agrees_with_independent_geometry']=bool(np.array_equal(logged_ok,computed_ok))
    if require_down:
        checks['palm_faces_down_in_all_active_samples']=bool(np.all(palm[active,2]<-.05))
    if require_web_up:
        checks['radial_web_side_faces_up_in_all_active_samples']=bool(np.all(web[active,2]>=min_web))
    if max_palm is not None:
        checks['palm_up_component_below_limit_in_all_active_samples']=bool(np.all(palm[active,2]<=max_palm))
    if bounds is not None:
        checks['both_tool_axes_in_angle_range_in_all_active_samples']=bool(
            tip_angle is not None and np.all((handle_angle[active]>=bounds[0])&(handle_angle[active]<=bounds[1])
                &(tip_angle[active]>=bounds[0])&(tip_angle[active]<=bounds[1])))
    def summary(values):
        if values is None:return None
        v=np.asarray(values)[active]
        return dict(initial=float(v[0]),minimum=float(v.min()),maximum=float(v.max()),median=float(np.median(v)),final=float(v[-1]))
    evidence=dict(method='Asset URDF FK of actual q plus fixed USD robot-base transform; actual handle and two jaw rigid-body poses.',
        kinematics_implementation_sha256=sha(ROOT/'src/kinematics.py'),
        palm_local_outward_normal=[0,1,0],palm_normal_calibration='Finger flexion direction and original palm STL surface normals establish local +Y as the outward palm side.',
        body_states_available=body_states_present,new_orientation_telemetry_present=has_new_fields,
        orientation_requirements_enabled=gated,require_palm_down=require_down,
        require_web_up=require_web_up,web_side_axis_hand_local=radial.tolist(),
        min_web_up_component=min_web if require_web_up else None,max_palm_up_component=max_palm,
        palm_z_strict_upper_bound=-.05 if require_down else None,tool_down_angle_range_deg=bounds,
        actual_tip_center_local_z_m=tip_z,actual_tip_length_m=tip_length,
        angle_definition='atan2(-axis_world_z, hypot(axis_world_x,axis_world_y)) in degrees; 0 is parallel to table, +90 vertically downward.',
        log_max_absolute_discrepancies=errors,active_samples=int(active.sum()),
        palm_z=summary(palm[:,2]),web_side_z=summary(web[:,2]),handle_angle_to_table_deg=summary(handle_angle),
        root_to_jaw_tip_centres_angle_to_table_deg=summary(tip_angle),
        active_orientation_acceptable_fraction=float(computed_ok[active].mean()),
        first_failing_active_frame=int(np.flatnonzero(active&~computed_ok)[0]) if np.any(active&~computed_ok) else None,
        source_of_success_gate='Independently reconstructed values; recorded orientation_acceptable is checked, never used as truth.',
        limitations=['Source URDF and fixed USD base define the robot frame; no human normal/angle ground truth is assumed.',
                    'Checks cover stored control samples, not every PhysX substep.',
                    'For legacy runs without actual jaw states, root-to-tip reconstruction is unavailable and no new orientation requirement is retroactively invented.'])
    return computed_ok,checks,evidence


def tool_geometry_audit(task,launch,usd):
    geo=task['tool_geometry'];length=float(geo.get('length_m',.14));tip_length=float(geo.get('tip_length_m',.015))
    width=float(geo.get('strip_width_m',.007));thickness=float(geo.get('strip_thickness_m',.001));handle_length=float(geo.get('handle_length_m',.006))
    bridge=launch['arguments'].get('bridge_width')
    if bridge is None:bridge=2*float(geo.get('hinge_offset_m',.0045))+thickness
    handle_width=float(geo.get('handle_width_m',width))
    boxes=usd['tool_collision_boxes'];expected={'Handle/Shape':([bridge,handle_width,handle_length],[0,0,-handle_length/2])}
    for side in ['Left','Right']:
        expected[side+'/Blade']=([thickness,width,length-tip_length],[0,0,(length-tip_length)/2])
        expected[side+'/Tip']=([thickness,min(.005,width),tip_length],[0,0,length-tip_length/2])
    extension=geo.get('handle_extension');extension_match=True
    if extension:
        start,end=[np.asarray(extension[k],float) for k in ['start_m','end_m']]
        length=float(np.linalg.norm(end-start));z=(end-start)/length
        x=np.array([1.,0.,0.])-z*z[0];x/=np.linalg.norm(x)
        rotation=np.column_stack((x,np.cross(z,x),z))
        expected['Handle/TailExtension']=([extension['width_x_m'],extension['width_y_m'],length],(.5*(start+end)).tolist())
        authored=boxes.get('Handle/TailExtension')
        extension_match=bool(authored and np.allclose(rotations_from_wxyz(np.array([authored['local_quaternion_wxyz']]))[0],rotation,atol=2e-6))
    elif 'Handle/TailExtension' in boxes:extension_match=False
    matches=extension_match and set(boxes)==set(expected) and all(np.allclose(boxes[n]['size_m'],size,rtol=0,atol=2e-7)
                and np.allclose(boxes[n]['local_translation_m'],position,rtol=0,atol=2e-7)
                for n,(size,position) in expected.items())
    masses={'handle':float(geo.get('handle_mass_kg',.003)),
            'left':float(geo.get('jaw_mass_kg',.006)),'right':float(geo.get('jaw_mass_kg',.006))}
    mass_match=all(math.isfinite(m) and m>0 and abs(float(usd['bodies'][name]['mass_kg'])-m)<1e-7 for name,m in masses.items())
    return {'USD_tool_collision_dimensions_match_task_geometry':bool(matches),'USD_tool_masses_match_configured_or_legacy_defaults':bool(mass_match)},dict(expected_boxes=expected,expected_mass_kg=masses)


def validate(run,opt):
    if opt.min_height<=0 or opt.min_duration<=0 or opt.contact_threshold<0 or opt.joint_tolerance<0:
        raise ValueError('Validation thresholds must have positive height/duration and nonnegative force/tolerance')
    rows=json.loads((run/'trajectory.json').read_text())
    evaluation=json.loads((run/'evaluation.json').read_text())
    launch=json.loads((run/'launch.json').read_text())
    task=json.loads((run/'task.json').read_text())
    usd=usd_audit(run/'final_scene.usda');source=source_audit(run,launch)
    reset_evidence={k:task.get(k) for k in ['initial_tip_clearance','reset_xy_range','closure_bias',
        'closure_scale','reset_snapshot','reset_snapshot_sha256','aperture_reward_weight',
        'reset_tool_quaternion_wxyz','require_palm_down','tool_down_angle_range_deg',
        'require_web_up','min_web_up_component','max_palm_up_component','require_tripod_support','palm_collision']}
    reset_evidence['pose_configuration']=launch['arguments'].get('pose')
    if task.get('reset_snapshot'):
        snapshot_path=Path(task['reset_snapshot'])
        if not snapshot_path.is_absolute(): snapshot_path=Path(__file__).resolve().parents[1]/snapshot_path
        snapshot=json.loads(snapshot_path.read_text())
        reset_evidence['snapshot_file_sha256']=sha(snapshot_path)
        reset_evidence['snapshot_hash_matches_task']=sha(snapshot_path)==task['reset_snapshot_sha256']
        reset_evidence['snapshot_recorded_hold']={k:snapshot.get(k) for k in
            ['source_run','physics_time_s','fixture_active','natural_hold']}
        reset_evidence['snapshot_evidence_scope']='Snapshot metadata from a separate free-hold run; not a replay of that run. Settled hold and preservation of the original pre-settling pose are distinct criteria.'
    t=np.asarray([r['time'] for r in rows],float)
    if len(t)<2: raise ValueError('At least two trajectory records are required')
    q=np.asarray([r['joint_pos'] for r in rows],float)
    cube=np.asarray([r['cube'] for r in rows],float)
    tool=np.asarray([r['tool'] for r in rows],float)
    actions=np.asarray([r['action'] for r in rows],float)
    forces=np.asarray([r['jaw_contact_N'] for r in rows],float)
    recorded_lift=np.asarray([r['cube_lift_m'] for r in rows],float)
    grip=np.asarray([r['tool_grip_error_m'] for r in rows],float)
    reward=np.asarray([r['reward'] for r in rows],float)
    if q.shape!=(len(t),18) or forces.shape!=(len(t),2) or cube.shape!=(len(t),13) or tool.shape!=(len(t),13):
        raise ValueError('Unexpected trajectory tensor dimensions')
    finite=all(np.isfinite(x).all() for x in [t,q,cube,tool,actions,forces,recorded_lift,grip,reward])
    if not finite: raise ValueError('Non-finite recorded numeric state/action/contact/reward')
    policy_dt=float(task['physics_dt'])*int(task['decimation'])
    reference_z=usd['table_top_world_z_m']+usd['block_size_m'][2]/2+.0001
    lift=cube[:,2]-reference_z
    fixture=np.asarray([r['fixture_active'] for r in rows],bool)
    done=np.asarray([r['done'] for r in rows],bool)
    active_end=int(np.flatnonzero(done)[0])+1 if done.any() else len(t)
    active=np.arange(len(t))<active_end
    orientation_ok,orientation_checks,orientation_evidence=orientation_audit(rows,task,usd,active)
    support_ok,support_checks,support_evidence=audit_support(rows,task,usd,active,Kinematics(ASSET/'robot.urdf'),rotations_from_wxyz)
    face_ok,face_checks,face_evidence=audit_face_pinch(rows,task,active,Kinematics(ASSET/'robot.urdf'),rotations_from_wxyz)
    geometry_checks,geometry_evidence=tool_geometry_audit(task,launch,usd)
    good=(lift>=opt.min_height)&(forces>opt.contact_threshold).all(-1)&(grip<.035)&~fixture&active&orientation_ok&support_ok&face_ok
    interval=longest_interval(good,t)
    urdf=ET.parse(ASSET/'robot.urdf').getroot()
    limits={e.get('name'):{k:float(e.find('limit').get(k)) for k in ['lower','upper','effort','velocity']}
            for e in urdf.findall('joint') if e.get('type')!='fixed'}
    joint_report={}
    for column,name in enumerate(task['joint_names']):
        limit=limits[name];minimum=float(q[:,column].min());maximum=float(q[:,column].max())
        violation=max(0.,limit['lower']-minimum,maximum-limit['upper'])
        joint_report[name]={**limit,'observed_min_rad':minimum,'observed_max_rad':maximum,
                            'max_position_violation_rad':violation}
    env='/World/envs/env_0';wrist=env+'/Robot/root_joint/wrist_3_link'
    expected_filters={(wrist,env+'/Robot/root_joint/'+name) for name in
                      ['right_hand_index_rota_link1','right_hand_thumb_rota_link1']}
    joints=usd['tool_block_connected_joints'];spring_reports=[]
    for j in joints:
        d=j['angular_drive']
        stiffness=d['stiffness']['value'];damping=d['damping']['value']
        spring_reports.append({'path':j['path'],'stiffness_Nm_per_rad':stiffness*180/math.pi if stiffness is not None else None,
                               'damping_Nms_per_rad':damping*180/math.pi if damping is not None else None,
                               'rest_angle_deg':d['targetPosition']['value']})
    body_checks=all(b['rigid_body_api'] and b['rigid_body_enabled'] and not b['kinematic_enabled']
                    and not b['disable_gravity']['value'] and b['mass_kg']>0 and b['collision_shapes']
                    for b in usd['bodies'].values())
    springs_ok=len(joints)==2
    for side in ['Left','Right']:
        expected={env+'/Tweezers/Handle',env+'/Tweezers/'+side}
        matching=[j for j in joints if set(j['body_paths'])==expected]
        springs_ok &= len(matching)==1
        if matching:
            j=matching[0];d=j['angular_drive']
            springs_ok &= (j['type']=='PhysicsRevoluteJoint' and j['joint_enabled'] and d['type']['value']=='force'
                and (d['stiffness']['value'] or 0)>0 and (d['damping']['value'] or 0)>=0
                and (d['maxForce']['value'] or 0)>0 and all(a['time_samples']==0 for a in d.values()))
    checks={
        'trajectory_numeric_data_finite':finite,
        'trajectory_clock_matches_policy_dt':bool(np.allclose(np.diff(t),policy_dt,rtol=1e-6,atol=1e-8)),
        'lift_recomputed_from_USD_table_and_block_matches_log':bool(np.max(np.abs(lift-recorded_lift))<2e-6),
        'lift_and_bilateral_contact_continuous_half_second':interval['observed_span_s']>=opt.min_duration-1e-9,
        'no_fixture_declared_in_trajectory':bool(not fixture.any()),
        'URDF_joint_position_limits':all(j['max_position_violation_rad']<=opt.joint_tolerance for j in joint_report.values()),
        'URDF_effort_limits_match_runtime_metadata':all(abs(float(task['robot_effort_limits_Nm'][n])-limits[n]['effort'])<1e-8 for n in task['joint_names']),
        'USD_tool_and_cube_dynamic_with_gravity_and_mass':bool(body_checks),
        'USD_scene_gravity_downward':usd['gravity_magnitude']>0 and np.allclose(usd['gravity_direction'],[0,0,-1]),
        'USD_robot_self_collision_enabled':bool(usd['robot_articulation_roots']) and all(r['enabled_self_collisions']['value'] for r in usd['robot_articulation_roots']),
        'USD_exactly_two_mechanical_pair_filters':set(map(tuple,usd['filtered_pairs']))==expected_filters and len(usd['filtered_pairs'])==2,
        'USD_no_tool_or_cube_fixed_attachment':not any(j['type']=='PhysicsFixedJoint' for j in joints),
        'USD_no_block_joint_attachment':not any(usd['bodies']['block']['path'] in j['body_paths'] for j in joints),
        'USD_only_two_constant_passive_tool_springs':bool(springs_ok),
        'USD_passive_spring_gains_match_launch':len(spring_reports)==2 and all(
            s['stiffness_Nm_per_rad'] is not None and s['damping_Nms_per_rad'] is not None
            and abs(s['stiffness_Nm_per_rad']-float(launch['arguments']['spring']))<1e-5
            and abs(s['damping_Nms_per_rad']-float(launch['arguments']['spring_damping']))<1e-5
            for s in spring_reports),
        **source['checks'],
        **orientation_checks,**geometry_checks,**support_checks,**face_checks,
        'recorded_episode_has_no_midrun_done_or_restart':not done[:-1].any(),
        'evaluation_mode_uses_checkpoint':launch['arguments'].get('evaluate') not in [None,'zero'],
    }
    if evaluation['episodes']==1:
        checks['evaluation_success_agrees_with_trajectory']=bool(evaluation['successes']==1)==checks['lift_and_bilateral_contact_continuous_half_second']
        # The evaluator initializes its running maximum at zero before stepping.
        # A resting block may settle slightly below that initial reference.
        checks['evaluation_max_lift_agrees_with_trajectory']=abs(float(evaluation['max_lift_m'][0])-max(0.,float(lift[:active_end].max())))<2e-6
    if task.get('reset_snapshot'):
        checks['reset_snapshot_hash_matches_task']=reset_evidence['snapshot_hash_matches_task']
    checkpoint=Path(evaluation['checkpoint'])
    if not checkpoint.is_absolute(): checkpoint=Path(__file__).resolve().parents[1]/checkpoint
    checkpoint_evidence={'path':str(checkpoint),'exists':checkpoint.exists(),'sha256':sha(checkpoint) if checkpoint.exists() else None}
    velocity=np.linalg.norm(cube[:,7:10],axis=-1)
    difference=np.linalg.norm(np.diff(cube[:,:3],axis=0)/np.diff(t)[:,None],axis=-1)
    report={'success':all(checks.values()),'run':str(run.resolve()),'checks':checks,
        'criteria':{'min_lift_m':opt.min_height,'min_continuous_observed_span_s':opt.min_duration,
                    'per_jaw_force_strictly_greater_than_N':opt.contact_threshold,'max_logged_grip_error_m':.035,
                    'joint_limit_tolerance_rad':opt.joint_tolerance,
                    'require_palm_down':task.get('require_palm_down',False),
                    'palm_normal_z_strict_upper_bound':-.05 if task.get('require_palm_down') else None,
                    'require_web_up':task.get('require_web_up',False),
                    'min_web_up_component':task.get('min_web_up_component'),
                    'max_palm_up_component':task.get('max_palm_up_component'),
                    'require_tripod_support':task.get('require_tripod_support',False),
                    'grasp_contact_threshold_N':task.get('grasp_contact_threshold_N'),
                    'support_vertical_threshold_N':task.get('support_vertical_threshold_N'),
                    'both_tool_axis_down_angle_range_deg':task.get('tool_down_angle_range_deg')},
        'trajectory_evidence':{'environment_index':0,'samples':len(t),'sample_period_s':policy_dt,
            'independent_lift_reference_world_z_m':reference_z,'task_reference_clearance_above_rest_m':.0001,
            'max_lift_m':float(lift.max()),'max_lift_log_discrepancy_m':float(np.max(np.abs(lift-recorded_lift))),
            'evaluation_max_lift_zero_floor_m':max(0.,float(lift[:active_end].max())),
            'first_sample_lift_m':float(lift[0]),'first_sample_jaw_contact_N':forces[0].tolist(),
            'longest_qualifying_interval':interval,'all_episode_bilateral_contact_fraction':float((forces>opt.contact_threshold).all(-1).mean()),
            'max_grip_error_m':float(grip.max()),'max_URDF_position_violation_rad':max(j['max_position_violation_rad'] for j in joint_report.values()),
            'joints':joint_report,'physics_reported_cube_speed_mean_mps':float(velocity.mean()),
            'position_difference_cube_speed_mean_mps':float(difference.mean())},
        'USD_structural_evidence':usd,'passive_spring_units':spring_reports,'saved_source_evidence':source,
        'independent_orientation_evidence':orientation_evidence,'tool_geometry_evidence':geometry_evidence,
        'independent_support_evidence':support_evidence,
        'independent_broad_face_pinch_evidence':face_evidence,
        'checkpoint_evidence':checkpoint_evidence,
        'reset_conditions':reset_evidence,
        'evaluation_summary':{k:evaluation[k] for k in ['checkpoint','episodes','successes','success_rate']},
        'all_evaluation_trajectories_available':evaluation['episodes']==1,
        'evidence_limits':[
            'Motion/contact continuity is verified at the recorded policy sample rate, not at every PhysX substep.',
            'Final USD proves authored physical structure; Fabric can leave its transform values at spawn poses.',
            'No task-time teleport/wrench and constant passive drives are supported by saved-source AST/hash inspection, not a runtime write-interception trace.',
            'Fixture_active is a declared log field; absence of attachments is independently checked in USD.',
            'URDF position limits are checked on recorded joint positions; measured joint torques and velocities were not saved.',
            'The logged grip translation is checked; the task orientation-retention metric is not included in trajectory.json.',
            'For multi-environment evaluations, trajectory.json records environment 0 only; aggregate success counts cannot be reconstructed for every environment.'],
        'artifact_sha256':{n:sha(run/n) for n in ['trajectory.json','evaluation.json','launch.json','task.json','final_scene.usda']},
        'URDF_sha256':sha(ASSET/'robot.urdf')}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+')
    parser.add_argument('--min-height',type=float,default=.054)
    parser.add_argument('--min-duration',type=float,default=.5)
    parser.add_argument('--contact-threshold',type=float,default=.003)
    parser.add_argument('--joint-tolerance',type=float,default=1e-4)
    opt=parser.parse_args();exit_code=0
    for name in opt.runs:
        run=Path(name)
        try:
            report=validate(run,opt)
            (run/'rl_validation.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'run':name,'success':report['success'],
                'max_lift_m':report['trajectory_evidence']['max_lift_m'],
                'qualifying_interval':report['trajectory_evidence']['longest_qualifying_interval'],
                'failed_checks':[k for k,v in report['checks'].items() if not v],
                'report':str(run/'rl_validation.json')},indent=2))
            exit_code=max(exit_code,0 if report['success'] else 1)
        except Exception as error:
            print(json.dumps({'run':name,'success':False,'evidence_error':str(error)},indent=2));exit_code=2
    return exit_code


if __name__=='__main__': sys.exit(main())
