"""UR5e + XHand passive tweezer task, developed from the supplied URDF/USD.

Only robot joint position targets are commanded after reset. The tool and block
are dynamic bodies; the tweezer joints are passive torsion springs.
"""
import argparse
import json
import sys
import hashlib
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import Kinematics, transform, ASSET

p = argparse.ArgumentParser()
p.add_argument('--out', default='runs/development')
p.add_argument('--pose', default='analysis/control_pose_b_clearance.json')
p.add_argument('--seconds', type=float, default=9.)
p.add_argument('--dt', type=float, default=1/240)
p.add_argument('--render', action='store_true')
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--hold-only', action='store_true')
p.add_argument('--fixed-handle', action='store_true', help='Explicit pre-held handle abstraction for diagnosis')
p.add_argument('--preload', type=float, default=.6, help='Reset fixture duration, removed before approach')
p.add_argument('--close', type=float, default=.25)
p.add_argument('--friction', type=float, default=1.4, help='Fingertip friction coefficient')
p.add_argument('--jaw-friction', type=float, default=.5)
p.add_argument('--spring', type=float, default=1.5)
p.add_argument('--pinch-z', type=float, default=.04)
p.add_argument('--hinge-offset', type=float, default=.0045)
p.add_argument('--open-angle', type=float, default=.05)
p.add_argument('--tool-feedback', action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--grip-index-link', default='right_hand_index_rota_link2')
p.add_argument('--track-block', action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--release-at', type=float, default=-1.)
p.add_argument('--release-close', type=float, default=-.3)
p.add_argument('--offset-x', type=float, default=0.)
p.add_argument('--offset-y', type=float, default=0.)
p.add_argument('--block-mass', type=float, default=.002)
args = p.parse_args()
out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
for source in [Path(__file__),ROOT/'src/kinematics.py']:
    shutil.copy2(source,out/source.name)
shutil.copy2(ROOT/args.pose,out/'control_pose.json')
pose = json.loads((ROOT / args.pose).read_text())

from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 960, 'height': 720,
                     'active_gpu': args.gpu, 'physics_gpu': args.gpu,
                     'multi_gpu': False, 'anti_aliasing': 2,
                     'renderer': 'RayTracedLighting', 'sync_loads': True})
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, PhysxSchema, Gf, Sdf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
import omni.usd
import imageio.v2 as imageio

world = World(stage_units_in_meters=1., physics_dt=args.dt, rendering_dt=1/30, backend='numpy')
stage = omni.usd.get_context().get_stage()
pc = world.get_physics_context()
pc.set_solver_type('TGS')
pc.set_broadphase_type('MBP')
pc.enable_gpu_dynamics(False)
pc.set_gravity(-9.81)
pc.enable_ccd(True)

def quat(rot):
    q=R.from_matrix(rot).as_quat()
    return np.r_[q[3],q[:3]]

def set_pose(prim, T):
    xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float,T[:3,3])))
    q=quat(T[:3,:3]);xf.AddOrientOp().Set(Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:]))))

def material(path, color, metallic=0., roughness=.4, friction=None):
    m=UsdShade.Material.Define(stage,path)
    s=UsdShade.Shader.Define(stage,path+'/Shader');s.CreateIdAttr('UsdPreviewSurface')
    s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    s.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metallic)
    s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
    if friction is not None:
        a=UsdPhysics.MaterialAPI.Apply(m.GetPrim());a.CreateStaticFrictionAttr(friction);a.CreateDynamicFrictionAttr(friction)
        a.CreateRestitutionAttr(0.)
        b=PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim());b.CreateFrictionCombineModeAttr('max')
    return m

metal=material('/World/Materials/Steel',(.46,.53,.61),.85,.24,args.jaw_friction)
finger_material=material('/World/Materials/Fingertip',(.8,.82,.85),.0,.4,args.friction)
orange=material('/World/Materials/Orange',(.95,.25,.035),.0,.3,.65)
tablemat=material('/World/Materials/Table',(.19,.23,.27),.1,.52,.7)
black=material('/World/Materials/Black',(.028,.04,.055),.25,.4,.8)

def box(path, size, pos=(0,0,0), rot=None, mat=None, collision=True):
    c=UsdGeom.Cube.Define(stage,path);c.CreateSizeAttr(1.)
    set_pose(c.GetPrim(),transform(pos, np.eye(3) if rot is None else rot))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float,size)))
    if mat:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
        UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(mat, materialPurpose='physics')
    if collision:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        ca=PhysxSchema.PhysxCollisionAPI.Apply(c.GetPrim());ca.CreateContactOffsetAttr(.0002);ca.CreateRestOffsetAttr(0.)
    return c.GetPrim()

def rigid(path,T,mass):
    pr=UsdGeom.Xform.Define(stage,path).GetPrim();set_pose(pr,T)
    UsdPhysics.RigidBodyAPI.Apply(pr)
    UsdPhysics.MassAPI.Apply(pr).CreateMassAttr(mass)
    a=PhysxSchema.PhysxRigidBodyAPI.Apply(pr)
    a.CreateSolverPositionIterationCountAttr(64);a.CreateSolverVelocityIterationCountAttr(8)
    a.CreateMaxDepenetrationVelocityAttr(.25);a.CreateEnableCCDAttr(True)
    a.CreateLinearDampingAttr(.01);a.CreateAngularDampingAttr(.01)
    return pr

TABLE=.72
BASE=np.array([0.,0.,TABLE])
box('/World/Table/Top',(1.10,.8,.04),(.35,0,TABLE-.02),mat=tablemat)
for x in [-.12,.82]:
    for y in [-.32,.32]:
        box(f'/World/Table/Leg_{int((x+1)*100)}_{int((y+1)*100)}',(.045,.045,.68),(x,y,.34),mat=black)
box('/World/Floor',(20,20,.05),(0,0,-.025),mat=tablemat)
light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(650.)
key=UsdLux.DistantLight.Define(stage,'/World/Key');key.CreateIntensityAttr(1000.);key.CreateAngleAttr(.7)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35,25,-25))

add_reference_to_stage(str(ASSET/'robot.usd'),'/World/Robot')
robotprim=stage.GetPrimAtPath('/World/Robot');set_pose(robotprim,transform(BASE))
# Deinstance in this stage only so contact offsets can be tuned on the source meshes.
for pr in list(Usd.PrimRange(robotprim)):
    if pr.IsInstance():pr.SetInstanceable(False)
colliders=[]
for pr in Usd.PrimRange(robotprim):
    if pr.HasAPI(UsdPhysics.CollisionAPI):
        colliders.append(str(pr.GetPath()))
        ca=PhysxSchema.PhysxCollisionAPI.Apply(pr);ca.CreateContactOffsetAttr(.0002);ca.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(pr).Bind(finger_material,materialPurpose='physics')
    if pr.HasAPI(UsdPhysics.RigidBodyAPI):
        a=PhysxSchema.PhysxRigidBodyAPI.Apply(pr);a.CreateDisableGravityAttr(True)
        a.CreateSolverPositionIterationCountAttr(64);a.CreateSolverVelocityIterationCountAttr(8)

kin=Kinematics()
hand_tool=np.array(pose['T_hand_tool'])
hand_tool[:3,3]+=(pose.get('tool_pinch_z',.075)-args.pinch_z)*hand_tool[:3,2]
q_open=pose['q_open']
q_closed=pose['q_closed']
toolrot=np.diag([1.,-1.,-1.])
tool_start=np.array([.48,0.,TABLE+.14+.12])
T_tool=transform(tool_start,toolrot)
T_hand=T_tool@np.linalg.inv(hand_tool);T_hand[:3,3]-=BASE
ik=kin.inverse_arm(T_hand)
assert ik['error']<1e-4,ik
q_init={**q_open,**ik['q']}
robot=world.scene.add(SingleArticulation('/World/Robot',name='robot'))

handle=rigid('/World/Tweezers/Handle',T_tool,.003)
box('/World/Tweezers/Handle/Bridge',(2*args.hinge_offset+.003,.007,.006),(0,0,-.003),mat=metal)
angle=args.open_angle
sep=args.hinge_offset
toolpaths=[];initial_tool_poses={}
for name,sgn in [('Left',-1),('Right',1)]:
    # local Z is the 140 mm longitudinal axis; spring-rest jaw angle is +/-angle.
    A=T_tool@transform([sgn*sep,0,0],R.from_euler('y',sgn*angle).as_matrix())
    path='/World/Tweezers/'+name;toolpaths.append(path)
    initial_tool_poses[path]=A
    rigid(path,A,.006)
    box(path+'/Blade',(.001,.007,.125),(0,0,.0625),mat=metal)
    box(path+'/Tip',(.001,.005,.015),(0,0,.1325),mat=metal)
    j=UsdPhysics.RevoluteJoint.Define(stage,'/World/Tweezers/Joint'+name)
    j.CreateBody0Rel().SetTargets([handle.GetPath()]);j.CreateBody1Rel().SetTargets([path])
    j.CreateLocalPos0Attr(Gf.Vec3f(sgn*sep,0,0));j.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
    j.CreateAxisAttr('Y');j.CreateLowerLimitAttr(-9.);j.CreateUpperLimitAttr(9.)
    d=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular');d.CreateTypeAttr('force')
    # USD angular drives use degree-based gains, convert Nm/rad to Nm/degree.
    d.CreateStiffnessAttr(args.spring*np.pi/180);d.CreateDampingAttr(.001*np.pi/180)
    d.CreateTargetPositionAttr(float(sgn*np.rad2deg(angle)));d.CreateMaxForceAttr(.2)

fixture=None
fixture_prim=None
if args.fixed_handle or args.preload>0:
    # Diagnostic abstraction is explicitly recorded in metadata and video labels.
    wrist='/World/Robot/root_joint/wrist_3_link'
    T_w=kin.fk(q_init)['wrist_3_link'];T_w[:3,3]+=BASE
    H=np.linalg.inv(T_w)@T_tool
    j=UsdPhysics.FixedJoint.Define(stage,'/World/Tweezers/PreheldHandle')
    j.CreateExcludeFromArticulationAttr(True)
    fixture=j
    fixture_prim=j.GetPrim()
    j.CreateBody0Rel().SetTargets([wrist]);j.CreateBody1Rel().SetTargets([handle.GetPath()])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float,H[:3,3])))
    q=quat(H[:3,:3]);j.CreateLocalRot0Attr(Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:]))))

cube_pos=np.array([.48+args.offset_x,args.offset_y,TABLE+.0061])
rigid('/World/Block',transform(cube_pos),args.block_mass)
box('/World/Block/Shape',(.012,.012,.012),mat=orange)
cube=world.scene.add(SingleRigidPrim('/World/Block',name='block'))
handle_obj=world.scene.add(SingleRigidPrim(str(handle.GetPath()),name='handle'))
left_obj=world.scene.add(SingleRigidPrim(toolpaths[0],name='left'))
right_obj=world.scene.add(SingleRigidPrim(toolpaths[1],name='right'))
force_view=world.scene.add(RigidPrim('/World/Block',name='block_forces',track_contact_forces=True,
    reset_xform_properties=False,contact_filter_prim_paths_expr=toolpaths,max_contact_count=64))
grip_view=world.scene.add(RigidPrim('/World/Tweezers/(Left|Right)',name='grip_forces',track_contact_forces=True,
    reset_xform_properties=False,contact_filter_prim_paths_expr=[
        '/World/Robot/root_joint/right_hand_thumb_rota_link2',
        '/World/Robot/root_joint/'+args.grip_index_link],max_contact_count=128))

def camera(name,eye,target,res=(960,720)):
    z=np.asarray(eye)-target;z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    c=Camera('/World/'+name,resolution=res,frequency=30)
    c.set_world_pose(position=np.array(eye),orientation=quat(np.column_stack([x,y,z])),camera_axes='usd')
    c.set_horizontal_aperture(.036);c.set_focal_length(.028);c.set_clipping_range(.01,100.)
    return c

cams=[]
if args.render:
    cams=[camera('Overview',[1.25,-1.15,1.30],np.array([.30,0,1.])),
          camera('Closeup',[.72,.35,1.08],np.array([.48,0,.86]))]
world.reset()
names=robot.dof_names
assert set(names)==set(kin.active),('Unexpected articulation DOFs; tool must remain passive',names)
q0=np.array([q_init.get(n,0.) for n in names],dtype=np.float32)
robot.set_joint_positions(q0)
robot.set_joint_velocities(np.zeros_like(q0))
robot.apply_action(ArticulationAction(joint_positions=q0))
kp=np.array([15000. if n in kin.arm else 10. for n in names],dtype=np.float32)
kd=np.array([700. if n in kin.arm else .3 for n in names],dtype=np.float32)
robot.get_articulation_controller().set_gains(kps=kp,kds=kd,save_to_usd=True)
for ob,T in [(handle_obj,T_tool),(left_obj,initial_tool_poses[toolpaths[0]]),(right_obj,initial_tool_poses[toolpaths[1]])]:
    ob.set_world_pose(position=T[:3,3],orientation=quat(T[:3,:3]))
    ob.set_linear_velocity(np.zeros(3));ob.set_angular_velocity(np.zeros(3))
cube.set_world_pose(position=cube_pos,orientation=np.array([1.,0.,0.,0.]))
cube.set_linear_velocity(np.zeros(3));cube.set_angular_velocity(np.zeros(3))
for c in cams:c.initialize()
meta={'args':vars(args),'robot_joints':names,'robot_colliders':colliders,'ik':ik,'q_initial':q_init,
      'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'run_tweezers.py',out/'kinematics.py',out/'control_pose.json']},
      'tool_model':'passive dual torsion spring, no jaw actuator',
      'block_constraints':[],'handle_fixed_to_hand':args.fixed_handle,
      'reset_fixture_release_s':args.preload if args.preload>0 else None,
      'robot_gravity_compensation':True,'initialization':'pre-held tool', 'table_height':TABLE}
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
stage.GetRootLayer().Export(str(out/'scene.usda'))
for _ in range(4):world.render()
for i,c in enumerate(cams):
    im=c.get_rgba()
    if im is not None and len(im):imageio.imwrite(out/f'initial_{i}.png',im[:,:,:3])
print('SCENE_READY',len(colliders),names,flush=True)

writers=[]
if args.render:
    writers=[imageio.get_writer(out/f'{s}.mp4',fps=30,codec='libx264',quality=8,macro_block_size=8) for s in ['overview','closeup']]
records=[];trajectories=[]
last_arm=np.array([q_init[n] for n in kin.arm])
grasp_estimate=hand_tool.copy()
clock_start=world.current_time
target_xy=np.array([.48,0.])
controller_fault=None
max_ik_error=0.
def smooth(t):
    t=np.clip(t,0.,1.);return t*t*(3-2*t)

for step in range(int(args.seconds/args.dt)):
    t=step*args.dt
    if fixture is not None and not args.fixed_handle and t>=args.preload:
        fixture.CreateJointEnabledAttr(False)
        print('RESET_FIXTURE_RELEASED',t,flush=True)
        fixture=None
    # Settle -> descend -> squeeze -> lift -> sustained hold.
    z=tool_start[2]-.118*smooth((t-1.)/2.)+.10*smooth((t-4.5)/2.)
    close=args.close*smooth((t-3.2)/.9)
    if args.release_at>0 and t>args.release_at:
        close+=(args.release_close-args.close)*smooth((t-args.release_at)/.6)
    if args.hold_only:z=tool_start[2];close=0.
    if step%4==0 and controller_fault is None:
        if args.tool_feedback and t>=max(.7,args.preload+.1):
            hand_now=kin.fk(dict(zip(names,robot.get_joint_positions())))['right_hand_link']
            hand_now[:3,3]+=BASE
            hp,hq=handle_obj.get_world_pose()
            tool_now=transform(hp,R.from_quat(np.r_[hq[1:],hq[0]]).as_matrix())
            estimate=np.linalg.inv(hand_now)@tool_now
            # Low-pass estimated tool-to-hand transform; all correction is robot joint motion.
            alpha=1-np.exp(-4*args.dt/.13)
            grasp_estimate[:3,3]+=alpha*(estimate[:3,3]-grasp_estimate[:3,3])
            dr=R.from_matrix(grasp_estimate[:3,:3].T@estimate[:3,:3]).as_rotvec()
            grasp_estimate[:3,:3]=grasp_estimate[:3,:3]@R.from_rotvec(alpha*dr).as_matrix()
        if args.track_block and t<3.2:
            target_xy+=.10*(cube.get_world_pose()[0][:2]-target_xy)
        T=transform([*target_xy,z],toolrot)@np.linalg.inv(grasp_estimate);T[:3,3]-=BASE
        sol=kin.inverse_arm(T,initial=last_arm)
        max_ik_error=max(max_ik_error,sol['error'])
        if sol['error']>.002:
            controller_fault='IK target unreachable; robot holds its last safe target'
            print('CONTROLLER_FAULT',controller_fault,sol['error'],flush=True)
        last_arm=np.array([sol['q'][n] for n in kin.arm])
        qcmd={n:(1-close)*q_open.get(n,0.)+close*q_closed.get(n,0.) for n in kin.hand}
        qcmd.update(sol['q'])
        target=np.array([qcmd.get(n,0.) for n in names],dtype=np.float32)
        target=np.clip(target,[kin.by_name[n]['lower'] for n in names],[kin.by_name[n]['upper'] for n in names])
        if controller_fault is None:
            robot.apply_action(ArticulationAction(joint_positions=target))
    do_render=args.render and step%round((1/30)/args.dt)==0
    world.step(render=False)
    if do_render:world.render()
    pos,ori=cube.get_world_pose();hp,hq=handle_obj.get_world_pose()
    forces=force_view.get_contact_force_matrix(dt=args.dt)
    grip_forces=grip_view.get_contact_force_matrix(dt=args.dt)
    rec={'t':t,'block':pos.tolist(),'handle':hp.tolist(),'force':forces.tolist(),
         'fixture_active':fixture is not None,
         'actual_fixture_joint_enabled':bool(fixture_prim.GetAttribute('physics:jointEnabled').Get()) if fixture_prim else False,
         'grip_force':grip_forces.tolist(),'block_velocity':cube.get_linear_velocity().tolist(),
         'phase':'release' if args.release_at>0 and t>args.release_at else 'settle' if t<1 else 'approach' if t<3.2 else 'squeeze' if t<4.5 else 'lift' if t<6.5 else 'hold'}
    records.append(rec)
    if step%round((1/30)/args.dt)==0:
        trajectories.append({'t':t,'joints':robot.get_joint_positions().tolist(),
            'block_pose':[pos.tolist(),ori.tolist()], 'handle_pose':[hp.tolist(),hq.tolist()],
            'left_pose':[x.tolist() for x in left_obj.get_world_pose()],
            'right_pose':[x.tolist() for x in right_obj.get_world_pose()]})
        if do_render:
            for c,w in zip(cams,writers):
                im=c.get_rgba()
                if im is not None and len(im):w.append_data(im[:,:,:3])
    if step%120==0:print('STEP',round(t,2),'cube',pos.round(5),'handle',hp.round(5),'force',np.linalg.norm(forces,axis=-1).round(3),'grip',np.linalg.norm(grip_forces,axis=-1).round(3),flush=True)

for w in writers:w.close()
(out/'telemetry.json').write_text(json.dumps(records))
(out/'trajectory.json').write_text(json.dumps(trajectories))
stage.GetRootLayer().Export(str(out/'final_scene.usda'))
heights=np.array([r['block'][2] for r in records]);held=heights[-min(len(heights),round(2/args.dt)):]
tail=records[-min(len(records),round(2/args.dt)):]
bilateral=np.mean([np.all(np.linalg.norm(r['force'],axis=-1)>.005) for r in tail])
motion_range=np.ptp(np.array([r['block'] for r in tail]),axis=0)
released=all(not r['actual_fixture_joint_enabled'] for r in tail)
passed=bool(controller_fault is None and args.seconds>=2 and np.min(held)>TABLE+.006+.075 and bilateral>=.95 and np.linalg.norm(motion_range)<.003 and released)
result={'success':passed,
        'controller_fault':controller_fault,'max_ik_error':max_ik_error,
        'max_lift_m':float(heights.max()-TABLE-.006),'last_2s_min_lift_m':float(held.min()-TABLE-.006),
        'last_2s_bilateral_contact_fraction':float(bilateral),'last_2s_position_range_m':motion_range.tolist(),
        'last_2s_fixture_disabled':released,
        'final_block_position':records[-1]['block'],'handle_fixed':args.fixed_handle,
        'physics_steps':len(records),'simulated_seconds':args.seconds,'world_time':world.current_time-clock_start}
(out/'result.json').write_text(json.dumps(result,indent=2))
print('RESULT',json.dumps(result),flush=True)
app.close()
