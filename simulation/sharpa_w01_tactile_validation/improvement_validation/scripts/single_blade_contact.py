"""Bounded-force, guided 1 mm blade coupon contact: rigid/compliant x skin rest offset.

The rail and bounded external actuator belong to a diagnostic fixture. They are
not a free-tweezers grasp controller. No runtime hand or coupon pose writes occur.
"""
from pathlib import Path
import argparse,json,sys,time,hashlib,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3]
RUN_SOURCE=Path(__file__).read_text()
sys.path.insert(0,str(ROOT/'src'))
parser=argparse.ArgumentParser()
parser.add_argument('--case',choices=['all','rigid_rest0','rigid_restneg','soft_rest0','soft_restneg'],default='all')
parser.add_argument('--output',default=str(ROOT/'sharpa_w01_tactile_validation/improvement_validation/single_blade/run_2x2'))
parser.add_argument('--approach-steps',type=int,default=64)
parser.add_argument('--step-um',type=float,default=25.)
parser.add_argument('--hold-s',type=float,default=.15)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf,Vt
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.sensors.ray_caster import patterns
from sharpa_tacmap import official_classes,box_triangles,file_evidence
from kinematics import Kinematics
SharpaTacmap,SharpaTacmapCfg,compat=official_classes()
from isaaclab.utils.warp import raycast_dynamic_meshes

FINGERS=['thumb','index','middle','ring','pinky']
DT=1/480
INITIAL_GAP=.0007
DRIVE_K=40.
DRIVE_D=1.
DRIVE_MAX=.08
FORCE_STOP=.06
DEPTH_STOP=.00025
OTHER_FINGER_STOP=.002
COUPON_SIZE=np.array([.001,.014,.020])

def cpu(t):return t.detach().cpu().numpy().copy()
def quat(R):return np.roll(Rotation.from_matrix(R).as_quat(),1)
def gfquat(q):return Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:])))

def raw_depth(sensor):
    _,d,_,_,_=raycast_dynamic_meshes(sensor._ray_starts_w,sensor._ray_directions_w,
        mesh_ids_wp=sensor._mesh_ids_wp,max_dist=sensor.cfg.max_distance,
        mesh_positions_w=sensor._mesh_positions_w,mesh_orientations_w=sensor._mesh_orientations_w,
        return_distance=True,return_normal=False)
    starts=sensor._ray_starts_w+sensor._ray_directions_w*(d.clamp_min(0.)-1e-4).unsqueeze(-1)
    _,d2,_,_,_=raycast_dynamic_meshes(starts,-sensor._ray_directions_w,
        mesh_ids_wp=sensor._mesh_ids_wp,max_dist=sensor.cfg.cpd_max_dist,
        mesh_positions_w=sensor._mesh_positions_w,mesh_orientations_w=sensor._mesh_orientations_w,
        return_distance=True,return_normal=False,return_mesh_id=False)
    keep=torch.isfinite(d)&torch.isfinite(d2)
    return cpu(torch.where(keep,d,torch.zeros_like(d))).reshape(240,240),int(keep.sum())

def material(stage,path,mu,soft=False):
    mat=UsdShade.Material.Define(stage,path);api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu);api.CreateDynamicFrictionAttr(mu);api.CreateRestitutionAttr(0.)
    px=PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim());px.CreateFrictionCombineModeAttr('max')
    px.CreateCompliantContactStiffnessAttr(280. if soft else 0.)
    px.CreateCompliantContactDampingAttr(20. if soft else 0.)
    return mat

def binding_record(prim):
    mat,rel=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    return {'collider':str(prim.GetPath()),'material':str(mat.GetPath()),
        'collider_attributes':{a.GetName():a.Get() for a in prim.GetAttributes() if a.GetName() in ['physics:approximation','physxCollision:contactOffset','physxCollision:restOffset'] or a.GetName().startswith('physxConvexDecompositionCollision:')},
        'material_attributes':{a.GetName():a.Get() for a in mat.GetPrim().GetAttributes() if a.GetName().startswith(('physics:','physxMaterial:'))}}

class Case:
    def __init__(self,stage,name,i,out,cfg,kin,efforts):
        self.name=name;self.out=out/name;self.out.mkdir(parents=True,exist_ok=False)
        self.soft=name.startswith('soft');self.rest=-.0001 if name.endswith('restneg') else 0.
        self.base=f'/World/Test_{i}';stage.DefinePrim(self.base,'Xform')
        self.root_pos=np.array([0.,i*.45,.4]);self.root_quat=np.array([1.,0.,0.,0.])
        self.q=dict(cfg['q_reference']);self.kin=kin
        usd=ROOT/'assets/sharpa_wave/official/wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.usda'
        sp=sim_utils.UsdFileCfg(usd_path=str(usd),activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=64,solver_velocity_iteration_count=8),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0002,rest_offset=0.))
        sp.func(self.base+'/Hand',sp,translation=tuple(self.root_pos),orientation=tuple(self.root_quat))
        skin=material(stage,self.base+'/Materials/Skin',1.4,self.soft)
        solid=material(stage,self.base+'/Materials/HandSolid',1.4,False)
        blade_mat=material(stage,self.base+'/Materials/Blade',.5,False)
        root=stage.GetPrimAtPath(self.base+'/Hand')
        for p in list(Usd.PrimRange(root)):
            if p.IsInstance():p.SetInstanceable(False)
        self.paths={};self.skinpaths=[]
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):self.paths[p.GetName()]=str(p.GetPath())
            if not p.HasAPI(UsdPhysics.CollisionAPI):continue
            is_skin='elastomer' in str(p.GetPath()).lower()
            UsdShade.MaterialBindingAPI.Apply(p).Bind(skin if is_skin else solid,materialPurpose='physics')
            c=PhysxSchema.PhysxCollisionAPI.Apply(p);c.CreateContactOffsetAttr(.0002);c.CreateRestOffsetAttr(self.rest if is_skin else 0.)
            if is_skin:
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexDecomposition')
                dc=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(p)
                dc.CreateErrorPercentageAttr(.5);dc.CreateHullVertexLimitAttr(64);dc.CreateMaxConvexHullsAttr(64)
                dc.CreateMinThicknessAttr(.0001);dc.CreateShrinkWrapAttr(True);dc.CreateVoxelResolutionAttr(2000000)
                self.skinpaths.append(str(p.GetPath()))
        self.hand=Articulation(ArticulationCfg(prim_path=self.base+'/Hand',spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(self.root_pos),rot=tuple(self.root_quat),joint_pos=self.q),
            actuators={'hand':ImplicitActuatorCfg(joint_names_expr=['.*'],stiffness=100.,damping=1.,effort_limit_sim={n:efforts[n] for n in self.q})}))
        maps=ROOT/'assets/sharpa_tacmap_w01_matched_v2'
        points=maps/'tactileSensor_map_4F_point_origin.npy';normals=maps/'tactileSensor_map_4F_normal_origin.npy'
        pp=np.load(points)*.001;nn=np.load(normals);self.surfacepoint=pp[120,120];self.surfacenormal=nn[120,120]/np.linalg.norm(nn[120,120])
        F=kin.fk(self.q,root='right_hand_flange')['right_index_elastomer']
        self.axis=F[:3,:3]@self.surfacenormal
        y=F[:3,:3]@np.array([1.,0.,0.]);y-=self.axis*np.dot(y,self.axis);y/=np.linalg.norm(y)
        self.blade_R=np.column_stack([self.axis,y,np.cross(self.axis,y)])
        self.blade_q=quat(self.blade_R)
        self.pad_reference=self.root_pos+F[:3,:3]@self.surfacepoint+F[:3,3]
        self.center0=self.pad_reference+self.axis*(COUPON_SIZE[0]/2+INITIAL_GAP)
        bp=self.base+'/Blade';prim=UsdGeom.Xform.Define(stage,bp).GetPrim()
        xf=UsdGeom.Xformable(prim);xf.AddTranslateOp().Set(Gf.Vec3d(*map(float,self.center0)));xf.AddOrientOp().Set(gfquat(self.blade_q))
        UsdPhysics.RigidBodyAPI.Apply(prim);UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.006)
        rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rb.CreateDisableGravityAttr(True)
        rb.CreateSolverPositionIterationCountAttr(64);rb.CreateSolverVelocityIterationCountAttr(8)
        rb.CreateMaxDepenetrationVelocityAttr(.2);rb.CreateLinearDampingAttr(.01);rb.CreateAngularDampingAttr(.01)
        sim_utils.activate_contact_sensors(bp,0.)
        v,f=box_triangles();v*=COUPON_SIZE
        m=UsdGeom.Mesh.Define(stage,bp+'/Mesh');m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*x) for x in v]));m.CreateFaceVertexCountsAttr([3]*len(f));m.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist());m.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.CollisionAPI.Apply(m.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
        bc=PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim());bc.CreateContactOffsetAttr(.0002);bc.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(blade_mat,materialPurpose='physics')
        rail=UsdPhysics.PrismaticJoint.Define(stage,self.base+'/Guide')
        rail.CreateBody1Rel().SetTargets([bp]);rail.CreateAxisAttr('X')
        rail.CreateLocalPos0Attr(Gf.Vec3f(*map(float,self.center0)));rail.CreateLocalRot0Attr(gfquat(self.blade_q))
        rail.CreateLocalPos1Attr(Gf.Vec3f(0.));rail.CreateLocalRot1Attr(gfquat([1,0,0,0]))
        rail.CreateLowerLimitAttr(-.003);rail.CreateUpperLimitAttr(.003);rail.CreateExcludeFromArticulationAttr(True)
        self.blade=RigidObject(RigidObjectCfg(prim_path=bp,spawn=None))
        self.contacts=[ContactSensor(ContactSensorCfg(prim_path=self.paths[f'right_{f}_elastomer'],update_period=0.,history_length=0,
            filter_prim_paths_expr=[bp],track_contact_points=True,max_contact_data_count_per_prim=128)) for f in FINGERS]
        sc=SharpaTacmapCfg(prim_path=self.paths['right_index_elastomer'],
            mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr=bp+'/Mesh',track_mesh_transforms=True)],
            pattern_cfg=patterns.GridPatternCfg(resolution=.01,size=(.5,.5)),
            offset=SharpaTacmapCfg.OffsetCfg(pos=(0.,0.,0.),rot=(1.,0.,0.,0.),convention='world'),
            points_npy=str(points),normals_npy=str(normals),resolution_step=1,correction_scale=.001,pts_offsets=0.,
            max_distance=.015,cpd_max_dist=.5,update_period=0.,debug_vis=False,debug_viz=False)
        self.sensor=SharpaTacmap(sc)
        self.metadata={'case':name,'fixture':'single-axis dynamic blade coupon; bounded external actuator, gravity disabled to isolate contact. Not a free tweezers task.',
            'case_parameters':{'skin_stiffness':280. if self.soft else 0.,'skin_damping':20. if self.soft else 0.,'skin_rest_m':self.rest,'skin_contact_m':.0002,'skin_mu':1.4,'blade_rest_m':0.,'blade_contact_m':.0002,'blade_mu':.5,'combine':'max'},
            'physics_dt_s':DT,'solver':'TGS64/8','self_collision':True,'global_gravity':[0.,0.,-9.81],'gravity_disabled_on':['hand','blade'],
            'joint_pd':{'stiffness':100.,'damping':1.,'effort_limits_Nm':{n:efforts[n] for n in self.q}},'joint_reference':self.q,
            'skin_proxy':'convexDecomposition; same original task decomposition parameters','blade_size_m':COUPON_SIZE.tolist(),'blade_mass_kg':.006,
            'surface_pixel':[120,120],'initial_gap_m':INITIAL_GAP,'axis_w':self.axis.tolist(),'blade_initial_center_w':self.center0.tolist(),
            'drive':{'type':'bounded explicit PD force on dynamic body, rail constrains other5 DOF','stiffness_N_m':DRIVE_K,'damping_Ns_m':DRIVE_D,'force_limit_N':DRIVE_MAX},
            'schedule':{'approach_steps':args.approach_steps,'step_um':args.step_um,'hold_s':args.hold_s,'warmup_s':1.,'final_release_hold_s':1.},
            'limits':{'filtered_index_force_N':FORCE_STOP,'CPD_depth_m':DEPTH_STOP,'other_finger_force_N':OTHER_FINGER_STOP},
            'runtime_pose_writes_after_initialization':0,'source_usd':file_evidence(usd),'surface_points':file_evidence(points),'surface_normals':file_evidence(normals),'compatibility':compat,
            'material_readback':[binding_record(stage.GetPrimAtPath(p)) for p in self.skinpaths+[bp+'/Mesh']]}
        (self.out/'metadata.json').write_text(json.dumps(self.metadata,indent=2,default=str))
        (self.out/'executed_script.py').write_text(RUN_SOURCE)
        np.savez_compressed(self.out/'geometry.npz',surface_points_m=pp,surface_normals=nn,blade_vertices_m=v,blade_faces=f)
        self.rows=[];self.phase='settle';self.target=0.;self.retreat_start=None;self.retreat_value=0.;self.stop_reason=None;self.force=0.;self.done=False

    def initialize(self):
        self.qtensor=torch.tensor([[self.q[n] for n in self.hand.joint_names]],device=args.device)
        self.hand.write_joint_state_to_sim(self.qtensor,torch.zeros_like(self.qtensor));self.hand.set_joint_position_target(self.qtensor)
        self.hand.reset();self.blade.reset()
        for c in self.contacts:c.reset()
        self.sensor.reset()
        self.metadata['actual_joint_effort_limits_Nm']=cpu(self.hand.data.joint_effort_limits[0]).tolist()
        self.metadata['joint_names']=self.hand.joint_names
        self.metadata['fixed_base']=self.hand.is_fixed_base
        (self.out/'metadata.json').write_text(json.dumps(self.metadata,indent=2,default=str))

    def command(self,t):
        duration=args.approach_steps*args.hold_s
        if self.retreat_start is None:
            if t<1.:self.phase='settle';self.target=0.
            elif t<1.+duration:
                self.phase='approach';j=min(args.approach_steps,int((t-1.)/args.hold_s)+1);self.target=-j*args.step_um*1e-6
            else:self.retreat_start=t;self.retreat_value=self.target;self.phase='release'
        if self.retreat_start is not None:
            elapsed=t-self.retreat_start
            self.phase='release' if elapsed<duration else 'released_hold'
            steps=int(elapsed/args.hold_s)+1
            self.target=min(0.,self.retreat_value+steps*args.step_um*1e-6)
            if elapsed>=duration+1.:self.done=True
        pos=self.blade.data.root_pos_w[0];vel=self.blade.data.root_lin_vel_w[0]
        axis=torch.tensor(self.axis,dtype=torch.float32,device=args.device)
        s=torch.dot(pos-torch.tensor(self.center0,dtype=torch.float32,device=args.device),axis)
        v=torch.dot(vel,axis)
        f=torch.clamp(DRIVE_K*(self.target-s)-DRIVE_D*v,-DRIVE_MAX,DRIVE_MAX)
        self.force=float(f);self.s=float(s)
        self.blade.set_external_force_and_torque((f*axis).reshape(1,1,3),torch.zeros((1,1,3),device=args.device),is_global=True)
        self.hand.set_joint_position_target(self.qtensor);self.hand.write_data_to_sim();self.blade.write_data_to_sim()

    def update(self,t,record):
        self.hand.update(DT);self.blade.update(DT)
        for c in self.contacts:c.update(DT)
        self.sensor.update(DT)
        forces=np.stack([cpu(c.data.force_matrix_w[0,0,0]) for c in self.contacts]);norm=np.linalg.norm(forces,axis=1)
        if self.phase=='approach' and self.retreat_start is None:
            if norm[1]>FORCE_STOP:self.stop_reason='index_force_limit'
            if np.max(norm[[0,2,3,4]])>OTHER_FINGER_STOP:self.stop_reason='unintended_other_finger_contact'
            if self.stop_reason:self.retreat_start=t;self.retreat_value=self.target
        if not record:return
        im=cpu(self.sensor.data.output['distance_along_normal'][0]).reshape(240,240)
        dep,hits=raw_depth(self.sensor)
        transforms=cpu(self.sensor._view.get_transforms()[0]);pose=np.r_[transforms[:3],transforms[3:][[3,0,1,2]]]
        bp=cpu(self.blade.data.root_state_w[0,:7]);rr=Rotation.from_quat(np.roll(pose[3:],-1)).as_matrix()
        br=Rotation.from_quat(np.roll(bp[3:],-1)).as_matrix()
        rel=np.r_[rr.T@(bp[:3]-pose[:3]),quat(rr.T@br)]
        self.rows.append({'time_s':t,'phase':self.phase,'target_displacement_m':self.target,'actual_rail_displacement_m':self.s,
            'drive_force_N':self.force,'gray_u8':im,'raw_CPD_m':dep,'CPD_kept_pixels':hits,'force_blade_w':forces,
            'sensor_pose_w':pose,'blade_pose_w':bp,'blade_pose_sensor':rel,'joint_pos':cpu(self.hand.data.joint_pos[0]),
            'joint_vel':cpu(self.hand.data.joint_vel[0]),'applied_joint_effort':cpu(self.hand.data.applied_torque[0])})
        if self.phase=='approach' and dep.max()>DEPTH_STOP and self.retreat_start is None:
            self.stop_reason='depth_limit';self.retreat_start=t;self.retreat_value=self.target

    def save(self):
        a={k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        np.savez_compressed(self.out/'trajectory.npz',**a)
        gray=a['gray_u8'].reshape(len(self.rows),-1).max(1);depth=a['raw_CPD_m'].reshape(len(self.rows),-1).max(1)
        ff=np.linalg.norm(a['force_blade_w'],axis=-1);tail=a['time_s']>a['time_s'][-1]-.5
        report={'case':self.name,'frames':len(self.rows),'simulation_time_s':float(a['time_s'][-1]),'stop_reason':self.stop_reason or 'completed_bounded_approach',
            'peak_gray':int(gray.max()),'peak_CPD_um':float(depth.max()*1e6),'peak_index_force_N':float(ff[:,1].max()),
            'peak_other_finger_force_N':float(ff[:,[0,2,3,4]].max()),'maximum_actual_rail_approach_mm':float(-a['actual_rail_displacement_m'].min()*1e3),
            'maximum_joint_deviation_rad':float(np.abs(a['joint_pos']-cpu(self.qtensor[0])).max()),
            'final_half_second_max_gray':int(gray[tail].max()),'final_half_second_max_force_N':float(ff[tail].max()),
            'finite_state':bool(np.isfinite(a['joint_pos']).all() and np.isfinite(a['blade_pose_w']).all()),
            'dynamic_pose_writes_during_test':0,'execution_sha256':hashlib.sha256(RUN_SOURCE.encode()).hexdigest()}
        report['contact_test_valid']=bool(report['finite_state'] and report['peak_index_force_N']>.001 and report['peak_index_force_N']<.12 and report['peak_other_finger_force_N']<OTHER_FINGER_STOP and report['final_half_second_max_force_N']<.001 and report['final_half_second_max_gray']==0)
        (self.out/'summary.json').write_text(json.dumps(report,indent=2));print('CASE_RESULT',json.dumps(report),flush=True)
        return report

def main():
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/sharpa_w01_natural_v6d_tacmap_matched_v2.json').read_text())
    urdf=ROOT/'assets/ur5e_sharpa_w01_right/robot.urdf'
    kin=Kinematics(urdf,palm_link=cfg['palm_link'],hand_root_link=cfg['hand_root_link'])
    efforts={j.get('name'):float(j.find('limit').get('effort')) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device,dt=DT,render_interval=24,gravity=(0.,0.,-9.81),
        physx=sim_utils.PhysxCfg(solver_type=1,min_position_iteration_count=64,max_position_iteration_count=64,
            min_velocity_iteration_count=8,max_velocity_iteration_count=8,enable_ccd=False,enable_stabilization=False,
            friction_offset_threshold=.001,friction_correlation_distance=.0005,gpu_max_rigid_contact_count=2**20,gpu_max_rigid_patch_count=2**18)))
    stage=omni.usd.get_context().get_stage()
    names=['rigid_rest0','rigid_restneg','soft_rest0','soft_restneg'] if args.case=='all' else [args.case]
    cases=[Case(stage,n,i,out,cfg,kin,efforts) for i,n in enumerate(names)]
    sim.reset()
    for c in cases:c.initialize()
    print('SINGLE_BLADE_READY',names,flush=True)
    wall=time.time();maxsteps=int((3.+2*args.approach_steps*args.hold_s)/DT)
    for i in range(maxsteps):
        t=(i+1)*DT
        for c in cases:c.command(t)
        sim.step(render=False)
        for c in cases:c.update(t,(i+1)%24==0)
        if (i+1)%480==0:print('PROGRESS',round(t,2),[(c.name,c.phase,round(c.force,5),c.stop_reason) for c in cases],flush=True)
        if all(c.done for c in cases):break
    results=[c.save() for c in cases]
    (out/'run_summary.json').write_text(json.dumps({'wall_time_s':time.time()-wall,'cases':results,'source_urdf':file_evidence(urdf),'source_config':file_evidence(ROOT/'configs/sharpa_w01_natural_v6d_tacmap_matched_v2.json')},indent=2))
    sim.clear_all_callbacks();sim.clear_instance()
try:main()
finally:app.close()
