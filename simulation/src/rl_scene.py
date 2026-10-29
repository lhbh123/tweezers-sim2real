"""Batched Isaac Lab contact scene for learned tweezer control.

Import this module only after AppLauncher/SimulationApp has initialized Kit.
This module contains no approach, squeeze, lift schedule, or reward function.
All task-time control is the supplied robot joint target tensor.
"""
from pathlib import Path
import json
import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
import omni.usd
from isaacsim.core.cloner import GridCloner
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
from isaaclab.utils.math import quat_apply, quat_mul

from kinematics import Kinematics, transform, ASSET

ROOT = Path(__file__).resolve().parents[1]


def _quat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return np.r_[q[3], q[:3]]


def _pose(prim, matrix):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, matrix[:3, 3])))
    q = _quat(matrix[:3, :3])
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))


class BatchedTweezersScene:
    """GPU tensor API around one UR5e/hand + passive tool + block per env.

    ``reset`` only changes selected environments and performs no simulation
    steps. Optional initial poses are local to each environment and use wxyz.
    ``step`` writes robot joint targets and advances ``decimation`` physics steps.
    ``get_state`` exposes live tensors; clone them when persistent history is needed.
    No fixture, kinematic handle, artificial tool wrench, or block attachment is
    created. Any future reset support must remain outside task-time stepping.
    """

    def __init__(self, num_envs=8, device="cuda:0", dt=1 / 240,
                 pose_json=None, tool_world_R=None, tool_position=(.48, 0., .98),
                 table_height=.72, env_spacing=1.8, spring=1.5,
                 finger_friction=1.4, jaw_friction=.5, block_mass=.002,
                 solver_iterations=16, simplify_arm_collisions=True,
                 initial_joint_positions=None, contact_offset=.0003,
                 spring_damping=.003, hand_stiffness=10., hand_damping=.3,
                 self_collisions=True, filter_mechanical_self_pairs=True,
                 tool_width=None, bridge_width=None, solver_velocity_iterations=2,
                 enable_grasp_support=False, enable_face_metrics=False,
                 table_size=(1.10, .80, .04), table_center_xy=(.35, 0.),
                 block_xy=None):
        self.num_envs, self.device, self.dt = int(num_envs), str(device), float(dt)
        self.table_height = float(table_height)
        self.table_size = np.asarray(table_size, dtype=float)
        self.table_center_xy = np.asarray(table_center_xy, dtype=float)
        if (self.table_size.shape != (3,) or not np.isfinite(self.table_size).all()
                or np.any(self.table_size <= 0.) or self.table_center_xy.shape != (2,)
                or not np.isfinite(self.table_center_xy).all()):
            raise ValueError('Table requires three positive sizes and a finite XY centre')
        self.self_collisions = bool(self_collisions)
        self.filter_mechanical_self_pairs = bool(filter_mechanical_self_pairs)
        self._solver_velocity_iterations = int(solver_velocity_iterations)
        self.grasp_contact_reader = None
        self._grasp_support_cache = None
        self.capture_grasp_substeps = False
        self.grasp_contact_substeps = []
        self.physics_step_callbacks = []
        self.pose_config = json.loads(Path(pose_json or ROOT / 'configs/human_grasp_precision_v2.json').read_text())
        # Optional soft-finger contact-area approximation in the native TGS
        # friction solver. These are collider properties, not an added wrench
        # or attachment. Zero preserves PhysX's default point-contact behavior.
        self.finger_torsional_patch_radius = float(self.pose_config.get('finger_torsional_patch_radius', 0.))
        self.finger_min_torsional_patch_radius = float(self.pose_config.get('finger_min_torsional_patch_radius', 0.))
        for value in (self.finger_torsional_patch_radius, self.finger_min_torsional_patch_radius):
            if not math.isfinite(value) or value < 0.:
                raise ValueError('Finger torsional patch radii must be finite nonnegative distances in metres')
        self.finger_torsional_patch_settings = {
            'torsional_patch_radius_m': self.finger_torsional_patch_radius,
            'min_torsional_patch_radius_m': self.finger_min_torsional_patch_radius,
            'solver': 'TGS',
            'model': 'PhysX single-anchor torsional friction; finite elastomer contact-area approximation',
            'colliders_env_0': [],
        }
        self.robot_asset_dir = Path(self.pose_config.get('robot_asset_dir', ASSET))
        if not self.robot_asset_dir.is_absolute():
            self.robot_asset_dir = ROOT / self.robot_asset_dir
        self.palm_link = self.pose_config.get('palm_link', 'right_hand_link')
        self.kin = Kinematics(self.robot_asset_dir / 'robot.urdf', palm_link=self.palm_link,
                              hand_root_link=self.pose_config.get('hand_root_link'),
                              arm_joint_names=self.pose_config.get('arm_joint_names'))
        self.sim = SimulationContext.instance()
        if self.sim is None:
            self.sim = SimulationContext(SimulationCfg(
                device=self.device, dt=self.dt, render_interval=8, use_fabric=True,
                physx=PhysxCfg(solver_type=1, min_position_iteration_count=solver_iterations,
                               max_position_iteration_count=solver_iterations,
                               min_velocity_iteration_count=self._solver_velocity_iterations,
                               max_velocity_iteration_count=max(4,self._solver_velocity_iterations),
                               enable_ccd=False, enable_stabilization=False,
                               friction_offset_threshold=.001, friction_correlation_distance=.0005,
                               gpu_max_rigid_contact_count=2**20,
                               gpu_max_rigid_patch_count=2**18)))
        self.stage = omni.usd.get_context().get_stage()
        self.env_paths = [f'/World/envs/env_{i}' for i in range(self.num_envs)]
        self.env_regex = '/World/envs/env_.*'
        self.stage.DefinePrim('/World/envs', 'Xform')
        self.stage.DefinePrim(self.env_paths[0], 'Xform')
        base = self.env_paths[0]
        self._contact_offset = float(contact_offset)
        self._solver_iterations = int(solver_iterations)

        def material(name, friction):
            mat = UsdShade.Material.Define(self.stage, '/World/Materials/' + name)
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(float(friction))
            api.CreateDynamicFrictionAttr(float(friction))
            api.CreateRestitutionAttr(0.)
            PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('max')
            return mat

        self._finger_mat = material('RLFinger', finger_friction)
        skin_contact = self.pose_config.get('tacmap_skin_contact')
        self.tacmap_skin_contact = None
        skin_material = None
        if skin_contact is not None:
            stiffness = float(skin_contact['stiffness'])
            damping = float(skin_contact['damping'])
            if stiffness <= 0 or damping < 0:
                raise ValueError('TacMap skin contact requires positive stiffness and nonnegative damping')
            skin_material = material('SharpaTacmapSkin', finger_friction)
            skin_api = PhysxSchema.PhysxMaterialAPI.Apply(skin_material.GetPrim())
            skin_api.CreateCompliantContactStiffnessAttr(stiffness)
            skin_api.CreateCompliantContactDampingAttr(damping)
            self.tacmap_skin_contact = dict(skin_contact, colliders_env_0=[])
        self._jaw_mat = material('RLSteel', jaw_friction)
        self._block_mat = material('RLPlastic', .65)
        self._table_mat = material('RLTable', .7)
        self._box(base + '/Table', self.table_size,
                  (*self.table_center_xy, table_height - .5*self.table_size[2]), self._table_mat)

        robot_prim = self.stage.DefinePrim(base + '/Robot', 'Xform')
        robot_prim.GetReferences().AddReference(str(self.robot_asset_dir / 'robot.usd'))
        _pose(robot_prim, transform([0., 0., table_height]))
        for prim in list(Usd.PrimRange(robot_prim)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(robot_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                self._collision(prim)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(self._finger_mat, materialPurpose='physics')
                is_hand_collision = any(name in str(prim.GetPath()).split('/') for name in self.kin.hand_links)
                is_elastomer = any(name.lower().startswith('elastomer') for name in str(prim.GetPath()).split('/'))
                if is_hand_collision and is_elastomer and skin_material is not None:
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(skin_material, materialPurpose='physics')
                    skin_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    skin_collision.CreateContactOffsetAttr(float(skin_contact['contact_offset_m']))
                    skin_collision.CreateRestOffsetAttr(float(skin_contact['rest_offset_m']))
                    self.tacmap_skin_contact['colliders_env_0'].append(str(prim.GetPath()))
                if is_hand_collision and is_elastomer and (self.finger_torsional_patch_radius or self.finger_min_torsional_patch_radius):
                    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision_api.CreateTorsionalPatchRadiusAttr(self.finger_torsional_patch_radius)
                    collision_api.CreateMinTorsionalPatchRadiusAttr(self.finger_min_torsional_patch_radius)
                    self.finger_torsional_patch_settings['colliders_env_0'].append(str(prim.GetPath()))
                if simplify_arm_collisions and not is_hand_collision:
                    if prim.IsA(UsdGeom.Mesh):
                        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
                palm_collision = self.pose_config.get('palm_collision')
                if palm_collision and '/collisions/'+self.palm_link+'/' in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh):
                    # Refine the supplied palm mesh without bridging its web concavity.
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
                    decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                    decomposition.CreateShrinkWrapAttr(bool(palm_collision.get('shrink_wrap',True)))
                    decomposition.CreateMaxConvexHullsAttr(int(palm_collision.get('max_convex_hulls',64)))
                    decomposition.CreateVoxelResolutionAttr(int(palm_collision.get('voxel_resolution',2000000)))
                    decomposition.CreateMinThicknessAttr(float(palm_collision.get('min_thickness_m',.0001)))
                    decomposition.CreateErrorPercentageAttr(float(palm_collision.get('error_percentage',1.)))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.CreateDisableGravityAttr(True)
                api.CreateSolverPositionIterationCountAttr(solver_iterations)
                api.CreateSolverVelocityIterationCountAttr(self._solver_velocity_iterations)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(self.self_collisions)

        if (self.finger_torsional_patch_radius or self.finger_min_torsional_patch_radius) and not self.finger_torsional_patch_settings['colliders_env_0']:
            raise ValueError('Finger torsional patch approximation requested, but no hand elastomer colliders were found')
        if skin_material is not None and not self.tacmap_skin_contact['colliders_env_0']:
            raise ValueError('TacMap compliant contact requested but no hand elastomer colliders found')

        world_rot = np.diag([1., -1., -1.]) if tool_world_R is None else np.asarray(tool_world_R)
        self.tool_initial_T = transform(tool_position, world_rot)
        hand_T = self.tool_initial_T @ np.linalg.inv(np.asarray(self.pose_config['T_hand_tool']))
        hand_T[:3, 3] -= [0., 0., table_height]
        ik = self.kin.inverse_arm(hand_T)
        q_dict = dict(self.pose_config['q_reference'] if 'q_reference' in self.pose_config else self.pose_config['q_open'])
        clearance = self.pose_config.get('arm_reset_table_clearance_m')
        if clearance is not None:
            # Spawn is a reset too. An arbitrary accurate IK branch can put the
            # forearm through the table and eject an otherwise valid hand grasp
            # on the first physics step. Apply the same source-mesh check used
            # by subsequent task resets before authoring the initial robot.
            from arm_clearance import ArmTableClearance
            wrist_to_hand = self.kin.fk(q_dict,root='wrist_3_link')[self.palm_link]
            wrist_T = hand_T @ np.linalg.inv(wrist_to_hand)
            initial = [ik['q'][n] for n in self.kin.arm]
            ik = ArmTableClearance(self.kin).inverse(wrist_T,initial,
                min_clearance=float(clearance),
                preferred_seed=self.pose_config.get('arm_reset_seed'),
                hand_joint_positions=q_dict)
        self.initial_arm_ik_evidence = ik
        if ik['error'] > 1e-4:
            raise RuntimeError(f'Initial arm IK is infeasible: {ik}')
        q_dict.update(ik['q'])
        if initial_joint_positions is not None:
            q_dict.update(initial_joint_positions)
        q_dict = {name: float(value) for name, value in q_dict.items()}
        self.initial_joint_positions = q_dict
        geo = self.pose_config.get('tool_geometry', {})
        self.hinge_offset = float(geo.get('hinge_offset_m', .0045))
        self.open_angle = float(geo.get('open_angle_rad', .05))
        tool_width = float(geo.get('strip_width_m', .007) if tool_width is None else tool_width)
        self.tool_width = tool_width
        self.tool_length = float(geo.get('length_m', .14))
        self.tip_length = float(geo.get('tip_length_m', .015))
        if not .08 <= self.tool_length <= .25 or not 0 < self.tip_length < self.tool_length:
            raise ValueError('Invalid physical tweezer dimensions')
        self.tip_local_z = self.tool_length - .5*self.tip_length
        self.tip_half_size = (.5*float(geo.get('strip_thickness_m', .001)),
                              .5*min(.005, tool_width), .5*self.tip_length)
        tool_thickness = float(geo.get('strip_thickness_m', .001))
        self._initial_tool_T = {'handle': self.tool_initial_T}
        self._jaw_local_poses = {}
        self.handle_length = float(geo.get('handle_length_m', .006))
        self.handle_width = float(geo.get('handle_width_m', tool_width))
        if not 0 < self.handle_width <= .04 or not 0 < self.handle_length <= .05:
            raise ValueError('Invalid physical tweezer joined-tail dimensions')
        handle_path = base + '/Tweezers/Handle'
        self._rigid(handle_path, self.tool_initial_T, float(geo.get('handle_mass_kg', .003)))
        self.bridge_width = float(2*self.hinge_offset+tool_thickness if bridge_width is None else bridge_width)
        self._box(handle_path + '/Shape', (self.bridge_width, self.handle_width, self.handle_length),
                  (0., 0., -.5*self.handle_length), self._jaw_mat)
        extension = geo.get('handle_extension')
        if extension:
            start,end = [np.asarray(extension[k],dtype=float) for k in ['start_m','end_m']]
            length = float(np.linalg.norm(end-start))
            widths = [float(extension[k]) for k in ['width_x_m','width_y_m']]
            if start.shape!=(3,) or end.shape!=(3,) or not .001<length<.08 or not all(0<w<.04 for w in widths):
                raise ValueError('Invalid physical joined-tail extension')
            z = (end-start)/length
            x = np.array([1.,0.,0.])-z*z[0]
            if np.linalg.norm(x)<1e-5: raise ValueError('Tail extension cannot be parallel to tool X')
            x /= np.linalg.norm(x)
            rotation = np.column_stack((x,np.cross(z,x),z))
            self._box(handle_path+'/TailExtension',(*widths,length),.5*(start+end),self._jaw_mat,rotation)
        for side, sign in [('Left', -1), ('Right', 1)]:
            path = base + '/Tweezers/' + side
            T = self.tool_initial_T @ transform([sign*self.hinge_offset, 0., 0.],
                                                 Rotation.from_euler('y', sign*self.open_angle).as_matrix())
            self._initial_tool_T[side.lower()] = T
            relative_T = np.linalg.inv(self.tool_initial_T) @ T
            self._jaw_local_poses[side.lower()] = torch.tensor(
                np.r_[relative_T[:3, 3], _quat(relative_T[:3, :3])],
                device=self.device, dtype=torch.float32)
            self._rigid(path, T, float(geo.get('jaw_mass_kg', .006)))
            blade_length = self.tool_length-self.tip_length
            self._box(path+'/Blade', (tool_thickness, tool_width, blade_length), (0., 0., .5*blade_length), self._jaw_mat)
            self._box(path+'/Tip', (tool_thickness, min(.005,tool_width), self.tip_length), (0., 0., self.tip_local_z), self._jaw_mat)
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, base + '/Tweezers/Joint' + side)
            joint.CreateBody0Rel().SetTargets([handle_path]); joint.CreateBody1Rel().SetTargets([path])
            joint.CreateLocalPos0Attr(Gf.Vec3f(sign*self.hinge_offset, 0., 0.))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
            joint.CreateAxisAttr('Y'); joint.CreateLowerLimitAttr(-12.); joint.CreateUpperLimitAttr(12.)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(float(spring * math.pi / 180))
            drive.CreateDampingAttr(float(spring_damping * math.pi / 180))
            drive.CreateTargetPositionAttr(float(sign * np.rad2deg(self.open_angle)))
            drive.CreateMaxForceAttr(.2)
        block_xy = np.asarray(tool_position[:2] if block_xy is None else block_xy, dtype=float)
        if block_xy.shape != (2,) or not np.isfinite(block_xy).all():
            raise ValueError('Block position requires a finite XY pair')
        self.block_initial_T = transform([*block_xy, table_height+.0061])
        self._rigid(base + '/Block', self.block_initial_T, block_mass)
        self._box(base+'/Block/Shape', (.012, .012, .012), (0., 0., 0.), self._block_mat)

        self.cloner = GridCloner(spacing=env_spacing)
        self.cloner.define_base_env('/World/envs')
        origins = self.cloner.clone(source_prim_path=base, prim_paths=self.env_paths,
                                     root_path='/World/envs/env_', replicate_physics=True,
                                     copy_from_source=True, enable_env_ids=True)
        self.env_origins = torch.as_tensor(origins, device=self.device, dtype=torch.float32)
        self.origins = self.env_origins
        # Model-specific mechanism exclusions. An alternate hand never inherits
        # the two legacy XHand housing exclusions implicitly.
        self.filtered_mechanical_pairs = []
        if self.filter_mechanical_self_pairs:
            legacy_pairs = [('wrist_3_link','right_hand_index_rota_link1'),
                            ('wrist_3_link','right_hand_thumb_rota_link1')]
            pairs = self.pose_config.get('mechanical_self_collision_pairs',
                legacy_pairs if self.robot_asset_dir.resolve() == ASSET.resolve() else [])
            for env_path in self.env_paths:
                bodies = {p.GetName():p for p in Usd.PrimRange(self.stage.GetPrimAtPath(env_path+'/Robot'))
                          if p.HasAPI(UsdPhysics.RigidBodyAPI)}
                for source_name, target_name in pairs:
                    if source_name not in bodies or target_name not in bodies:
                        raise RuntimeError(f'Mechanical self-pair body missing: {(source_name,target_name)}')
                    source, target = bodies[source_name], bodies[target_name]
                    UsdPhysics.FilteredPairsAPI.Apply(source).CreateFilteredPairsRel().AddTarget(target.GetPath())
                    self.filtered_mechanical_pairs.append((str(source.GetPath()),str(target.GetPath())))
        efforts = {joint.get('name'): float(joint.find('limit').get('effort'))
                   for joint in self.kin.root.findall('joint') if joint.find('limit') is not None}
        self.effort_limits = efforts
        self.robot = Articulation(ArticulationCfg(
            prim_path=self.env_regex + '/Robot', spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., table_height), joint_pos=q_dict),
            actuators={
                'arm': ImplicitActuatorCfg(joint_names_expr=list(self.kin.arm),
                                            stiffness=15000., damping=700.,
                                            effort_limit_sim={n: efforts[n] for n in self.kin.arm}),
                'hand': ImplicitActuatorCfg(joint_names_expr=list(self.kin.hand),
                                             stiffness=hand_stiffness, damping=hand_damping,
                                             effort_limit_sim={n: efforts[n] for n in self.kin.hand})}))
        self.objects = {}
        for key, suffix in [('handle', '/Tweezers/Handle'), ('left', '/Tweezers/Left'),
                            ('right', '/Tweezers/Right'), ('block', '/Block')]:
            self.objects[key] = RigidObject(RigidObjectCfg(prim_path=self.env_regex+suffix, spawn=None))
        self.handle, self.left, self.right, self.block = [self.objects[n] for n in ['handle', 'left', 'right', 'block']]
        self.block_contact = ContactSensor(ContactSensorCfg(
            prim_path=self.env_regex+'/Block', update_period=0., history_length=0,
            filter_prim_paths_expr=[self.env_regex+'/Tweezers/Left', self.env_regex+'/Tweezers/Right'],
            max_contact_data_count_per_prim=16))
        self.sim.reset()
        self.joint_names = list(self.robot.joint_names)
        self.names = self.joint_names
        if len(self.joint_names) != len(self.kin.active) or set(self.joint_names) != set(self.kin.active):
            raise RuntimeError(f'Robot DOFs differ from URDF: expected {self.kin.active}, got {self.joint_names}')
        self.arm_joint_indices = [self.joint_names.index(n) for n in self.kin.arm]
        self.hand_joint_indices = [self.joint_names.index(n) for n in self.kin.hand]
        self.wrist_body_index = self.robot.body_names.index('wrist_3_link')
        # The fixed hand base is merged into wrist_3_link by PhysX. Recover its
        # true frame through the asset's fixed transform, not a body-name guess.
        fk_initial = self.kin.fk(q_dict)
        wrist_to_palm = np.linalg.inv(fk_initial['wrist_3_link']) @ fk_initial[self.palm_link]
        self.wrist_to_palm_position = torch.tensor(wrist_to_palm[:3,3],device=self.device,dtype=torch.float32)
        self.wrist_to_palm_quaternion = torch.tensor(_quat(wrist_to_palm[:3,:3]),device=self.device,dtype=torch.float32)
        self.wrist_jacobian_index = self.wrist_body_index - int(self.robot.is_fixed_base)
        self.default_joint_pos = torch.tensor([q_dict[n] for n in self.joint_names], device=self.device,
                                               dtype=torch.float32).repeat(self.num_envs, 1)
        self._all_ids = torch.arange(self.num_envs, device=self.device)
        self._contact_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.default_object_states = {}
        for key, matrix in {**self._initial_tool_T, 'block': self.block_initial_T}.items():
            state = torch.zeros((self.num_envs, 13), device=self.device)
            state[:, :3] = torch.as_tensor(matrix[:3, 3], device=self.device) + self.env_origins
            state[:, 3:7] = torch.as_tensor(_quat(matrix[:3, :3]), device=self.device)
            self.default_object_states[key] = state
        if enable_grasp_support or enable_face_metrics:
            from grasp_contacts import TripodContactReader
            region = self.pose_config.get('web_contact_region')
            if not region and not enable_face_metrics:
                raise ValueError('Grasp support monitoring requires an explicit hand-local web contact region')
            self.grasp_contact_reader = TripodContactReader(self, capacity=max(32768, self.num_envs*256),
                                                            web_region=region, enable_face_metrics=enable_face_metrics)
        self.reset()

    def _collision(self, prim):
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(self._contact_offset); api.CreateRestOffsetAttr(0.)

    def _box(self, path, size, position, material, rotation=None):
        cube = UsdGeom.Cube.Define(self.stage, path); cube.CreateSizeAttr(1.)
        _pose(cube.GetPrim(), transform(position,rotation))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()); self._collision(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
        return cube

    def _rigid(self, path, matrix, mass):
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim(); _pose(prim, matrix)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        api.CreateSolverPositionIterationCountAttr(self._solver_iterations)
        api.CreateSolverVelocityIterationCountAttr(self._solver_velocity_iterations)
        api.CreateMaxDepenetrationVelocityAttr(.2)
        api.CreateLinearDampingAttr(.01); api.CreateAngularDampingAttr(.01)
        api.CreateSleepThresholdAttr(0.)
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        return prim

    def reset(self, env_ids=None, joint_pos=None, object_states=None, local_object_poses=None,
              *, q=None, tool_rootpose=None, blockpos=None):
        """Reset selected envs; object_states are world-frame [position,wxyz,v,w].

        local_object_poses optionally maps handle/left/right/block to (K,7)
        poses relative to env origins. It does not automatically move linked jaws
        when only the handle is supplied: reset all three together for coherent joints.
        """
        ids = self._all_ids if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if q is not None:
            if joint_pos is not None: raise ValueError('Pass q or joint_pos, not both')
            joint_pos = q
        if tool_rootpose is not None or blockpos is not None:
            object_states = dict(object_states or {})
            if tool_rootpose is not None:
                pose = torch.as_tensor(tool_rootpose, device=self.device, dtype=torch.float32)
                state = torch.zeros((len(ids), 13), device=self.device)
                state[:, :7] = pose
                object_states['handle'] = state
                for name, local in self._jaw_local_poses.items():
                    jaw = torch.zeros_like(state)
                    jaw[:, :3] = pose[:, :3] + quat_apply(pose[:, 3:7], local[:3].expand(len(ids), -1))
                    jaw[:, 3:7] = quat_mul(pose[:, 3:7], local[3:7].expand(len(ids), -1))
                    object_states[name] = jaw
            if blockpos is not None:
                state = self.default_object_states['block'][ids].clone()
                state[:, :3] = torch.as_tensor(blockpos, device=self.device, dtype=torch.float32)
                object_states['block'] = state
        q = self.default_joint_pos[ids] if joint_pos is None else joint_pos
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.robot.set_joint_position_target(q, env_ids=ids)
        self.robot.reset(ids)
        for name, obj in self.objects.items():
            state = self.default_object_states[name][ids].clone()
            if object_states and name in object_states:
                state = object_states[name]
            if local_object_poses and name in local_object_poses:
                state[:, :7] = local_object_poses[name]
                state[:, :3] += self.env_origins[ids]
            obj.write_root_state_to_sim(state, env_ids=ids)
            obj.reset(ids)
        self.block_contact.reset(ids)
        self._contact_valid[ids] = False
        self._grasp_support_cache = None
        self.robot.write_data_to_sim()
        self.update(0.)
        return self.get_state()

    def update(self, dt=None):
        dt = self.dt if dt is None else dt
        self.robot.update(dt)
        for obj in self.objects.values(): obj.update(dt)
        self.block_contact.update(dt, force_recompute=True)

    def step(self, joint_targets, decimation=1, render=False, *, substeps=None):
        if substeps is not None: decimation = substeps
        self.robot.set_joint_position_target(joint_targets)
        accumulated = None
        self.grasp_contact_substeps = []
        for substep in range(int(decimation)):
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.update()
            for callback in self.physics_step_callbacks:
                callback(self.dt)
            if self.grasp_contact_reader is not None:
                state = self.get_state()
                values = self.grasp_contact_reader.read_support_tensors(state,self.dt,with_web=True)
                if accumulated is None: accumulated={k:v.float().clone() for k,v in values.items()}
                else:
                    for k,v in values.items():accumulated[k].add_(v)
                if self.capture_grasp_substeps:
                    self.grasp_contact_substeps.append({
                        'time_within_control_s':(substep+1)*self.dt,
                        'joint_pos':state['joint_pos'][0].cpu().tolist(),
                        **{key:state[body+'_root_state_w'][0].cpu().tolist() for key,body in
                           [('tool','handle'),('left','left'),('right','right')]},
                        'grasp_contact_points':self.grasp_contact_reader.audit_env_points(state,self.dt,env_id=0)})
        self._contact_valid.fill_(True)
        self._grasp_support_cache = {k:v/int(decimation) for k,v in accumulated.items()} if accumulated is not None else None
        if render: self.sim.render()
        return self.get_state()

    def get_jacobians(self):
        """PhysX world-frame Jacobians; fixed-base body indexing excludes base."""
        return self.robot.root_physx_view.get_jacobians()

    def get_wrist_jacobian(self):
        return self.get_jacobians()[:, self.wrist_jacobian_index, :, self.arm_joint_indices]

    def read_grasp_support(self, state=None):
        """Actual contact forces on the tool, cached until the next step/reset.

        Teleported rows are masked until their next physics step, just like the
        block contact sensor. This method neither applies forces nor moves bodies.
        """
        if self.grasp_contact_reader is None:
            return None
        if self._grasp_support_cache is None:
            values = self.grasp_contact_reader.read_support_tensors(
                self.get_state() if state is None else state, self.dt, with_web=True)
            self._grasp_support_cache = {
                name: value*self._contact_valid.reshape((self.num_envs,)+(1,)*(value.ndim-1))
                for name,value in values.items()}
        return self._grasp_support_cache

    def get_state(self):
        data = self.robot.data
        # Teleporting rigid bodies does not recompute PhysX contacts. Do not
        # expose previous-episode impulses until the next physical step.
        contact_forces = self.block_contact.data.force_matrix_w[:, 0] * self._contact_valid[:, None, None]
        wrist_quaternion = data.body_quat_w[:, self.wrist_body_index]
        palm_quaternion = quat_mul(wrist_quaternion,self.wrist_to_palm_quaternion.expand(self.num_envs,4))
        palm_position = data.body_pos_w[:,self.wrist_body_index]+quat_apply(wrist_quaternion,self.wrist_to_palm_position.expand(self.num_envs,3))
        return {'joint_pos': data.joint_pos, 'joint_vel': data.joint_vel,
                'body_pos_w': data.body_pos_w, 'body_quat_w': data.body_quat_w,
                'wrist_pos_w': data.body_pos_w[:, self.wrist_body_index],
                'wrist_quat_w': data.body_quat_w[:, self.wrist_body_index],
                'palm_pos_w': palm_position,
                'palm_quat_w': palm_quaternion,
                'tool_root_state_w': self.handle.data.root_state_w,
                'handle_root_state_w': self.handle.data.root_state_w,
                'left_root_state_w': self.left.data.root_state_w,
                'right_root_state_w': self.right.data.root_state_w,
                'block_root_state_w': self.block.data.root_state_w,
                'cube_root_state_w': self.block.data.root_state_w,
                'block_jaw_contact_forces_w': contact_forces,
                'cube_jaw_contact_forces_w': contact_forces,
                'env_origins': self.env_origins}

    def close(self):
        """Detach Isaac Lab's timeline-stop renderer before SimulationApp.close."""
        self.sim.clear_all_callbacks()
        SimulationContext.clear_instance()
