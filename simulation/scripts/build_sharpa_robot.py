"""Build UR5e + official Sharpa Wave W01, optionally importing into Isaac USD.

Run with system Python to build the URDF and provenance. Use isaac-python with
--import-usd (or --import-only) for the separate, headless USD import process.
The 20 mm adapter is an explicit simulation design, not a certified real mount.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_W01_URDF_SHA256 = '261c1a5284e04bc9690e0f5ac39d7a7cace0a16d335d663a1d3c7fcf4e4daf31'
sys.path.insert(0, str(ROOT / 'src'))
from kinematics import ASSET, Kinematics, transform


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_mesh(source, value):
    if value.startswith('package://'):
        _, relative = value[len('package://'):].split('/', 1)
        return source.parent / relative
    if value.startswith('file://'):
        return Path(value[7:])
    return source.parent / value


def build(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arm_path, hand_path = args.arm_urdf.resolve(), args.hand_urdf.resolve()
    robot = ET.Element('robot', name='ur5e_sharpa_w01_right')
    meshes = {}
    for source in [arm_path, hand_path]:
        parsed = ET.parse(source).getroot()
        for element in parsed:
            if element.tag not in ('link', 'joint', 'material'):
                continue
            node = copy.deepcopy(element)
            for mesh in node.findall('.//mesh'):
                path = resolve_mesh(source, mesh.get('filename')).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f'Missing source mesh: {path}')
                meshes[str(path)] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
                mesh.set('filename', os.path.relpath(path, output))
            robot.append(node)
    links = [node.get('name') for node in robot.findall('link')]
    joints = robot.findall('joint')
    if len(links) != len(set(links)):
        raise ValueError('UR5e and hand source link names collide')
    if 'right_hand_flange' not in links or 'tool0' not in links:
        raise ValueError('Expected the official W01 with_flange hand and a UR5e tool0 frame')
    length, radius, density = args.adapter_length, .030, 2700.
    if not 0 < length <= .10:
        raise ValueError('Adapter length must be in (0, 0.10] metres')
    mass = density*math.pi*radius**2*length
    inertia_xy = mass*(3*radius**2+length**2)/12
    inertia_z = .5*mass*radius**2
    adapter = ET.SubElement(robot, 'link', name='sharpa_sim_adapter')
    inertial = ET.SubElement(adapter, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=f'0 0 {length/2}', rpy='0 0 0')
    ET.SubElement(inertial, 'mass', value=str(mass))
    ET.SubElement(inertial, 'inertia', ixx=str(inertia_xy), iyy=str(inertia_xy),
                  izz=str(inertia_z), ixy='0', ixz='0', iyz='0')
    for kind in ['visual', 'collision']:
        element = ET.SubElement(adapter, kind)
        ET.SubElement(element, 'origin', xyz=f'0 0 {length/2}', rpy='0 0 0')
        ET.SubElement(ET.SubElement(element, 'geometry'), 'cylinder', radius=str(radius), length=str(length))
        if kind == 'visual':
            material = ET.SubElement(element, 'material', name='sharpa_adapter_metal')
            ET.SubElement(material, 'color', rgba='0.24 0.27 0.31 1')
    for name, parent, child, xyz, rpy in [
        ('tool0_sharpa_adapter', 'tool0', 'sharpa_sim_adapter', [0,0,0], [0,0,0]),
        ('sharpa_adapter_hand', 'sharpa_sim_adapter', 'right_hand_flange', [0,0,length], args.mount_rpy),
    ]:
        joint = ET.SubElement(robot, 'joint', name=name, type='fixed')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str,xyz)), rpy=' '.join(map(str,rpy)))
    children = [j.find('child').get('link') for j in robot.findall('joint')]
    roots = set(links+['sharpa_sim_adapter'])-set(children)
    if roots != {'base_link'} or len(children) != len(set(children)):
        raise ValueError(f'Combined model must be one tree rooted at base_link: {roots}')
    ET.indent(robot, space='  ')
    target = output/'robot.urdf'
    ET.ElementTree(robot).write(target, encoding='utf-8', xml_declaration=True)
    kin = Kinematics(target, palm_link='right_hand_C_MC', hand_root_link='right_hand_flange')
    if len(kin.arm) != 6 or len(kin.hand) != 22:
        raise ValueError(f'Expected UR5e 6 + W01 22 DOFs: {kin.active}')
    fk = kin.fk()
    by_child = {j['child']:j for j in kin.joints}
    frames = {}
    for link in fk:
        ancestor = link
        while ancestor in by_child and by_child[ancestor]['kind'] == 'fixed':
            ancestor = by_child[ancestor]['parent']
        if ancestor != link:
            frames[link] = {'rigid_body_link': ancestor,
                            'body_T_frame': (np.linalg.inv(fk[ancestor])@fk[link]).tolist()}
    limits = {j.get('name'):{k:float(v) for k,v in j.find('limit').attrib.items()}
              for j in robot.findall('joint') if j.get('type') != 'fixed'}
    report = {
        'schema_version':1, 'model':'UR5e + Sharpa Wave W01 right hand',
        'status':'urdf_built_not_physics_validated', 'w02_asset':False,
        'arm_source':str(arm_path), 'arm_source_sha256':sha256(arm_path),
        'hand_source':str(hand_path), 'hand_source_sha256':sha256(hand_path),
        'official_hand_commit':'0d19cac602f46456b819e4b6a2c09a74982c9a3e' if sha256(hand_path)==OFFICIAL_W01_URDF_SHA256 else None,
        'official_hand_urdf_hash_verified':sha256(hand_path)==OFFICIAL_W01_URDF_SHA256,
        'official_hand_url':'https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/blob/0d19cac602f46456b819e4b6a2c09a74982c9a3e/wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.urdf' if sha256(hand_path)==OFFICIAL_W01_URDF_SHA256 else None,
        'robot_asset_dir':str(output), 'palm_link':kin.palm_link,
        'hand_root_link':kin.hand_root_link, 'joint_names':kin.active,
        'arm_joint_names':kin.arm, 'hand_joint_names':kin.hand, 'joint_limits':limits,
        'mechanical_self_collision_pairs':[],
        'mount':{'parent':'tool0','child':'right_hand_flange','xyz_m':[0,0,length],
                 'rpy_rad':args.mount_rpy,'adapter_radius_m':radius,'adapter_length_m':length,
                 'adapter_mass_kg':mass,'scope':'Explicit simulation adapter; physical bolt/electrical compatibility not validated.'},
        'fixed_frame_map':frames, 'source_meshes':meshes, 'urdf_sha256':sha256(target),
        'source_collision_links':[link.get('name') for link in robot.findall('link') if link.findall('collision')],
        'notes':['Official hand link geometry, joint limits and inertias are retained.',
                 'Only mesh references and the explicit fixed mounting transform are added.',
                 'Tool/block attachments are not part of this robot asset.'],
    }
    (output/'robot.build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'urdf':str(target),'active_joints':len(kin.active),'source_meshes':len(meshes)}),flush=True)
    return report


def import_usd(args, report):
    output = args.output.resolve()/'robot.usd'
    audit_path = output.with_suffix('.import.json')
    audit_path.write_text(json.dumps({'status':'importing','urdf_sha256':report['urdf_sha256']},indent=2)+'\n')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':True,'renderer':'RaytracedLighting','active_gpu':args.device,
                         'physics_gpu':args.device,'multi_gpu':False})
    try:
        import omni.kit.app
        import omni.kit.commands
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.asset.importer.urdf',True)
        ok, options = omni.kit.commands.execute('URDFCreateImportConfig')
        if not ok:raise RuntimeError('URDFCreateImportConfig failed')
        for key,value in {'fix_base':True,'merge_fixed_joints':True,'self_collision':True,
                          'convex_decomp':True,'collision_from_visuals':False,'distance_scale':1.,
                          'create_physics_scene':False,'make_default_prim':True,
                          'default_drive_strength':0.,'default_position_drive_damping':0.}.items():
            setattr(options,key,value)
        ok, articulation = omni.kit.commands.execute('URDFParseAndImportFile',
            urdf_path=str(output.with_suffix('.urdf')),import_config=options,
            dest_path=str(output),get_articulation_root=True)
        if not ok:raise RuntimeError('Official W01 + UR5e import failed')
        for _ in range(3):app.update()
        stage = Usd.Stage.Open(str(output))
        root = stage.GetDefaultPrim()
        UsdGeom.SetStageMetersPerUnit(stage,1.)
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        deinstanced=[]
        while True:
            instances=[p for p in Usd.PrimRange(root) if p.IsInstance()]
            if not instances:break
            for p in instances:
                deinstanced.append(str(p.GetPath()));p.SetInstanceable(False)
        bodies,joints,colliders,roots={},{},[],[]
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies[prim.GetName()]=prim
                body=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                body.CreateDisableGravityAttr(False)
                body.CreateSolverPositionIterationCountAttr(64)
                body.CreateSolverVelocityIterationCountAttr(8)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(str(prim.GetPath()))
                PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(True)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(str(prim.GetPath()))
                contact=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                contact.CreateContactOffsetAttr(.0002);contact.CreateRestOffsetAttr(0.)
                if 'right_' in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
                    dec=PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                    dec.CreateHullVertexLimitAttr(64);dec.CreateMaxConvexHullsAttr(64)
                    dec.CreateVoxelResolutionAttr(2000000);dec.CreateErrorPercentageAttr(.5)
                    dec.CreateMinThicknessAttr(.0001);dec.CreateShrinkWrapAttr(True)
            if prim.IsA(UsdPhysics.RevoluteJoint):joints[prim.GetName()]=prim
        if set(joints)!=set(report['joint_names']) or len(roots)!=1:
            raise RuntimeError(f'Invalid imported articulation: joints={list(joints)}, roots={roots}')
        coverage={n:[str(p.GetPath()) for p in Usd.PrimRange(b) if p.HasAPI(UsdPhysics.CollisionAPI)]
                  for n,b in bodies.items()}
        expected_collision_bodies={report['fixed_frame_map'].get(n,{}).get('rigid_body_link',n)
                                   for n in report['source_collision_links']}
        missing=[n for n in expected_collision_bodies if not coverage.get(n)]
        if missing:raise RuntimeError(f'Imported rigid bodies lost source colliders: {missing}')
        # The official thumb CMC/MCP virtual-axis links have inertia and joints
        # but no source collision geometry. Preserve that topology explicitly.
        collider_free_bodies=[n for n,paths in coverage.items() if not paths]
        for name,prim in joints.items():
            arm=name in report['arm_joint_names'];limit=report['joint_limits'][name]
            drive=UsdPhysics.DriveAPI.Apply(prim,'angular');drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr((15000. if arm else 10.)*math.pi/180)
            drive.CreateDampingAttr((700. if arm else .3)*math.pi/180)
            drive.CreateMaxForceAttr(limit['effort']);drive.CreateTargetPositionAttr(0.)
            drive.CreateTargetVelocityAttr(0.)
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(limit['velocity']*180/math.pi)
        frame_paths={n:str(b.GetPath()) for n,b in bodies.items()}
        for name,info in report['fixed_frame_map'].items():
            body=bodies[info['rigid_body_link']]
            path=str(body.GetPath())+'/__frames/'+name
            xf=UsdGeom.Xform.Define(stage,path);matrix=info['body_T_frame']
            xf.ClearXformOpOrder();xf.AddTransformOp().Set(Gf.Matrix4d(*[matrix[j][i] for i in range(4) for j in range(4)]))
            frame_paths[name]=path
        stage.GetRootLayer().Save()
        audit={'schema_version':1,'status':'imported_not_physics_validated','usd':str(output),
               'urdf_sha256':report['urdf_sha256'],'default_prim':str(root.GetPath()),
               'articulation_roots':roots,'returned_articulation_path':articulation,
               'active_joint_count':len(joints),'rigid_body_count':len(bodies),
               'collider_count':len(colliders),'rigid_body_paths':{n:str(b.GetPath()) for n,b in bodies.items()},
               'frame_paths':frame_paths,'fixed_frame_map':report['fixed_frame_map'],
               'collision_coverage':coverage,'deinstanced_prims':deinstanced,
               'official_collider_free_virtual_bodies':collider_free_bodies,
               'notes':['Official W01 model; not W02.','No physics timeline was played during import.',
                        'One world anchor at the UR5e base; hand attached through fixed mount only.']}
        output.with_suffix('.import.json').write_text(json.dumps(audit,indent=2)+'\n')
        print('SHARPA_IMPORT_RESULT',json.dumps({k:audit[k] for k in ['usd','active_joint_count','rigid_body_count','collider_count']}),flush=True)
    except Exception as error:
        import traceback
        audit_path.write_text(json.dumps({'status':'import_failed','error':str(error),
            'urdf_sha256':report['urdf_sha256']},indent=2)+'\n')
        traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush()
        raise
    finally:
        app.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm-urdf',type=Path,default=ASSET/'ur5e.urdf')
    p.add_argument('--hand-urdf',type=Path,default=ROOT/'assets/sharpa_wave/official/wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.urdf')
    p.add_argument('--output',type=Path,default=ROOT/'assets/ur5e_sharpa_w01_right')
    p.add_argument('--adapter-length',type=float,default=.020)
    p.add_argument('--mount-rpy',type=float,nargs=3,default=[0.,0.,0.])
    p.add_argument('--device',type=int,default=1)
    p.add_argument('--import-usd',action='store_true')
    p.add_argument('--import-only',action='store_true')
    args=p.parse_args()
    report=json.loads((args.output/'robot.build.json').read_text()) if args.import_only else build(args)
    if args.import_usd or args.import_only:import_usd(args,report)


if __name__=='__main__':main()
