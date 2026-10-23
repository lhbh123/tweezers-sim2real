#!/usr/bin/env python3
"""Official TacMap geometry controls, separate from the physical pick task.

Runs the unchanged SharpaTacmap class for TH and 4F surface assets against
closed boxes. These intentionally fixed fixtures validate geometric sensing;
they do not provide evidence of physical hand contact or task success.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--resolution-step', type=int, default=1, choices=[1, 3])
parser.add_argument('--surface-directory', type=Path,
                    help='CAD-derived maps with manifest.json; default uses unchanged official maps')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdPhysics, Vt
import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import patterns
from sharpa_tacmap import official_classes, box_triangles, file_evidence


def main():
    SharpaTacmap, SharpaTacmapCfg, compatibility = official_classes()
    directory = (args.surface_directory.resolve() if args.surface_directory else
                 ROOT/'assets/sharpa_tactile/assets/tactilesensor_map')
    surface_provenance = {'kind': 'unmodified_official_TacMap_surface', 'directory': str(directory)}
    if args.surface_directory:
        manifest = directory/'manifest.json'
        surface_provenance = {'kind': 'current_W01_CAD_derived_surface', 'directory': str(directory),
                              'manifest': file_evidence(manifest),
                              'details': json.loads(manifest.read_text())}
    simulation = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1/480, device=args.device, gravity=(0., 0., 0.), render_interval=1))
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, '/World/envs/env_0')
    cases = []
    source_points, source_normals = [], []
    vertices, triangles = box_triangles()
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    outward = np.array([0., -1., 0.])
    half = np.array([.012, .012, .003])
    for model_index, suffix in enumerate(['TH', '4F']):
        origin = '' if suffix == 'TH' else '_origin'
        points_path = directory/f'tactileSensor_map_{suffix}_point{origin}.npy'
        normals_path = directory/f'tactileSensor_map_{suffix}_normal{origin}.npy'
        full_points = np.load(points_path).astype(np.float32)*np.float32(.001)
        full_normals = np.load(normals_path)
        valid = np.linalg.norm(full_normals, axis=-1) > .5
        support = np.where(valid, full_points@outward, -np.inf)
        index = np.unravel_index(support.argmax(), support.shape)
        support_point = full_points[index].astype(float)
        source_points.append(full_points[::args.resolution_step, ::args.resolution_step])
        source_normals.append(full_normals[::args.resolution_step, ::args.resolution_step])
        for case_index, depth in enumerate([-.001, .0001, .0005, .001]):
            name = suffix+('_clear' if depth < 0 else f'_indent_{depth*1000:g}mm')
            sensor_pos = np.array([len(cases)*.08, 0., .3])
            prim_name = 'Control_'+name.replace('.', 'p')
            sensor_path = '/World/envs/env_0/'+prim_name+'_sensor'
            prim = UsdGeom.Xform.Define(stage, sensor_path)
            prim.AddTranslateOp().Set(Gf.Vec3d(*sensor_pos))
            body = UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
            body.CreateRigidBodyEnabledAttr(True)
            body.CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(.001)
            center = sensor_pos+support_point+outward*(half[2]-depth)
            mesh_path = '/World/envs/env_0/'+prim_name+'_box'
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            world_points = ((vertices*(2*half))@rotation.T+center).astype(np.float32)
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, x)) for x in world_points]))
            mesh.CreateFaceVertexCountsAttr([3]*len(triangles))
            mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            cfg = SharpaTacmapCfg(
                prim_path=sensor_path,
                mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr=mesh_path, track_mesh_transforms=False)],
                update_period=0., pattern_cfg=patterns.GridPatternCfg(resolution=.01, size=(.5, .5)),
                offset=SharpaTacmapCfg.OffsetCfg(pos=(0.,0.,0.), rot=(1.,0.,0.,0.), convention='world'),
                data_types=['distance_along_normal'], points_npy=str(points_path), normals_npy=str(normals_path),
                resolution_step=args.resolution_step, correction_scale=.001, pts_offsets=0.,
                max_distance=.015, cpd_max_dist=.5, debug_vis=False, debug_viz=False)
            cases.append({'name': name, 'model': suffix, 'model_index': model_index,
                'nominal_surface_plane_indent_m': depth, 'sensor_path': sensor_path,
                'sensor_pose_w': [*sensor_pos, 1., 0., 0., 0.], 'box_path': mesh_path,
                'boxes': [{'center_m': center.tolist(), 'half_extents_m': half.tolist(),
                           'rotation_matrix': rotation.tolist()}],
                'box_mesh_vertices_w_m': world_points.tolist(), 'box_mesh_triangles': triangles.tolist(),
                'support_point_local_m': support_point.tolist(), 'outward_normal': outward.tolist(),
                'config_object': cfg,
                'surface_assets': [file_evidence(points_path), file_evidence(normals_path)]})
    simulation.reset()
    simulation.step(render=False)
    sensors = []
    for row in cases:
        cfg = row.pop('config_object')
        sensor = SharpaTacmap(cfg)
        sensor._initialize_impl()
        sensor._is_initialized = True
        method = type(sensor)._update_buffers_impl
        row.update({'config': cfg.to_dict(), 'class_name': type(sensor).__name__,
            'class_module': type(sensor).__module__,
            'class_mro': [c.__module__+'.'+c.__name__ for c in type(sensor).__mro__],
            'implementation_file': str(Path(inspect.getfile(type(sensor))).resolve()),
            'implementation_sha256': file_evidence(inspect.getfile(type(sensor)))['sha256'],
            'update_method_module': method.__module__, 'update_method_qualname': method.__qualname__,
            'update_method_file': str(Path(inspect.getfile(method)).resolve()),
            'update_method_file_sha256': file_evidence(inspect.getfile(method))['sha256'],
            'official_update_calls': 0})
        sensors.append(sensor)
    outputs, counters, starts, directions = [], [], [], []
    resolution = 240//args.resolution_step
    for sensor, row in zip(sensors, cases):
        images, frames = [], []
        for repeat in range(2):
            before = int(sensor._frame[0])
            sensor.update(1/30, force_recompute=True)
            result = sensor.data.output['distance_along_normal']
            after = int(sensor._frame[0])
            if after != before+1 or result.dtype != torch.uint8:
                raise RuntimeError('Official sensor did not update exactly once')
            images.append(result.reshape(resolution, resolution).cpu().numpy().copy())
            frames.append(after)
            row['official_update_calls'] += 1
        outputs.append(images); counters.append(frames)
        starts.append(sensor._ray_starts_w[0].reshape(resolution, resolution, 3).cpu().numpy())
        directions.append(sensor._ray_directions_w[0].reshape(resolution, resolution, 3).cpu().numpy())
        print('TACMAP_CONTROL_CASE', json.dumps({'name': row['name'], 'range': [int(np.min(images)),int(np.max(images))],
                                              'nonzero': int(np.count_nonzero(images[0]))}), flush=True)
    metadata = {'schema_version': 1, 'kind': 'independent_official_TacMap_closed_box_geometry_controls',
        'official_repository': 'https://github.com/sharpa-robotics/sharpa-tacmap',
        'official_commit': 'e4f8827407e15625368ef0b9cf5a72c44304e16b',
        'official_sources': [file_evidence(ROOT/'assets/sharpa_tactile'/name) for name in
                            ['tacmap_sensor/sharpa_tacmap_vbts.py', 'tacmap_sensor/sharpa_tacmap_cfg.py', 'torch_jit_utils.py']],
        'compatibility': compatibility, 'cases': cases, 'resolution_step': args.resolution_step,
        'surface_provenance': surface_provenance,
        'fixture_present': True, 'geometry_control_only': True, 'force_map_input_used': False,
        'task_success_claimed': False, 'physical_contact_or_compliance_claimed': False,
        'sampling': 'two repeated official updates of each static geometric fixture',
        'nominal_indent_definition': 'Closed-box near plane is positioned relative to the maximum outward-Y projection of valid source surface points. Curved-surface normal depths vary spatially and are checked by independent analytic ray/box geometry.',
        'script': file_evidence(__file__), 'adapter': file_evidence(ROOT/'src/sharpa_tacmap.py'),
        'arguments': vars(args), 'device': args.device}
    archive = args.out/'controls.npz'
    np.savez_compressed(archive, deformation_uint8=np.asarray(outputs), sensor_frame_counter=np.asarray(counters),
        ray_starts_w=np.asarray(starts), ray_directions_w=np.asarray(directions),
        source_points_m=np.asarray(source_points), source_normals=np.asarray(source_normals),
        sensor_pose_w=np.asarray([x['sensor_pose_w'] for x in cases]),
        case_names=np.asarray([x['name'] for x in cases]), metadata_json=np.asarray(json.dumps(metadata, default=str)))
    metadata['npz_sha256'] = file_evidence(archive)['sha256']
    (args.out/'controls.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
    stage.Export(str(args.out/'fixed_geometry_controls.usda'))
    print('TACMAP_CONTROLS_READY', archive, flush=True)
    simulation.stop()


try:
    main()
except BaseException:
    import traceback
    error = traceback.format_exc()
    print(error, file=sys.stderr, flush=True)
    (args.out/'error.txt').write_text(error)
finally:
    app.close()
