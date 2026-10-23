"""Run the unmodified upstream SharpaTacmap sensor on this task's live bodies.

The adapter only supplies coordinate transforms, exact triangle equivalents of
authored boxes, recording and provenance. The official class owns ray casting,
CPD, quantization and Gaussian filtering. It never reads the force-map output.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'assets/sharpa_tactile/tacmap_sensor').is_dir())
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
_DIAGNOSTIC_LOCK = threading.RLock()


def file_evidence(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def official_classes():
    # Imported only after SimulationApp is running.
    from sharpa_tacmap_compat import enable_compat
    compatibility = enable_compat()
    package_name = 'sharpa_official_tacmap'
    if package_name not in sys.modules:
        package_dir = ROOT / 'assets/sharpa_tactile'
        spec = importlib.util.spec_from_file_location(package_name, package_dir / '__init__.py',
                                                     submodule_search_locations=[str(package_dir)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    module = importlib.import_module(package_name + '.tacmap_sensor')
    return module.SharpaTacmap, module.SharpaTacmapCfg, compatibility


def box_triangles():
    vertices = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                         [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], float) / 2
    faces = np.array([[0,2,1], [0,3,2], [4,5,6], [4,6,7],
                      [0,1,5], [0,5,4], [3,7,6], [3,6,2],
                      [0,4,7], [0,7,3], [1,2,6], [1,6,5]], int)
    return vertices, faces


def add_tool_ray_meshes(scene, env_id=0):
    """Exact closed triangle copies; no collision/body/mass APIs are added."""
    from pxr import Gf, UsdGeom, Vt
    vertices, faces = box_triangles()
    rows = []
    for body in ('Handle', 'Left', 'Right'):
        path = scene.env_paths[env_id] + '/Tweezers/' + body
        prim = scene.stage.GetPrimAtPath(path)
        points, triangles, originals = [], [], []
        for child in prim.GetChildren():
            if not child.IsA(UsdGeom.Cube):
                continue
            cube = UsdGeom.Cube(child)
            transform = UsdGeom.Xformable(child).GetLocalTransformation()
            local = vertices * float(cube.GetSizeAttr().Get())
            local = np.array([transform.Transform(Gf.Vec3d(*point)) for point in local])
            triangles.extend((faces + len(points)).tolist())
            points.extend(local.tolist())
            originals.append(str(child.GetPath()))
        if not originals:
            raise ValueError('No authored tool boxes at ' + path)
        mesh_path = path + '/TacmapRaySurface'
        mesh = UsdGeom.Mesh.Define(scene.stage, mesh_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
        mesh.CreateFaceVertexIndicesAttr(np.asarray(triangles).reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateVisibilityAttr('invisible')
        rows.append({'body_path': path, 'mesh_path': mesh_path,
                     'source_box_paths': originals, 'vertices_body_m': points,
                     'triangles': triangles, 'collision_api': False,
                     'geometry_sha256': hashlib.sha256(np.asarray(points, '<f8').tobytes()
                        + np.asarray(triangles, '<i4').tobytes()).hexdigest()})
    return rows


class OfficialDiagnosticTap:
    """Observe one unmodified official update without repeating any operation.

    Scoped wrappers return the exact original tensors, never replace arguments,
    and restore module bindings even if the sensor raises. The quantizer mutates
    its input from metres to millimetres, so capture its input before invoking it.
    This is for sequential single-environment recording, not concurrent updates.
    """
    def __init__(self, sensor, capture_rays=False):
        self.namespace = type(sensor)._update_buffers_impl.__globals__
        self.capture_rays = bool(capture_rays)
        self.rays = []
        self.raw_cpd = self.preblur = None

    def __enter__(self):
        _DIAGNOSTIC_LOCK.acquire()
        try:
            self.original_ray = self.namespace['raycast_dynamic_meshes']
            self.original_quantize = self.namespace['deform_quantize']
        except BaseException:
            _DIAGNOSTIC_LOCK.release()
            raise
        def raycast(*args, **kwargs):
            result = self.original_ray(*args, **kwargs)
            row = {'distance': result[1].detach().clone(),
                   'mesh_id': result[4].detach().clone() if result[4] is not None else None}
            if self.capture_rays:
                row['starts'] = args[0].detach().clone()
                if not self.rays:
                    row['directions'] = args[1].detach().clone()
            self.rays.append(row)
            return result
        def quantize(value):
            if self.raw_cpd is not None:
                raise RuntimeError('Diagnostic tap expected exactly one official quantization')
            self.raw_cpd = value.detach().clone()
            result = self.original_quantize(value)
            self.preblur = result.detach().clone()
            return result
        self.namespace['raycast_dynamic_meshes'] = raycast
        self.namespace['deform_quantize'] = quantize
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.namespace['raycast_dynamic_meshes'] = self.original_ray
            self.namespace['deform_quantize'] = self.original_quantize
        finally:
            _DIAGNOSTIC_LOCK.release()
        return False

    def arrays(self, resolution):
        if len(self.rays) != 2 or self.raw_cpd is None or self.preblur is None:
            raise RuntimeError('Expected two original ray casts and one quantization during one sensor update')
        def array(value, vector=False):
            shape = (resolution, resolution, 3) if vector else (resolution, resolution)
            return value.reshape(*shape).cpu().numpy().copy()
        result = {'first_ray_depth_m': array(self.rays[0]['distance']),
                  'reverse_ray_depth_m': array(self.rays[1]['distance']),
                  'raw_cpd_depth_m': array(self.raw_cpd),
                  'preblur_uint8': array(self.preblur)}
        for index, label in [(0, 'first'), (1, 'reverse')]:
            ids = self.rays[index]['mesh_id']
            if ids is not None:
                result[label + '_ray_mesh_id'] = array(ids)
        if self.capture_rays:
            result.update(ray_starts_w=array(self.rays[0]['starts'], True),
                          ray_directions_w=array(self.rays[0]['directions'], True),
                          reverse_ray_starts_w=array(self.rays[1]['starts'], True))
        return result


class OfficialTacmapRecorder:
    def __init__(self, scene, *, resolution_step=1, env_id=0, diagnostics=False, diagnostic_rays=False):
        import torch
        from pxr import Usd, UsdPhysics
        from isaaclab.sensors.ray_caster import patterns

        if resolution_step not in (1, 3):
            raise ValueError('Unmodified official Gaussian kernel supports selected step 1 or 3')
        SharpaTacmap, SharpaTacmapCfg, compatibility = official_classes()
        self.scene = scene
        self.meshes = add_tool_ray_meshes(scene, env_id)
        self.sensors, self.transforms, self.rows = [], [], []
        self.times, self.control_steps, self.images, self.counters, self.poses = [], [], [], [], []
        self.target_poses = []
        self.capture_diagnostics = bool(diagnostics or diagnostic_rays)
        self.capture_diagnostic_rays = bool(diagnostic_rays)
        self.diagnostic_frames = []
        self.valid, self.surface_points, self.surface_normals = [], [], []
        base = scene.env_paths[env_id]
        bodies = {p.GetName(): str(p.GetPath())
                  for p in Usd.PrimRange(scene.stage.GetPrimAtPath(base + '/Robot'))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        self.resolution = 240 // resolution_step
        self.device = scene.device
        from sharpa_tactile import resolve_surface_directory, inspect_contact_geometry
        configured_directory = scene.pose_config.get('tacmap_surface_directory')
        directory = resolve_surface_directory(scene)
        surface_provenance = {'kind': 'unmodified_official_TacMap_surface', 'directory': str(directory)}
        if configured_directory:
            manifest = directory / 'manifest.json'
            if not manifest.is_file():
                raise ValueError('Derived TacMap surface requires a provenance manifest: ' + str(manifest))
            surface_provenance = {'kind': 'current_W01_CAD_derived_surface',
                                  'directory': str(directory), 'manifest': file_evidence(manifest),
                                  'details': json.loads(manifest.read_text())}
        for finger in FINGERS:
            distal, skin = f'right_{finger}_DP', f'right_{finger}_elastomer'
            attached = skin if skin in bodies else distal
            transform = np.eye(4) if attached == skin else scene.kin.fk(root=distal)[skin]
            q = Rotation.from_matrix(transform[:3, :3]).as_quat()
            offset_quat = np.r_[q[3], q[:3]]
            suffix = 'TH' if finger == 'thumb' else '4F'
            origin = '' if finger == 'thumb' else '_origin'
            points = directory / f'tactileSensor_map_{suffix}_point{origin}.npy'
            normals = directory / f'tactileSensor_map_{suffix}_normal{origin}.npy'
            cfg = SharpaTacmapCfg(
                prim_path=bodies[attached],
                mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(
                    prim_expr=row['mesh_path'], track_mesh_transforms=True) for row in self.meshes],
                update_period=0.,
                pattern_cfg=patterns.GridPatternCfg(resolution=.01, size=(.5, .5)),
                offset=SharpaTacmapCfg.OffsetCfg(pos=tuple(float(v) for v in transform[:3, 3]),
                    rot=tuple(float(v) for v in offset_quat), convention='world'),
                data_types=['distance_along_normal'], points_npy=str(points),
                normals_npy=str(normals), resolution_step=resolution_step,
                correction_scale=.001, pts_offsets=0., max_distance=.015,
                cpd_max_dist=.5, debug_vis=False, debug_viz=False)
            sensor = SharpaTacmap(cfg)
            # The task has already started physics before attaching readers.
            # Initialize directly so initialization exceptions cannot be swallowed
            # by SensorBase's timeline callback.
            sensor._initialize_impl()
            sensor._is_initialized = True
            self.sensors.append(sensor)
            self.transforms.append(transform)
            raw_points = np.load(points)[::resolution_step, ::resolution_step]
            raw_normals = np.load(normals)[::resolution_step, ::resolution_step]
            self.surface_points.append(raw_points * .001)
            self.surface_normals.append(raw_normals)
            self.valid.append(np.linalg.norm(raw_normals, axis=-1) > .5)
            method = type(sensor)._update_buffers_impl
            self.rows.append({
                'finger': finger, 'class_name': type(sensor).__name__,
                'class_module': type(sensor).__module__,
                'class_mro': [c.__module__ + '.' + c.__name__ for c in type(sensor).__mro__],
                'implementation_file': str(Path(inspect.getfile(type(sensor))).resolve()),
                'implementation_sha256': file_evidence(inspect.getfile(type(sensor)))['sha256'],
                'update_method_module': method.__module__,
                'update_method_qualname': method.__qualname__,
                'update_method_file': str(Path(inspect.getfile(method)).resolve()),
                'update_method_file_sha256': file_evidence(inspect.getfile(method))['sha256'],
                'config': cfg.to_dict(), 'attached_body': bodies[attached],
                'attached_to_elastomer': transform.tolist(),
                'surface_assets': [file_evidence(points), file_evidence(normals)],
                'official_update_calls': 0})
        self.metadata = {
            'schema_version': 2, 'backend': 'official SharpaTacmap',
            'official_repository': 'https://github.com/sharpa-robotics/sharpa-tacmap',
            'official_commit': 'e4f8827407e15625368ef0b9cf5a72c44304e16b',
            'official_sources': [file_evidence(ROOT / 'assets/sharpa_tactile' / name)
                for name in ('tacmap_sensor/sharpa_tacmap_vbts.py',
                             'tacmap_sensor/sharpa_tacmap_cfg.py', 'torch_jit_utils.py')],
            'compatibility': compatibility, 'sensors': self.rows, 'ray_targets': self.meshes,
            'resolution_step': resolution_step, 'sampling': 'render_endpoint',
            'surface_provenance': surface_provenance,
            'output_semantics': 'quantized_blurred_normal_penetration',
            'output_units': 'official uint8 code 0..255; not N, pressure, or raw millimetres',
            'postprocessing': 'None beyond the unmodified upstream quantization and GaussianBlur',
            'force_map_input_used': False, 'runtime_object_writes': False,
            'contact_physics': (getattr(scene, 'tacmap_skin_contact', None) or
                'Existing task rigid-body PhysX contact/materials unchanged; no fitted compliance added'),
            'adapter': file_evidence(__file__),
            'contact_geometry': inspect_contact_geometry(scene,
                {row['finger']: row['attached_body'] for row in self.rows},
                [row['body_path'] for row in self.meshes]),
            'diagnostics': {'enabled': self.capture_diagnostics,
                'actual_world_rays_recorded': self.capture_diagnostic_rays,
                'method': 'Read-only scoped taps around the two existing ray casts and the one existing quantizer call; no additional ray casts, sensor updates, or quantization.',
                'raw_cpd_depth_m': 'Exact GPU tensor entering official deform_quantize, copied before its in-place millimetre conversion; geometric distance, not pressure.',
                'preblur_uint8': 'Exact output of official deform_quantize before the original GaussianBlur.',
                'ray_depth_miss_value': 'Positive infinity, preserved from original GPU ray casts.',
                'reverse_mesh_id_limitation': 'Official second pass uses return_mesh_id=False; no reverse shape/mesh identity is invented.',
                'thread_scope': 'Sequential single-environment recorder; wrappers restored after each update.'},
        }

    def record(self, time_s, control_step):
        import torch
        from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
        from isaaclab.utils.math import combine_frame_transforms
        elapsed = float(time_s) - (self.times[-1] if self.times else 0.)
        if elapsed <= 0:
            raise ValueError('TacMap record times must increase')
        images, counters, poses, diagnostics = [], [], [], []
        for sensor, row in zip(self.sensors, self.rows):
            before = int(sensor._frame[0])
            if self.capture_diagnostics:
                with OfficialDiagnosticTap(sensor, self.capture_diagnostic_rays) as tap:
                    sensor.update(elapsed, force_recompute=True)
                    output = sensor.data.output['distance_along_normal']
                diagnostics.append(tap.arrays(self.resolution))
            else:
                sensor.update(elapsed, force_recompute=True)
                output = sensor.data.output['distance_along_normal']
            after = int(sensor._frame[0])
            if after != before + 1 or output.dtype != torch.uint8:
                raise RuntimeError('Official sensor update was skipped or duplicated')
            row['official_update_calls'] += 1
            images.append(output.reshape(self.resolution, self.resolution).cpu().numpy().copy())
            counters.append(after)
            position, quaternion = obtain_world_pose_from_view(sensor._view, None)
            position, quaternion = position.reshape(-1, 3), quaternion.reshape(-1, 4)
            position, quaternion = combine_frame_transforms(position, quaternion,
                sensor._offset_pos, sensor._offset_quat)
            poses.append(torch.cat((position, quaternion), dim=-1).reshape(-1, 7)[0].cpu().numpy())
        self.times.append(float(time_s)); self.control_steps.append(int(control_step))
        self.images.append(np.stack(images)); self.counters.append(counters); self.poses.append(np.stack(poses))
        if diagnostics:
            keys = set(diagnostics[0])
            if any(set(item) != keys for item in diagnostics):
                raise RuntimeError('Diagnostic fields differ between official sensors')
            self.diagnostic_frames.append({key: np.stack([item[key] for item in diagnostics]) for key in keys})
        targets = [torch.cat((sensor._mesh_positions_w, sensor._mesh_orientations_w), dim=-1)
                   .reshape(-1, 7).cpu().numpy() for sensor in self.sensors]
        if any(not np.allclose(targets[0], value, atol=1e-6, rtol=0) for value in targets[1:]):
            raise RuntimeError('Five official sensors disagree about live ray-target poses')
        self.target_poses.append(targets[0].copy())

    def save(self, path):
        path = Path(path)
        if not self.times:
            raise RuntimeError('No actual TacMap frames recorded')
        self.metadata.update({'frame_count': len(self.times), 'finger_names': list(FINGERS),
                              'frame_time_range_s': [self.times[0], self.times[-1]]})
        # Config classes include Python class references; preserve their qualified
        # representation in JSON while recording every numeric parameter directly.
        metadata_json = json.dumps(self.metadata, ensure_ascii=False, indent=2, default=str)
        diagnostic_data = ({'diagnostic_' + key: np.stack([frame[key] for frame in self.diagnostic_frames])
                            for key in self.diagnostic_frames[0]} if self.diagnostic_frames else {})
        np.savez_compressed(path, frame_time_s=np.asarray(self.times),
            sample_control_step=np.asarray(self.control_steps, dtype=np.int64),
            deformation_uint8=np.stack(self.images), sensor_frame_counter=np.asarray(self.counters),
            sensor_pose_w=np.stack(self.poses), finger_names=np.asarray(FINGERS),
            ray_target_pose_w=np.stack(self.target_poses),
            surface_valid=np.stack(self.valid), surface_points_m=np.stack(self.surface_points),
            surface_normals=np.stack(self.surface_normals), metadata_json=np.asarray(metadata_json), **diagnostic_data)
        self.metadata['npz_sha256'] = file_evidence(path)['sha256']
        path.with_suffix('.json').write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2, default=str)+'\n')
