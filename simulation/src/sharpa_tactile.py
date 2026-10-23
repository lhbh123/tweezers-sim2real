"""Record PhysX fingertip contacts and project them onto official W01 surfaces.

This is a force/contact visualizer, not a TacMap deformation or hardware sensor
emulator. Import is safe outside Isaac; construct a reader after sim.reset().
Call record() after EVERY physics step. save() computes time-weighted means over
video-frame intervals, retaining all underlying substep forces/contact points.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Probe runs import their frozen source/ copies. Resolve the data directory from
# ancestors so those copies use the same checksummed official calibration files.
ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent / "assets/sharpa_tactile/assets/tactilesensor_map").is_dir()),
            Path(__file__).resolve().parents[1])
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
RAW_COLUMNS = ("sample_index", "finger_index", "filter_index", "world_x_m",
               "world_y_m", "world_z_m", "sensor_x_m", "sensor_y_m",
               "sensor_z_m", "force_world_x_N", "force_world_y_N",
               "force_world_z_N", "separation_m", "taxel_index",
               "nearest_surface_distance_m")


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def resolve_surface_directory(scene, override=None):
    """Resolve one geometry-map choice shared by force binning and TacMap."""
    configured = override if override is not None else scene.pose_config.get('tacmap_surface_directory')
    if configured is None:
        return ROOT / 'assets/sharpa_tactile/assets/tactilesensor_map'
    path = Path(configured)
    return (path if path.is_absolute() else ROOT / path).resolve()


def inspect_contact_geometry(scene, sensor_paths, filter_paths):
    """Read actual USD collider/material bindings without changing the stage.

    RigidPrim contact buffers identify rigid-body/filter pairs, not individual
    shapes. A collider inventory documents that limitation rather than guessing
    whether a contact belongs to the skin from a nearest-taxel distance.
    """
    from pxr import Usd, UsdPhysics, UsdShade
    rows = []
    for role, path in [(finger, path) for finger, path in sensor_paths.items()] + [
            ('target', path) for path in filter_paths]:
        root = scene.stage.GetPrimAtPath(path)
        if not root:
            raise ValueError('Missing contact body while recording provenance: ' + path)
        colliders = []
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            material, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            material_prim = material.GetPrim() if material else None
            shape_attributes = ('physics:approximation', 'physxCollision:contactOffset',
                                'physxCollision:restOffset', 'physxConvexDecompositionCollision:errorPercentage',
                                'physxConvexDecompositionCollision:minThickness')
            material_attributes = ('physics:staticFriction', 'physics:dynamicFriction', 'physics:restitution',
                                   'physxMaterial:frictionCombineMode', 'physxMaterial:compliantContactStiffness',
                                   'physxMaterial:compliantContactDamping')
            def read_attributes(source, names):
                return {name: source.GetAttribute(name).Get() if source and source.GetAttribute(name) else None
                        for name in names}
            colliders.append({'path': str(prim.GetPath()), 'usd_type': prim.GetTypeName(),
                'anatomical_label_from_path': ('skin' if 'elastomer' in str(prim.GetPath()).lower()
                                               else 'non_skin_or_target'),
                'shape_attributes': read_attributes(prim, shape_attributes),
                'bound_material_path': str(material.GetPath()) if material else None,
                'material_binding_relation': str(relation.GetPath()) if relation else None,
                'material_attributes': read_attributes(material_prim, material_attributes)})
        rows.append({'role': role, 'rigid_body_path': path, 'colliders': colliders})
    return {'bodies': rows, 'capture_time': 'recorder_initialization',
            'unset_attribute_means': 'Not authored or no schema value; no engine default is inferred.',
            'contact_shape_attribution_available': False,
            'attribution_limitation': 'RigidPrim get_contact_force_data/get_friction_data exposes body/filter pairs but no collider IDs. Anatomical labels inventory authored shapes; they do not identify the shape that generated each measured contact.',
            'collision_geometry_limitation': 'USD approximation settings are recorded; cooked PhysX hull vertices are not exported by this inventory.'}


class SurfaceTaxelMap:
    """Nearest-surface force binning; no smoothing, pressure or deformation model."""

    def __init__(self, finger, resolution=60, max_distance_m=.003, directory=None):
        if finger not in FINGERS:
            raise ValueError(f"Unknown W01 finger: {finger}")
        if resolution < 1 or 240 % resolution:
            raise ValueError("resolution must divide the official 240-pixel surface")
        directory = Path(directory or ROOT / "assets/sharpa_tactile/assets/tactilesensor_map")
        suffix = "TH" if finger == "thumb" else "4F"
        origin = "" if finger == "thumb" else "_origin"
        point_file = directory / f"tactileSensor_map_{suffix}_point{origin}.npy"
        normal_file = directory / f"tactileSensor_map_{suffix}_normal{origin}.npy"
        stride = 240 // resolution
        self.points_m = np.load(point_file)[::stride, ::stride].astype(np.float64) * .001
        normals = np.load(normal_file)[::stride, ::stride]
        self.valid = (np.linalg.norm(normals, axis=-1) > .5) & np.isfinite(self.points_m).all(-1)
        self.indices = np.flatnonzero(self.valid)
        if not len(self.indices):
            raise ValueError(f"No valid sensor points in {point_file}")
        self.tree = cKDTree(self.points_m.reshape(-1, 3)[self.indices])
        self.resolution = resolution
        self.max_distance_m = float(max_distance_m)
        if not 0 < self.max_distance_m < .02:
            raise ValueError("max_distance_m must be positive and below 20 mm")
        self.metadata = {"point_file": str(point_file), "normal_file": str(normal_file),
                         "point_sha256": hashlib.sha256(point_file.read_bytes()).hexdigest(),
                         "normal_sha256": hashlib.sha256(normal_file.read_bytes()).hexdigest()}

    def project(self, points_sensor_m):
        points = np.asarray(points_sensor_m).reshape(-1, 3)
        distance, nearest = self.tree.query(points)
        index = self.indices[nearest]
        return np.where(distance <= self.max_distance_m, index, -1), distance


class SharpaTactileReader:
    """One environment, five sensor frames, filtered physical object contacts.

    sensor_paths identifies one rigid body per finger. With fixed-joint merging,
    use right_*_DP and supply sensor_transforms[finger] = DP_T_elastomer. A
    contact farther than max_mapping_distance_m from the official skin surface
    remains in raw/body-force data and is explicitly excluded from skin maps.
    """

    def __init__(self, sensor_paths, filter_paths, *, sensor_transforms=None,
                 resolution=60, max_mapping_distance_m=.003, max_contact_count=2048,
                 surface_directory=None):
        from isaacsim.core.prims import RigidPrim
        if set(sensor_paths) != set(FINGERS):
            raise ValueError(f"sensor_paths must contain exactly {FINGERS}")
        if not filter_paths or len(set(filter_paths)) != len(filter_paths):
            raise ValueError("Provide distinct, nonempty object filter paths")
        self.sensor_paths = {name: str(sensor_paths[name]) for name in FINGERS}
        self.filter_paths = list(map(str, filter_paths))
        self.maps = [SurfaceTaxelMap(name, resolution, max_mapping_distance_m, surface_directory)
                     for name in FINGERS]
        self.transforms = np.asarray([(sensor_transforms or {}).get(name, np.eye(4)) for name in FINGERS])
        if self.transforms.shape != (5, 4, 4) or not np.isfinite(self.transforms).all():
            raise ValueError("sensor_transforms must contain finite 4x4 matrices")
        self.views = []
        self.max_contact_count = int(max_contact_count)
        for name in FINGERS:
            view = RigidPrim(self.sensor_paths[name], name="sharpa_tactile_" + name,
                             track_contact_forces=True, reset_xform_properties=False,
                             contact_filter_prim_paths_expr=self.filter_paths,
                             max_contact_count=self.max_contact_count)
            view.initialize()
            if len(view.prim_paths) != 1:
                raise ValueError(f"Expected one rigid body for {name}: {view.prim_paths}")
            self.views.append(view)
        self.samples = []
        self.normal_records = []
        self.friction_records = []

    @classmethod
    def from_scene(cls, scene, env_id=0, **kwargs):
        """Resolve preserved elastomers or merged DP bodies in an imported scene."""
        from pxr import Usd, UsdPhysics
        base = scene.env_paths[env_id]
        bodies = {prim.GetName(): str(prim.GetPath())
                  for prim in Usd.PrimRange(scene.stage.GetPrimAtPath(base + "/Robot"))
                  if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        paths, transforms = {}, {}
        for finger in FINGERS:
            skin, distal = f"right_{finger}_elastomer", f"right_{finger}_DP"
            name = skin if skin in bodies else distal
            if name not in bodies:
                raise ValueError(f"No sensor rigid body for {finger}")
            paths[finger] = bodies[name]
            transforms[finger] = np.eye(4) if name == skin else scene.kin.fk(root=distal)[skin]
        filters = kwargs.pop("filter_paths", [base + "/Tweezers/" + body
                                              for body in ("Handle", "Left", "Right")])
        directory = resolve_surface_directory(scene, kwargs.pop('surface_directory', None))
        reader = cls(paths, filters, sensor_transforms=transforms, surface_directory=directory, **kwargs)
        reader.contact_model = getattr(scene, "finger_torsional_patch_settings", {})
        reader.surface_directory = str(directory)
        manifest = directory / 'manifest.json'
        reader.surface_provenance = {'directory': str(directory),
            'kind': 'derived_surface_with_manifest' if manifest.is_file() else 'configured_surface_arrays',
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None}
        reader.contact_geometry = inspect_contact_geometry(scene, paths, filters)
        return reader

    def _contact_rows(self, raw, finger, pose, sample_index, friction=False):
        if raw is None:
            raise RuntimeError("PhysX did not provide contact buffers")
        raw = [_numpy(x) for x in raw]
        if friction:
            force, point, count, start = raw
            separation = np.full((len(force), 1), np.nan)
        else:
            magnitude, point, normal, separation, count, start = raw
            force = magnitude * normal
        count, start = count.reshape(-1), start.reshape(-1)
        if int(count.sum()) >= self.max_contact_count:
            raise RuntimeError("Contact buffer reached capacity; increase max_contact_count")
        rows = []
        for pair, (number, begin) in enumerate(zip(count, start)):
            number, begin = int(number), int(begin)
            if number == 0:
                continue
            if begin < 0 or begin + number > len(force):
                raise RuntimeError("Invalid PhysX contact-buffer range")
            indexes = np.arange(begin, begin + number)
            if not np.isfinite(force[indexes]).all() or not np.isfinite(point[indexes]).all():
                raise RuntimeError("Nonfinite PhysX contact data; do not replace it with zero tactile readings")
            indexes = indexes[np.linalg.norm(force[indexes], axis=-1) > 1e-9]
            if not len(indexes):
                continue
            world = point[indexes]
            local = (world - pose[:3, 3]) @ pose[:3, :3]
            taxel, distance = self.maps[finger].project(local)
            rows.append(np.column_stack((np.full(len(indexes), sample_index),
                                         np.full(len(indexes), finger),
                                         np.full(len(indexes), pair % len(self.filter_paths)),
                                         world, local, force[indexes], separation[indexes].reshape(-1),
                                         taxel, distance)))
        return np.concatenate(rows) if rows else np.empty((0, len(RAW_COLUMNS)), np.float64)

    def record(self, time_s, dt):
        """Capture immediately after one physics step, before any reset/write."""
        time_s, dt = float(time_s), float(dt)
        if not np.isfinite(time_s) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Finite time_s and positive physics dt are required")
        if self.samples and abs(time_s - self.samples[-1]["time_s"] - dt) > max(1e-7, dt * .001):
            raise ValueError("Tactile record() must be called once per consecutive physics substep")
        sample_index = len(self.samples)
        sample = {"time_s": time_s, "dt_s": dt, "sensor_pose_w": np.zeros((5, 7))}
        for key in ("normal", "friction"):
            sample[key + "_load_N"] = np.zeros(5)
            sample[key + "_force_w_N"] = np.zeros((5, 3))
            sample[key + "_unmapped_load_N"] = np.zeros(5)
            sample[key + "_contact_count"] = np.zeros(5, dtype=np.int32)
        for finger, view in enumerate(self.views):
            position, quaternion = map(_numpy, view.get_world_poses())
            rigid_pose = np.eye(4)
            rigid_pose[:3, :3] = Rotation.from_quat(quaternion[0][[1, 2, 3, 0]]).as_matrix()
            rigid_pose[:3, 3] = position[0]
            pose = rigid_pose @ self.transforms[finger]
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            sample["sensor_pose_w"][finger] = np.r_[pose[:3, 3], q[3], q[:3]]
            for key, raw, records in (
                    ("normal", view.get_contact_force_data(dt=dt), self.normal_records),
                    ("friction", view.get_friction_data(dt=dt), self.friction_records)):
                points = self._contact_rows(raw, finger, pose, sample_index, key == "friction")
                if len(points):
                    vectors = points[:, 9:12]
                    loads = np.linalg.norm(vectors, axis=-1)
                    sample[key + "_load_N"][finger] = loads.sum()
                    sample[key + "_force_w_N"][finger] = vectors.sum(0)
                    sample[key + "_unmapped_load_N"][finger] = loads[points[:, 13] < 0].sum()
                    sample[key + "_contact_count"][finger] = len(points)
                    records.append(points)
        self.samples.append(sample)
        return sample

    def save(self, path, frame_times_s=None, fps=30.):
        """Write safe, pickle-free NPZ plus physical semantics and provenance JSON.

        Each frame stores the dt-weighted mean over (previous frame time, frame
        time]. At the first frame the interval begins at the first physics sample
        start. All source substeps and sparse contact records are also retained.
        """
        if not self.samples:
            raise ValueError("No physical samples have been recorded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"substep_" + key: np.asarray([sample[key] for sample in self.samples])
                for key in self.samples[0]}
        times, dts = data["substep_time_s"], data["substep_dt_s"]
        start = times[0] - dts[0]
        if frame_times_s is None:
            count = max(1, int(np.ceil((times[-1] - start) * fps)))
            frame_times_s = np.minimum(start + np.arange(1, count + 1) / fps, times[-1])
        frame_times = np.asarray(frame_times_s, np.float64)
        if (frame_times.ndim != 1 or len(frame_times) == 0 or not np.isfinite(frame_times).all()
                or np.any(np.diff(frame_times) <= 0) or frame_times[0] <= start
                or frame_times[-1] > times[-1] + 1e-6):
            raise ValueError("Frame times must increase within the recorded physical interval")
        begins = np.r_[start, frame_times[:-1]]
        durations = frame_times - begins
        resolution = self.maps[0].resolution
        overlaps = []
        data["frame_time_s"], data["frame_interval_s"] = frame_times, durations
        data['endpoint_sample_time_s'] = np.zeros(len(frame_times))
        data["finger_names"] = np.asarray(FINGERS)
        data["surface_valid"] = np.stack([surface.valid for surface in self.maps])
        data["surface_points_m"] = np.stack([surface.points_m for surface in self.maps])
        data["raw_columns"] = np.asarray(RAW_COLUMNS)
        for key in ("normal", "friction"):
            data['substep_' + key + '_resultant_N'] = np.linalg.norm(data['substep_' + key + '_force_w_N'], axis=-1)
            for suffix in ("load_N", "force_w_N", "unmapped_load_N"):
                source = data["substep_" + key + "_" + suffix]
                data[key + "_" + suffix] = np.zeros((len(frame_times),) + source.shape[1:])
            data[key + "_peak_load_N"] = np.zeros((len(frame_times), 5))
            data[key + '_mean_resultant_N'] = np.zeros((len(frame_times), 5))
            data[key + '_endpoint_load_N'] = np.zeros((len(frame_times), 5))
            data[key + '_endpoint_force_w_N'] = np.zeros((len(frame_times), 5, 3))
            data[key + "_map_N"] = np.zeros((len(frame_times), 5, resolution, resolution), np.float32)
        for frame, (left, right, duration) in enumerate(zip(begins, frame_times, durations)):
            weight = np.maximum(0., np.minimum(times, right) - np.maximum(times - dts, left)) / duration
            selected = np.flatnonzero(weight > 1e-10)
            if not len(selected) or not np.isclose(weight.sum(), 1., atol=1e-4):
                raise ValueError(f"Incomplete physical sampling in frame interval {frame}")
            overlaps.append((selected, weight[selected]))
            data['endpoint_sample_time_s'][frame] = times[selected[-1]]
            for key in ("normal", "friction"):
                for suffix in ("load_N", "force_w_N", "unmapped_load_N"):
                    data[key + "_" + suffix][frame] = np.tensordot(
                        weight[selected], data["substep_" + key + "_" + suffix][selected], axes=1)
                data[key + "_peak_load_N"][frame] = data["substep_" + key + "_load_N"][selected].max(0)
                data[key + '_mean_resultant_N'][frame] = np.tensordot(
                    weight[selected], data['substep_' + key + '_resultant_N'][selected], axes=1)
                data[key + '_endpoint_load_N'][frame] = data['substep_' + key + '_load_N'][selected[-1]]
                data[key + '_endpoint_force_w_N'][frame] = data['substep_' + key + '_force_w_N'][selected[-1]]
        for key, records in (("normal", self.normal_records), ("friction", self.friction_records)):
            data[key + '_resultant_N'] = np.linalg.norm(data[key + '_force_w_N'], axis=-1)
            data[key + '_endpoint_resultant_N'] = np.linalg.norm(data[key + '_endpoint_force_w_N'], axis=-1)
            raw = np.concatenate(records) if records else np.empty((0, len(RAW_COLUMNS)))
            data[key + "_records"] = raw
            if len(raw):
                sample_ids, finger_ids, taxels = raw[:, [0, 1, 13]].astype(np.int64).T
                magnitude = np.linalg.norm(raw[:, 9:12], axis=-1)
                for frame, (selected, weights) in enumerate(overlaps):
                    keep = ((sample_ids >= selected[0]) & (sample_ids <= selected[-1]) & (taxels >= 0))
                    if not keep.any():
                        continue
                    factor = weights[np.searchsorted(selected, sample_ids[keep])]
                    flat = data[key + "_map_N"][frame].reshape(-1)
                    np.add.at(flat, finger_ids[keep] * resolution**2 + taxels[keep], magnitude[keep] * factor)
            mapped = data[key + "_map_N"].sum((-1, -2))
            expected = data[key + "_load_N"] - data[key + "_unmapped_load_N"]
            if not np.allclose(mapped, expected, rtol=2e-5, atol=2e-6):
                raise RuntimeError("Projected map does not conserve the assigned physical contact load")
        metadata = {
            "format_version": 2, "model": "SharpaWave W01", "sensor_type": "PhysX rigid contact force projection",
            "source": "RigidPrim.get_contact_force_data/get_friction_data(dt=physics_dt)",
            "sample_count": len(times), "frame_count": len(frame_times),
            "finger_names": list(FINGERS), "sensor_paths": self.sensor_paths, "filter_paths": self.filter_paths,
            "rigid_to_elastomer_transforms": self.transforms.tolist(), "quaternion_convention": "wxyz",
            "surface_assets": [surface.metadata for surface in self.maps],
            "surface_directory": getattr(self, 'surface_directory', str(Path(self.maps[0].metadata['point_file']).parent)),
            "surface_provenance": getattr(self, 'surface_provenance', None),
            "contact_geometry": getattr(self, 'contact_geometry', None),
            "max_mapping_distance_m": self.maps[0].max_distance_m,
            "mapping_threshold_scope": "Nearest-surface correspondence tolerance for visualization; not a collision activation threshold, penetration depth, or hardware spatial accuracy.",
            "surface_mapping_approximation": "The official Tacmap arrays and current W01 collision meshes are different geometric representations; nearest-surface assignment is approximate.",
            "physics_contact_model": getattr(self, "contact_model", {}),
            "force_direction": "Forces act on each sensor rigid body; vectors in world coordinates.",
            "normal_load_N": "Sum of magnitudes of solver normal contact vectors; not norm of their resultant.",
            "force_statistic_definitions": {
                '*_load_N': 'Time-weighted mean of sum of contact-vector magnitudes.',
                '*_force_w_N': 'Time-weighted mean of the summed contact vector in world coordinates.',
                '*_resultant_N': 'Norm of the time-weighted mean contact vector; not sum of magnitudes.',
                '*_mean_resultant_N': 'Time-weighted mean of the norm of each substep contact resultant.',
                '*_endpoint_*': 'Last physical sample whose interval overlaps this frame; exact time is endpoint_sample_time_s. Compare with endpoint TacMap only when timestamps coincide; no interpolation.',
                'substep_*_resultant_N': 'Norm of the summed contact vector at each physical sample.'},
            "friction_load_N": "Sum of magnitudes at solver friction anchors; not a shear-pressure calibration.",
            "map_definition": "N per displayed bin, nearest point on the configured elastomer map, no smoothing; mapping does not alter raw forces.",
            "map_time_definition": "Physics-dt-weighted mean over previous-to-current frame interval; peaks separate.",
            "unmapped_definition": "Body contact farther than the mapping threshold; retained but excluded from skin map.",
            "not_simulated": ["TacMap depth/deformation", "optical tactile RGB", "hardware pressure/shear calibration"],
            "not_recorded": ["solver contact torsional moments", "complete six-axis contact wrench"],
            "raw_columns": list(RAW_COLUMNS), "friction_record_separation": "NaN; friction API does not provide separation",
        }
        alignment = ROOT / "analysis/sharpa_grasp/v2_tactile_map_alignment.json"
        legacy_directory = ROOT / 'assets/sharpa_tactile/assets/tactilesensor_map'
        if alignment.exists() and Path(self.maps[0].metadata['point_file']).parent.resolve() == legacy_directory.resolve():
            metadata["source_mesh_alignment_example"] = {
                "source_file": str(alignment), "sha256": hashlib.sha256(alignment.read_bytes()).hexdigest(),
                "tested_patches": json.loads(alignment.read_text()),
                "scope": "Selected v2 thumb/index patches only; these distances are not a global error bound or a hardware calibration.",
            }
        data["metadata_json"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(path, **data)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
