#!/usr/bin/env python3
"""Independently validate official Sharpa TacMap output and task acceptance.

No Isaac session is started. The official class/source provenance, runtime
sensor counters, saved geometry controls and original task gates are separate
checks. Zero-valued fingers during a task are allowed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']
UPSTREAM_TREE = ROOT / 'analysis/sharpa_research/sharpa-tacmap_tree.json'
OFFICIAL_ROOT = ROOT / 'assets/sharpa_tactile'
CORE_FILES = ['tacmap_sensor/sharpa_tacmap_vbts.py',
              'tacmap_sensor/sharpa_tacmap_cfg.py', 'torch_jit_utils.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def evidence_files(value):
    """Find recorded source evidence without assuming a compatibility layout."""
    found = []
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            found.append(value)
        for item in value.values():
            found.extend(evidence_files(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(evidence_files(item))
    return found


class Audit:
    def __init__(self):
        self.checks = {}
        self.warnings = []

    def check(self, name, passed, **details):
        self.checks[name] = {'passed': bool(passed), **details}

    def close(self, name, actual, expected, atol=1e-6):
        a, b = np.asarray(actual), np.asarray(expected)
        same = a.shape == b.shape
        self.check(name, same and np.allclose(a, b, atol=atol, rtol=0),
                   actual_shape=list(a.shape), expected_shape=list(b.shape),
                   maximum_error=float(np.max(np.abs(a-b))) if same and a.size else None,
                   absolute_tolerance=atol)


def longest(mask, dt):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return float(np.max(edges[1::2]-edges[::2], initial=0)*dt)


def quantize_and_blur(depth_m, resolution_step):
    """Independent NumPy equivalent of the published quantizer and blur.

    GaussianBlur on uint8 rounds its convolution before conversion to uint8.
    Normal quantization truncates to uint8 before that convolution.
    """
    from scipy.ndimage import convolve1d
    mm = np.asarray(depth_m, dtype=np.float32)*np.float32(1000.)
    quantized = np.where(mm < np.float32(.5), mm/np.float32(.005),
                         (mm-np.float32(.5))/np.float32(.03)+np.float32(100.))
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    kernel = 9 // int(resolution_step)
    if kernel < 1 or kernel % 2 != 1:
        raise ValueError('Official 9//resolution_step Gaussian kernel must be positive and odd')
    x = np.arange(kernel, dtype=np.float32)-(kernel-1)/2
    weights = np.exp(-.5*(x/np.float32(1.5))**2)
    weights /= weights.sum()
    # scipy mirror matches torch reflect padding (edge itself not duplicated).
    value = convolve1d(quantized.astype(np.float32), weights, axis=-1, mode='mirror')
    value = convolve1d(value, weights, axis=-2, mode='mirror')
    return np.rint(value).clip(0, 255).astype(np.uint8)


def ray_box_depth(starts, directions, center, half_extents, rotation, max_distance):
    """First positive ray/closed-box surface hit using analytic slab geometry."""
    starts = (np.asarray(starts)-center) @ rotation
    directions = np.asarray(directions) @ rotation
    parallel = np.abs(directions) < 1e-12
    outside_parallel = parallel & (np.abs(starts) > half_extents)
    divisor = np.where(parallel, 1., directions)
    t1 = (-half_extents-starts)/divisor
    t2 = (half_extents-starts)/divisor
    near = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(axis=-1)
    far = np.where(parallel, np.inf, np.maximum(t1, t2)).min(axis=-1)
    hit = np.where(near >= 0., near, far)
    valid = (~outside_parallel.any(axis=-1)) & (far >= np.maximum(near, 0.))
    valid &= (hit >= 0.) & (hit <= max_distance) & np.isfinite(hit)
    valid &= np.linalg.norm(directions, axis=-1) > 1e-10
    return np.where(valid, hit, np.inf)


def closed_box_cpd(starts, directions, boxes, max_distance, cpd_max_distance):
    """Two-pass analytic CPD reference for the official ray-casting algorithm."""
    def cast(s, n, distance):
        result = np.full(s.shape[:-1], np.inf)
        for box in boxes:
            rotation = np.asarray(box.get('rotation_matrix', np.eye(3)), dtype=float)
            result = np.minimum(result, ray_box_depth(s, n, np.asarray(box['center_m']),
                                np.asarray(box['half_extents_m']), rotation, distance))
        return result
    first = cast(starts, directions, max_distance)
    safe = np.where(np.isfinite(first), np.maximum(first, 0.)-1e-4, 0.)
    second = cast(starts+directions*safe[..., None], -directions, cpd_max_distance)
    return np.where(np.isfinite(first) & np.isfinite(second), first, 0.).astype(np.float32)


def derived_geometry(audit, model, prefix):
    """Check exported points against recorded barycentrics and the binary CAD STL."""
    from itertools import permutations
    from scipy.spatial import cKDTree
    with np.load(model['correspondence']['path'], allow_pickle=False) as archive:
        d = {key: archive[key] for key in archive.files}
    original = np.load(model['source_map']['path']).astype(float)*.001
    original_normals = np.load(model['source_normals']['path'])
    old_valid = np.linalg.norm(original_normals, axis=-1) > .5
    points = np.load(model['derived_points']['path']).astype(float)*.001
    normals = np.load(model['derived_normals']['path'])
    valid = np.linalg.norm(normals, axis=-1) > .5
    selected = d.get('accepted', np.ones(int(old_valid.sum()), dtype=bool))
    audit.check(prefix+'mask_matches_correspondence', np.array_equal(valid, d['valid'])
                and not np.any(valid & ~old_valid) and np.array_equal(valid[old_valid], selected),
                original_valid_count=int(old_valid.sum()), derived_valid_count=int(valid.sum()))
    audit.close(prefix+'source_points_match_original_map', d['original_points_m'], original[old_valid], 1e-10)
    audit.close(prefix+'exported_points_equal_unoffset_source_projection', points[valid], d['derived_points_m'][selected], 2e-9)
    audit.close(prefix+'exported_normals_equal_source_interpolation', normals[valid], d['derived_normals'][selected], 1e-7)
    audit.close(prefix+'valid_normals_are_unit', np.linalg.norm(normals[valid], axis=-1), np.ones(int(valid.sum())), 2e-6)
    audit.check(prefix+'unsupported_cells_have_zero_normals', not np.any(normals[old_valid & ~valid]))
    bary = d['barycentric']; triangles = d['source_triangle_vertices_m']
    audit.close(prefix+'barycentric_projection', np.sum(triangles*bary[..., None], axis=1), d['derived_points_m'], 2e-9)
    audit.close(prefix+'barycentric_sum', bary.sum(axis=-1), np.ones(len(bary)), 1e-7)
    audit.check(prefix+'projection_inside_source_triangle', bary.min() >= -1e-7 and bary.max() <= 1+1e-7,
                minimum=float(bary.min()), maximum=float(bary.max()))
    # Direct binary STL parsing avoids dependence on the generator's trimesh processing.
    raw = Path(model['source_elastomer_mesh']['path']).read_bytes()
    count = int(np.frombuffer(raw, dtype='<u4', count=1, offset=80)[0])
    if len(raw) != 84+50*count:
        raise ValueError('Independent derived-map check expects the original binary CAD STL')
    dtype = np.dtype([('normal','<f4',(3,)), ('vertices','<f4',(3,3)), ('attribute','<u2')])
    source_triangles = np.frombuffer(raw, dtype=dtype, count=count, offset=84)['vertices'].astype(float)
    _, indexes = cKDTree(source_triangles.mean(axis=1)).query(triangles.mean(axis=1))
    matched = source_triangles[indexes]
    errors = np.stack([np.linalg.norm(triangles-matched[:, order], axis=-1).max(axis=-1)
                       for order in permutations(range(3))])
    nearest = errors.min(axis=0)
    audit.check(prefix+'correspondence_triangles_belong_to_actual_CAD', nearest.max() < 1e-9,
                maximum_corner_error_m=float(nearest.max()), triangle_count=len(triangles))
    outward = np.cross(matched[:,1]-matched[:,0], matched[:,2]-matched[:,0])
    outward /= np.linalg.norm(outward, axis=-1, keepdims=True)
    alignment = np.sum(outward*d['derived_normals'], axis=-1)
    audit.check(prefix+'derived_normals_point_out_of_source_CAD', np.all(alignment[selected] > 0),
                minimum_outward_dot=float(alignment[selected].min()))
    rules = model.get('acceptance_rules')
    if rules:
        displacement = np.linalg.norm(d['derived_points_m']-original[old_valid], axis=-1)
        n = original_normals[old_valid].astype(float); n /= np.linalg.norm(n, axis=-1, keepdims=True)
        angles = np.degrees(np.arccos(np.clip(np.sum(n*d['derived_normals'], axis=-1), -1, 1)))
        expected = (displacement <= rules['maximum_displacement_m']) & (angles <= rules['maximum_normal_angle_deg'])
        audit.check(prefix+'mask_uses_declared_geometry_only_rules', np.array_equal(selected, expected)
                    and rules.get('selection_uses_task_or_force') is False,
                    maximum_displacement_m=float(displacement[selected].max()),
                    maximum_normal_angle_deg=float(angles[selected].max()))


def surface_provenance(audit, metadata, records, prefix=''):
    """Separate untouched original maps from declared, traceable CAD derivations."""
    tree = json.loads(UPSTREAM_TREE.read_text())
    blobs = {x['path']: x['sha'] for x in tree['tree'] if x['type'] == 'blob'}
    proof = metadata.get('surface_provenance', {'kind': 'unmodified_official_TacMap_surface'})
    kind = proof.get('kind')
    derived = kind == 'current_W01_CAD_derived_surface'
    audit.check(prefix+'surface_provenance_kind_known', derived or kind == 'unmodified_official_TacMap_surface',
                kind=kind)
    declared_outputs = set()
    if derived:
        item = proof.get('manifest', {})
        path = Path(item.get('path', ''))
        present = path.is_file() and sha(path) == item.get('sha256')
        audit.check(prefix+'derived_manifest_hash', present, path=str(path), sha256=item.get('sha256'))
        manifest = json.loads(path.read_text()) if present else {}
        audit.check(prefix+'derived_manifest_matches_recorded_details', manifest == proof.get('details'))
        audit.check(prefix+'derived_surface_not_algorithm_or_force_fit',
                    manifest.get('official_sensor_algorithm_modified') is False
                    and manifest.get('force_fitting') is False
                    and manifest.get('arbitrary_outward_offset_m') == 0.)
        models = manifest.get('models', [])
        audit.check(prefix+'derived_two_surface_models', [x.get('model') for x in models] == ['TH', '4F'])
        for index, evidence in enumerate(evidence_files(manifest)):
            evidence_path = Path(evidence['path'])
            audit.check(prefix+'derived_evidence_hash_'+str(index),
                        evidence_path.is_file() and sha(evidence_path) == evidence['sha256'],
                        path=str(evidence_path), sha256=evidence['sha256'])
        audit.check(prefix+'derived_generator_and_CAD_evidence_present',
                    bool(manifest.get('generator', {}).get('sha256')) and
                    len(models) == 2 and all(x.get('source_elastomer_mesh', {}).get('sha256') for x in models))
        for model in models:
            audit.check(prefix+'derived_'+model['model']+'_no_data_fit_or_offset',
                        model.get('trajectory_or_force_input_used') is False
                        and model.get('arbitrary_outward_offset_m') == 0.)
            for key in ['derived_points', 'derived_normals']:
                evidence = model.get(key, {})
                declared_outputs.add((str(Path(evidence.get('path', '')).resolve()), evidence.get('sha256')))
            for key in ['source_map', 'source_normals']:
                evidence = model.get(key, {}); source = Path(evidence.get('path', ''))
                relative = str(source.resolve().relative_to(OFFICIAL_ROOT.resolve())) if source.is_relative_to(OFFICIAL_ROOT) else None
                audit.check(prefix+'derived_'+model['model']+'_'+key+'_official_blob',
                            source.is_file() and git_blob_sha(source.read_bytes()) == blobs.get(relative))
            derived_geometry(audit, model, prefix+'derived_'+model['model']+'_')
        audit.warnings.append('Input surface maps are derived from current W01 CAD with recorded geometry provenance; they are not the original published TacMap map assets or a hardware-taxel calibration.')
    checked = set()
    for record in records:
        for asset in record.get('surface_assets', []):
            path = Path(asset['path']).resolve()
            if path in checked:
                continue
            checked.add(path)
            digest_ok = path.is_file() and sha(path) == asset.get('sha256')
            if derived:
                accepted = (str(path), asset.get('sha256')) in declared_outputs
                audit.check(prefix+'derived_runtime_surface_'+path.name, digest_ok and accepted,
                            path=str(path), sha256=asset.get('sha256'))
            else:
                relative = str(path.relative_to(OFFICIAL_ROOT.resolve())) if path.is_relative_to(OFFICIAL_ROOT.resolve()) else None
                digest = git_blob_sha(path.read_bytes()) if path.is_file() else None
                audit.check(prefix+'official_surface_'+path.name, digest_ok and digest == blobs.get(relative),
                            path=str(path), sha256=asset.get('sha256'), git_blob_sha=digest)
    audit.check(prefix+'four_surface_assets_verified', len(checked) == 4)
    return proof


def provenance(audit, metadata, run):
    tree = json.loads(UPSTREAM_TREE.read_text())
    blobs = {x['path']: x['sha'] for x in tree['tree'] if x['type'] == 'blob'}
    files = []
    for relative in CORE_FILES:
        path = OFFICIAL_ROOT / relative
        actual_blob = git_blob_sha(path.read_bytes())
        audit.check('official_blob_'+relative, actual_blob == blobs[relative],
                    file=str(path), actual_git_blob_sha=actual_blob,
                    expected_git_blob_sha=blobs[relative], sha256=sha(path))
        files.append({'repository_path': relative, 'local_path': str(path), 'sha256': sha(path),
                      'git_blob_sha': actual_blob})
    source = (OFFICIAL_ROOT/CORE_FILES[0]).read_text()
    parsed = ast.parse(source)
    cls = next(x for x in parsed.body if isinstance(x, ast.ClassDef) and x.name == 'SharpaTacmap')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_update_buffers_impl')
    calls = [x.func.id for x in ast.walk(method) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)]
    audit.check('official_core_uses_two_geometry_raycasts_and_quantization',
                calls.count('raycast_dynamic_meshes') == 2 and 'deform_quantize' in calls,
                raycast_calls=calls.count('raycast_dynamic_meshes'))
    sensors = metadata.get('sensors', [])
    audit.check('official_backend_and_no_force_map_input_declared',
                metadata.get('backend') == 'official SharpaTacmap' and metadata.get('force_map_input_used') is False)
    audit.check('official_commit_matches_pinned_tree', metadata.get('official_commit') == tree['sha'])
    for item in metadata.get('official_sources', []):
        path = Path(item['path'])
        audit.check('recorded_source_hash_'+path.name, path.is_file() and sha(path) == item['sha256'])
    adapter = metadata.get('adapter', {})
    adapter_path = Path(adapter.get('path', ''))
    archived_adapter = run/'source'/adapter_path.name
    candidates = [archived_adapter, adapter_path]
    matching = [p for p in candidates if p.is_file() and sha(p) == adapter.get('sha256')]
    audit.check('adapter_actual_source_hash_recorded', bool(matching),
                verified_file=str(matching[0]) if matching else None,
                recorded_sha256=adapter.get('sha256'))
    compatibility_files = evidence_files(metadata.get('compatibility', {}))
    audit.check('runtime_engine_source_evidence_present', len(compatibility_files) > 0,
                evidence_count=len(compatibility_files))
    for index, item in enumerate(compatibility_files):
        path = Path(item['path'])
        audit.check('runtime_engine_file_hash_'+str(index), path.is_file() and sha(path) == item['sha256'],
                    path=str(path), recorded_sha256=item['sha256'])
    audit.check('runtime_five_sensor_records', len(sensors) == 5 and
                [x.get('finger') for x in sensors] == FINGERS)
    for item in sensors:
        finger = item.get('finger', 'unknown')
        method_file = Path(item.get('update_method_file', ''))
        known_path = method_file.is_file() and method_file.resolve() == (OFFICIAL_ROOT/CORE_FILES[0]).resolve()
        audit.check(finger+'_official_runtime_update_method',
                    known_path and item.get('update_method_qualname') == 'SharpaTacmap._update_buffers_impl'
                    and item.get('update_method_file_sha256') == sha(OFFICIAL_ROOT/CORE_FILES[0]),
                    recorded_class=item.get('class_name'), recorded_module=item.get('class_module'),
                    method_file=str(method_file), method_qualname=item.get('update_method_qualname'))
        audit.check(finger+'_runtime_config_recorded', isinstance(item.get('config'), dict)
                    and all(k in item['config'] for k in ['points_npy', 'normals_npy', 'resolution_step',
                                                         'max_distance', 'cpd_max_dist', 'correction_scale', 'pts_offsets']))
        audit.check(finger+'_official_method_called', int(item.get('official_update_calls', 0)) > 0,
                    recorded_calls=item.get('official_update_calls'))
    surfaces = surface_provenance(audit, metadata, sensors)
    return {'repository': 'https://github.com/sharpa-robotics/sharpa-tacmap',
            'pinned_tree': tree['sha'], 'files': files, 'runtime_sensors': sensors,
            'surface_provenance': surfaces}


def geometry_controls(audit, path, task_metadata=None):
    path = path/'controls.npz' if path.is_dir() else path
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    metadata = json.loads(str(data['metadata_json']))
    cases = metadata['cases']; images = data['deformation_uint8']
    surface_provenance(audit, metadata, cases, 'control_')
    step = int(metadata['resolution_step']); size = 240//step
    audit.check('geometry_controls_are_separate_fixed_fixtures',
                metadata.get('geometry_control_only') is True and metadata.get('fixture_present') is True
                and metadata.get('task_success_claimed') is False and metadata.get('force_map_input_used') is False)
    audit.check('geometry_control_shape_uint8', images.shape == (8, 2, size, size) and images.dtype == np.uint8,
                shape=list(images.shape), dtype=str(images.dtype))
    audit.check('geometry_control_case_names', data['case_names'].tolist() == [c['name'] for c in cases])
    for index, item in enumerate(evidence_files(metadata.get('compatibility', {}))):
        source = Path(item['path'])
        audit.check('control_runtime_engine_source_hash_'+str(index),
                    source.is_file() and sha(source) == item['sha256'], path=str(source))
    if task_metadata:
        task_evidence = {(x['path'], x['sha256']) for x in evidence_files(task_metadata.get('compatibility', {}))}
        control_evidence = {(x['path'], x['sha256']) for x in evidence_files(metadata.get('compatibility', {}))}
        audit.check('control_and_task_runtime_engine_sources_identical', bool(task_evidence) and task_evidence == control_evidence)
    reference_cases = [(model, depth) for model in ['TH', '4F'] for depth in [-.001, .0001, .0005, .001]]
    audit.check('geometry_controls_cover_both_surface_models_and_all_depths',
                [(c['model'], c['nominal_surface_plane_indent_m']) for c in cases] == reference_cases)
    counter = data['sensor_frame_counter']
    audit.check('geometry_controls_official_frame_advancement', counter.shape == (8,2) and
                (counter > 0).all() and (np.diff(counter, axis=1) == 1).all())
    audit.check('geometry_controls_repeat_deterministically', np.array_equal(images[:, 0], images[:, 1]))
    rows = []
    for index, case in enumerate(cases):
        cfg = case['config']; name = case['name']; model_index = case['model_index']
        core = OFFICIAL_ROOT/CORE_FILES[0]
        audit.check('control_'+name+'_official_method',
                    Path(case['update_method_file']).resolve() == core.resolve()
                    and case['update_method_file_sha256'] == sha(core)
                    and case['update_method_qualname'] == 'SharpaTacmap._update_buffers_impl'
                    and case['class_name'] == 'SharpaTacmap' and case['official_update_calls'] == 2)
        points = np.load(cfg['points_npy'])[::step, ::step]*np.float32(cfg['correction_scale'])
        normals = np.load(cfg['normals_npy'])[::step, ::step]
        audit.close('control_'+name+'_archived_source_points', data['source_points_m'][model_index], points, 1e-10)
        audit.close('control_'+name+'_archived_source_normals', data['source_normals'][model_index], normals, 1e-10)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)+1e-12
        inward = -normals/norms
        pose = np.asarray(case['sensor_pose_w'])
        audit.close('control_'+name+'_identity_pose_rotation', pose[3:], [1,0,0,0], 1e-10)
        audit.close('control_'+name+'_actual_ray_starts', data['ray_starts_w'][index],
                    points+cfg['pts_offsets']*inward+pose[:3], 1e-7)
        audit.close('control_'+name+'_actual_ray_directions', data['ray_directions_w'][index], inward, 2e-6)
        # USD stores Vec3f vertices. Use the archived authored float32 vertices,
        # not nominal double-precision centers, to avoid inventing depth error.
        box = case['boxes'][0]; rotation = np.asarray(box['rotation_matrix'])
        vertices = np.asarray(case['box_mesh_vertices_w_m'], dtype=float)
        local_vertices = vertices@rotation
        lower, upper = local_vertices.min(axis=0), local_vertices.max(axis=0)
        exact_box = {'rotation_matrix': rotation.tolist(),
                     'center_m': (((lower+upper)/2)@rotation.T).tolist(),
                     'half_extents_m': ((upper-lower)/2).tolist()}
        depth = closed_box_cpd(data['ray_starts_w'][index], data['ray_directions_w'][index], [exact_box],
                               cfg['max_distance'], cfg['cpd_max_dist'])
        expected = quantize_and_blur(depth, step)
        observed = images[index, 0]
        error = np.abs(expected.astype(np.int16)-observed.astype(np.int16))
        audit.check('control_'+name+'_independent_geometric_response', error.max() <= 2 and error.mean() <= .1,
                    maximum_uint8_code_error=int(error.max()), mean_uint8_code_error=float(error.mean()),
                    mismatching_pixels=int(np.count_nonzero(error)), tolerance='max 2 codes and mean <=0.1 code',
                    expected_depth_max_m=float(depth.max()))
        clear = case['nominal_surface_plane_indent_m'] < 0
        audit.check('control_'+name+('_clear_zero' if clear else '_pressed_nonzero'),
                    not np.any(observed) if clear else np.any(observed > 0),
                    maximum_code=int(observed.max()), nonzero_pixels=int(np.count_nonzero(observed)))
        if task_metadata:
            sensors = task_metadata['sensors']; task_sensor = sensors[0 if case['model']=='TH' else 1]
            keys = ['points_npy', 'normals_npy', 'resolution_step', 'correction_scale', 'pts_offsets',
                    'max_distance', 'cpd_max_dist', 'data_types']
            audit.check('control_'+name+'_same_sensor_algorithm_configuration_as_task',
                        all(cfg[k] == task_sensor['config'][k] for k in keys)
                        and case['update_method_file_sha256'] == task_sensor['update_method_file_sha256'], compared_fields=keys)
        rows.append({'case': name, 'model': case['model'], 'nominal_indent_m': case['nominal_surface_plane_indent_m'],
                     'actual_peak_code': int(observed.max()), 'nonzero_pixels': int(np.count_nonzero(observed)),
                     'independent_depth_max_m': float(depth.max()), 'max_code_difference': int(error.max()),
                     'mean_code_difference': float(error.mean())})
    for model in ['TH', '4F']:
        selected = [r for r in rows if r['model'] == model]
        peaks = [r['actual_peak_code'] for r in selected]
        pixels = [r['nonzero_pixels'] for r in selected]
        audit.check('control_'+model+'_response_increases_with_indent',
                    all(b>a for a,b in zip(peaks, peaks[1:])) and all(b>=a for a,b in zip(pixels,pixels[1:])),
                    peak_codes=peaks, nonzero_pixels=pixels)
    return {'archive': str(path), 'sha256': sha(path), 'cases': rows,
            'scope': 'Geometric sensing controls using fixed fixtures, not physical-contact or task-success evidence.'}


def physical_task(audit, run, reference_run, require_success):
    path = run/'trajectory.npz'
    if not path.exists():
        audit.check('physical_task_trajectory_present', not require_success)
        return None
    with np.load(path, allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    result = json.loads((run/'evaluation.json').read_text())
    task = result['task']; cfg = task['place_config']; dt = float(d['dt'])
    block, tip = d['block'], d['tip']
    origin = d['origins']; goal = d['goal']+origin
    lift = block[:, :, 2]-origin[None, :, 2]-(.72+.0061)
    xy = np.linalg.norm(block[:, :, :2]-goal[None, :, :2], axis=-1)
    root_tip = tip-d['tool'][:, :, :3]
    root_angle = np.degrees(np.arctan2(-root_tip[:, :, 2], np.linalg.norm(root_tip[:, :, :2], axis=-1)))
    low, high = task['tool_down_angle_range_deg']
    angle = (d['normal_angle'] >= low) & (d['normal_angle'] <= high) & (root_angle >= low) & (root_angle <= high)
    frame = np.ones_like(angle)
    if task.get('require_palm_down'):
        frame &= d['palm_normal'][:, :, 2] < -.05
    if task.get('require_web_up'):
        frame &= d['web_axis'][:, :, 2] >= task['min_web_up_component']
    if task.get('max_palm_up_component') is not None:
        frame &= d['palm_normal'][:, :, 2] <= task['max_palm_up_component']
    valid = (d['retained'] < .035) & (d['orientation_error'] < .3) & d['face'] & angle & frame
    lifting = d['active'] & valid & (lift > .045) & d['both']
    stable = d['active'] & (d['phase'] == 9) & valid & (xy < cfg['goal_tolerance_m']) & (np.abs(lift) < .0015)
    stable &= (np.linalg.norm(block[:, :, 7:10], axis=-1) < .01) & (np.linalg.norm(block[:, :, 10:13], axis=-1) < .5)
    stable &= (d['forces'].max(axis=-1) < .003) & (tip[:, :, 2]-block[:, :, 2] > .035)
    rows = []
    for env, claimed in enumerate(result['success_per_env']):
        count = 0; lifted = False; eligible = stable[:, env].copy()
        for step in range(len(block)):
            count = count+1 if lifting[step, env] else 0
            lifted |= count >= round(.25/dt)
            eligible[step] &= lifted
        duration = longest(eligible, dt)
        success = duration >= cfg['stable_seconds']-1e-8
        rows.append({'environment': env, 'claimed': claimed, 'reconstructed': success,
                     'longest_valid_lift_s': longest(lifting[:, env], dt),
                     'longest_valid_phase9_stability_after_lift_s': duration})
    audit.check('physical_task_claims_independently_reconstructed', all(x['claimed'] == x['reconstructed'] for x in rows))
    if require_success:
        audit.check('requested_physical_task_success', all(x['reconstructed'] for x in rows))
    if reference_run:
        reference = json.loads((reference_run/'task.json').read_text())
        fields = ['success_definition', 'tool_down_angle_range_deg', 'require_face_pinch',
                  'require_palm_down', 'require_web_up', 'min_web_up_component', 'max_palm_up_component']
        audit.check('task_acceptance_unchanged_from_reference', all(task.get(k) == reference.get(k) for k in fields),
                    reference=str(reference_run), compared_fields=fields)
        audit.check('task_control_parameters_unchanged_from_reference', task['place_config'] == reference['place_config'])
    validation_path = run/'validation.json'
    if validation_path.exists():
        v = json.loads(validation_path.read_text())
        audit.check('separate_physical_audit_consistent', v.get('all_claims_consistent') and
                    v.get('no_runtime_object_control_calls') and v.get('archived_source_hashes_match'))
        usd = v.get('USD_audit', {})
        audit.check('dynamic_objects_without_attachments', usd.get('all_objects_dynamic_with_gravity') and
                    not usd.get('forbidden_tool_robot_or_block_joints', ['missing']))
    elif require_success:
        audit.check('separate_physical_audit_present', False,
                    instruction='Run analysis/validate_pick_place.py RUN --geometry before final TacMap validation')
    return {'rows': rows, 'successes': sum(x['reconstructed'] for x in rows),
            'trajectory_sha256': sha(path), 'evaluation_sha256': sha(run/'evaluation.json')}


def audit_run(run, require_video=False, require_task_success=False, reference_run=None, controls=None):
    audit = Audit()
    with np.load(run/'tacmap.npz', allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    metadata = json.loads(str(d['metadata_json']))
    times = d['frame_time_s']; maps = d['deformation_uint8']; valid = d['surface_valid']
    frames = len(times)
    audit.check('nonempty_strictly_increasing_finite_times', frames > 0 and times.ndim == 1 and
                times.dtype == np.float64 and np.isfinite(times).all() and (np.diff(times) > 0).all())
    audit.check('five_fingers_in_declared_order', d['finger_names'].tolist() == FINGERS)
    shape_ok = maps.ndim == 4 and maps.shape[:2] == (frames, 5) and maps.shape[2] == maps.shape[3]
    audit.check('deformation_shape_uint8_finite_range', shape_ok and maps.dtype == np.uint8 and
                np.isfinite(maps).all() and (maps >= 0).all() and (maps <= 255).all(), shape=list(maps.shape), dtype=str(maps.dtype))
    if not shape_ok:
        raise ValueError('Expected deformation_uint8[F,5,H,W]')
    audit.check('surface_valid_shape_boolean', valid.shape == maps.shape[1:] and valid.dtype == bool,
                valid_cells_by_finger=valid.sum(axis=(-1, -2)).tolist())
    proof = provenance(audit, metadata, run)
    for index, sensor in enumerate(metadata.get('sensors', [])):
        cfg = sensor.get('config', {})
        if not all(k in cfg for k in ['resolution_step', 'normals_npy']):
            continue
        step = int(cfg['resolution_step'])
        normals = np.load(cfg['normals_npy'])[::step, ::step]
        expected = np.linalg.norm(normals, axis=-1) > 1e-10
        audit.check(FINGERS[index]+'_surface_mask_matches_official_normal_domain',
                    valid[index].shape == expected.shape and np.array_equal(valid[index], expected))
        audit.check(FINGERS[index]+'_resolution_matches_config', maps.shape[-2:] == (240//step, 240//step))
        audit.check(FINGERS[index]+'_official_calls_match_saved_frames', sensor.get('official_update_calls') == frames)
    counters = d.get('sensor_frame_counter')
    audit.check('official_sensor_frame_counters_recorded', counters is not None)
    if counters is not None:
        audit.check('five_fingers_updated_once_per_saved_frame', counters.shape == (frames, 5) and
                    np.issubdtype(counters.dtype, np.integer) and (counters > 0).all() and
                    np.array_equal(np.diff(counters, axis=0), np.ones((max(frames-1, 0), 5), dtype=int)),
                    first=counters[0].tolist(), last=counters[-1].tolist())
    poses = d.get('sensor_pose_w')
    if poses is not None:
        audit.check('sensor_poses_finite_shape', poses.shape == (frames, 5, 7) and np.isfinite(poses).all())
        audit.close('sensor_pose_unit_quaternions', np.linalg.norm(poses[:, :, 3:], axis=-1), np.ones((frames, 5)), 2e-5)
    trajectory_path = run/'trajectory.npz'
    if trajectory_path.exists():
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            steps = trajectory['render_control_steps']; dt = float(trajectory['dt'])
            if not len(steps) and not require_video:
                steps = d['sample_control_step']
                audit.close('TacMap_matches_sampled_control_endpoints_without_camera', times, (steps+1)*dt)
            else:
                audit.close('TacMap_matches_camera_control_endpoints', times, (steps+1)*dt)
            if 'sample_control_step' in d:
                audit.close('saved_control_step_indexes', d['sample_control_step'], steps, 0)
            targets = d.get('ray_target_pose_w')
            audit.check('actual_dynamic_raycast_target_poses_recorded', targets is not None)
            if targets is not None:
                expected = np.stack([trajectory[name][steps, 0, :7] for name in ['tool','left','right']], axis=1)
                audit.close('raycast_target_positions_follow_actual_dynamic_bodies', targets[:, :, :3], expected[:, :, :3], 3e-6)
                dots = np.abs(np.sum(targets[:, :, 3:]*expected[:, :, 3:], axis=-1))
                audit.close('raycast_target_rotations_follow_actual_dynamic_bodies', dots, np.ones((frames,3)), 3e-6)
    if (run/'tactile.npz').exists():
        with np.load(run/'tactile.npz', allow_pickle=False) as tactile:
            audit.close('TacMap_and_PhysX_force_frames_synchronized', times, tactile['frame_time_s'])
            if poses is not None and 'substep_sensor_pose_w' in tactile:
                physical_times = tactile['substep_time_s']
                indexes = np.searchsorted(physical_times, times-1e-8)
                audit.close('sensor_pose_sample_times_match_actual_physics', physical_times[indexes], times, 1e-6)
                expected = tactile['substep_sensor_pose_w'][indexes]
                audit.close('TacMap_sensor_positions_follow_actual_distal_body_plus_skin_offset', poses[:, :, :3], expected[:, :, :3], 3e-6)
                dots = np.abs(np.sum(poses[:, :, 3:]*expected[:, :, 3:], axis=-1))
                audit.close('TacMap_sensor_rotations_follow_actual_distal_body_plus_skin_offset', dots, np.ones((frames,5)), 3e-6)
    camera_stats = []
    manifest_path = run/'camera_manifest.json'
    if require_video:
        audit.check('camera_manifest_present', manifest_path.exists())
    if manifest_path.exists():
        import subprocess
        for camera in json.loads(manifest_path.read_text()):
            path = run/(camera['name']+'.mp4')
            try:
                import imageio.v2 as imageio
                with imageio.get_reader(path, format='FFMPEG') as reader:
                    count = reader.count_frames(); fps = reader.get_meta_data()['fps']
            except ModuleNotFoundError:
                # Rendering dependencies live in the project environment; keep its
                # Python extension modules isolated from the caller's interpreter.
                script = ('import sys,json,imageio.v2 as i; r=i.get_reader(sys.argv[1],format="FFMPEG"); '
                          'print(json.dumps([r.count_frames(),r.get_meta_data()["fps"]])); r.close()')
                probe = subprocess.run([str(ROOT/'.venv-human/bin/python'), '-c', script, str(path)],
                                       check=True, capture_output=True, text=True)
                count, fps = json.loads(probe.stdout)
            audit.check('camera_'+camera['name']+'_frame_count', count == frames, camera_frames=count, TacMap_frames=frames)
            camera_stats.append({'path': str(path), 'frames': count, 'fps': fps})
    task = physical_task(audit, run, reference_run, require_task_success)
    stats = [{'finger': finger, 'minimum': int(maps[:, i].min()), 'maximum': int(maps[:, i].max()),
              'frames_with_nonzero_values': int(np.any(maps[:, i] != 0, axis=(-1, -2)).sum()),
              'nonzero_pixel_fraction': float(np.count_nonzero(maps[:, i])/maps[:, i].size)} for i, finger in enumerate(FINGERS)]
    audit.warnings.extend(['Zero task output for a finger is allowed; every finger is not required to deform.',
        'Published uint8 TacMap values are quantized and spatially blurred ray-derived depth responses, not forces in N or calibrated physical skin displacement.',
        'surface_valid describes valid input normals. Official Gaussian blur can spread output beyond that input mask.',
        'Source hashes, runtime method/counter records and geometry controls form the provenance evidence; an output heatmap alone cannot prove its computing backend.'])
    controls_result = None
    if controls is None:
        audit.check('independent_geometry_controls_supplied', False)
    else:
        audit.check('independent_geometry_controls_supplied', True, path=str(controls))
        controls_result = geometry_controls(audit, controls, metadata)
    return {'valid': all(x['passed'] for x in audit.checks.values()), 'run': str(run),
            'tacmap_frames': frames, 'finger_statistics': stats, 'provenance': proof,
            'physical_task': task, 'geometry_controls': controls_result,
            'cameras': camera_stats, 'checks': audit.checks, 'warnings': audit.warnings,
            'tacmap_npz_sha256': sha(run/'tacmap.npz'), 'validator_sha256': sha(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-video', action='store_true')
    parser.add_argument('--require-task-success', action='store_true')
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--controls', type=Path)
    parser.add_argument('--controls-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.controls_only:
            audit = Audit()
            controls = geometry_controls(audit, args.run.resolve())
            report = {'valid': all(x['passed'] for x in audit.checks.values()),
                      'geometry_controls': controls, 'checks': audit.checks, 'warnings': audit.warnings,
                      'validator_sha256': sha(__file__)}
        else:
            report = audit_run(args.run.resolve(), args.require_video, args.require_task_success,
                               args.reference_run, args.controls)
    except (ValueError, KeyError, OSError, IndexError, TypeError) as exc:
        report = {'valid': False, 'run': str(args.run), 'error': str(exc)}
    output = args.output or args.run/('controls_validation.json' if args.controls_only else 'tacmap_validation.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'valid': report['valid'], 'report': str(output), 'frames': report.get('tacmap_frames'),
                      'failed_checks': [k for k,v in report.get('checks', {}).items() if not v['passed']],
                      'error': report.get('error')}, ensure_ascii=False))
    raise SystemExit(0 if report['valid'] else 1)


if __name__ == '__main__':
    main()
