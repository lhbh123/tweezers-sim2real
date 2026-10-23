"""Offline per-body TacMap CPD extension on a recorded physical trajectory.

Usage: replay_per_target_tacmap.py RUN --output NEW_OUTPUT_DIR [--frames 399]
No simulator is started and no source run is modified. The two ray passes are
restricted to one body at a time; accepted positive depths are combined by min.
Max is saved as a sensitivity comparison, not an asserted physical thickness.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file())
sys.path.insert(0, str(ROOT/'analysis'))
from validate_sharpa_tacmap import ray_box_depth, quantize_and_blur, closed_box_cpd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def per_body_cpd(starts, directions, boxes, max_distance, cpd_max_distance):
    """Keep the official -100 um second-ray start and finite-pair condition."""
    def cast(s, n, distance):
        result = np.full(s.shape[:-1], np.inf)
        for center, half, rotation in boxes:
            result = np.minimum(result, ray_box_depth(s, n, center, half, rotation, distance))
        return result
    first = cast(starts, directions, max_distance)
    advance = np.where(np.isfinite(first), np.maximum(first, 0.)-1e-4, 0.)
    reverse = cast(starts+advance[..., None]*directions, -directions, cpd_max_distance)
    depth = np.where(np.isfinite(first)&np.isfinite(reverse), first, 0.).astype(np.float32)
    outside_distance = np.full(first.shape, np.inf)
    for center, half, rotation in boxes:
        local = (starts-center)@rotation
        excess = np.abs(local)-half
        outside_distance = np.minimum(outside_distance, np.linalg.norm(np.maximum(excess, 0.), axis=-1))
    return depth, outside_distance


def geometry_checks():
    checks = []
    eye = np.eye(3); start = np.array([[0., 0., 0.]])
    direction = np.array([[1., 0., 0.]])
    box = lambda x, half: (np.array([x, 0., 0.]), np.array([half, .005, .005]), eye)
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    centered = [box(0., .0005)]
    a, outside = per_body_cpd(start, direction, centered, .015, .5)
    official = closed_box_cpd(start, direction, [{'center_m': x[0], 'half_extents_m': x[1], 'rotation_matrix': x[2]} for x in centered], .015, .5)
    check('single-body result equals official analytic two-pass result', np.array_equal(a, official) and np.isclose(a[0], .0005))
    thin, _ = per_body_cpd(start, direction, [box(0., 20e-6)], .015, .5)
    check('40 um thin body preserves official epsilon-induced no-hit, without inside fill', thin[0] == 0)
    left, right = [box(.0015, .0005)], [box(-.0015, .0005)]
    d1, _ = per_body_cpd(start, direction, left, .015, .5)
    d2, _ = per_body_cpd(start, direction, right, .015, .5)
    mixed, outside = per_body_cpd(start, direction, left+right, .015, .5)
    check('separate-body gap is rejected by both independent pairs', d1[0] == 0 and d2[0] == 0)
    check('same-body disconnected-component limitation remains and is diagnosed', mixed[0] > 0 and outside[0] > 1e-6)
    a, _ = per_body_cpd(start, direction, [box(0., .0005)], .015, .5)
    b, _ = per_body_cpd(start, direction, [box(0., .0007)], .015, .5)
    check('overlap sensitivity keeps nearest 0.5 mm and farther 0.7 mm distinct', np.isclose(min(a[0], b[0]), .0005) and np.isclose(max(a[0], b[0]), .0007))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run'); p.add_argument('--output', required=True)
    p.add_argument('--frames', nargs='+', type=int)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--boundary-tolerance-um', type=float, default=1.)
    args = p.parse_args()
    run = Path(args.run).resolve(); out = Path(args.output).resolve()
    if out.exists(): raise FileExistsError('Refusing to overwrite an existing output directory: '+str(out))
    if args.workers < 1: raise ValueError('--workers must be positive')
    checks = geometry_checks(); out.mkdir(parents=True)
    started = time.monotonic()
    source = run/'tacmap.npz'
    source_hash_before = sha(source)
    with np.load(source) as z:
        meta_array = z['metadata_json']
        meta = json.loads(str(meta_array))
        names = [str(x) for x in z['finger_names']]
        times = z['frame_time_s']; steps = z['sample_control_step']
        sensor_poses = z['sensor_pose_w']; target_poses = z['ray_target_pose_w']
        points = z['surface_points_m'].astype(np.float32)
        normals = z['surface_normals'].astype(np.float32)
        original_gray = z['deformation_uint8']; original_raw = z['diagnostic_raw_cpd_depth_m']
    frames = np.array(sorted(set(args.frames if args.frames is not None else range(len(times)))), dtype=np.int64)
    if not len(frames) or frames.min() < 0 or frames.max() >= len(times): raise ValueError('Invalid frame indices')
    nf, nk, h, w = len(frames), len(names), *original_gray.shape[-2:]
    nb = len(meta['ray_targets']); tolerance = args.boundary_tolerance_um*1e-6
    local_boxes = []
    for target in meta['ray_targets']:
        vv = np.asarray(target['vertices_body_m'], np.float32).astype(float)
        if len(vv)%8: raise ValueError('This analytic implementation supports archived closed boxes only')
        group = []
        for j in range(0, len(vv), 8):
            lo, hi = vv[j:j+8].min(0), vv[j:j+8].max(0)
            if any(len(np.unique(vv[j:j+8, axis])) != 2 for axis in range(3)):
                raise ValueError('Expected axis-aligned body-local box vertices')
            group.append(((lo+hi)/2, (hi-lo)/2))
        local_boxes.append(group)
    normalized = -normals/(np.linalg.norm(normals, axis=-1, keepdims=True)+np.float32(1e-12))
    per_body = np.zeros((nf, nk, nb, h, w), np.float32)
    min_depth = np.zeros((nf, nk, h, w), np.float32)
    max_depth = np.zeros_like(min_depth)
    min_gray = np.zeros((nf, nk, h, w), np.uint8); max_gray = np.zeros_like(min_gray)
    selected_body = np.full((nf, nk, h, w), -1, np.int8)
    outside_packed = np.zeros((nf, nk, nb, (h*w+7)//8), np.uint8)
    rows = []
    print(json.dumps({'state': 'loaded', 'frames': nf, 'fingers': names, 'bodies': nb,
                      'workers': args.workers, 'geometry_checks': checks}), flush=True)

    def work(item):
        output_frame, frame = item
        grouped = []
        for group, pose in zip(local_boxes, target_poses[frame]):
            r = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
            grouped.append([(center@r.T+pose[:3], half, r) for center, half in group])
        local_rows = []
        for k, name in enumerate(names):
            cfg = meta['sensors'][k]['config']; pose = sensor_poses[frame, k]
            sr = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
            starts = (points[k]+np.float32(cfg['pts_offsets'])*normalized[k])@sr.T+pose[:3]
            directions = normalized[k]@sr.T
            outside_any = np.full((h, w), np.inf); body_rows = []
            for body, boxes in enumerate(grouped):
                depth, distance = per_body_cpd(starts, directions, boxes, cfg['max_distance'], cfg['cpd_max_dist'])
                per_body[output_frame, k, body] = depth
                positive = depth > 0
                strict_outside = positive&(distance > 0)
                robust_outside = positive&(distance > tolerance)
                outside_packed[output_frame, k, body] = np.packbits(robust_outside.ravel(), bitorder='little')
                outside_any = np.minimum(outside_any, distance)
                body_rows.append({'body_id': body, 'positive_count': int(positive.sum()),
                                  'positive_outside_own_boxes_count': int(strict_outside.sum()),
                                  'positive_outside_own_boxes_by_more_than_tolerance_count': int(robust_outside.sum()),
                                  'outside_positive_depth_max_mm': float(depth[robust_outside].max()*1e3) if robust_outside.any() else 0.,
                                  'outside_origin_distance_max_um': float(distance[robust_outside].max()*1e6) if robust_outside.any() else 0.})
            db = per_body[output_frame, k]
            valid = np.where(db > 0, db, np.inf)
            near = valid.min(0); near = np.where(np.isfinite(near), near, 0.).astype(np.float32)
            far = db.max(0)
            min_depth[output_frame, k] = near; max_depth[output_frame, k] = far
            selected_body[output_frame, k] = np.where(near > 0, valid.argmin(0), -1).astype(np.int8)
            min_gray[output_frame, k] = quantize_and_blur(near, cfg['resolution_step'])
            max_gray[output_frame, k] = quantize_and_blur(far, cfg['resolution_step'])
            raw = original_raw[frame, k]
            local_rows.append({'source_frame': int(frame), 'output_frame': output_frame, 'time_s': float(times[frame]),
                               'finger': name, 'original_raw_positive_count': int((raw > 0).sum()),
                               'original_raw_positive_outside_all_boxes_by_more_than_tolerance_count': int(((raw > 0)&(outside_any > tolerance)).sum()),
                               'min_positive_count': int((near > 0).sum()), 'max_positive_count': int((far > 0).sum()),
                               'original_raw_peak_mm': float(raw.max()*1e3), 'min_raw_peak_mm': float(near.max()*1e3),
                               'max_raw_peak_mm': float(far.max()*1e3),
                               'original_gray_peak': int(original_gray[frame, k].max()),
                               'min_gray_peak': int(min_gray[output_frame, k].max()),
                               'max_gray_peak': int(max_gray[output_frame, k].max()),
                               'multiple_positive_bodies_count': int(((db > 0).sum(0) > 1).sum()),
                               'min_max_depth_different_count': int((near != far).sum()),
                               'min_max_gray_max_difference': int(np.abs(min_gray[output_frame, k].astype(int)-max_gray[output_frame, k].astype(int)).max()),
                               'bodies': body_rows})
        return local_rows

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, item) for item in enumerate(frames)]
        for complete, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if complete == 1 or complete%50 == 0 or complete == nf:
                print(json.dumps({'completed_frames': complete, 'total_frames': nf,
                                  'elapsed_s': round(time.monotonic()-started, 2)}), flush=True)
    rows.sort(key=lambda r: (r['source_frame'], names.index(r['finger'])))
    summary = {}
    for k, name in enumerate(names):
        rr = [r for r in rows if r['finger'] == name]
        summary[name] = {'original_gray_peak': int(original_gray[frames, k].max()),
                         'min_gray_peak': int(min_gray[:, k].max()), 'max_gray_peak': int(max_gray[:, k].max()),
                         'original_raw_peak_mm': float(original_raw[frames, k].max()*1e3),
                         'min_raw_peak_mm': float(min_depth[:, k].max()*1e3),
                         'max_raw_peak_mm': float(max_depth[:, k].max()*1e3),
                         'original_robust_outside_positive_pixel_frames': sum(r['original_raw_positive_outside_all_boxes_by_more_than_tolerance_count'] for r in rr),
                         'per_body_robust_outside_positive_pixel_frames': sum(b['positive_outside_own_boxes_by_more_than_tolerance_count'] for r in rr for b in r['bodies']),
                         'min_max_different_pixel_frames': sum(r['min_max_depth_different_count'] for r in rr)}
    payload = dict(source_frame_index=frames, frame_time_s=times[frames], sample_control_step=steps[frames],
                   finger_names=np.asarray(names), body_paths=np.asarray([x['body_path'] for x in meta['ray_targets']]),
                   per_body_cpd_depth_m=per_body, min_positive_cpd_depth_m=min_depth,
                   max_positive_cpd_depth_m=max_depth, min_positive_gray_uint8=min_gray,
                   max_positive_gray_uint8=max_gray, min_selected_body_id=selected_body,
                   per_body_robust_outside_origin_mask_packbits=outside_packed,
                   original_gray_uint8=original_gray[frames], sensor_pose_w=sensor_poses[frames],
                   ray_target_pose_w=target_poses[frames])
    payload_hashes = {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': array_sha(value)} for key, value in payload.items()}
    np.savez_compressed(out/'per_target_tacmap.npz', **payload)
    original_inputs = {'metadata_json': meta_array, 'surface_points_m': points,
                       'surface_normals': normals, 'sensor_pose_w': sensor_poses,
                       'ray_target_pose_w': target_poses, 'frame_time_s': times,
                       'sample_control_step': steps, 'deformation_uint8': original_gray,
                       'diagnostic_raw_cpd_depth_m': original_raw}
    source_dir = out/'source'; source_dir.mkdir()
    for path in [Path(__file__), ROOT/'analysis/validate_sharpa_tacmap.py']:
        shutil.copy2(path, source_dir/path.name)
    source_hash_after = sha(source)
    report = {'schema_version': 1, 'state': 'complete', 'method': 'offline_per_body_paired_rays_min_positive',
              'run': str(run), 'source_tacmap_npz_sha256': source_hash_after,
              'source_tacmap_npz_sha256_before': source_hash_before,
              'source_unchanged': source_hash_before == source_hash_after, 'source_frame_count': len(times), 'output_frame_count': nf,
              'selected_source_frames': frames.tolist(), 'finger_names': names,
              'body_id_order': [x['body_path'] for x in meta['ray_targets']],
              'second_ray_epsilon_m': -1e-4, 'outside_diagnostic_tolerance_um': args.boundary_tolerance_um,
              'per_body_robust_outside_masks': {'array': 'per_body_robust_outside_origin_mask_packbits',
                                               'bitorder': 'little', 'unpacked_shape_suffix': [h, w]},
              'input_array_hashes': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': array_sha(value)} for key, value in original_inputs.items()},
              'output_arrays': payload_hashes, 'summary': summary, 'rows': rows,
              'geometry_boundary_checks': checks,
              'source_files_sha256': {path.name: sha(path) for path in source_dir.iterdir()},
              'elapsed_seconds': time.monotonic()-started,
              'semantics_and_limits': [
                  'Offline observation extension on saved physical poses. No new simulation, force change, attachment or motion command is performed.',
                  'Official raw tacmap.npz is preserved. These arrays are not claimed to be the original official sensor output.',
                  'Each body independently uses the published forward ray, -100 um shifted reverse ray, distance limits and finite-hit-pair condition.',
                  'The official piecewise uint8 quantization and 9x9 sigma1.5 Gaussian reflect blur are reproduced by the previously validated independent NumPy implementation.',
                  'Primary min ignores absent/zero body depths. It selects the nearest accepted positive exit; max is a separate sensitivity view. Neither is asserted as union thickness or measured deformation.',
                  'A finger is not permanently assigned to one body: all configured bodies remain eligible, preserving handle/other-jaw contacts during motion.',
                  'Same-body tests cannot eliminate gaps between disconnected or nonconvex components in one body. Positive outside-own-box samples are retained and separately flagged, never silently clipped.',
                  'The -100 um reverse origin can reject real inside points in very thin objects; this official behavior is retained, not filled using an inside mask.',
                  'World rays use saved float32 points/normals and scipy float64 rotation reconstruction; they are CPU predictions, not exact original GPU ray tensors.',
                  'Only authored closed-box target geometry is supported by this script; skin collision geometry and force calibration are outside its scope.',
                  'Depth maxima and gray values are geometric quantities, not pressure or normal force. Pixel counts are not calibrated contact area.'
              ]}
    (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'state': 'complete', 'output': str(out), 'summary': summary,
                      'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
