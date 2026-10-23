"""Cross-check selected actual task frames by independent analytic box geometry.

Uses saved live sensor/target poses, archived ray-target vertices and the exact
declared surface maps. No measured force or tactile image is used to calculate
the predicted response; recorded images only select peak frames and compare.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from validate_sharpa_tacmap import closed_box_cpd, quantize_and_blur, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    with np.load(args.run/'tacmap.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data['metadata_json']))
    images = data['deformation_uint8']
    frames = set(np.linspace(0, len(images)-1, min(len(images), args.samples), dtype=int))
    for finger in range(5):
        frames.add(int(images[:, finger].max(axis=(-1,-2)).argmax()))
        frames.add(int((images[:, finger] != 0).sum(axis=(-1,-2)).argmax()))
    maps = []
    for sensor in metadata['sensors']:
        cfg = sensor['config']; step = cfg['resolution_step']
        points = np.load(cfg['points_npy'])[::step, ::step].astype(np.float32)*np.float32(cfg['correction_scale'])
        normals = np.load(cfg['normals_npy'])[::step, ::step].astype(np.float32)
        inward = -normals/(np.linalg.norm(normals, axis=-1, keepdims=True)+np.float32(1e-12))
        maps.append((points+np.float32(cfg['pts_offsets'])*inward, inward, cfg))
    rows = []
    for frame in sorted(frames):
        boxes = []
        for target, pose in zip(metadata['ray_targets'], data['ray_target_pose_w'][frame]):
            rotation = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
            vertices = np.asarray(target['vertices_body_m'], dtype=np.float32).astype(float)
            # Each exact authored box has eight consecutive vertices.
            assert len(vertices) == len(target['source_box_paths'])*8
            for offset in range(0, len(vertices), 8):
                chunk = vertices[offset:offset+8]
                lower, upper = chunk.min(axis=0), chunk.max(axis=0)
                boxes.append({'center_m': (((lower+upper)/2)@rotation.T+pose[:3]).tolist(),
                              'half_extents_m': ((upper-lower)/2).tolist(),
                              'rotation_matrix': rotation.tolist()})
        for finger, (points, inward, cfg) in enumerate(maps):
            pose = data['sensor_pose_w'][frame, finger]
            rotation = Rotation.from_quat(np.roll(pose[3:], -1)).as_matrix()
            starts = points@rotation.T+pose[:3]
            directions = inward@rotation.T
            depth = closed_box_cpd(starts, directions, boxes, cfg['max_distance'], cfg['cpd_max_dist'])
            predicted = quantize_and_blur(depth, cfg['resolution_step'])
            observed = images[frame, finger]
            error = np.abs(predicted.astype(np.int16)-observed.astype(np.int16))
            rows.append({'frame': int(frame), 'time_s': float(data['frame_time_s'][frame]),
                         'finger': str(data['finger_names'][finger]),
                         'maximum_code_error': int(error.max()), 'mean_code_error': float(error.mean()),
                         'expected_peak_code': int(predicted.max()), 'observed_peak_code': int(observed.max()),
                         'expected_nonzero_pixels': int(np.count_nonzero(predicted)),
                         'observed_nonzero_pixels': int(np.count_nonzero(observed)),
                         'passed': bool(error.max() <= 2 and error.mean() <= .1)})
    result = {'valid': all(row['passed'] for row in rows), 'run': str(args.run),
              'selected_frames': sorted(map(int, frames)), 'finger_frame_checks': len(rows),
              'scope': 'Selected task frames, including uniform timeline samples and each finger peak and maximal nonzero area; not every saved frame.',
              'method': 'Independent NumPy analytic ray/closed-box CPD from saved live transforms, then published quantization and Gaussian kernel. Force values are never inputs.',
              'tolerance': 'maximum <=2 uint8 codes and mean <=0.1 code per finger frame, allowing float32 engine/float64 analytic transform differences',
              'maximum_code_error': max(row['maximum_code_error'] for row in rows),
              'maximum_mean_code_error': max(row['mean_code_error'] for row in rows),
              'tacmap_npz_sha256': sha(args.run/'tacmap.npz'), 'script_sha256': sha(__file__), 'rows': rows}
    output = args.run/'independent_tacmap_task_geometry.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['valid','finger_frame_checks','maximum_code_error','maximum_mean_code_error']}), flush=True)
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
