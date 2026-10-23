"""Describe geometric TacMap response versus independently recorded PhysX force.

Nonzero geometric response is not treated as evidence of physical support.
This analysis reads existing archives only and does not alter acceptance gates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from validate_sharpa_tacmap import sha


def spans(mask, times):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [{'first_frame': int(left), 'last_frame': int(right-1), 'frames': int(right-left),
             'first_endpoint_time_s': float(times[left]), 'last_endpoint_time_s': float(times[right-1])}
            for left, right in zip(edges[::2], edges[1::2])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    with np.load(args.run/'tacmap.npz', allow_pickle=False) as archive:
        times = archive['frame_time_s']; names = archive['finger_names'].tolist()
        codes = archive['deformation_uint8'].max(axis=(-1,-2))
        pixels = (archive['deformation_uint8'] > 0).sum(axis=(-1,-2))
    with np.load(args.run/'tactile.npz', allow_pickle=False) as archive:
        keys = ['frame_time_s', 'frame_interval_s', 'substep_time_s', 'substep_dt_s',
                'normal_load_N', 'friction_load_N', 'substep_normal_load_N', 'substep_friction_load_N',
                'substep_normal_contact_count', 'substep_friction_contact_count']
        physical = {key: archive[key] for key in keys}
    assert np.allclose(times, physical['frame_time_s'], atol=1e-8, rtol=0)
    threshold = 1e-9
    force = physical['substep_normal_load_N']+physical['substep_friction_load_N']
    mean_force = physical['normal_load_N']+physical['friction_load_N']
    subtime = physical['substep_time_s']; subdt = physical['substep_dt_s']
    endpoints = np.searchsorted(subtime, times-1e-8)
    assert np.allclose(subtime[endpoints], times, atol=1e-6, rtol=0)
    substep_max = np.zeros_like(mean_force)
    selections = []
    for frame, time in enumerate(times):
        left = time-physical['frame_interval_s'][frame]
        overlap = np.maximum(0., np.minimum(subtime, time)-np.maximum(subtime-subdt, left))
        selected = np.flatnonzero(overlap/physical['frame_interval_s'][frame] > 1e-10)
        assert len(selected)
        selections.append(selected)
        substep_max[frame] = force[selected].max(axis=0)
    rows = []
    for finger, name in enumerate(names):
        nonzero = codes[:, finger] > 0
        masks = {'frame_mean_force_zero': nonzero & (mean_force[:, finger] <= threshold),
                 'endpoint_substep_force_zero': nonzero & (force[endpoints, finger] <= threshold),
                 'all_frame_substeps_force_zero': nonzero & (substep_max[:, finger] <= threshold)}
        row = {'finger': name, 'TacMap_nonzero_frames': int(nonzero.sum()),
               'TacMap_peak_code': int(codes[:, finger].max()),
               'maximum_nonzero_pixels': int(pixels[:, finger].max()),
               'nonzero_time_spans': spans(nonzero, times)}
        for key, mask in masks.items():
            row[key] = {'frames': int(mask.sum()), 'time_spans': spans(mask, times),
                        'frame_indexes': np.flatnonzero(mask).tolist()}
        if name == 'middle':
            union = masks['frame_mean_force_zero'] | masks['endpoint_substep_force_zero'] | masks['all_frame_substeps_force_zero']
            row['mismatch_frame_details'] = []
            for frame in np.flatnonzero(union):
                selected = selections[frame]
                counts = physical['substep_normal_contact_count'][selected, finger]+physical['substep_friction_contact_count'][selected, finger]
                row['mismatch_frame_details'].append({'frame': int(frame), 'endpoint_time_s': float(times[frame]),
                    'TacMap_peak_code': int(codes[frame, finger]), 'TacMap_nonzero_pixels': int(pixels[frame, finger]),
                    'frame_mean_normal_plus_friction_load_N': float(mean_force[frame, finger]),
                    'endpoint_normal_plus_friction_load_N': float(force[endpoints[frame], finger]),
                    'maximum_substep_normal_plus_friction_load_N': float(substep_max[frame, finger]),
                    'first_substep_time_s': float(subtime[selected[0]]), 'last_substep_time_s': float(subtime[selected[-1]]),
                    'substep_count': len(selected), 'maximum_raw_contact_count': int(counts.max())})
        rows.append(row)
    result = {'run': str(args.run), 'frames': len(times), 'force_zero_tolerance_N': threshold,
              'force_definition': 'Sum of normal and friction vector magnitudes from all raw merged-DP/tool contact points and anchors, including unmapped contacts; this avoids resultant-force cancellation and skin-map coverage loss.',
              'time_definition': 'TacMap is the render endpoint. Frame PhysX means cover (previous frame time, endpoint]; endpoint and every contributing physics substep are separately evaluated.',
              'interpretation': 'A geometric ray-depth response can occur without any solver force because source-CAD geometry and collision proxies differ. TacMap nonzero does not prove skin contact, positive upward force or physical support.',
              'finger_statistics': rows, 'script_sha256': sha(__file__),
              'source_sha256': {name: sha(args.run/name) for name in ['tacmap.npz','tactile.npz']}}
    output = args.run/'tacmap_force_alignment.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'report': str(output), 'summary': [{key: row[key] for key in ['finger','TacMap_nonzero_frames','TacMap_peak_code']} for row in rows],
                      'middle_zero_force_counts': {key: rows[2][key]['frames'] for key in ['frame_mean_force_zero','endpoint_substep_force_zero','all_frame_substeps_force_zero']}}), flush=True)


if __name__ == '__main__':
    main()
