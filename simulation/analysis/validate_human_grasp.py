"""Continuous source-triangle versus finite beam audit of a grasp reference.

This complements, and does not replace, PhysX collision-hull/dynamic testing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from kinematics import ASSET, Kinematics, stl_triangles


def triangle_box_overlap(triangles, center, half_size, epsilon=1e-12):
    """Separating-axis triangle/AABB test, all 13 axis families.

    Each triangle is tested continuously, including triangles whose vertices
    are all outside the box. Degenerate candidate axes cannot separate.
    """
    t = np.asarray(triangles) - center
    extent = np.asarray(half_size)
    overlap = np.all((t.max(axis=1) >= -extent-epsilon) &
                     (t.min(axis=1) <= extent+epsilon), axis=1)
    edges = np.stack([t[:,1]-t[:,0], t[:,2]-t[:,1], t[:,0]-t[:,2]], axis=1)
    axes = [np.cross(edges[:,0], edges[:,1])]
    for edge in range(3):
        for world_axis in np.eye(3):
            axes.append(np.cross(edges[:,edge], world_axis))
    for axis in axes:
        projection = np.einsum('nij,nj->ni', t, axis)
        radius = np.abs(axis) @ extent
        overlap &= (projection.min(axis=1) <= radius+epsilon) & (projection.max(axis=1) >= -radius-epsilon)
    return overlap


def audit(path):
    data = json.loads(path.read_text())
    q = data['q_reference']
    T_tool = np.asarray(data['T_hand_tool'])
    k = Kinematics()
    Ts = k.fk(q, root='right_hand_link')
    g = data['tool_geometry']
    intended = {data['contact_targets'][f]['link'] for f in ['thumb', 'index']}
    intersections = []
    for link in k.root.findall('link'):
        name = link.get('name')
        mesh = link.find('visual/geometry/mesh')
        if name not in Ts or mesh is None:
            continue
        triangles = stl_triangles(ASSET/mesh.get('filename'))
        T = np.linalg.inv(T_tool) @ Ts[name]
        triangles = triangles @ T[:3,:3].T + T[:3,3]
        for sign in [-1, 1]:
            R = Rotation.from_rotvec([0, sign*g['open_angle_rad'], 0]).as_matrix()
            tr = (triangles - [sign*g['hinge_offset_m'], 0, 0]) @ R
            overlaps = triangle_box_overlap(tr, [0, 0, g['length_m']/2],
                                             [g['strip_thickness_m']/2, g['strip_width_m']/2, g['length_m']/2])
            if overlaps.any():
                intersections.append({'link':name, 'arm_sign':sign,
                                      'intersecting_triangles':int(overlaps.sum()),
                                      'intersecting_triangle_vertex_z_range_m':[float(tr[overlaps,:,2].min()), float(tr[overlaps,:,2].max())],
                                      'intended_contact_link':name in intended})
    margins = {n:float(min(q[n]-k.by_name[n]['lower'], k.by_name[n]['upper']-q[n])) for n in k.hand}
    soft_valid = all(data['soft_bounds'][n][0]-1e-8 <= q[n] <= data['soft_bounds'][n][1]+1e-8 for n in k.hand)
    points = {}
    for f in ['thumb','index']:
        c = data['contact_targets'][f]
        T = np.linalg.inv(T_tool) @ Ts[c['link']]
        points[f] = T[:3,:3] @ np.asarray(c['point_local']) + T[:3,3]
    separation = points['index'] - points['thumb']
    return {'config_path':str(path.resolve()), 'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'method':'13-axis SAT: each source STL triangle against each oriented finite beam box',
            'tool_intersections':intersections,
            'no_noncontact_triangle_intersections':all(i['intended_contact_link'] for i in intersections),
            'joint_hard_limit_margins_rad':margins,
            'minimum_hard_limit_margin_rad':min(margins.values()),
            'all_reference_joints_inside_soft_bounds':soft_valid,
            'contact_points_in_tool_frame_m':{f:p.tolist() for f,p in points.items()},
            'closing_axis_contact_separation_m':float(abs(separation[0])),
            'axial_contact_separation_m':float(abs(separation[2])),
            'euclidean_contact_separation_m':float(np.linalg.norm(separation)),
            'evidence_frames':data['human_evidence']['selected_frame_indices'],
            'evidence_times_s':data['human_evidence']['selected_time_range_s'],
            'no_world_3d_strong_supervision':all(not f['world_3d_used_for_strong_supervision'] for f in data['human_evidence']['frame_diagnostics']),
            'limitations':['Intended contact patches overlap slightly to initialize preload.',
                           'Uses source visual STL surfaces; simulator convex collision hulls can differ.',
                           'Does not certify inter-finger self-collision or dynamic stability.',
                           'No initial middle-finger tool contact is asserted.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'configs/human_grasp_precision.json')
    p.add_argument('--output', type=Path, default=ROOT/'analysis/human_grasp_validation.json')
    a = p.parse_args()
    result = audit(a.config)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
