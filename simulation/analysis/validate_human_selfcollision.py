"""Conservative inter-link convex hull overlap audit for a grasp reference."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from kinematics import ASSET, Kinematics, stl_triangles
from check_mesh_pair import mesh_intersections

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, default=ROOT/'configs/human_grasp_precision_v2.json')
p.add_argument('--output', type=Path, default=ROOT/'analysis/human_grasp_v2_selfcollision.json')
a = p.parse_args()
data = json.loads(a.config.read_text())
k = Kinematics()
Ts = k.fk(data['q_reference'], root='right_hand_link')
adjacent = {frozenset((j['parent'], j['child'])) for j in k.joints}
bodies = []
source_triangles = {}
for link in k.root.findall('link'):
    name = link.get('name')
    mesh = link.find('visual/geometry/mesh')
    if name not in Ts or mesh is None:
        continue
    triangles = stl_triangles(ASSET/mesh.get('filename'))
    triangles = triangles @ Ts[name][:3,:3].T + Ts[name][:3,3]
    source_triangles[name] = triangles
    v = np.unique(triangles.reshape(-1,3), axis=0)
    hull = ConvexHull(v)
    bodies.append((name, v.min(axis=0), v.max(axis=0), hull.equations))
checks = []
for i, (n1, lo1, hi1, e1) in enumerate(bodies):
    for n2, lo2, hi2, e2 in bodies[i+1:]:
        if frozenset((n1,n2)) in adjacent:
            continue
        if np.any(np.maximum(lo1,lo2) > np.minimum(hi1,hi2)):
            checks.append({'links':[n1,n2], 'separated':True, 'method':'disjoint AABBs'})
            continue
        eq = np.concatenate([e1,e2])
        # Largest ball in the intersection of two convex hull halfspaces.
        # A negative optimum proves the two source meshes cannot intersect.
        result = linprog([0,0,0,-1], A_ub=np.column_stack([eq[:,:3],np.ones(len(eq))]),
                         b_ub=-eq[:,3], bounds=[(None,None)]*4, method='highs')
        checks.append({'links':[n1,n2], 'separated':bool(result.success and result.x[3] <= 1e-7),
                       'method':'convex hull halfspace intersection',
                       'max_intersection_ball_radius_m':float(result.x[3]) if result.success else None,
                       'solver_success':bool(result.success)})
        if not checks[-1]['separated']:
            exact = mesh_intersections(source_triangles[n1], source_triangles[n2])
            checks[-1]['exact_surface_intersection_count'] = exact['intersection_pair_count']
            checks[-1]['exact_triangle_pair_tests'] = exact['triangle_pair_tests']
out = {'config':str(a.config), 'checked_nonadjacent_pairs':len(checks),
       'all_nonadjacent_source_mesh_hulls_separated':all(c['separated'] for c in checks),
       'all_nonadjacent_source_mesh_surfaces_disjoint':all(c['separated'] or c.get('exact_surface_intersection_count') == 0 for c in checks),
       'overlap_candidates':[c for c in checks if not c['separated']], 'checks':checks,
       'limitations':['Directly connected URDF links excluded because their mechanical joint geometry overlaps by design.',
                      'Convex hull overlap candidates receive continuous triangle-pair SAT narrow phase.',
                      'Surface separation does not independently check fully contained disconnected shells.',
                      'Uses source STL geometry, not PhysX collider cooking.']}
a.output.write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
