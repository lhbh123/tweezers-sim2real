"""Offline source/cooked-hull comparison; never changes a collision setting."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/kinematics.py').is_file());OUT=Path(__file__).resolve().parent
P=argparse.ArgumentParser();P.add_argument('geometry_directory');args=P.parse_args();folder=Path(args.geometry_directory).resolve();meta=json.loads((folder/'metadata.json').read_text())
with np.load(folder/'meshes.npz') as z:data={k:z[k] for k in z.files}
with np.load(OUT/'cold_contact_source_geometry.npz') as z:contacts={k:z[k] for k in z.files}
q=lambda a:[float(x) for x in np.quantile(a,[0,.5,.95,1])]
rows=[];arrays={}
for row in meta['colliders']:
 if not row['hulls']:continue
 finger=next((f for f in ['thumb','index','middle'] if 'right_'+f+'_' in row['collider_path']),None)
 key=row['key'];v=data[key+'_source_vertices'];faces=data[key+'_source_triangles'];source=trimesh.Trimesh(vertices=v*1000,faces=faces,process=False)
 hulls=[];equations=[]
 for h in row['hulls']:
  hv=data[h['key']+'_vertices'];hf=data[h['key']+'_triangles'];hulls.append(trimesh.Trimesh(vertices=hv*1000,faces=hf,process=False))
  equations.append(ConvexHull(hv).equations)
 aggregate=trimesh.util.concatenate(hulls)
 def membership(points):return np.minimum.reduce([np.max(points@eq[:,:3].T+eq[:,3],axis=1) for eq in equations])
 # Sample exported source vertices. This is a containment check, not Hausdorff distance.
 source_points=np.unique(v,axis=0);margin=membership(source_points)
 cooked_vertices=np.vstack([data[h['key']+'_vertices'] for h in row['hulls']]);_,h_to_source,_=trimesh.proximity.closest_point(source,cooked_vertices*1000)
 result={'collider_path':row['collider_path'],'surface_class':row['surface_class'],'hull_count':len(hulls),'source_unique_vertices':len(source_points),'source_vertex_union_plane_margin_um_quantiles':q(margin*1e6),'source_vertices_outside_cooked_union_gt_1um':int((margin>1e-6).sum()),'all_hull_vertices_to_source_CAD_um_quantiles':q(h_to_source*1000),'all_hull_vertex_distance_caveat':'Includes internal decomposition vertices; not exterior Hausdorff error.'}
 if finger and row['mesh_to_rigid_matrix_row_vector'] is not None:
  # Saved shell-local coordinates are in the merged distal rigid (DP) frame.
  dp=contacts[finger+'_raw_local_shell_m'];transform=np.asarray(row['mesh_to_rigid_matrix_row_vector']);local=(np.c_[dp,np.ones(len(dp))]@np.linalg.inv(transform))[:,:3]
  source_closest,source_dist,_=trimesh.proximity.closest_point(source,local*1000);hull_closest,hull_dist,_=trimesh.proximity.closest_point(aggregate,local*1000);cm=membership(local)
  result['archived_cold_contacts']={'finger':finger,'points':len(local),'distance_to_exported_source_mesh_um_quantiles':q(source_dist*1000),'distance_to_any_cooked_hull_surface_um_quantiles':q(hull_dist*1000),'union_plane_margin_um_quantiles':q(cm*1e6),'outside_union_gt_1um_count':int((cm>1e-6).sum()),'near_any_hull_surface_lt_20um_fraction':float((hull_dist<=.020).mean()),'caveat':'The contact points come from a separate archived cold run. Their DP-local coordinates are compared to the current exported collider geometry; this assumes the same robot asset and collision parameters. Internal hull boundaries are included in closest-surface distance.'}
  arrays[key+'_contact_local_m']=local;arrays[key+'_contact_to_source_um']=source_dist*1000;arrays[key+'_contact_to_any_hull_um']=hull_dist*1000;arrays[key+'_contact_union_plane_margin_um']=cm*1e6
 rows.append(result)
result={'geometry_directory':str(folder),'quantile_order':['min','median','p95','max'],'distance_algorithm':'Triangle proximity in mm-scaled coordinates. Plane margins use a re-hulled convex envelope of returned vertices, which can differ from returned native polygon faces; actual surface distances use the returned polygon triangles. It does not change simulation geometry.','limitations':['The native API representation may be regenerated from stage/cooking cache; it is not a resident solver shape-memory dump.','Positive plane margin is a re-hulled envelope diagnostic, not exact native polygon containment or Euclidean distance. See matched_v2_cooked_normal_offset.json for measured envelope/native differences.','Distance to the aggregate hull surfaces includes internal partition boundaries and is not an exterior union Hausdorff distance.','Contact comparison must be interpreted with the recorded mesh-to-rigid transform and identical underlying robot geometry.'],'rows':rows,'sources_sha256':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'metadata.json',folder/'meshes.npz',OUT/'cold_contact_source_geometry.npz',Path(__file__)]}}
name=folder.parent.name+'_cooked_geometry';(OUT/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(OUT/(name+'.npz'),**arrays);print(json.dumps(rows,ensure_ascii=False,indent=2))
