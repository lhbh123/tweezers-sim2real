"""Derive TacMap input surfaces from current W01 CAD without tactile data.

Published map indexing is retained. Points are projected onto actual source
triangles by their global Euclidean nearest point. No outward offset is applied.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[2]

def evidence(path):
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def nearest_compatible(mesh,point,normal):
    eligible=np.flatnonzero(mesh.face_normals@normal>.05)
    projected=trimesh.triangles.closest_point(mesh.triangles[eligible],np.repeat(point[None],len(eligible),axis=0))
    i=np.argmin(np.linalg.norm(projected-point,axis=-1))
    return projected[i],eligible[i]

def build(suffix,dest):
    origin=''if suffix=='TH'else'_origin'
    source=ROOT/'assets/sharpa_tactile/assets/tactilesensor_map'
    pp=source/f'tactileSensor_map_{suffix}_point{origin}.npy'
    npth=source/f'tactileSensor_map_{suffix}_normal{origin}.npy'
    meshpath=ROOT/f'assets/sharpa_wave/official/wave_01/right_sharpa_wave/meshes/elastomer_HB1_{suffix}.STL'
    original=np.load(pp);normals=np.load(npth);valid=np.linalg.norm(normals,axis=-1)>.5
    p=original[valid].astype(float)*.001;n=normals[valid].astype(float);n/=np.linalg.norm(n,axis=-1,keepdims=True)
    mesh=trimesh.load_mesh(meshpath,process=True)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    locations=np.zeros_like(p);faces=np.full(len(p),-1,int);distance=np.full(len(p),np.inf);methods=np.zeros(len(p),np.uint8)
    # Exact global nearest points avoid long grazing-ray matches to distant faces.
    for start in range(0,len(p),512):
        stop=min(start+512,len(p))
        cp,dd,ff=trimesh.proximity.closest_point(mesh,p[start:stop])
        locations[start:stop]=cp;distance[start:stop]=dd;faces[start:stop]=ff;methods[start:stop]=3
        if start%8192==0:print('MAP_CLOSEST',suffix,start,len(p),flush=True)
    bary=trimesh.triangles.points_to_barycentric(mesh.triangles[faces],locations)
    vertexnormals=mesh.vertex_normals[mesh.faces[faces]]
    mappednormals=np.sum(vertexnormals*bary[:,:,None],axis=1)
    mappednormals/=np.linalg.norm(mappednormals,axis=-1,keepdims=True)
    # A smooth normal must point outward of the containing source triangle.
    assert np.all(np.sum(mappednormals*mesh.face_normals[faces],axis=-1)>0)
    angles=np.degrees(np.arccos(np.clip(np.sum(n*mappednormals,axis=-1),-1,1)))
    faceangles=np.degrees(np.arccos(np.clip(np.sum(n*mesh.face_normals[faces],axis=-1),-1,1)))
    # Geometry-only acceptance rules, fixed before any task-output prediction.
    accepted=(distance<=.0025)&(angles<=60.)
    derived_valid=np.array(valid,copy=True);derived_valid[valid]=accepted
    outputpoints=np.array(original,copy=True);outputnormals=np.array(normals,copy=True)
    outputpoints[valid]=0.;outputnormals[valid]=0.
    outputpoints[derived_valid]=(locations[accepted]*1000).astype(np.float32)
    outputnormals[derived_valid]=mappednormals[accepted].astype(np.float32)
    assert np.array_equal(outputpoints[~valid],original[~valid])
    assert np.array_equal(outputnormals[~valid],normals[~valid])
    reconstructed=np.sum(mesh.triangles[faces]*bary[:,:,None],axis=1)
    triangle_error=np.linalg.norm(reconstructed-locations,axis=-1)
    p_out=dest/pp.name;n_out=dest/npth.name
    np.save(p_out,outputpoints);np.save(n_out,outputnormals)
    np.savez_compressed(dest/f'{suffix}_correspondence.npz',original_valid=valid,valid=derived_valid,accepted=accepted,
        rejected_distance=distance>.0025,rejected_normal_angle=angles>60.,
        original_points_m=p,derived_points_m=locations,original_normals=n,derived_normals=mappednormals,
        source_face_index=faces,source_face_vertex_indices=mesh.faces[faces],barycentric=bary,
        source_triangle_vertices_m=mesh.triangles[faces],method=methods,displacement_m=locations-p,
        normal_angle_deg=angles,source_face_normal_angle_deg=faceangles)
    return {'model':suffix,'units':{'points':'mm','normals':'unit vectors'},'shape':list(original.shape),
            'original_valid_points':len(p),'valid_points':int(accepted.sum()),
            'original_invalid_domain_unchanged':True,'unsupported_added_points':int((~accepted).sum()),
            'retained_surface_fraction':float(accepted.mean()),'arbitrary_outward_offset_m':0.,
            'rejected_distance_count':int(np.sum(distance>.0025)),
            'rejected_normal_angle_count':int(np.sum(angles>60.)),
            'rejected_both_count':int(np.sum((distance>.0025)&(angles>60.))),
            'trajectory_or_force_input_used':False,'source_map':evidence(pp),'source_normals':evidence(npth),
            'source_elastomer_mesh':evidence(meshpath),'derived_points':evidence(p_out),'derived_normals':evidence(n_out),
            'correspondence':evidence(dest/f'{suffix}_correspondence.npz'),
            'method_codes':{'3':'exact global nearest source-triangle point'},
            'method_counts':{'3':int(len(p))},
            'missing_correspondences':int(np.sum(faces<0)),
            'displacement_mm_quantiles_0_50_95_99_100':(np.quantile(distance,[0,.5,.95,.99,1])*1000).tolist(),
            'normal_angle_deg_quantiles_0_50_95_99_100':np.quantile(angles,[0,.5,.95,.99,1]).tolist(),
            'normal_angle_over_90_count':int(np.sum(angles>90)),
            'acceptance_rules':{'maximum_displacement_m':.0025,'maximum_normal_angle_deg':60.,'selection_uses_task_or_force':False},
            'accepted_displacement_mm_quantiles_0_50_95_99_100':(np.quantile(distance[accepted],[0,.5,.95,.99,1])*1000).tolist(),
            'accepted_normal_angle_deg_quantiles_0_50_95_99_100':np.quantile(angles[accepted],[0,.5,.95,.99,1]).tolist(),
            'accepted_normal_angle_over_90_count':int(np.sum(angles[accepted]>90)),
            'source_triangle_barycentric_min_max':[float(bary.min()),float(bary.max())],
            'source_triangle_reconstruction_error_max_m':float(triangle_error.max()),
            'export_float32_coordinate_error_max_m':float(np.linalg.norm(outputpoints[derived_valid].astype(float)*.001-locations[accepted],axis=-1).max())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'assets/sharpa_tacmap_w01_matched_v2');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report={'label':'Derived W01 CAD-matched TacMap input surfaces, not the original published maps',
            'official_sensor_algorithm_modified':False,'arbitrary_outward_offset_m':0.,'force_fitting':False,
            'method':'Preserve original 240x240 indexing and all original invalid cells; exact global nearest point on current W01 source triangles; barycentric interpolated outward vertex normals. Reject displacement>2.5mm or normal-angle>60deg as additional unsupported cells. This is geometric reprojection, not a bijective or hardware-taxel calibration.',
            'generator':evidence(Path(__file__)),'models':[build(s,a.out)for s in ['TH','4F']]}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MATCHED_MAP_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
