"""Compare published points with paired upstream USD and current W01 surfaces."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
import trimesh
ROOT=Path(__file__).resolve().parents[2]

def q(x):return (np.quantile(x,[0,.05,.5,.95,1])*1000).tolist() if len(x) else []
def distances(mesh,points):
    return trimesh.proximity.closest_point(mesh,points)

def main():
    old=np.load(ROOT/'analysis/sharpa_tacmap/official_elastomer_meshes.npz')
    task=np.load(ROOT/'runs/sharpa_w01_natural_v6d_video/tactile.npz')
    report={'method':'Source geometry only. Original right-hand map coordinates are compared in each elastomer sensor frame. Positive map-normal projection means closest surface lies outward of map point. No scene, offset, force, or sensor output changed.','quantiles_order':[0,.05,.5,.95,1],'models':{}}
    for finger_id,finger,suffix in [(0,'thumb','TH'),(1,'index','4F'),(2,'middle','4F')]:
        folder=ROOT/'assets/sharpa_tactile/assets/tactilesensor_map';origin=''if suffix=='TH'else'_origin'
        pp=folder/f'tactileSensor_map_{suffix}_point{origin}.npy';npth=folder/f'tactileSensor_map_{suffix}_normal{origin}.npy'
        points=np.load(pp).reshape(-1,3)*.001;normals=np.load(npth).reshape(-1,3);valid=np.linalg.norm(normals,axis=-1)>.5
        ids=np.flatnonzero(valid);points=points[valid];normals=normals[valid];normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
        raw=task['normal_records'];raw=raw[raw[:,1]==finger_id];dd,ii=cKDTree(points).query(raw[:,6:9]);v=raw[:,6:9]-points[ii]
        unique=np.unique(ii);pick=np.unique(np.r_[np.linspace(0,len(points)-1,256).astype(int),unique[np.linspace(0,len(unique)-1,min(len(unique),512)).astype(int)]])
        row={'map_points':len(points),'sampled_surface_points':len(pick),'raw_contact_records':len(raw),
             'raw_contact_to_dense_map_distance_mm':q(dd),'raw_contact_minus_map_outward_normal_projection_mm':q(np.sum(v*normals[ii],axis=-1)),
             'raw_contact_tangential_map_distance_mm':q(np.linalg.norm(v-np.sum(v*normals[ii],axis=-1)[:,None]*normals[ii],axis=-1)),
             'surfaces':{},'map_sha256':hashlib.sha256(pp.read_bytes()).hexdigest()}
        asset_finger='thumb'if suffix=='TH'else'index'
        meshes={kind:trimesh.Trimesh(old[f'{asset_finger}_{kind}_points'],old[f'{asset_finger}_{kind}_faces'],process=True)for kind in ['visual','collision']}
        source=ROOT/f'assets/sharpa_wave/official/wave_01/right_sharpa_wave/meshes/elastomer_HB1_{suffix}.STL'
        meshes['current_visual_and_collision']=trimesh.load_mesh(source,process=True)
        for kind,mesh in meshes.items():
            cp,dist,face=distances(mesh,points[pick]);projection=np.sum((cp-points[pick])*normals[pick],axis=-1)
            rc,rd,rf=distances(mesh,raw[np.linspace(0,len(raw)-1,min(len(raw),512)).astype(int),6:9])
            row['surfaces'][kind]={'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),
                                  'map_to_surface_distance_mm':q(dist),'surface_minus_map_normal_projection_mm':q(projection),
                                  'outward_projection_positive_fraction':float(np.mean(projection>1e-5)),
                                  'sampled_raw_contact_to_surface_distance_mm':q(rd)}
        report['models'][finger]=row
        print('SURFACE_DONE',finger,flush=True)
    report['source_anchors']={'official_environment':'assets/sharpa_tactile/env_cfg.py:73-87','compliant_contact_stiffness':280.,'compliant_contact_damping':20.,'robot_rest_offset_m':-.0001,'presser_rest_offset_m':-.0005,
                               'interpretation':'Official example uses compliant rigid-body contact so objects can geometrically penetrate sensor surface; this is not a finite-element deformation solve. Current rigid task lacks that compliant material. Geometry mismatch must be measured against paired USD rather than assumed.'}
    out=ROOT/'analysis/sharpa_tacmap/surface_alignment.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main()
