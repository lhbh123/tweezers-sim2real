"""Read-only geometry prediction for official maps on recorded live poses."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis'))
from validate_sharpa_tacmap import closed_box_cpd,quantize_and_blur

DATA={}
def matrix(pose):
    return Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix()

def one(index):
    d=DATA; sensorposes=d['sensor'][index]; targetposes=d['target'][index]
    boxes=[]
    for body,local in enumerate(d['local_boxes']):
        r=matrix(targetposes[body]);t=targetposes[body,:3]
        for center,half in local:
            boxes.append({'center_m':center@r.T+t,'half_extents_m':half,'rotation_matrix':r})
    out=[]
    for finger in range(5):
        points=d['points'][finger]; valid=d['valid'][finger]
        r=matrix(sensorposes[finger]);t=sensorposes[finger,:3]
        starts=points@r.T+t
        directions=-d['normals'][finger]@r.T
        lengths=np.linalg.norm(directions,axis=-1,keepdims=True)
        directions=directions/(lengths+1e-12)
        depth=closed_box_cpd(starts,directions,boxes,.015,.5)
        image=quantize_and_blur(depth.reshape(240,240),1)
        gap=np.full(len(points),np.inf); deepest=np.zeros(len(points))
        closest_box=np.zeros(len(points),int)
        for b,box in enumerate(boxes):
            q=np.abs((starts-box['center_m'])@box['rotation_matrix'])-box['half_extents_m']
            sd=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
            take=sd<gap;gap[take]=sd[take];closest_box[take]=b
        ix=np.flatnonzero(valid)[np.argmin(gap[valid])]
        out.append({'finger':d['fingers'][finger],
                    'raw_cpd_nonzero_taxels':int(np.count_nonzero(depth[valid])),
                    'raw_cpd_max_mm':float(np.max(depth[valid])*1000),
                    'output_nonzero_taxels':int(np.count_nonzero(image)),
                    'output_max_code':int(image.max()),
                    'map_points_inside_tool':int(np.count_nonzero(gap[valid]<0)),
                    'minimum_signed_box_clearance_mm':float(gap[ix]*1000),
                    'nearest_map_index':int(ix),'nearest_box_index':int(closest_box[ix]),
                    'nearest_map_local_m':points[ix].tolist(),
                    'nearest_map_world_m':starts[ix].tolist()})
    return {'control_step':int(d['steps'][index]),'time_s':float(d['times'][index]),
            'phase':int(d['phase'][index]),'fingers':out}

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--source',type=Path,default=ROOT/'runs/sharpa_w01_official_tacmap_smoke_v3');p.add_argument('--stride',type=int,default=1);p.add_argument('--workers',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--map-directory',type=Path);p.add_argument('--representative',action='store_true');a=p.parse_args()
    meta=json.loads((a.source/'tacmap.json').read_text());maps=np.load(a.source/'tacmap.npz');tr=np.load(a.run/'trajectory.npz');ta=np.load(a.run/'tactile.npz')
    n=len(tr['phase']);steps=np.arange(0,n,a.stride); decimation=int(round(float(tr['dt'])/np.median(ta['substep_dt_s'])))
    if a.representative:
        steps=np.unique(np.concatenate([np.flatnonzero(tr['phase'][:,0]==phase)[[0,len(np.flatnonzero(tr['phase'][:,0]==phase))//2,-1]] for phase in np.unique(tr['phase'])]))
    assert len(ta['substep_time_s'])==n*decimation
    local=[]
    for target in meta['ray_targets']:
        points=np.asarray(target['vertices_body_m']);pieces=[]
        for k in range(0,len(points),8):
            q=points[k:k+8];lo=q.min(0);hi=q.max(0)
            assert all(np.all(np.isclose(v,lo,atol=1e-9)|np.isclose(v,hi,atol=1e-9)) for v in q)
            pieces.append(((lo+hi)/2,(hi-lo)/2))
        local.append(pieces)
    surface_points=maps['surface_points_m'];surface_normals=maps['surface_normals'];surface_valid=maps['surface_valid']
    if a.map_directory:
        points=[];normals=[]
        for suffix,origin in [('TH',''),('4F','_origin'),('4F','_origin'),('4F','_origin'),('4F','_origin')]:
            points.append(np.load(a.map_directory/f'tactileSensor_map_{suffix}_point{origin}.npy')*.001)
            normals.append(np.load(a.map_directory/f'tactileSensor_map_{suffix}_normal{origin}.npy'))
        surface_points=np.stack(points);surface_normals=np.stack(normals)
        derived_valid=np.linalg.norm(surface_normals,axis=-1)>.5
        assert not np.any(derived_valid & ~surface_valid)
        surface_valid=derived_valid
    DATA.update(sensor=ta['substep_sensor_pose_w'][steps*decimation+decimation-1],
                target=np.stack([tr[k][steps,0,:7] for k in ['tool','left','right']],axis=1),
                points=surface_points.reshape(5,-1,3).astype(float),
                normals=surface_normals.reshape(5,-1,3).astype(float),
                valid=surface_valid.reshape(5,-1),fingers=maps['finger_names'].tolist(),
                local_boxes=local,steps=steps,times=(steps+1)*float(tr['dt']),phase=tr['phase'][steps,0])
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
        for i,row in enumerate(pool.map(one,range(len(steps)),chunksize=5)):
            rows.append(row)
            if i%100==0:print('GEOMETRY_PROGRESS',i,len(steps),flush=True)
    summary={}
    for f,name in enumerate(DATA['fingers']):
        vals=[x['fingers'][f] for x in rows]; ix=min(range(len(vals)),key=lambda i:vals[i]['minimum_signed_box_clearance_mm'])
        summary[name]={'states':len(vals),'states_with_inside_map_points':sum(x['map_points_inside_tool']>0 for x in vals),
                       'states_with_nonzero_official_reference_output':sum(x['output_nonzero_taxels']>0 for x in vals),
                       'max_code':max(x['output_max_code'] for x in vals),'max_raw_cpd_mm':max(x['raw_cpd_max_mm'] for x in vals),
                       'min_clearance_mm':vals[ix]['minimum_signed_box_clearance_mm'],'nearest_state':rows[ix]}
    result={'run':str(a.run.resolve()),'method':'Full 240x240 official original surface; actual recorded elastomer-frame poses at each selected control endpoint; exact authored closed-box geometry; independent analytic two-pass ray/box CPD and nonlinear quantization/Gaussian blur. Predictions, not a rerun of SharpaTacmap.',
            'control_stride':a.stride,'control_samples':len(rows),'physics_substeps_per_control':decimation,
            'surface_asset_source':str(a.map_directory.resolve()) if a.map_directory else str(a.source.resolve()),'representative_phase_first_middle_last':a.representative,'summary':summary,'states':rows,
            'evidence_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in [a.run/'trajectory.npz',a.run/'tactile.npz',a.source/'tacmap.json']}}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('GEOMETRY_COMPLETE',str(a.out),json.dumps({k:{x:v[x]for x in ['max_code','min_clearance_mm','states_with_inside_map_points']}for k,v in summary.items()}),flush=True)

if __name__=='__main__':main()
