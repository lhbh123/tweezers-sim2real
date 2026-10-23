"""Independent halfspace normal-line intersection with the union of native cooked hulls."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'src/kinematics.py').is_file())
run=ROOT/'sharpa_w01_tactile_validation/improvement_validation/runs/compliant_v1_full_validation'
meta=json.loads((run/'collision_geometry/metadata.json').read_text())
mesh=np.load(run/'collision_geometry/meshes.npz')
pose=json.loads((run/'source/sharpa_w01_tacmap_compliant_v1.json').read_text())
oldposepath=ROOT/'sharpa_w01_tactile_validation/improvement_validation/runs/cold_soft_rest0_v3/source/soft_rest0_closure_v2_witness.json'
oldpose=json.loads(oldposepath.read_text())
rows=[];arrays={};files=[run/'collision_geometry/metadata.json',run/'collision_geometry/meshes.npz',oldposepath,Path(__file__)]
for finger,suffix,key in [('index','4F','collider_1'),('thumb','TH','collider_9')]:
    row=next(r for r in meta['colliders'] if r['key']==key)
    origin='_origin' if suffix=='4F' else ''
    pp=ROOT/pose['tacmap_surface_directory']/('tactileSensor_map_'+suffix+'_point'+origin+'.npy')
    npth=ROOT/pose['tacmap_surface_directory']/('tactileSensor_map_'+suffix+'_normal'+origin+'.npy')
    p=np.load(pp).astype(np.float64)*.001;n=np.load(npth).astype(np.float64);valid=np.linalg.norm(n,axis=-1)>.5
    shape=valid.shape;p=p[valid];n=n[valid];n/=np.linalg.norm(n,axis=-1,keepdims=True)
    equations=[ConvexHull(mesh[h['key']+'_vertices']).equations for h in row['hulls']]
    outer=np.full(len(p),-np.inf);intersections=np.zeros(len(p),int);inside=np.zeros(len(p),bool)
    for eq in equations:
        value=p@eq[:,:3].T+eq[:,3];rate=n@eq[:,:3].T
        parallel=np.abs(rate)<1e-12
        with np.errstate(divide='ignore',invalid='ignore'): t=-value/rate
        lo=np.max(np.where(rate< -1e-12,t,-np.inf),axis=1)
        hi=np.min(np.where(rate> 1e-12,t,np.inf),axis=1)
        hit=(hi>=lo)&~np.any(parallel&(value>1e-10),axis=1)
        outer=np.maximum(outer,np.where(hit,hi,-np.inf));intersections+=hit
        inside|=np.max(value,axis=1)<=1e-9
    hit=np.isfinite(outer)
    # Check the chosen outer exit is outside every hull just beyond the endpoint.
    # This rules out the internal partition faces without constructing a mesh Boolean union.
    exitp=p[hit]+outer[hit,None]*n[hit]
    beyond=exitp+1e-7*n[hit]
    outside_beyond=np.ones(len(beyond),bool)
    on_any_boundary=np.zeros(len(beyond),bool)
    for eq in equations:
        outside_beyond &= np.max(beyond@eq[:,:3].T+eq[:,3],axis=1)>-1e-10
        on_any_boundary |= np.abs(np.max(exitp@eq[:,:3].T+eq[:,3],axis=1))<1e-9
    assert outside_beyond.all() and on_any_boundary.all()
    # Query the returned polygon faces as well: re-hulling their vertices is
    # measurably different near a few nonplanar/rounded polygon boundaries.
    hull_meshes=[trimesh.Trimesh(vertices=mesh[h['key']+'_vertices']*1000,faces=mesh[h['key']+'_triangles'],process=False) for h in row['hulls']]
    aggregate=trimesh.util.concatenate(hull_meshes)
    native_outer=np.full(len(p),-np.inf)
    for begin in range(0,len(p),1024):
        end=min(begin+1024,len(p));orig=(p[begin:end]-.1*n[begin:end])*1000
        loc,ray,_=aggregate.ray.intersects_location(orig,n[begin:end],multiple_hits=True)
        dist=np.einsum('ij,ij->i',loc*.001-p[begin:end][ray],n[begin:end][ray])
        np.maximum.at(native_outer,begin+ray,dist)
    native_hit=np.isfinite(native_outer)
    both=hit&native_hit
    values=native_outer[native_hit]*1e6
    sample_ids=np.flatnonzero(native_hit)
    chosen=np.unique(np.r_[sample_ids[np.linspace(0,len(sample_ids)-1,40).astype(int)],sample_ids[np.argmin(values)],sample_ids[np.argmax(values)]])
    triangles=aggregate.triangles*.001;e1=triangles[:,1]-triangles[:,0];e2=triangles[:,2]-triangles[:,0];direct=[]
    for sample in chosen:
        h=np.cross(n[sample],e2);det=np.einsum('ij,ij->i',e1,h)
        inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-16)
        ss=p[sample]-triangles[:,0];u=np.einsum('ij,ij->i',ss,h)*inv
        q=np.cross(ss,e1);v=q@n[sample]*inv;t=np.einsum('ij,ij->i',e2,q)*inv
        ok=(np.abs(det)>1e-16)&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)
        direct.append(np.max(t[ok]))
    direct_error=float(np.max(abs(np.array(direct)-native_outer[chosen]))*1e6)
    assert direct_error<1e-4
    source_path=ROOT/'assets/sharpa_wave/official/wave_01/right_sharpa_wave/meshes'/('elastomer_HB1_'+suffix+'.STL')
    src=trimesh.load_mesh(source_path,process=False)
    source_exact=np.array_equal(mesh[key+'_source_vertices'],src.vertices)
    image=np.full(shape,np.nan);image[valid]=np.where(native_hit,native_outer*1e6,np.nan)
    arrays[finger+'_normal_line_outer_exit_um']=image;arrays[finger+'_valid']=valid
    arrays[finger+'_line_intersects_union']=np.zeros(shape,bool);arrays[finger+'_line_intersects_union'][valid]=native_hit
    quant=lambda a:[float(x) for x in np.quantile(a,[0,.5,.95,1])]
    result={'finger':finger,'collider_path':row['collider_path'],'valid_map_samples':int(valid.sum()),'intersecting_samples':int(native_hit.sum()),'missing_normal_line_intersections':int((~native_hit).sum()),'all_native_hull_meshes_watertight':bool(all(h.is_watertight for h in hull_meshes)),'source_vertices_exactly_equal_STL':source_exact,'signed_outer_exit_offset_um_quantiles':quant(values),'absolute_outer_exit_offset_um_quantiles':quant(abs(values)),'fraction_abs_offset_gt_20um':float((abs(values)>20).mean()),'fraction_abs_offset_gt_100um':float((abs(values)>100).mean()),'fraction_abs_offset_gt_500um':float((abs(values)>500).mean()),'independent_reconstructed_convex_envelope_comparison':{'common_hit_count':int(both.sum()),'hit_status_mismatches':int((hit!=native_hit).sum()),'native_vs_reconstructed_absolute_offset_um_quantiles':quant(abs(native_outer[both]-outer[both])*1e6),'reconstructed_exit_then_100nm_outward_all_outside_envelope':bool(outside_beyond.all()),'reconstructed_exit_on_envelope_boundary_all':bool(on_any_boundary.all()),'caveat':'The convex envelope of returned vertices is not identical to the returned polygon faces. Main offsets use returned polygon faces.'}}
    rows.append(result);files.extend([pp,npth,source_path])
    result['independent_all_triangle_Moller_Trumbore_spotcheck']={'samples':len(chosen),'includes_min_and_max_offset':True,'max_absolute_error_um':direct_error,'threshold_um':1e-4,'passed':direct_error<1e-4}
report={'method':'For each valid matched_v2 CAD sample p and unit outward normal n, cast from p-100mm*n through all API-returned convex polygon triangles. Select the furthest finite intersection along n among every hull. This is the outer exit of their union, so internal hull partition faces cannot be chosen ahead of the exterior. Negative offset means cooked outer exit lies inward of the source CAD sample. An independent halfspace envelope of the same vertices is reported separately; its small discrepancies are retained.','metric_not':'This is an outward normal-line offset, not a nearest Euclidean distance or a Hausdorff error. Source map normals define the query; absent intersections are reported separately. Convex decomposition gaps and disconnected intervals may occur.','quantile_order':['min','median','p95','max'],'rows':rows,'same_configuration_evidence':{k:oldpose[k]==pose[k] for k in ['robot_asset_dir','tacmap_surface_directory','tacmap_skin_contact','mechanical_self_collision_pairs']},'cross_run_boundary':'Cold-v3 did not archive asset hashes. Same robot path and unchanged scene/FK source hashes support comparability but do not prove historical asset byte identity.','export_scope':'The v1 full task was interrupted for runtime cost; its before-step native cooking export completed successfully. These geometry results do not imply task completion.','sources_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(OUT/'matched_v2_cooked_normal_offset.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
np.savez_compressed(OUT/'matched_v2_cooked_normal_offset.npz',**arrays)
fig,ax=plt.subplots(2,2,figsize=(11,9))
for col,finger in enumerate(['thumb','index']):
    im=ax[0,col].imshow(arrays[finger+'_normal_line_outer_exit_um'],cmap='coolwarm',vmin=-750,vmax=750);ax[0,col].set_title(finger+' : cooked union exit − CAD sample');ax[0,col].set_xlabel('map column');ax[0,col].set_ylabel('map row');fig.colorbar(im,ax=ax[0,col],label='Signed offset along outward map normal (µm)',extend='both')
    vals=arrays[finger+'_normal_line_outer_exit_um'];vals=vals[np.isfinite(vals)]
    ax[1,col].hist(vals,bins=80);ax[1,col].axvline(0,color='black',lw=.8);ax[1,col].set_title('All valid intersecting map samples');ax[1,col].set_xlabel('Signed normal-line offset (µm)');ax[1,col].set_ylabel('Sample count');ax[1,col].grid(alpha=.2)
    row=next(r for r in rows if r['finger']==finger);v=row['absolute_outer_exit_offset_um_quantiles']
    ax[1,col].text(.03,.95,f'Absolute offset (µm)\nMedian {v[1]:.1f}; P95 {v[2]:.1f}\nMax {v[3]:.1f}\nNo intersection: {row["missing_normal_line_intersections"]}',transform=ax[1,col].transAxes,va='top',fontsize=9)
fig.suptitle('Native cooked skin vs matched_v2 CAD surface: union exterior, not internal hull faces',fontsize=13)
fig.text(.5,.015,'Negative: cooked surface is inward along the source outward normal. This metric is not Euclidean Hausdorff distance.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(OUT/'matched_v2_cooked_normal_offset.png',dpi=150)
print(json.dumps(rows,indent=2))
