"""Read-only source-CAD and TacMap coverage audit of saved tensor contacts."""
from pathlib import Path
import json,sys,hashlib,importlib.util
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'analysis/validate_sharpa_tacmap.py').is_file());OUT=Path(__file__).resolve().parent
run=ROOT/'sharpa_w01_tactile_validation/improvement_validation/runs/cold_soft_rest0_v3';report=json.loads((run/'report.json').read_text());launch=json.loads((run/'launch.json').read_text());pose_path=run/'source'/Path(launch['arguments']['pose']).name;pose=json.loads(pose_path.read_text());origins=np.asarray(report['env_origins']);snapshot_path=next(run.glob('*_snapshot.json'));arm_base=np.asarray(json.loads(snapshot_path.read_text())['arm_base_world_xyz'])
spec=importlib.util.spec_from_file_location('frozen_kinematics',run/'source/kinematics.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);kin=module.Kinematics(Path(pose['robot_asset_dir'])/'robot.urdf',palm_link=pose['palm_link'],hand_root_link=pose['hand_root_link'])
with np.load(run/'timeseries.npz') as z:times=z['time'];joint_q=z['q']
contacts={finger:[] for finger in ['thumb','index','middle']};maxdt=0.
for line in (run/'raw_contact_points.jsonl').read_text().splitlines():
 event=json.loads(line);t=event['time_s'];frame=int(np.argmin(abs(times-t)));maxdt=max(maxdt,float(abs(times[frame]-t)));env=event['env'];Ts=kin.fk(dict(zip(report['joint_names'],joint_q[frame,env])));base=origins[env]+arm_base
 for point in event['points']['normal_contacts']:
  finger=point['role']
  if finger not in contacts or np.linalg.norm(point['force_on_tool_N'])<=1e-8:continue
  world=np.asarray(point['point_world']);normal=np.asarray(point['normal_world']);sep=float(point['separation_m']);skin=Ts['right_'+finger+'_elastomer'];shell=Ts['right_'+finger+'_DP'];local=lambda w,T:(w-T[:3,3]-base)@T[:3,:3]
  contacts[finger].append({'time':t,'skin_raw':local(world,skin),'shell_raw':local(world,shell),'skin_minus_sep_n':local(world-sep*normal,skin),'skin_plus_sep_n':local(world+sep*normal,skin),'normal_local':normal@skin[:3,:3],'force':float(np.linalg.norm(point['force_on_tool_N'])),'separation':sep,'world_raw':world,'tool':point['tool']})
assert maxdt<1e-6
stats=[];arrays={};references=[]
def closest_mm(mesh, points):
    scaled=mesh.copy();scaled.apply_scale(1000.)
    closest,distance,face=trimesh.proximity.closest_point(scaled,np.asarray(points)*1000.)
    return closest*.001,distance*.001,face

quantiles=lambda a:[float(v) for v in np.quantile(a,[0,.5,.95,1])]
for finger,points in contacts.items():
 suffix='TH' if finger=='thumb' else '4F';meshroot=ROOT/'assets/sharpa_wave/official/wave_01/right_sharpa_wave/meshes';skinpath=meshroot/('elastomer_HB1_'+suffix+'.STL');shellpath=meshroot/('DP_HB1_'+suffix+'.STL');skin=trimesh.load_mesh(skinpath,process=False);shell=trimesh.load_mesh(shellpath,process=False)
 origin='' if finger=='thumb' else '_origin';folder=ROOT/pose['tacmap_surface_directory'];mp=folder/('tactileSensor_map_'+suffix+'_point'+origin+'.npy');mn=folder/('tactileSensor_map_'+suffix+'_normal'+origin+'.npy');map_points=np.load(mp)*.001;mapnorm=np.load(mn);valid=np.linalg.norm(mapnorm,axis=-1)>.5;tree=cKDTree(map_points[valid])
 raw=np.stack([p['skin_raw'] for p in points]);normal=np.stack([p['normal_local'] for p in points]);normal/=np.linalg.norm(normal,axis=1,keepdims=True);closest,distance,face=closest_mm(skin,raw);_,sdistance,_=closest_mm(shell,np.stack([p['shell_raw'] for p in points]));mdistance,mi=tree.query(raw);assert np.all(mdistance>=distance-2e-9), 'Nearest discrete sample cannot be closer than continuous CAD';sample_normal=mapnorm[valid][mi];sample_normal/=np.linalg.norm(sample_normal,axis=1,keepdims=True);fn=skin.face_normals[face]
 cos=np.einsum('ij,ij->i',normal,fn);mcos=np.einsum('ij,ij->i',normal,sample_normal);normal_agree=np.einsum('ij,ij->i',fn,sample_normal)
 ang=lambda c:np.degrees(np.arccos(np.clip(abs(c),0,1)))
 shifted={}
 for field in ['skin_minus_sep_n','skin_plus_sep_n']:
  _,dist,_=closest_mm(skin,np.stack([p[field] for p in points]));shifted[field]=quantiles(dist*1e6)
 row={'finger':finger,'normal_contact_points':len(points),'quantile_order':['min','median','p95','max'],'raw_to_skin_CAD_distance_um':quantiles(distance*1e6),'raw_to_shell_CAD_distance_um':quantiles(sdistance*1e6),'raw_to_nearest_valid_map_sample_distance_um':quantiles(mdistance*1e6),'raw_to_skin_CAD_under_20um_fraction':float((distance<=20e-6).mean()),'skin_closer_than_shell_by_50um_fraction':float((distance+50e-6<sdistance).mean()),'shell_closer_than_skin_by_50um_fraction':float((sdistance+50e-6<distance).mean()),'signed_contact_normal_dot_skin_face_normal':quantiles(cos),'unsigned_contact_normal_angle_to_skin_face_deg':quantiles(ang(cos)),'unsigned_contact_normal_angle_to_nearest_map_normal_deg':quantiles(ang(mcos)),'unsigned_nearest_map_normal_angle_to_skin_face_deg':quantiles(ang(normal_agree)),'shifted_candidates_to_skin_CAD_distance_um':shifted,'force_weighted_mean_raw_to_skin_CAD_distance_um':float(np.average(distance*1e6,weights=[p['force'] for p in points]))}
 stats.append(row)
 for key,value in [('raw_local_skin_m',raw),('raw_local_shell_m',np.stack([p['shell_raw'] for p in points])),('raw_world_m',np.stack([p['world_raw'] for p in points])),('source_closest_skin_m',closest),('raw_to_skin_m',distance),('raw_to_shell_m',sdistance),('raw_to_map_m',mdistance),('normal_local_skin',normal),('skin_face_normal',fn),('map_nearest_normal',sample_normal),('time_s',np.array([p['time'] for p in points])),('force_N',np.array([p['force'] for p in points])),('separation_m',np.array([p['separation'] for p in points]))]:arrays[finger+'_'+key]=value
 references.extend([skinpath,shellpath,mp,mn])
result={'scope':'Source geometry proximity and normal audit of actual tensor contact points from cold_soft_rest0_v3. No new simulation.','surface':'matched_v2 CAD reprojection; no change to original map indexing or physics geometry.','distance_algorithm':'Trimesh exact triangle proximity in millimetre-scaled coordinates to avoid inappropriate metre-scale absolute tolerances; exported distances converted back to metres. Discrete-map distance >= continuous-CAD distance is asserted within 2 nm.','maximum_contact_vs_joint_sample_time_error_s':maxdt,'stats':stats,'limitations':['These contact arrays identify a distal body, not the actual collider. Nearest-CAD classification is geometric evidence only and does not prove which PhysX shape generated a point.','Raw contact position and both separation-shifted candidates are assessed separately; no universal witness-point convention is assumed.','A contact normal is a separating/constraint direction and need not equal a smooth skin normal, especially at edges or under convex decomposition.','Distance to a discrete TacMap sample includes sample spacing and excluded coverage, so it is not a direct CAD projection error.','Source STL is not the cooked convex union. That comparison requires the separately exported native collision representation.'],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(references+[run/'raw_contact_points.jsonl',run/'timeseries.npz',pose_path,snapshot_path,run/'source/kinematics.py',Path(__file__)])}}
np.savez_compressed(OUT/'cold_contact_source_geometry.npz',**arrays);(OUT/'cold_contact_source_geometry.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
fig,ax=plt.subplots(2,3,figsize=(13,8));bins=np.geomspace(.01,1e4,65)
for col,finger in enumerate(contacts):
 a=ax[0,col]
 for key,label in [('raw_to_skin_m','skin CAD'),('raw_to_shell_m','shell CAD'),('raw_to_map_m','map sample')]:a.hist(np.maximum(arrays[finger+'_'+key]*1e6,.01),bins=bins,histtype='step',label=label)
 a.set_xscale('log');a.set_title(finger+' actual normal contact points');a.set_xlabel('Closest distance (um)');a.set_ylabel('Point count');a.legend(fontsize=8);a.grid(alpha=.2)
 a=ax[1,col];cos=np.einsum('ij,ij->i',arrays[finger+'_normal_local_skin'],arrays[finger+'_skin_face_normal']);a.hist(ang(cos),bins=np.linspace(0,90,31));a.set_xlabel('Unsigned normal angle to source skin face (deg)');a.set_ylabel('Point count');a.grid(alpha=.2)
fig.suptitle('Cold hold: actual tensor contacts vs source CAD and matched_v2 samples',fontsize=15);fig.text(.5,.015,'Geometric proximity does not identify the generating collider or replace a cooked-hull audit.',ha='center');fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'cold_contact_source_geometry.png',dpi=140);plt.close(fig)
print(json.dumps(stats,indent=2))
