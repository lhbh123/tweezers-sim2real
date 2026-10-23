"""Inspect source meshes and measured force points in a completed hold snapshot."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import NaturalFit,transform,module

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--candidate',default='close_release75');p.add_argument('--out',type=Path);a=p.parse_args();out=a.out or ROOT/'analysis/sharpa_grasp/natural_v6'/a.run.name;out.mkdir(exist_ok=True,parents=True)
 report=json.loads((a.run/'report.json').read_text());env=next(i for i,c in enumerate(report['candidates'])if c['name']==a.candidate)
 with np.load(a.run/'timeseries.npz',allow_pickle=False)as d:q=d['q'][-1,env];bodies=d['bodies'][-1,env];time=float(d['time'][-1]);up=float(d['role_support_z_N'][-1,env,2]);last_time=d['time']>=time-1.;mean_up=float(d['role_support_z_N'][last_time,env,2].mean())
 f=NaturalFit();audit=f.audit;assert report['joint_names']==audit.names;qs=dict(zip(audit.names,q));origin=np.asarray(report['env_origins'][env]);states=dict(zip(['tool','left','right'],bodies));bounds=audit.state(q,states,origin,True)
 fk=f.kin.fk(qs);Wroot=transform(origin+[0,0,.72])@fk[f.kin.hand_root_link];Wpalm=transform(origin+[0,0,.72])@fk[f.kin.palm_link];f.Wpalm=Wpalm[:3,:3];f.T=np.linalg.inv(Wpalm)@module.pose_transform(states['tool'])
 raw=[json.loads(s)for s in(a.run/'raw_contact_points.jsonl').read_text().splitlines()];raw=[r for r in raw if r['env']==env][-1];assert abs(raw['time_s']-time)<1e-5
 contacts=[c for c in raw['points']['normal_contacts']if c['role']=='middle'];friction=[c for c in raw['points']['friction_anchors']if c['role']=='middle'];source_details=[]
 for contact in contacts:
  pw=np.asarray(contact['point_world']);pr=Wroot[:3,:3].T@(pw-Wroot[:3,3]);detail={'measured_normal_contact':contact,'source_surfaces':{}}
  for name in ['right_middle_elastomer','right_middle_DP']:
   obj=audit.objects[name];R=np.asarray(obj.getRotation());t=np.asarray(obj.getTranslation());local=R.T@(pr-t);closest,distance,triangle=trimesh.proximity.closest_point(audit.meshes[name],[local]);cp=R@closest[0]+t;cw=Wroot[:3,:3]@cp+Wroot[:3,3];normal=Wroot[:3,:3]@R@audit.meshes[name].face_normals[triangle[0]]
   detail['source_surfaces'][name]={'distance_m':float(distance[0]),'closest_world_m':cw.tolist(),'closest_local_m':closest[0].tolist(),'triangle_index':int(triangle[0]),'triangle_normal_world':normal.tolist()}
  source_details.append(detail)
 ss=audit.examine(bounds,audit.self_pairs);ts=audit.examine(bounds,audit.nontarget_pairs);mid=audit.examine(bounds,f.middle_pairs)
 centers={}
 for name in ['right_middle_DP','right_middle_elastomer','right_index_DP','right_index_elastomer']:
  ob=audit.objects[name];pr=np.asarray(ob.getRotation())@audit.meshes[name].bounds.mean(0)+np.asarray(ob.getTranslation());centers[name]=(Wroot[:3,:3]@pr+Wroot[:3,3]).tolist()
 index=f.cfg['contact_targets']['index'];index_world=transform(origin+[0,0,.72])@fk[index['link']];index_point=index_world[:3,:3]@np.asarray(index['point_local'])+index_world[:3,3]
 geometry={'middle_DP_center_below_index_DP_m':centers['right_index_DP'][2]-centers['right_middle_DP'][2],'source_centers_world_m':centers,'index_source_contact_target_world_m':index_point.tolist(),'middle_contact_below_index_source_target_m':[float(index_point[2]-c['point_world'][2])for c in contacts],'middle_contact_below_local_jaw_axis_m':[-c.get('radial_offset_world_z_m',float('nan'))for c in contacts]}
 meshes={}
 for name,m in {**audit.meshes,**audit.tool_meshes}.items():
  ob=audit.objects[name];mesh=m.copy();mesh.apply_transform(transform(np.asarray(ob.getTranslation()),np.asarray(ob.getRotation())));mesh.apply_transform(Wroot);mesh.vertices-=Wpalm[:3,3];meshes[name]=mesh
 cloud=np.vstack([m.vertices for m in meshes.values()]);center=(cloud.min(0)+cloud.max(0))/2;rad=np.ptp(cloud,axis=0).max()*.53
 contact_points=[np.asarray(c['point_world'])-Wpalm[:3,3]for c in contacts];fig=plt.figure(figsize=(16,12))
 for i,(elev,azim,title)in enumerate([(10,-80,'Actual gravity side view'),(20,25,'Thumb/index broad-face pinch'),(15,165,'Opposite view / curled ring and pinky'),(10,-50,'Measured middle contact close-up')]):
  ax=fig.add_subplot(2,2,i+1,projection='3d');ax.set_proj_type('ortho')
  for name,m in meshes.items():
   color=next((v for k,v in f.render_colors.items()if k in name),'#bac0c9')if name not in audit.tool_meshes else '#626080';stride=max(1,len(m.faces)//2500)
   ax.add_collection3d(Poly3DCollection(m.triangles[::stride]*1000,facecolor=color,edgecolor='none',alpha=.96))
  for c in contacts:
   pos=np.asarray(c['point_world'])-Wpalm[:3,3];force=np.asarray(c['force_on_tool_N']);ax.scatter(*pos*1000,c='#e02a20',s=50,depthshade=False);ax.quiver(*pos*1000,*(force*30),color='#e02a20',linewidth=2)
  for c in friction:
   pos=np.asarray(c['point_world'])-Wpalm[:3,3];force=np.asarray(c['force_on_tool_N']);ax.quiver(*pos*1000,*(force*30),color='#123fcb',linewidth=2)
  ctr=(np.mean(contact_points,axis=0)if i==3 and contact_points else center);rr=.045 if i==3 else rad
  for axis,j in [('x',0),('y',1),('z',2)]:getattr(ax,'set_'+axis+'lim')((ctr[j]-rr)*1000,(ctr[j]+rr)*1000)
  ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim);ax.set_title(title);ax.set_xlabel('World X (mm)');ax.set_ylabel('World Y (mm)');ax.set_zlabel('World Z (mm)')
 fig.suptitle(f'Actual {a.candidate} final state at {time:.2f} s — source mesh reconstruction\nMeasured middle upward force {up:.3f} N; final 1 s mean {mean_up:.3f} N',fontsize=15)
 fig.text(.5,.02,'Green: middle; purple/pink: ring/pinky. Red point/arrow: measured normal contact/force. Blue arrow: measured friction force.\nArrow scale 30 mm/N. Joint FK source reconstruction may differ from PhysX convex contact; source overlap is reported separately.',ha='center',fontsize=10);fig.subplots_adjust(left=.01,right=.99,top=.91,bottom=.08,wspace=.02,hspace=.04);fig.savefig(out/'actual_source_views.png',dpi=145);plt.close(fig);trimesh.Scene(meshes).export(out/'actual_source_geometry.glb')
 result={'run':str(a.run),'candidate':a.candidate,'environment':env,'time_s':time,'timeseries_sha256':sha(a.run/'timeseries.npz'),'script_sha256':sha(__file__),'force_point_time_s':raw['time_s'],'middle_final_up_N':up,'middle_last_1s_mean_up_N':mean_up,'middle_normal_contacts':source_details,'middle_friction_anchors':friction,'posture_geometry':geometry,'final_self_source_check':ss,'final_non_target_source_check':ts,'final_middle_source_check':mid,'last_1s_broad_face_qualification':report['candidates'][env]['final_1s_both_assigned_broad_face_fraction'],'limitations':['Measured contact robot rigid-body name is merged DP; original elastomer versus rigid shell attribution is estimated from source triangle distance and normal.','FCL witness depth is not a maximum solid penetration depth; see PhysX measured separation and original surface distances.','This is one final state plus separately reported final-second forces, not full trajectory clearance or task success.']};(out/'actual_source_inspection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
