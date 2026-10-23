"""Fit natural inactive fingers and optional under-tool middle support for W01.

Keeps the verified thumb/index grasp and hand-to-tool transform. All screening
uses official source collision meshes and the existing seven mechanical pairs.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,itertools,json,sys,time
import numpy as np
from scipy.spatial.transform import Rotation
import fcl,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from kinematics import transform
spec=importlib.util.spec_from_file_location('source_audit',ROOT/'analysis/sharpa_grasp/audit_trajectory_mesh.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NaturalFit:
 def __init__(self):
  self.cfg=json.loads((ROOT/'configs/sharpa_w01_task_v4.json').read_text())
  metadata=json.loads((ROOT/'runs/sharpa_w01_video_v5_hard_case/task.json').read_text())
  self.audit=module.SourceMeshAudit(metadata);self.kin=self.audit.kin
  self.T=np.asarray(self.cfg['T_hand_tool']);self.Wtool=np.asarray(self.cfg['preferred_tool_world_R'])
  self.Wpalm=self.Wtool@self.T[:3,:3].T
  self.middle_pairs=[(h,t) for h in self.audit.keys if 'middle' in h for t in self.audit.tool_meshes]
  self.middle_distal_pairs=[(h,t) for h in self.audit.keys if h in ['right_middle_DP','right_middle_elastomer'] for t in ['left_blade','right_blade']]
  self.mid_self_pairs=[(a,b) for a,b in self.audit.self_pairs if 'middle' in a or 'middle' in b]
  self.render_colors={'thumb':'#e7a94e','index':'#47a9c4','middle':'#70b77c','ring':'#a797bd','pinky':'#d893b1'}
 def pose(self,others,closure=.75,middle_residual=0.):
  q={n:self.cfg['q_reference'][n]+closure*self.cfg['closure_delta'][n] for n in self.cfg['q_reference']}
  q.update(others);q['right_middle_PIP']+=middle_residual
  F=self.kin.fk(q);base=transform([0,0,.72]);T=base@F[self.kin.palm_link]@self.T
  poses={'tool':T}
  angle=self.cfg['tool_geometry']['open_angle_rad']*(1-closure)
  for side,sign in [('left',-1),('right',1)]:
   poses[side]=T@transform([sign*self.cfg['tool_geometry']['hinge_offset_m'],0,0],Rotation.from_euler('y',sign*angle).as_matrix())
  states={n:np.r_[T[:3,3],np.roll(Rotation.from_matrix(T[:3,:3]).as_quat(),1),np.zeros(6)] for n,T in poses.items()}
  bounds=self.audit.state(np.asarray([q.get(n,0.) for n in self.audit.names]),states,np.zeros(3),True)
  return q,bounds
 def closest(self,pairs):
  best=None
  for a,b in pairs:
   result=fcl.DistanceResult();distance=float(fcl.distance(self.audit.objects[a],self.audit.objects[b],fcl.DistanceRequest(enable_nearest_points=True),result))
   if best is None or distance<best['distance_m']:
    best={'pair':[a,b],'distance_m':distance,'points_root_m':[np.asarray(v).tolist() for v in result.nearest_points]}
  return best
 def support(self,q):
  r=self.closest(self.middle_distal_pairs)
  rootpalm=self.kin.fk(q,root=self.kin.hand_root_link)[self.kin.palm_link]
  p=np.asarray(r['points_root_m']);palm=(p-rootpalm[:3,3])@rootpalm[:3,:3]
  delta=self.Wpalm@(palm[1]-palm[0]);tool=(palm-self.T[:3,3])@self.T[:3,:3]
  r.update(points_palm_m=palm.tolist(),points_tool_m=tool.tolist(),tool_minus_middle_world_m=delta.tolist(),vertical_gap_m=float(delta[2]),horizontal_gap_m=float(np.linalg.norm(delta[:2])),middle_below_tool=bool(delta[2]>0),tool_side_y_m=float(tool[1,1]),tool_axial_z_m=float(tool[1,2]))
  fk=self.kin.fk(q,root=self.kin.hand_root_link);index=self.cfg['contact_targets']['index'];it=fk[index['link']];ip=it[:3,:3]@np.asarray(index['point_local'])+it[:3,3]
  root_world=self.Wpalm@rootpalm[:3,:3].T
  r['middle_witness_below_index_contact_m']=float((root_world@(ip-p[0]))[2])
  centers={}
  for name in ['right_middle_DP','right_middle_elastomer','right_index_DP','right_index_elastomer']:
   obj=self.audit.objects[name];local=self.audit.meshes[name].bounds.mean(0);center=np.asarray(obj.getRotation())@local+np.asarray(obj.getTranslation())
   centers[name]=(root_world@(center-rootpalm[:3,3])).tolist()
  r['source_mesh_aabb_centers_world_relative_to_palm_m']=centers
  r['middle_DP_center_below_index_DP_center_m']=float(centers['right_index_DP'][2]-centers['right_middle_DP'][2])
  return r
 def check(self,others,closures=(.65,.75,.85,1.),residuals=(-.005,0,.005),full=True):
  rows=[]
  for closure,residual in itertools.product(closures,residuals):
   q,b=self.pose(others,closure,residual)
   ss=self.audit.examine(b,self.audit.self_pairs if full else self.mid_self_pairs)
   ts=self.audit.examine(b,self.audit.nontarget_pairs if full else self.middle_pairs)
   support=self.support(q)
   rows.append({'closure':closure,'middle_PIP_residual':residual,'self_pairs':[h['pair'] for h in ss['intersections']],'non_target_tool_pairs':[h['pair'] for h in ts['intersections']],'self_gap_m':ss['min_distance_m'],'non_target_tool_gap_m':ts['min_distance_m'],'support':support})
  bad=sum(len(r['self_pairs'])+len(r['non_target_tool_pairs']) for r in rows)
  supports=[r['support'] for r in rows]
  return {'collision_count':bad,'min_self_gap_m':min(r['self_gap_m'] for r in rows),'min_tool_gap_m':min(r['non_target_tool_gap_m'] for r in rows),'middle_support_gap_range_m':[min(r['distance_m'] for r in supports),max(r['distance_m'] for r in supports)],'middle_vertical_gap_range_m':[min(r['vertical_gap_m'] for r in supports),max(r['vertical_gap_m'] for r in supports)],'rows':rows}
 def plot(self,others,out,closure=.75):
  q,_=self.pose(others,closure);F=self.kin.fk(q,root=self.kin.hand_root_link);palm=F[self.kin.palm_link]
  R=self.Wpalm@palm[:3,:3].T;t=-R@palm[:3,3]
  ms={}
  for name,local in {**self.audit.meshes,**self.audit.tool_meshes}.items():
   obj=self.audit.objects[name];T=transform(np.asarray(obj.getTranslation()),np.asarray(obj.getRotation()))
   m=local.copy();m.apply_transform(T);m.vertices=m.vertices@R.T+t;ms[name]=m
  support=self.support(q);ps=np.asarray(support['points_root_m'])@R.T+t
  cloud=np.vstack([m.vertices for m in ms.values()]);lo=cloud.min(0);hi=cloud.max(0);center=(lo+hi)/2;radius=max(hi-lo)*.52
  fig=plt.figure(figsize=(17,13),facecolor='white')
  for i,(elev,azim,title) in enumerate([(12,-80,'Side / middle finger below blade'),(22,20,'Oblique broad-face pinch'),(20,160,'Opposite side / folded fingers'),(80,-80,'Top / collision clearance')]):
   ax=fig.add_subplot(2,2,i+1,projection='3d');ax.set_proj_type('ortho')
   for n,m in ms.items():
    color=next((v for k,v in self.render_colors.items() if k in n),'#b8bdc5') if n not in self.audit.tool_meshes else '#626080'
    fs=m.triangles;stride=max(1,len(fs)//1800)
    ax.add_collection3d(Poly3DCollection(fs[::stride]*1000,facecolor=color,edgecolor='none',alpha=.92))
   ax.plot(*ps.T*1000,c='red',lw=2);ax.scatter(*ps.T*1000,c=['#167d2d','#a51010'],s=28,depthshade=False)
   for axis,j in [('x',0),('y',1),('z',2)]:getattr(ax,'set_'+axis+'lim')((center[j]-radius)*1000,(center[j]+radius)*1000)
   ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim);ax.set_xlabel('World X (mm)');ax.set_ylabel('World Y (mm)');ax.set_zlabel('World Z (mm)');ax.set_title(title)
  fig.suptitle('W01 natural posture candidate | fixed thumb/index wide-face pinch\nMiddle optional support below tool; ring/pinky curled | closure '+str(closure),fontsize=16)
  fig.text(.5,.02,f"Green: middle finger   Red segment: nearest middle-to-blade gap {support['distance_m']*1000:.2f} mm (vertical {support['vertical_gap_m']*1000:.2f} mm).\nSource collision meshes and gravity-aligned views; static candidate, no dynamic success claim.",ha='center',fontsize=11)
  fig.subplots_adjust(left=.01,right=.99,bottom=.075,top=.91,wspace=.02,hspace=.06);fig.savefig(out/'views.png',dpi=145);plt.close(fig)
  trimesh.Scene(ms).export(out/'geometry.glb')
  return dict(world_tool_rotation=self.Wtool.tolist(),world_palm_rotation=self.Wpalm.tolist(),support_points_world_relative_to_palm_m=ps.tolist(),tool_root_world_relative_to_palm_m=(self.Wpalm@self.T[:3,3]).tolist())


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['explore','explore_lower','curl','finish'],default='explore');p.add_argument('--out',default='analysis/sharpa_grasp/natural_v6');a=p.parse_args();out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
 fitter=NaturalFit();records=[];start=time.monotonic()
 if a.stage=='explore':
  # Start with moderately flexed middle finger, varying inward curl and lateral placement.
  for fe,pip,dip,aa in itertools.product([.35,.4,.45,.50,.55,.6,.65,.70,.75,.8,.85],[.5,.7,.9,1.1,1.3,1.5],[.2,.4,.6,.8],[.16,.18,.20,.22,.24,.26]):
   others={f'right_middle_{j}':v for j,v in [('MCP_FE',fe),('PIP',pip),('DIP',dip),('MCP_AA',aa)]}
   q,b=fitter.pose(others,.75);ss=fitter.audit.examine(b,fitter.mid_self_pairs);ts=fitter.audit.examine(b,fitter.middle_pairs)
   if ss['intersections'] or ts['intersections'] or ss['min_distance_m']<.0005 or ts['min_distance_m']<.0007:continue
   support=fitter.support(q)
   if not support['middle_below_tool'] or support['tool_side_y_m']<.0065:continue
   distance=support['distance_m']
   score=abs(distance-.002)+max(0,.0006-support['vertical_gap_m'])*3+max(0,distance-.004)*3
   records.append({'other_joints':others,'score':score,'support':support,'self_gap_m':ss['min_distance_m'],'tool_gap_m':ts['min_distance_m']})
  records.sort(key=lambda r:r['score']);(out/'middle_candidates.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({'middle_candidates':len(records),'best':records[:5],'seconds':time.monotonic()-start},indent=2),flush=True)
 if a.stage=='explore_lower':
  # The closest local blade point alone does not establish an under-index pose.
  for fe,pip,dip,aa in itertools.product([-.05,.05,.15,.25,.35],[.7,.9,1.1,1.3],[.2,.4,.6,.8],[.16,.17,.18,.19,.20]):
   others={f'right_middle_{j}':v for j,v in zip(['MCP_FE','PIP','DIP','MCP_AA'],[fe,pip,dip,aa])};q,b=fitter.pose(others)
   ss=fitter.audit.examine(b,fitter.mid_self_pairs);ts=fitter.audit.examine(b,fitter.middle_pairs)
   if ss['intersections'] or ts['intersections'] or ss['min_distance_m']<.0005 or ts['min_distance_m']<.001:continue
   support=fitter.support(q)
   if not support['middle_below_tool'] or support['tool_side_y_m']<.0065 or support['distance_m']>.004:continue
   records.append({'other_joints':others,'score':abs(support['distance_m']-.002)+max(0,.042-support['tool_axial_z_m'])*.2,'support':support,'self_gap_m':ss['min_distance_m'],'tool_gap_m':ts['min_distance_m']})
  records.sort(key=lambda r:r['score']);(out/'middle_lower_candidates.json').write_text(json.dumps(records,indent=2)+'\n');print('LOWER_CANDIDATES',len(records),flush=True)
 if a.stage=='curl':
  # A wider middle gap is acceptable for optional backup and gives a clear lower
  # stair-step relative to the actual index contact, without moving the pinch.
  mid=dict(zip(['right_middle_'+s for s in ['MCP_FE','PIP','DIP','MCP_AA']],[.5,.9,.4,.2]))
  for rfe,rpip,raa,pfe,ppip,paa in itertools.product([.65,.8,1.],[.8,1.1],[.16,.22,.28],[.8,1.,1.2],[.9,1.2],[.16,.22,.28]):
   others={**mid,**dict(zip(['right_ring_'+s for s in ['MCP_FE','PIP','DIP','MCP_AA']],[rfe,rpip,.55,raa])),**dict(zip(['right_pinky_'+s for s in ['MCP_FE','PIP','DIP','MCP_AA','CMC']],[pfe,ppip,.6,paa,.1]))}
   q,b=fitter.pose(others);ss=fitter.audit.examine(b,fitter.audit.self_pairs);ts=fitter.audit.examine(b,fitter.audit.nontarget_pairs)
   if ss['intersections'] or ts['intersections'] or ss['min_distance_m']<.0005:continue
   records.append({'joints':others,'selfgap':ss['min_distance_m'],'toolgap':ts['min_distance_m'],'score':abs(rfe-.8)+abs(rpip-1.1)+abs(raa-.2)+abs(pfe-1)+abs(ppip-1.2)+abs(paa-.2)})
  records.sort(key=lambda r:r['score']);(out/'curl_candidates.json').write_text(json.dumps(records,indent=2)+'\n');print('CURL_CANDIDATES',len(records),flush=True)
 if a.stage=='finish':
  curls=json.loads((out/'curl_candidates.json').read_text())[0]['joints']
  variants={'a':curls,'b':dict(curls,right_middle_MCP_FE=.4,right_middle_PIP=1.1,right_middle_DIP=.4,right_middle_MCP_AA=.18)}
  for label,others in variants.items():
   folder=out/('candidate_'+label);folder.mkdir(exist_ok=True)
   audit=fitter.check(others,closures=tuple(np.arange(.65,1.001,.05)),residuals=(-.005,0,.005))
   (folder/'static_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
   if audit['collision_count']:print('rejected',label,audit['collision_count'],flush=True);continue
   q,_=fitter.pose(others);support=fitter.support(q)
   cfg=copy.deepcopy(fitter.cfg)
   cfg['method']='Natural W01 posture: fixed verified thumb/index broad-face pinch, lower middle backup and curled ring/pinky; official source mesh FCL screening'
   for key in ['q_reference','q_open','q_closed']:cfg[key].update(others)
   cfg['grasp_topology']['middle']='Optional lower-side backup; finite initial gap, no sustained contact requirement'
   cfg['grasp_topology']['inactive_fingers']='Ring and pinky curled naturally toward the palm'
   cfg['backup_support_geometry']={'semantics':'Static nearest-source-mesh clearance under preferred gravity orientation; not a measured physical support force','closure':.75,'source':str((folder/'static_audit.json').relative_to(ROOT)),'nearest':support,'closure_range':[.65,1.0],'middle_PIP_residual_range_rad':[-.005,.005],'summary':{k:v for k,v in audit.items() if k!='rows'}}
   cfg['natural_posture_provenance']={'base_config':'configs/sharpa_w01_task_v4.json','script':'analysis/sharpa_grasp/fit_natural_support.py','thumb_index_transform_unchanged':True,'official_mechanical_filters_unchanged':True,'dynamic_validation':'pending'}
   path=ROOT/f'configs/sharpa_w01_natural_v6{label}.json';path.write_text(json.dumps(cfg,indent=2)+'\n');print('CONFIG_READY',path, json.dumps({k:v for k,v in audit.items() if k!='rows'}),flush=True)
   coords=fitter.plot(others,folder);(folder/'view_geometry.json').write_text(json.dumps(coords,indent=2)+'\n')
   retract=dict(others,right_middle_MCP_FE=-.12,right_middle_MCP_AA=.12,right_middle_PIP=.04,right_middle_DIP=.04)
   ra=fitter.check(retract,closures=tuple(np.arange(.65,1.001,.05)),residuals=(-.005,0,.005));middle_gaps=[]
   for c in np.arange(.65,1.001,.05):
    _,bounds=fitter.pose(retract,float(c));middle_gaps.append(fitter.audit.examine(bounds,fitter.middle_pairs)['min_distance_m'])
   ra['min_middle_to_tool_gap_m']=min(middle_gaps)
   (folder/'retracted_audit.json').write_text(json.dumps(ra,indent=2)+'\n')
   if not ra['collision_count'] and min(middle_gaps)>.01:
    rc=copy.deepcopy(cfg)
    for key in ['q_reference','q_open','q_closed']:rc[key].update(retract)
    rc['backup_support_geometry']={'semantics':'Matched counterfactual: only middle finger retracted away from tool','min_middle_to_tool_gap_m':min(middle_gaps),'audit':str((folder/'retracted_audit.json').relative_to(ROOT))}
    (ROOT/f'configs/sharpa_w01_natural_v6{label}_middle_retracted.json').write_text(json.dumps(rc,indent=2)+'\n')
    print('RETRACT_READY',label,min(middle_gaps),flush=True)


if __name__=='__main__':main()
