"""Quantify source-surface penetration into tool boxes, separately from FCL witnesses.

For each source triangle, a small linear program maximizes the signed interior
distance to the six box planes over all barycentric points on that triangle.
This is an exact directed surface-into-box depth (up to solver tolerance), not
a solid intersection volume or the minimum translation to separate two bodies.
"""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
import trimesh
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'analysis/sharpa_grasp'))
from fit_natural_support import module,transform

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
 x=np.asarray(x,float);return {'count':len(x),'min':float(x.min()),'max':float(x.max()),'median':float(np.median(x))}if len(x)else{'count':0}
def surface_box_depth(mesh,Ts,Tb,box,exact=True):
 center=box.bounds.mean(0);half=np.ptp(box.bounds,axis=0)/2;T=np.linalg.inv(Tb)@Ts;p=(mesh.vertices@T[:3,:3].T+T[:3,3]-center)*1000;h=half*1000
 signed=np.min(h-np.abs(p),axis=1);best=max(0.,float(signed.max()));vertex_best=best;best_point=p[int(np.argmax(signed))]if best>0 else None;best_triangle=None
 tri=p[mesh.faces];candidates=np.flatnonzero(np.all(tri.min(1)<=h,axis=1)&np.all(tri.max(1)>=-h,axis=1));inside=signed>0
 if exact:
  for j in candidates:
   V=tri[j];A=np.vstack([np.c_[V.T,np.ones(3)],np.c_[-V.T,np.ones(3)]]);b=np.r_[h,h]
   r=linprog([0,0,0,-1],A_ub=A,b_ub=b,A_eq=[[1,1,1,0]],b_eq=[1],bounds=[(0,None)]*3+[(0,None)],method='highs')
   if r.success and r.x[3]>best:best=float(r.x[3]);best_point=r.x[:3]@V;best_triangle=int(j)
 point=None if best_point is None else best_point/1000+center
 return {'max_source_surface_inside_box_depth_m':best/1000,'max_source_vertex_inside_box_depth_m':vertex_best/1000,'source_vertices_inside_box':int(inside.sum()),'triangle_AABB_candidates':len(candidates),'maximizer_source_triangle_index':best_triangle,'maximizer_tool_body_local_m':None if point is None else point.tolist(),'half_thickness_m':float(half[0]),'exact_triangle_surface_optimization':exact}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();report=json.loads((a.run/'report.json').read_text());pose=a.run/'source'/Path(report['arguments']['pose']).name;cfg=json.loads(pose.read_text());audit=module.SourceMeshAudit(dict(cfg,robot_model=cfg['model'],joint_names=report['joint_names'],arm_joint_names=None));old=json.loads((a.run/'source_mesh_probe_audit.json').read_text());origins=np.asarray(report['env_origins']);
 with np.load(a.run/'timeseries.npz',allow_pickle=False)as d:q=d['q'];bodies=d['bodies'];times=d['time']
 def frame(step,env):
  states=dict(zip(['tool','left','right'],bodies[step,env]));audit.state(q[step,env],states,origins[env],True);return states
 def matrix(key):
  o=audit.objects[key];return transform(np.asarray(o.getTranslation()),np.asarray(o.getRotation()))
 began=time.monotonic();rows=[];env_summary=[]
 for env in range(q.shape[1]):
  hits=[r for r in old['intersections']if r['environment']==env];selected=set([0,len(times)-1]+[int(np.argmin(abs(times-t)))for t in [.1,.15,.2,.3,.5,.75,1.,1.5]])
  if hits:
   selected.add(hits[0]['step']);selected.add(max(hits,key=lambda r:max(h['max_fcl_witness_depth_m']for h in r['non_target_tool']))['step'])
  pairs=sorted(set(tuple(h['pair'])for r in hits for h in r['non_target_tool']if 'middle'in h['pair'][0]));vertex_max={};vertex_rows=[]
  for step in range(len(times)):
   frame(step,env)
   for hand,tool in pairs:
    r=surface_box_depth(audit.meshes[hand],matrix(hand),matrix(tool),audit.tool_meshes[tool],False);key=hand+'/'+tool;vertex_rows.append((step,key,r['max_source_vertex_inside_box_depth_m']))
    if key not in vertex_max or r['max_source_vertex_inside_box_depth_m']>vertex_max[key][1]:vertex_max[key]=(step,r['max_source_vertex_inside_box_depth_m'])
  selected.update(v[0]for v in vertex_max.values());erows=[]
  for step in sorted(selected):
   frame(step,env)
   for hand,tool in pairs:
    r=surface_box_depth(audit.meshes[hand],matrix(hand),matrix(tool),audit.tool_meshes[tool]);r.update(environment=env,step=step,time_s=float(times[step]),pair=[hand,tool]);rows.append(r);erows.append(r)
  env_summary.append({'environment':env,'candidate':report['candidates'][env]['name'],'all_state_vertex_scan_samples':len(times),'max_all_state_source_vertex_depth_m':max((x[2]for x in vertex_rows),default=0.),'exact_triangle_surface_optimized_states':len(selected),'max_selected_exact_surface_depth_m':max((r['max_source_surface_inside_box_depth_m']for r in erows),default=0.),'final_exact_surface_depths':[r for r in erows if r['step']==len(times)-1]});print(json.dumps(env_summary[-1]),flush=True)
 normal=[]
 for line in (a.run/'raw_contact_points.jsonl').open():
  raw=json.loads(line);env=raw['env'];step=int(np.argmin(abs(times-raw['time_s'])));frame(step,env);fk=audit.kin.fk(dict(zip(audit.names,q[step,env])));Wroot=transform(origins[env]+[0,0,audit.base_height])@fk[audit.kin.hand_root_link]
  for c in raw['points']['normal_contacts']:
   if c['role']!='middle':continue
   pr=Wroot[:3,:3].T@(np.asarray(c['point_world'])-Wroot[:3,3]);r={'environment':env,'time_s':raw['time_s'],'tool':c['tool'],'separation_m':c.get('separation_m'),'force_on_tool_N':c['force_on_tool_N']}
   for hand in ['right_middle_DP','right_middle_elastomer']:
    T=matrix(hand);pl=T[:3,:3].T@(pr-T[:3,3]);r[hand+'_source_distance_m']=float(trimesh.proximity.closest_point(audit.meshes[hand],[pl])[1][0])
   normal.append(r)
 raw_summary={str(env):{key:stats([r[key]for r in normal if r['environment']==env and r[key]is not None])for key in ['separation_m','right_middle_DP_source_distance_m','right_middle_elastomer_source_distance_m']}for env in range(q.shape[1])}
 result={'method':__doc__,'source_run':str(a.run),'timeseries_sha256':sha(a.run/'timeseries.npz'),'source_audit_sha256':sha(a.run/'source_mesh_probe_audit.json'),'raw_contact_points_sha256':sha(a.run/'raw_contact_points.jsonl'),'script_sha256':sha(__file__),'summary':env_summary,'selected_exact_samples':rows,'raw_normal_contacts':normal,'raw_normal_summary':raw_summary,'wall_seconds':time.monotonic()-began,'limitations':['All saved physics states receive source-vertex depth checks; exact triangle-interior optimization is performed at the listed diagnostic states, not every state.','Depth is one-sided source surface inside the rectangular tool solid. It is not overlap volume, whole-body penetration, or an elastic strain estimate.','Source pose reconstruction here uses recorded joint FK; the probe did not record actual sensor poses.','PhysX separation concerns solver contact points, which can lie elsewhere than the deepest source-surface point.','No additional collision exclusions are introduced, and the original FCL intersection findings remain unchanged.']};out=a.run/'source_overlap_depth.json';out.write_text(json.dumps(result,indent=2)+'\n');print('REPORT',out,flush=True);print(json.dumps(raw_summary,indent=2),flush=True)
if __name__=='__main__':main()
