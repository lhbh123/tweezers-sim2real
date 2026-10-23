"""Read a completed probe's 480 Hz states and independently audit source meshes.

This adds an audit report; it never changes probe data, physics, or collision filters.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
import fcl

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('source_audit',ROOT/'analysis/sharpa_grasp/audit_trajectory_mesh.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--stride',type=int,default=1);p.add_argument('--out',type=Path);a=p.parse_args()
 if a.stride<1:p.error('stride must be positive')
 report=json.loads((a.run/'report.json').read_text());pose=a.run/'source'/Path(report['arguments']['pose']).name
 cfg=json.loads(pose.read_text());meta=dict(cfg,robot_model=cfg['model'],joint_names=report['joint_names'],arm_joint_names=None)
 audit=mod.SourceMeshAudit(meta)
 with np.load(a.run/'timeseries.npz',allow_pickle=False)as d:q=d['q'];bodies=d['bodies'];times=d['time'];closure=d['closure']
 origins=np.asarray(report['env_origins']);nsteps,nenvs=q.shape[:2]
 if bodies.shape!=(nsteps,nenvs,3,13):raise ValueError('Unexpected body state shape')
 indices=sorted(set(range(0,nsteps,a.stride))|{nsteps-1});started=time.monotonic();envs=[];hits=[]
 middle_pairs=[(h,t)for h,t in audit.nontarget_pairs if 'middle' in h]
 for env in range(nenvs):
  summary={'environment':env,'candidate':report['candidates'][env]['name'],'sample_count':len(indices),'self_intersection_samples':0,'non_target_tool_intersection_samples':0,'middle_tool_intersection_samples':0,'min_self_distance_m':float('inf'),'min_non_target_tool_distance_m':float('inf'),'min_middle_tool_distance_m':float('inf')}
  for step in indices:
   states=dict(zip(['tool','left','right'],bodies[step,env]));bounds=audit.state(q[step,env],states,origins[env],True)
   ss=audit.examine(bounds,audit.self_pairs);ts=audit.examine(bounds,audit.nontarget_pairs);ms=audit.examine(bounds,middle_pairs)
   for field,result in [('self',ss),('non_target_tool',ts),('middle_tool',ms)]:
    summary[field+'_intersection_samples']+=bool(result['intersections'])
    key='min_'+field+'_distance_m'
    if result['min_distance_m']<summary[key]:summary[key]=result['min_distance_m'];summary[key+'_at']={'step':step,'time_s':float(times[step]),'pair':result['nearest_pair'],'closure':float(closure[step,env])}
   if ss['intersections'] or ts['intersections']:
    item={'environment':env,'step':step,'time_s':float(times[step]),'self':ss['intersections'],'non_target_tool':ts['intersections']}
    fk=audit.kin.fk(dict(zip(audit.names,q[step,env])));Wroot=mod.transform(origins[env]+[0,0,audit.base_height])@fk[audit.kin.hand_root_link]
    for h in item['non_target_tool']:
     if 'middle' not in h['pair'][0]:continue
     body=audit.tool_body[h['pair'][1]];Tw=mod.pose_transform(states[body]);points=np.asarray(h['witness_points_hand_root_m']);pw=points@Wroot[:3,:3].T+Wroot[:3,3];pl=(pw-Tw[:3,3])@Tw[:3,:3]
     h['middle_contact_witnesses_world_m']=pw.tolist();h['middle_contact_witnesses_tool_body_local_m']=pl.tolist()
     h['lower_side_y_sign']=1 if Tw[2,1]<0 else -1
     h['classification']='Middle source mesh intersects tool; separately recorded, never exempted. Check actual lower-side contact force and source witness before interpreting support.'
    hits.append(item)
  envs.append(summary);print(json.dumps(summary),flush=True)
 result={'schema_version':1,'run':str(a.run.resolve()),'timeseries_sha256':sha(a.run/'timeseries.npz'),'probe_report_sha256':sha(a.run/'report.json'),'source_pose':str(pose),'source_pose_sha256':sha(pose),'source_urdf_sha256':sha(audit.urdf),'script_sha256':sha(__file__),'source_audit_sha256':sha(ROOT/'analysis/sharpa_grasp/audit_trajectory_mesh.py'),'method':'FCL original URDF source triangles at saved joint positions and independently recorded tool body poses. Same fixed/adjacent source assemblies and seven official mechanical pairs only. No middle-tool exemptions.','sampling':{'stride':a.stride,'saved_physics_steps':nsteps,'num_environments':nenvs,'exact_states':len(indices)*nenvs,'saved_dt_s':report['physics_dt_s'],'all_saved_steps_checked':a.stride==1},'official_filters':audit.filter_source,'excluded_self_pairs':audit.excluded_pairs,'summary':envs,'intersections':hits,'wall_seconds':time.monotonic()-started,'limitations':['FCL triangle witness depth is not a reliable maximum volumetric penetration depth.','Original triangles differ from PhysX convex colliders; FK does not include joint solver compliance.','This audit does not establish real middle support force or tactile-map validity.','Discrete saved state checks are not continuous collision detection.']}
 output=a.out or a.run/'source_mesh_probe_audit.json';output.write_text(json.dumps(result,indent=2)+'\n');print('AUDIT_READY',str(output),result['wall_seconds'],flush=True)

if __name__=='__main__':main()
