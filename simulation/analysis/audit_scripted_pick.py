"""Independent actual-state/source/USD audit of one scripted pick environment.

Not an RL certification. Missing raw point/contact substeps are explicitly
reported; logged support/bilateral Booleans cannot substitute for raw evidence.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis')]
from kinematics import Kinematics,ASSET,stl_triangles,transform,xyz
from audit_tripod_settled import boxes_for_config
spec=importlib.util.spec_from_file_location('mainvalidator',ROOT/'scripts/validate_rl_run.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--env',type=int,default=25);a=p.parse_args();run=a.run;env=a.env
x=np.load(run/'trajectory_arrays.npz');task=json.loads((run/'task.json').read_text());launch=json.loads((run/'launch.json').read_text());cfg=json.loads((run/'source'/Path(launch['arguments']['pose']).name).read_text());names=task['joint_names'];kin=Kinematics();dt=task['decimation']*task['physics_dt'];times=np.arange(1,len(x['q'])+1)*dt
v.USD_AUDIT=v.USD_AUDIT.replace("env='/World/envs/env_0'",f"env='/World/envs/env_{env}'");usd=v.usd_audit(run/'final_scene.usda');base=np.asarray(usd['robot_base_world_matrix_gf_row_vectors']).T;table=usd['table_top_world_z_m'];blockhalf=usd['block_size_m'][2]/2
vertices={}
for link in kin.root.findall('link'):
 name=link.get('name')
 if name in ['base_link','base_link_inertia']:continue
 parts=[]
 for c in link.findall('collision'):
  mesh=c.find('geometry/mesh')
  if mesh is None:continue
  points=stl_triangles(ASSET/mesh.get('filename')).reshape(-1,3)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ');o=c.find('origin');M=np.eye(4) if o is None else transform(xyz(o.get('xyz')),Rotation.from_euler('xyz',xyz(o.get('rpy'))).as_matrix());parts.append(points@M[:3,:3].T+M[:3,3])
 if parts:
  points=np.unique(np.concatenate(parts),axis=0);vertices[name]=points[ConvexHull(points).vertices]
mins={n:float('inf') for n in vertices};minstate={};palms=[];rows=[];violations={n:0. for n in names}
for i,q in enumerate(x['q'][:,env]):
 F=kin.fk(dict(zip(names,q)))
 for name,points in vertices.items():
  T=base@F[name];clearance=float((points@T[2,:3]+T[2,3]).min()-table)
  if clearance<mins[name]:mins[name]=clearance;minstate[name]={'frame':i,'time':float(times[i])}
 for j,n in enumerate(names):violations[n]=max(violations[n],kin.by_name[n]['lower']-q[j],q[j]-kin.by_name[n]['upper'])
 palms.append(base@F['right_hand_link']);rows.append({'time':float(times[i]),'joint_pos':q.tolist(),'tool':x['tool'][i,env].tolist(),'left':x['left'][i,env].tolist(),'right':x['right'][i,env].tolist(),'cube':x['block'][i,env].tolist()})
rows_path=run/f'env{env}_extracted_actual_trajectory.json';rows_path.write_text(json.dumps({'joint_names':names,'source_npz_sha256':hashlib.sha256((run/'trajectory_arrays.npz').read_bytes()).hexdigest(),'selected_environment':env,'frames':rows},indent=2))
R={n:Rotation.from_quat(x[n][:,env][:,[4,5,6,3]]).as_matrix() for n in ['tool','left','right']};toolmins={};toolclearance={}
for body,part,c,h,E in boxes_for_config(cfg):
 n='tool' if body=='handle' else body;c=np.asarray(c);h=np.asarray(h);corners=np.array([[dx,dy,dz] for dx in [-h[0],h[0]] for dy in [-h[1],h[1]] for dz in [-h[2],h[2]]])+c;corners=corners@E[:3,:3].T+E[:3,3];world=np.einsum('tij,kj->tki',R[n],corners)+x[n][:,env,None,:3];toolclearance[body+'/'+part]=world[:,:,2].min(axis=1)-table;toolmins[body+'/'+part]=float(toolclearance[body+'/'+part].min())
physical_lift=x['block'][:,env,2]-(table+blockhalf+.0001);threshold=task['target_lift'];tip=.5*(x['left'][:,env,:3]+R['left']@np.array([0.,0.,cfg['tool_geometry']['length_m']-.0075])+x['right'][:,env,:3]+R['right']@np.array([0.,0.,cfg['tool_geometry']['length_m']-.0075]));tipaxis=tip-x['tool'][:,env,:3];tipaxis/=np.linalg.norm(tipaxis,axis=1,keepdims=True);tableangle=np.degrees(np.arctan2(-R['tool'][:,2,2],np.linalg.norm(R['tool'][:,:2,2],axis=1)));tipangle=np.degrees(np.arctan2(-tipaxis[:,2],np.linalg.norm(tipaxis[:,:2],axis=1)));H=np.stack(palms);web=np.asarray(task['grasp_topology']['web_edge_axis_local']);webz=(H[:,:3,:3]@web)[:,2];palmz=H[:,2,1]
# Necessary recorded force inequalities only: raw contact points, ROI and underside
# counts are absent, and recorded bilateral is not an independent force check.
force_necessary=(x['strength'][:,env,:3]>.003).all(1)&(x['support_z'][:,env,2]>.003)&(x['web_z'][:,env]>.003)
logged_qual=(physical_lift>=.054)&x['both'][:,env]&x['support'][:,env]
hashes={name:hashlib.sha256((run/'source'/Path(name).name).read_bytes()).hexdigest()==expected for name,expected in launch['source_sha256'].items()};calls=[]
class Visitor(ast.NodeVisitor):
 def __init__(self,file):self.file=file;self.stack=[]
 def visit_FunctionDef(self,node):self.stack.append(node.name);self.generic_visit(node);self.stack.pop()
 def visit_Call(self,node):
  calls.append({'file':self.file,'function':self.stack[-1] if self.stack else '<module>','line':node.lineno,'call':ast.unparse(node.func),'auto_reset':next((ast.literal_eval(k.value) for k in node.keywords if k.arg=='auto_reset'),None)});self.generic_visit(node)
for f in (run/'source').glob('*.py'):Visitor(f.name).visit(ast.parse(f.read_text()))
writes=[c for c in calls if c['call'].endswith(('write_root_state_to_sim','write_root_pose_to_sim','write_root_velocity_to_sim','write_joint_state_to_sim','set_world_poses','set_joint_positions'))];external=[c for c in calls if any(w in c['call'].lower() for w in ['apply_force','apply_torque','external_force','external_wrench'])];steps=[c for c in calls if c['call']=='task.step'];joints=usd['tool_block_connected_joints'];fixed=[j for j in joints if j['type']=='PhysicsFixedJoint'];velocity=x['block'][:,env,7:10];fd=np.diff(x['block'][:,env,:3],axis=0)/dt
report={'algorithm':'scripted mechanical feasibility, NOT RL','environment':env,'frames':len(times),'sample_dt_s':dt,'artifacts_sha256':{f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ['trajectory_arrays.npz','task.json','launch.json','final_scene.usda']},'source_hashes_match':hashes,
 'checks':{'source_hashes_match':all(hashes.values()),'all18_recorded_joint_limits':max(violations.values())<1e-4,'all_source_moving_robot_meshes_clear_table':min(mins.values())>0,'all_actual_tool_boxes_clear_table':min(toolmins.values())>0,'cube_lift_matches_geometric_height':float(np.max(abs(physical_lift-x['lift'][:,env])))<2e-6,'no_tool_or_cube_fixed_joint_in_USD':not fixed,'only_two_passive_tool_springs_in_USD':len(joints)==2 and all(j['type']=='PhysicsRevoluteJoint' and j['angular_drive']['type']['value']=='force' and j['angular_drive']['stiffness']['value']>0 and all(z['time_samples']==0 for z in j['angular_drive'].values()) for j in joints),'source_state_writes_only_in_reset':bool(writes) and all(c['function']=='reset' for c in writes),'source_no_external_tool_force':not external,'script_disables_auto_reset':bool(steps) and all(c['auto_reset'] is False for c in steps),'source_FK_web_and_tool_orientation_all_frames':bool((webz>=task['min_web_up_component']).all() and (palmz<=task['max_palm_up_component']).all() and (tableangle>=20).all() and (tableangle<=60).all() and (tipangle>=20).all() and (tipangle<=60).all())},
 'height_evidence':{'max_lift_m':float(physical_lift.max()),'final_lift_m':float(physical_lift[-1]),'above54mm_contiguous_interval':v.longest_interval(physical_lift>=.054,times),'above54mm_with_RECORDED_bilateral_and_support_interval':v.longest_interval(logged_qual,times),'max_frame_position_delta_m':float(np.linalg.norm(np.diff(x['block'][:,env,:3],axis=0),axis=1).max()),'position_difference_vs_mean_recorded_velocity_max_mps':float(np.linalg.norm(fd-.5*(velocity[:-1]+velocity[1:]),axis=1).max())},
 'recorded_force_necessary_conditions':{'last_second_fraction':float(force_necessary[-60:].mean()),'logged_support_last_second':float(x['support'][-60:,env].mean()),'mean_middle_Fz_last_second_N':float(x['support_z'][-60:,env,2].mean()),'mean_web_Fz_last_second_N':float(x['web_z'][-60:,env].mean())},
 'source_mesh_table_clearance_by_link_m':mins,'lowest_source_mesh_states':minstate,'actual_tool_box_clearance_by_part_m':toolmins,'joint_limit_violation_rad':violations,
 'orientation':{'web_z_min_max':[float(webz.min()),float(webz.max())],'palm_z_min_max':[float(palmz.min()),float(palmz.max())],'tool_angle_deg_min_max':[float(tableangle.min()),float(tableangle.max())],'root_tip_angle_deg_min_max':[float(tipangle.min()),float(tipangle.max())]},
 'state_write_calls':writes,'external_force_calls':external,'task_step_calls':steps,'USD_evidence':usd,
 'evidence_limits':['No raw normal/friction points or per-substep samples were saved in this experiment; the full four-role support gate cannot be independently reconstructed.','Bilateral contact is a saved Boolean without jaw-force values; no independent bilateral-force certificate.','Geometry is verified at all300 recorded60Hz poses, not between samples.','USD and AST demonstrate authored structure/write paths; no runtime write interception trace.','Nonadjacent hand surface audit is a separate derived-trajectory report; source tail overlap remains real contact geometry.']}
report['tool_table_negative_clearance_events']={}
for part,clearance in toolclearance.items():
 bad=np.flatnonzero(clearance<0)
 if bad.size:
  minimum=int(np.argmin(clearance));report['tool_table_negative_clearance_events'][part]={'count':int(bad.size),'first_time_s':float(times[bad[0]]),'last_time_s':float(times[bad[-1]]),'minimum_frame':minimum,'minimum_time_s':float(times[minimum]),'minimum_clearance_m':float(clearance[minimum])}
lifted=physical_lift>=.054
report['minimum_tool_table_clearance_while_cube_above54mm_m']=float(np.stack(list(toolclearance.values()))[:,lifted].min()) if lifted.any() else None
hand_audit_path=run/f'env{env}_hand_geometry_audit.json'
if hand_audit_path.exists():
 hand_audit=json.loads(hand_audit_path.read_text());derived_sha=hashlib.sha256(rows_path.read_bytes()).hexdigest()
 report['nonadjacent_hand_source_geometry_audit']={'path':str(hand_audit_path),'sha256':hashlib.sha256(hand_audit_path.read_bytes()).hexdigest(),'extracted_trajectory_sha256':derived_sha,'audit_trajectory_sha256_matches':hand_audit['trajectory_sha256']==derived_sha,'summary':hand_audit['summary'],'scope':'All recorded poses; this linked audit does not certify geometry between the recorded samples.'}
(run/f'env{env}_independent_pick_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':report['checks'],'height':report['height_evidence'],'minimum_robot_clearance_m':min(mins.values()),'minimum_tool_clearance_m':min(toolmins.values()),'orientation':report['orientation']},indent=2))
