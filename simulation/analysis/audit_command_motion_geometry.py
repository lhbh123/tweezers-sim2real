"""Recorded actual-joint source-mesh table-clearance audit for actuation probe."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis')]
from kinematics import Kinematics,ASSET,stl_triangles,transform,xyz
from audit_tripod_settled import boxes_for_config
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();data=np.load(a.run/'records.npz');ini=json.loads((a.run/'initial_state.json').read_text());cfg=json.loads((a.run/'source/pose.json').read_text());kin=Kinematics();names=ini['joint_names'];qs=data['q_actual'];origins=np.asarray(ini['env_origins']);vertices={}
for link in kin.root.findall('link'):
 name=link.get('name');c=link.find('collision');mesh=link.find('collision/geometry/mesh')
 if mesh is None or name in ['base_link','base_link_inertia']:continue
 v=np.unique(stl_triangles(ASSET/mesh.get('filename')).reshape(-1,3),axis=0)
 if len(v)>4:v=v[ConvexHull(v).vertices]
 o=c.find('origin');T=transform(xyz(o.get('xyz')),Rotation.from_euler('xyz',xyz(o.get('rpy'))).as_matrix()) if o is not None else np.eye(4)
 vertices[name]=v@T[:3,:3].T+T[:3,3]
minimum={name:float('inf') for name in vertices};lowest_state={}
for step in range(len(qs)):
 for env in range(qs.shape[1]):
  fk=kin.fk(dict(zip(names,qs[step,env])))
  for name,v in vertices.items():
   T=fk[name];z=float(np.min(v@T[2,:3]+T[2,3]))
   if z<minimum[name]:minimum[name]=z;lowest_state[name]={'time_s':float(data['time'][step]),'env':env}
# Base is mounted at tabletop z=.72; source robot z relative to base is table clearance.
body_states=data['objects'];toolclear={};body_index={'handle':0,'left':1,'right':2}
for body,part,c,h,E in boxes_for_config(cfg):
 c=np.asarray(c);h=np.asarray(h);corners=np.array([[x,y,z] for x in [-h[0],h[0]] for y in [-h[1],h[1]] for z in [-h[2],h[2]]])+c;corners=corners@E[:3,:3].T+E[:3,3]
 s=body_states[:,:,body_index[body]];shape=s.shape;R=Rotation.from_quat(s.reshape(-1,13)[:,[4,5,6,3]]).as_matrix().reshape(*shape[:2],3,3)
 world=np.einsum('teij,kj->teki',R,corners)+s[:,:,:3,None].transpose(0,1,3,2)
 toolclear[body+'/'+part]=float(np.min(world[:,:,:,2]-origins[None,:,None,2]-.72))
limits={n:{'lower':kin.by_name[n]['lower'],'upper':kin.by_name[n]['upper']} for n in names}
violations={n:max(0.,limits[n]['lower']-float(qs[:,:,j].min()),float(qs[:,:,j].max())-limits[n]['upper']) for j,n in enumerate(names)}
handR=Rotation.from_quat(data['palm_quat_w'].reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(*qs.shape[:2],3,3);rel=np.einsum('teji,tej->tei',handR,body_states[:,:,0,:3]-data['palm_pos_w'])
ref=rel[0];relative_range=np.ptp(rel,axis=0);report={'source_URDF_sha256':hashlib.sha256((ASSET/'robot.urdf').read_bytes()).hexdigest(),'physics_trajectory_sha256':hashlib.sha256((a.run/'records.npz').read_bytes()).hexdigest(),'recorded_control_frames':len(qs),'environments':qs.shape[1],
 'minimum_source_robot_mesh_table_clearance_by_link_m':minimum,'lowest_state_by_link':lowest_state,'minimum_source_tool_box_table_clearance_by_part_m':toolclear,
 'joint_limit_violation_rad':violations,'max_actual_handle_in_hand_axis_range_m_per_env':relative_range.max(1).tolist(),
 'checks':{'recorded_actual_URDF_limits':max(violations.values())<1e-4,'recorded_source_hand_arm_table_clearance_positive':min(minimum.values())>0,'recorded_source_tool_table_clearance_positive':min(toolclear.values())>0},
 'scope':'Every recorded actual joint/body pose is checked at60Hz; contact forces against table were separately read every480Hz physics step. Source STL table-clearance extrema use convex-hull vertices of the exact source mesh, which preserve linear height extrema. This is not a full self-collision audit.'}
(a.run/'source_motion_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':report['checks'],'robot_min_m':min(minimum.values()),'tool_min_m':min(toolclear.values()),'max_relative_handle_axis_range_m':float(relative_range.max())},indent=2))
