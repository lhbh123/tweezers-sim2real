"""Source-STL check at before/loose/recovered states of a native catch probe."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis')]
from audit_distal_geometry import Auditor
from tripod_tail_surface_gap import min_distance
from kinematics import transform
from validate_human_grasp import triangle_box_overlap
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=json.load(open(a.run/'report.json'));d=np.load(a.run/'timeseries.npz');A=Auditor();rows=[]
for ts in [.8,1.5,2.5,4.5]:
 k=int(np.abs(d['time']-ts).argmin());q=dict(zip(r['joint_names'],d['q'][k,0]));mesh,hull=A.geometry(q);H=A.kin.fk(q)['right_hand_link'];H[:3,3]+=np.array(r['env_origins'][0])+[0,0,.72];gaps=[]
 for j,side in [(1,'left'),(2,'right')]:
  st=d['bodies'][k,0,j];W=transform(st[:3],Rotation.from_quat(st[[4,5,6,3]]).as_matrix());M=np.linalg.inv(H)@W;tri=(mesh['right_hand_mid_link2']-M[:3,3])@M[:3,:3];lo=np.array([-.0005,-.007,0]);hi=np.array([.0005,.007,.165]);gap,pm,pt=min_distance(tri,lo,hi);inter=triangle_box_overlap(tri,(lo+hi)/2,(hi-lo)/2)
  gaps.append(dict(body=side,gap_m=float(gap),middle_minus_tool_closest_vertical_m=float((W[:3,:3]@(pm-pt))[2]),source_triangle_box_intersections=int(inter.sum()),middle_point_world=(W[:3,:3]@pm+W[:3,3]).tolist(),tool_point_world=(W[:3,:3]@pt+W[:3,3]).tolist()))
 rows.append(dict(time_s=float(d['time'][k]),middle_q_rad={n:q[n].item() for n in ['right_hand_mid_joint1','right_hand_mid_joint2']},self_collision=A.self_collision(mesh,hull),jaw_gaps=gaps,middle_under_count=int(d['middle_under_contact_count'][k,0]),middle_up_N=float(d['role_support_z_N'][k,0,2])))
(a.run/'middle_geometry_audit.json').write_text(json.dumps(dict(source='Original URDF collision STL, not PhysX cooked hulls',sampled_times_only=True,states=rows),indent=2));print(json.dumps(rows,indent=2))
