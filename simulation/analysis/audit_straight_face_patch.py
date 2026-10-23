"""Source-finger penetration into each finite outer wide-face half-space.

Clips original source triangles to the blade Y/Z footprint, then computes exact
linear extrema in X. This is a source-surface depth, not PhysX contact depth.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis')]
from audit_distal_geometry import Auditor
from export_straight_face_pinch import bodies,boxes
from tripod_tail_surface_gap import min_distance
from validate_human_grasp import triangle_box_overlap

def clipped_triangle_vertices(tri,width,length):
    broad=np.all((tri.max(1)[:,[1,2]]>=np.array([-width/2,0]))&(tri.min(1)[:,[1,2]]<=np.array([width/2,length])),axis=1)
    vertices=[]
    for t in tri[broad]:
        poly=list(t)
        for axis,sign,bound in [(1,1,width/2),(1,-1,width/2),(2,1,length),(2,-1,0)]:
            new=[]
            if not poly:break
            for a,b in zip(poly,[*poly[1:],poly[0]]):
                da=sign*a[axis]-bound;db=sign*b[axis]-bound
                if da<=0:new.append(a)
                if (da<=0)!=(db<=0):new.append(a+(b-a)*(da/(da-db)))
            poly=new
        vertices.extend(poly)
    return np.asarray(vertices)

def main(args):
    a=Auditor();cfg=json.loads(args.config.read_text());T=np.asarray(cfg['T_hand_tool']);geo=cfg['tool_geometry'];Rh=np.asarray(cfg['preferred_hand_world_R']);reports=[]
    for label,q,angle in [('open',cfg['q_reference'],geo['open_angle_rad']),('closed',cfg['q_closed'],cfg['geometry_validation']['closed_assumed_jaw_angle_rad'])]:
        mesh,_=a.geometry(q);B=bodies(T,angle);pairs=[]
        for side,sgn,role,link in [('right',1,'thumb','right_hand_thumb_rota_link2'),('left',-1,'index','right_hand_index_rota_link2')]:
            M=B[side];tri=(mesh[link]-M[:3,3])@M[:3,:3];v=clipped_triangle_vertices(tri,geo['strip_width_m'],geo['length_m']-.015)
            depths=.5*geo['strip_thickness_m']-sgn*v[:,0];i=int(depths.argmax());contact=v[i].copy();contact[0]=sgn*.5*geo['strip_thickness_m']
            pairs.append(dict(role=role,jaw=side,maximum_source_halfspace_penetration_m=float(depths[i]),source_point_jaw_local_m=v[i].tolist(),projected_face_point_jaw_local_m=contact.tolist(),width_edge_margin_m=float(geo['strip_width_m']/2-abs(contact[1])),axial_edge_margin_m=float(min(contact[2],geo['length_m']-.015-contact[2])),method='Triangle polygons clipped to finite Y/Z wide-face footprint; extremal signed X distance. Not a volume-penetration measure.'))
        backup=[]
        for link in ['right_hand_mid_link1','right_hand_mid_link2']:
            for body,part,c,h in boxes(geo['length_m'],geo['strip_width_m'],geo['handle_length_m']):
                M=B[body];tri=(mesh[link]-M[:3,3])@M[:3,:3];hits=int(triangle_box_overlap(tri,c,h).sum());gap,pf,pt=min_distance(tri,c-h,c+h);pfh=M[:3,:3]@pf+M[:3,3];pth=M[:3,:3]@pt+M[:3,3]
                backup.append(dict(link=link,tool_body=body,part=part,source_gap_m=0. if hits else float(gap),intersection_triangles=hits,nearest_finger_point_hand_m=pfh.tolist(),nearest_tool_point_hand_m=pth.tolist(),finger_minus_tool_world_z_m=float((Rh@(pfh-pth))[2])))
        reports.append(dict(state=label,angle_rad=angle,face_patches=pairs,closest_middle_backup=min(backup,key=lambda x:x['source_gap_m']),middle_backup_all_parts=backup))
    output=args.output or ROOT/'analysis'/f'{args.config.stem}_patch_audit.json'
    output.write_text(json.dumps({'config':str(args.config),'source':'Original URDF/STL surfaces and nominal kinematic blade transforms; no dynamics claim','states':reports},indent=2)+'\n')
    print(json.dumps({'output':str(output),'states':[{k:s[k] for k in ['state','face_patches','closest_middle_backup']} for s in reports]},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('--output',type=Path);main(p.parse_args())
