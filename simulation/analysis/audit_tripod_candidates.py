from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis')]
from kinematics import transform
from audit_distal_geometry import Auditor
from validate_human_grasp import triangle_box_overlap

def tool_audit(a,c,length=.16):
 meshes,hulls=a.geometry(c['q']);T=np.array(c['T']);width=c['width'];bodies={'handle':T,'left':T@transform([-.0045,0,0],Rotation.from_euler('y',-.05).as_matrix()),'right':T@transform([.0045,0,0],Rotation.from_euler('y',.05).as_matrix())}
 boxes=[('handle','bridge',[0,0,-.003],[.006,width/2,.003])]
 for n in ['left','right']:boxes +=[(n,'blade',[0,0,(length-.015)/2],[.0005,width/2,(length-.015)/2]),(n,'tip',[0,0,length-.0075],[.0005,.0025,.0075])]
 hits=[]
 for body,part,center,half in boxes:
  b=bodies[body]
  for link,mesh in meshes.items():
   tri=(mesh-b[:3,3])@b[:3,:3];center=np.asarray(center);half=np.asarray(half)
   broad=np.all((tri.max(1)>=center-half)&(tri.min(1)<=center+half),axis=1)
   if not broad.any():continue
   overlap=triangle_box_overlap(tri[broad],center,half);count=int(overlap.sum())
   if count:
    vertices=tri[broad][overlap].reshape(-1,3);inside=np.all((vertices>center-half)&(vertices<center+half),axis=1)
    depth=np.min(half-np.abs(vertices[inside]-center),axis=1) if inside.any() else np.array([0.])
    hits.append(dict(link=link,body=body,part=part,triangles=count,max_inside_vertex_depth_m=float(depth.max()),intended=link in ['right_hand_thumb_rota_link2','right_hand_index_rota_link2','right_hand_mid_link2']))
 return dict(seed=c['seed'],self_collision=a.self_collision(meshes,hulls),tool_intersections=hits,no_noncontact_tool_intersections=all(x['intended'] for x in hits))
if __name__=='__main__':
 a=Auditor();c=json.loads((ROOT/'analysis/tripod_raw_candidates.json').read_text());out=[]
 for item in c[:20]:
  r=tool_audit(a,item);out.append(r);print(item['seed'],r,flush=True)
 (ROOT/'analysis/tripod_candidate_audit.json').write_text(json.dumps(out,indent=2)+'\n')
