"""Exact feature distances between visible bridge box and source palm triangles."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis')]
from audit_distal_geometry import Auditor
from validate_human_grasp import triangle_box_overlap

def min_distance(tri,lo,hi):
 pts=tri.reshape(-1,3);box=np.clip(pts,lo,hi);dd=((pts-box)**2).sum(1);j=dd.argmin();best=(float(dd[j]),pts[j],box[j])
 def add(pa,pb):
  nonlocal best
  d=((pa-pb)**2).sum(1);i=d.argmin()
  if d[i]<best[0]:best=(float(d[i]),pa[i],pb[i])
 corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
 A=tri[:,0];U=tri[:,1]-A;V=tri[:,2]-A;n=np.cross(U,V);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
 uu=(U*U).sum(1);uv=(U*V).sum(1);vv=(V*V).sum(1);den=uu*vv-uv*uv
 for p in corners:
  pp=np.broadcast_to(p,A.shape);projected=pp-((pp-A)*n).sum(1)[:,None]*n;W=projected-A;wu=(W*U).sum(1);wv=(W*V).sum(1)
  a=(wu*vv-wv*uv)/np.maximum(den,1e-30);b=(wv*uu-wu*uv)/np.maximum(den,1e-30);valid=(a>=0)&(b>=0)&(a+b<=1)&(den>1e-24)
  if valid.any():add(projected[valid],pp[valid])
  for k in range(3):
   a=tri[:,k];d=tri[:,(k+1)%3]-a;s=np.clip(((pp-a)*d).sum(1)/np.maximum((d*d).sum(1),1e-30),0,1);add(a+s[:,None]*d,pp)
 edges=[(corners[i],corners[j]) for i in range(8) for j in range(i+1,8) if np.sum(corners[i]!=corners[j])==1]
 for k in range(3):
  a=tri[:,k];u=tri[:,(k+1)%3]-a;aa=(u*u).sum(1)
  for c,d in edges:
   v=d-c;w=a-c;bb=u@v;cc=v@v;dd=(u*w).sum(1);ee=w@v;den=aa*cc-bb*bb
   s=(bb*ee-cc*dd)/np.maximum(den,1e-30);t=(aa*ee-bb*dd)/np.maximum(den,1e-30);valid=(den>1e-24)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
   if valid.any():add(a[valid]+s[valid,None]*u[valid],c+t[valid,None]*v)
 return np.sqrt(best[0]),best[1],best[2]

if __name__=='__main__':
 cfg=json.loads((ROOT/'configs/human_grasp_tripod_seed57.json').read_text());a=Auditor();T=np.array(cfg['T_hand_tool']);Rh=np.array(cfg['preferred_hand_world_R']);web=np.array(cfg['grasp_topology']['web_anchor_local']);mesh=a.meshes['right_hand_link'];tri=(mesh-T[:3,3])@T[:3,:3];records=[]
 for width in [.012,.014,.016,.017,.018]:
  for length in [.006,.012,.016,.020,.025]:
   center=np.array([0,0,-length/2]);half=np.array([.006,width/2,length/2]);intersects=triangle_box_overlap(tri,center,half);gap,pm,pb=min_distance(tri,center-half,center+half);pmh=T[:3,:3]@pm+T[:3,3];pbh=T[:3,:3]@pb+T[:3,3]
   rec=dict(handle_width_m=width,handle_length_m=length,surface_distance_m=float(gap),source_palm_intersecting_triangles=int(intersects.sum()),closest_palm_point_hand=pmh.tolist(),closest_tail_point_hand=pbh.tolist(),tail_minus_palm_gap_vector_hand=(pbh-pmh).tolist(),tail_minus_palm_gap_vector_world=(Rh@(pbh-pmh)).tolist(),closest_palm_distance_to_web_anchor_m=float(np.linalg.norm(pmh-web)))
   records.append(rec);print(width,length,round(gap*1000,4),int(intersects.sum()),flush=True)
 (ROOT/'analysis/tripod_tail_surface_gap.json').write_text(json.dumps(dict(source_config='configs/human_grasp_tripod_seed57.json',method=__doc__,candidates=records),indent=2)+'\n')
