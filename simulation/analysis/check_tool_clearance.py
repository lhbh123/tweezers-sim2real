import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from kinematics import *
from scipy.spatial import cKDTree
k=Kinematics(); data=json.loads(Path('analysis/natural_pinch_candidates.json').read_text())[0]
q=data['joints'];Ts=k.fk(q,root='right_hand_link')
c=np.array(data['center']);X=np.array(data['normal_separation']);Z=np.array([-.5,.245,-.831]);Z-=X*(Z@X);Z/=np.linalg.norm(Z)
Y=np.cross(Z,X);R=np.column_stack([X,Y,Z]);T=transform(c-.075*Z,R)
print('T',T)
for e in k.root.findall('link'):
 name=e.get('name')
 if name not in Ts:continue
 m=e.find('visual/geometry/mesh')
 if m is None:continue
 tri=stl_triangles(ASSET/m.get('filename'));v=tri.reshape(-1,3)@Ts[name][:3,:3].T+Ts[name][:3,3]
 vt=(v-T[:3,3])@R
 for sign in [-1,1]:
  # closest point on tapered rail centerline
  A=np.array([0.,0.,0.]);B=np.array([sign*(data['distance']/2-.0005)/.075*.14,0,.14]);d=B-A
  s=np.clip((vt-A)@d/(d@d),0,1);dist=np.linalg.norm(vt-A-s[:,None]*d,axis=1)
  idx=np.argmin(dist)
  if dist[idx]<.01: print(name,sign,'mindist',round(dist[idx]*1000,2),'toolz',round(s[idx]*140,1),'point',vt[idx].round(5))
data.update(T_hand_tool=T.tolist(),tool_axis_in_hand=Z.tolist(),tool_width_axis_in_hand=Y.tolist(),tool_pinch_z=.075)
Path('analysis/tool_grasp.json').write_text(json.dumps(data,indent=2))
