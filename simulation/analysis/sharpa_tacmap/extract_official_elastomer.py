"""Read-only extraction of the upstream demonstration's actual sensor meshes."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,'/cpfs/user/lvzhaoyang/apps/isaac-sim-5.0.0/extscache/omni.usd.libs-1.0.1+8131b85d.lx64.r.cp311')
from pxr import Usd,UsdGeom,Gf
ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'assets/sharpa_tactile/assets/sharpawave/right_sharpa_wave.usda'
stage=Usd.Stage.Open(str(path));out={};rows=[]
for finger in ['thumb','index']:
    root=stage.GetPrimAtPath('/Right_Sharpa_Wave/right_'+finger+'_elastomer')
    root_inverse=UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim);relative=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())*root_inverse
        points=np.array([relative.Transform(Gf.Vec3d(*p))for p in mesh.GetPointsAttr().Get()])
        counts=mesh.GetFaceVertexCountsAttr().Get();indices=mesh.GetFaceVertexIndicesAttr().Get();faces=[];offset=0
        for n in counts:
            for i in range(1,n-1):faces.append([indices[offset],indices[offset+i],indices[offset+i+1]])
            offset+=n
        kind='collision' if '/collisions/' in str(prim.GetPath()) else 'visual'
        key=finger+'_'+kind;out[key+'_points']=points;out[key+'_faces']=np.array(faces)
        rows.append({'key':key,'usd_path':str(path),'prim_path':str(prim.GetPath()),'sensor_prim_path':str(root.GetPath()),'bounds_m':[points.min(0).tolist(),points.max(0).tolist()]})
np.savez(ROOT/'analysis/sharpa_tacmap/official_elastomer_meshes.npz',**out)
(ROOT/'analysis/sharpa_tacmap/official_elastomer_meshes.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
