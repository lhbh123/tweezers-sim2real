"""Project recorded robot/object geometry through the actual saved USD cameras.

This checks image boundaries, not visibility through occluders. Hand meshes
come from the supplied original URDF. Tool/block boxes and cameras come from
the rendered run's USD, transformed by each recorded physical state.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from kinematics import ASSET,Kinematics,stl_triangles,transform,xyz

USD_SOURCE=r'''
import json,sys
import numpy as np
sys.path.insert(0,sys.argv[2])
from pxr import Usd,UsdGeom
s=Usd.Stage.Open(sys.argv[1]);d={"cameras":{},"objects":{}}
T=lambda p:np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
for name in ['overview','closeup','side','top']:
 p=s.GetPrimAtPath('/World/RL_'+name);c=UsdGeom.Camera(p)
 d['cameras'][name]={'world_T':T(p).tolist(),'projection':str(c.GetProjectionAttr().Get()),'horizontal_aperture':float(c.GetHorizontalApertureAttr().Get()),'vertical_aperture':float(c.GetVerticalApertureAttr().Get()),'focal_length':float(c.GetFocalLengthAttr().Get()),'clipping_range':list(c.GetClippingRangeAttr().Get())}
b='/World/envs/env_0'
d['robot_base_world_T']=T(s.GetPrimAtPath(b+'/Robot')).tolist()
for key,path in [('tool',b+'/Tweezers/Handle'),('left',b+'/Tweezers/Left'),('right',b+'/Tweezers/Right'),('cube',b+'/Block')]:
 p=s.GetPrimAtPath(path);root_inv=np.linalg.inv(T(p));boxes=[]
 for q in Usd.PrimRange(p):
  if q.IsA(UsdGeom.Cube):boxes.append({'size':float(UsdGeom.Cube(q).GetSizeAttr().Get()),'relative_T':(root_inv@T(q)).tolist()})
 d['objects'][key]=boxes
print(json.dumps(d))
'''


def transformed(points,T):
    return points@T[:3,:3].T+T[:3,3]


def state_T(state):
    q=np.asarray(state[3:7]);return transform(state[:3],Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix())


def project(points,camera):
    local=transformed(points,np.linalg.inv(np.array(camera['world_T'])))
    depth=-local[:,2]
    if camera['projection']=='orthographic':
        ndc=local[:,:2]*2/(np.array([camera['horizontal_aperture'],camera['vertical_aperture']])*.1)
    else:
        ndc=local[:,:2]/depth[:,None]*2*camera['focal_length']/np.array([camera['horizontal_aperture'],camera['vertical_aperture']])
    return np.c_[.5*(ndc[:,0]+1)*960,.5*(1-ndc[:,1])*720],depth


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--usd-python',default='/cpfs/user/lvzhaoyang/apps/bin/isaac-python');a=p.parse_args()
    sim=Path(os.environ.get('ISAAC_SIM_PATH','/cpfs/user/lvzhaoyang/apps/isaac-sim-5.0.0'))
    usd=next(iter(sorted((sim/'extscache').glob('omni.usd.libs-*'))))
    environment=os.environ.copy();environment['LD_LIBRARY_PATH']=str(usd/'bin')+':'+environment.get('LD_LIBRARY_PATH','')
    output=subprocess.check_output([str(sim/'python.sh'),'-c',USD_SOURCE,str(a.run/'final_scene.usda'),str(usd)],text=True,env=environment)
    scene=json.loads(next(line for line in output.splitlines() if line.startswith('{"cameras"')))
    rows=json.loads((a.run/'trajectory.json').read_text());task=json.loads((a.run/'task.json').read_text())
    k=Kinematics();meshes={};base=np.array(scene['robot_base_world_T'])
    for link in k.root.findall('link'):
        name=link.get('name')
        if not name.startswith('right_hand'):continue
        visual=link.find('visual');mesh=visual.find('geometry/mesh') if visual is not None else None
        if mesh is None:continue
        origin=visual.find('origin');T=transform(xyz(origin.get('xyz')),Rotation.from_euler('xyz',xyz(origin.get('rpy'))).as_matrix())
        vertices=stl_triangles(ASSET/mesh.get('filename')).reshape(-1,3)
        vertices=vertices[ConvexHull(vertices).vertices]
        if mesh.get('scale'):vertices*=xyz(mesh.get('scale'))
        meshes[name]=transformed(vertices,T)
    corners=np.array(list(itertools.product([-.5,.5],repeat=3)))
    local_objects={name:np.concatenate([transformed(corners*box['size'],np.array(box['relative_T'])) for box in boxes]) for name,boxes in scene['objects'].items()}
    totals={name:{key:dict(minimum_boundary_margin_px=1e9,frames_outside=[],extents_px=[1e9,1e9,-1e9,-1e9]) for key in ['hand','tool','cube']} for name in scene['cameras']}
    for i,row in enumerate(rows):
        if row.get('active') is False:continue
        fk=k.fk(dict(zip(task['joint_names'],row['joint_pos'])))
        hand=np.concatenate([transformed(v,base@fk[name]) for name,v in meshes.items()])
        objects={key:transformed(v,state_T(row[key])) for key,v in local_objects.items()}
        groups={'hand':hand,'tool':np.concatenate([objects[key] for key in ['tool','left','right']]),'cube':objects['cube']}
        for name,camera in scene['cameras'].items():
            for key,points in groups.items():
                uv,depth=project(points,camera);lo=uv.min(0);hi=uv.max(0);margin=float(min(lo[0],lo[1],960-hi[0],720-hi[1]))
                item=totals[name][key];item['minimum_boundary_margin_px']=min(item['minimum_boundary_margin_px'],margin)
                e=item['extents_px'];item['extents_px']=[float(min(e[0],lo[0])),float(min(e[1],lo[1])),float(max(e[2],hi[0])),float(max(e[3],hi[1]))]
                if margin<0 or depth.min()<=camera['clipping_range'][0]:item['frames_outside'].append(i)
    report={'method':'original hand STL convex-hull vertices through URDF FK; actual USD object boxes through recorded rigid states; saved USD camera projections',
            'run':str(a.run),'trajectory_sha256':hashlib.sha256((a.run/'trajectory.json').read_bytes()).hexdigest(),'usd_sha256':hashlib.sha256((a.run/'final_scene.usda').read_bytes()).hexdigest(),
            'records_checked':len(rows),'resolution':[960,720],'cameras':scene['cameras'],'framing':totals,
            'geometry_within_saved_usd_projection':all(not item['frames_outside'] for camera in totals.values() for item in camera.values()),
            'render_framing_accepted':None,
            'projection_model_validated_against_render':False,
            'limitations':['Saved USD camera parameters have disagreed with actual rendered framing in this project; a geometry-only pass is NOT a rendered-framing pass. Actual pixels require independent inspection.','Boundary containment does not establish freedom from occlusion.','Hand geometry is asset URDF/STL; tool/block geometry is actual saved USD primitive geometry.','Checks recorded frames, not unrecorded physics substeps.']}
    target=a.run/'delivery_review/camera_framing_audit.json';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report,indent=2));print(json.dumps({'path':str(target),'geometry_within_saved_usd_projection':report['geometry_within_saved_usd_projection'],'render_framing_accepted':None,'framing':totals},indent=2))


if __name__=='__main__':main()
