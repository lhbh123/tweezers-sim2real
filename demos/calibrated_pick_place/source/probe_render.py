"""Native cameras for a measured physics diagnostic; no state replay or posing."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
class ProbeRenderer:
 def __init__(self,scene,config,out):
  from pxr import UsdLux,UsdGeom,UsdShade,Sdf,Gf
  from isaacsim.core.utils.extensions import enable_extension
  enable_extension('isaacsim.sensors.camera')
  from isaacsim.sensors.camera import Camera
  import imageio.v2 as imageio
  light=UsdLux.DomeLight.Define(scene.stage,'/World/ProbeDome');light.CreateIntensityAttr(850.)
  key=UsdLux.DistantLight.Define(scene.stage,'/World/ProbeKey');key.CreateIntensityAttr(1200.);UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-25,30,0))
  for name,color,metal in [('RLSteel',(.5,.55,.62),.8),('RLPlastic',(.95,.22,.03),0),('RLTable',(.16,.21,.27),.1)]:
   mat=UsdShade.Material.Get(scene.stage,'/World/Materials/'+name);shader=UsdShade.Shader.Define(scene.stage,str(mat.GetPath())+'/Surface');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metal);shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.35);mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
  for e in range(scene.num_envs):
   for suffix,name in [('/Table','RLTable'),('/Block/Shape','RLPlastic'),('/Tweezers/Handle/Shape','RLSteel'),('/Tweezers/Left/Blade','RLSteel'),('/Tweezers/Right/Blade','RLSteel'),('/Tweezers/Left/Tip','RLSteel'),('/Tweezers/Right/Tip','RLSteel')]:
    prim=scene.stage.GetPrimAtPath(scene.env_paths[e]+suffix)
    if prim:UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(scene.stage,'/World/Materials/'+name))
  self.scene=scene;self.cameras=[];self.writers=[];self.follow=[];self.motion=[];self.out=out;manifest=[]
  for v in json.load(open(config)):
   self.follow.append(v if v.get('follow_tip') else None)
   origin=scene.env_origins[v.get('env',0)].cpu().numpy();eye=np.array(v['eye'])+origin;target=np.array(v['target'])+origin;z=eye-target;z/=np.linalg.norm(z);up=[0,1,0] if abs(z[2])>.999 else [0,0,1];x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x);q=Rotation.from_matrix(np.column_stack([x,y,z])).as_quat();cam=Camera('/World/Probe_'+v['name'],resolution=(960,720),frequency=30);cam.set_world_pose(position=eye,orientation=np.r_[q[3],q[:3]],camera_axes='usd');cam.set_horizontal_aperture(.036);cam.set_focal_length(v.get('focal_length_m',.055));cam.set_clipping_range(.01,100);cam.initialize();self.cameras.append(cam);self.writers.append(imageio.get_writer(out/(v['name']+'.mp4'),fps=30,codec='libx264',quality=8));manifest.append({**v,'eye_world':eye.tolist(),'target_world':target.tolist(),'quaternion_wxyz':np.r_[q[3],q[:3]].tolist()})
  (out/'camera_manifest.json').write_text(json.dumps(manifest,indent=2))
  for _ in range(4):scene.sim.render()
 def frame(self):
  moving=[]
  for cam,config in zip(self.cameras,self.follow):
   if config is None:continue
   from isaaclab.utils.math import quat_apply
   import torch
   state=self.scene.get_state();e=config.get('env',0)
   point=torch.tensor([0.,0.,self.scene.tip_local_z],device=self.scene.device).expand(self.scene.num_envs,3)
   tips=[state[n+'_root_state_w'][:,:3]+quat_apply(state[n+'_root_state_w'][:,3:7],point) for n in ['left','right']]
   tip=((tips[0]+tips[1])*.5)[e].cpu().numpy()
   target=tip+np.array(config['target_offset']);eye=target+np.array(config['eye_offset'])
   z=eye-target;z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
   q=Rotation.from_matrix(np.column_stack([x,y,z])).as_quat();cam.set_world_pose(position=eye,orientation=np.r_[q[3],q[:3]],camera_axes='usd')
   moving.append(dict(name=config['name'],eye_world=eye.tolist(),target_world=target.tolist(),actual_tip_world=tip.tolist()))
  self.motion.append(moving)
  self.scene.sim.render()
  for cam,writer in zip(self.cameras,self.writers):
   rgba=cam.get_rgba()
   if rgba is None or rgba.size==0:raise RuntimeError('Empty native camera frame')
   writer.append_data(rgba[:,:,:3])
 def close(self):
  for writer in self.writers:writer.close()
  (self.out/'camera_motion.json').write_text(json.dumps(self.motion))
