"""Project actual recorded contacts onto a validated original side-camera frame.

No contacts are selected from pixels. Camera projection is first checked against
the independently recorded cube pose and original pixels; failed views are refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from kinematics import Kinematics,ASSET,transform,xyz,stl_triangles

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rotation(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def camera_from_usd(run,name):
 source=run/'final_scene.usda';text=source.read_text();block=text.split('def Camera "RL_'+name+'"',1)[1].split('\n    }',1)[0]
 scalar=lambda key:float(re.search(r'float '+key+r' = ([0-9.eE+\-]+)',block).group(1))
 vec=lambda key:np.fromstring(re.search(re.escape(key)+r' = \(([^)]*)\)',block).group(1),sep=',')
 return dict(eye=vec('double3 xformOp:translate'),R=rotation(vec('quatd xformOp:orient')),fx=960*scalar('focalLength')/scalar('horizontalAperture'),fy=720*scalar('focalLength')/scalar('verticalAperture'),source=str(source),source_sha256=sha(source))
def project(points,c):
 v=(np.asarray(points)-c['eye'])@c['R']
 if np.any(v[:,2]>=0):raise ValueError('Projected point behind camera')
 return np.c_[480+c['fx']*v[:,0]/-v[:,2],360-c['fy']*v[:,1]/-v[:,2]]
def orange_bounds(im):
 r,g,b=np.asarray(im).astype(float).transpose(2,0,1);mask=(r>140)&(r-g>25)&(r-b>70)&(g>b*1.15);y,x=np.nonzero(mask)
 if len(x)<20:raise ValueError('Insufficient orange cube pixels for camera check')
 return np.array([x.min(),y.min(),x.max(),y.max()])
def arrow(d,start,end,color,width=5):
 d.line([start,end],fill=color,width=width);a=np.asarray(start,dtype=float);b=np.asarray(end,dtype=float);v=a-b;v/=np.linalg.norm(v);per=np.array([-v[1],v[0]])
 d.polygon([tuple(b),tuple(b+15*v+6*per),tuple(b+15*v-6*per)],fill=color)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'runs/rl_tripod_neartable_hold_render07');p.add_argument('--video-frame',type=int,default=60);p.add_argument('--output',type=Path,default=ROOT/'outputs/ur5e_xhand_natural_tripod_contact_points.png');args=p.parse_args()
 run=args.run;records=json.loads((run/'trajectory.json').read_text());task=json.loads((run/'task.json').read_text());row=args.video_frame*2;r=records[row];impath=run/'visual_review'/f'side_{args.video_frame:04d}.png';raw=Image.open(impath).convert('RGB');camera=camera_from_usd(run,'side')
 # Verify archived capture cadence, since source frame0 is post-control-step0.
 if 'step % max(1, round(1/(30*env.dt))) == 0' not in (run/'train_rl.py').read_text():raise ValueError('Unknown capture cadence')
 if not np.isclose(task['physics_dt']*task['decimation'],1/60):raise ValueError('Expected60Hz source control cadence')
 cube=np.asarray(r['cube']);corners=np.array([[x,y,z] for x in [-.006,.006] for y in [-.006,.006] for z in [-.006,.006]])@rotation(cube[3:7]).T+cube[:3]
 cube_uv=project(corners,camera);expected=np.r_[cube_uv.min(0),cube_uv.max(0)];observed=orange_bounds(raw);error=observed-expected
 if max(abs(error))>3:raise ValueError(f'Actual pixel projection mismatch: {error}; annotation refused')
 # Select direct force contacts from this final physics substep, not averages or image guesses.
 rawcontacts=r['grasp_contact_points']['normal_contacts'];roles=['thumb','index','middle','web'];selected=[]
 for role in roles:
  candidates=[c for c in rawcontacts if (c['role']==role if role!='web' else c['tool']=='handle' and c['within_web_region']) and np.linalg.norm(c['force_on_tool_N'])>1e-6]
  if not candidates:raise ValueError(f'No contemporaneous {role} contact in selected physical substep')
  contact=max(candidates,key=lambda c:np.linalg.norm(c['force_on_tool_N']));selected.append(dict(label=role,**contact))
 uv=project([c['point_world'] for c in selected],camera)
 for contact,point in zip(selected,uv):contact['pixel_xy']=point.tolist();contact['visibility']='Projection of recorded collider contact; not visibility/ray tested; contact surface may be occluded.'
 font=lambda n:ImageFont.truetype(str(ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'),n)
 canvas=Image.new('RGB',(2400,1450),'#f4f6f9');d=ImageDraw.Draw(canvas)
 d.text((40,20),'真实接触点：由仿真坐标投影到原始侧视图',font=font(46),fill='#17243a')
 d.text((40,85),f'自然握持验证 · 零动作控制 · 仿真 {r["time"]:.6f} s；未夹起方块。',font=font(31),fill='#986211')
 d.text((40,145),'未标注原始侧视帧',font=font(31),fill='#17243a');d.text((1220,145),'同帧局部放大 + 日志接触点投影',font=font(31),fill='#17243a')
 canvas.paste(raw.resize((1140,855),Image.Resampling.LANCZOS),(40,195))
 crop=(270,380,675,690);detail=raw.crop(crop);detail.thumbnail((1140,855),Image.Resampling.LANCZOS)
 # Enlarge explicitly; maintain aspect ratio. No image reconstruction or inpainting.
 scale=min(1140/(crop[2]-crop[0]),855/(crop[3]-crop[1]));size=(round((crop[2]-crop[0])*scale),round((crop[3]-crop[1])*scale));detail=raw.crop(crop).resize(size,Image.Resampling.LANCZOS);offset=(1220+(1140-size[0])//2,195+(855-size[1])//2);canvas.paste(detail,offset)
 colors={'thumb':'#b22d52','index':'#2366a9','middle':'#ad741c','web':'#7753af'}
 texts={'thumb':'拇指','index':'食指','middle':'中指下托接触','web':'镊尾—虎口区域'}
 # Label locations are layout choices; all arrow endpoints are projected source coordinates.
 labelpos={'thumb':(2070,430),'index':(1270,500),'middle':(2050,790),'web':(1270,220)}
 leaderstart={'thumb':(2065,469),'index':(1390,548),'middle':(2050,811),'web':(1550,257)}
 for contact in selected:
  role=contact['label'];point=(np.asarray(contact['pixel_xy'])-np.array(crop[:2]))*scale+np.array(offset);color=colors[role]
  arrow(d,leaderstart[role],tuple(point),color);d.ellipse((point[0]-9,point[1]-9,point[0]+9,point[1]+9),outline=color,width=4)
  x,y=labelpos[role];text=texts[role];box=d.textbbox((x,y),text,font=font(31));d.rounded_rectangle((box[0]-8,box[1]-5,box[2]+8,box[3]+5),6,fill='#f4f6f9');d.text((x,y),text,font=font(31),fill=color)
 lines=['圆圈是同一物理子步的真实碰撞接触点，不是根据画面猜测；接触面可能被手指或工具遮挡。',f'投影直接使用存档侧视相机参数；已用真实方块姿态核对图像边界，最大误差 {max(abs(error)):.2f} 像素。','右图只对同一帧做裁切、缩放和标注；没有重画手、镊子或方块。','箭头标接触位置，不表示接触力方向。净向上支撑另见 16.7 ms 控制周期均值日志。','本帧有四处角色接触，不代表每个物理子步都持续四点接触，也不代表完整夹块任务成功。']
 for i,line in enumerate(lines):d.text((40,1090+i*62),line,font=font(28),fill='#536179')
 args.output.parent.mkdir(exist_ok=True,parents=True);canvas.save(args.output)
 # Independent visual projection diagnostic: source hand silhouette over original frame.
 kin=Kinematics();q=dict(zip(task['joint_names'],r['joint_pos']));fk=kin.fk(q);mask=Image.new('L',raw.size);md=ImageDraw.Draw(mask)
 for link in kin.root.findall('link'):
  name=link.get('name')
  if not name.startswith('right_hand'):continue
  for visual in link.findall('visual'):
   mesh=visual.find('geometry/mesh')
   if mesh is None:continue
   tri=stl_triangles(ASSET/mesh.get('filename'));origin=visual.find('origin');T=fk[name]@transform(xyz(origin.get('xyz')),Rotation.from_euler('xyz',xyz(origin.get('rpy'))).as_matrix());T[:3,3]+=[0,0,.72]
   if mesh.get('scale'):tri*=xyz(mesh.get('scale'))
   points=tri@T[:3,:3].T+T[:3,3];projected=project(points.reshape(-1,3),camera).reshape(-1,3,2)
   for t in projected:md.polygon([tuple(v) for v in t],fill=255)
 edge=np.array(mask)-np.array(mask.filter(ImageFilter.MinFilter(3)));overlay=np.array(raw).copy();overlay[edge>0]=[0,190,90];debug=run/'visual_review/source_hand_projection_check.png';Image.fromarray(overlay).save(debug)
 report=dict(run=str(run),source_trajectory_sha256=sha(run/'trajectory.json'),source_raw_frame=str(impath),source_raw_frame_sha256=sha(impath),video_frame=args.video_frame,trajectory_row=row,physics_time_s=r['time'],contact_time='final physics substep of same post-step camera frame; no temporal mixing',camera=dict(name='side',fx=camera['fx'],fy=camera['fy'],eye=camera['eye'].tolist(),R=camera['R'].tolist(),source=camera['source'],source_sha256=camera['source_sha256']),projection_check=dict(method='Known12mm cube corners transformed with actual recorded cube pose versus orange source pixels; no fitted calibration parameters.',projected_bbox=expected.tolist(),actual_color_bbox=observed.tolist(),edge_errors_px=error.tolist(),maximum_abs_edge_error_px=float(max(abs(error))),accepted=True),contacts=selected,detail_crop_source_xyxy=list(crop),detail_scale=scale,source_hand_overlay_diagnostic=str(debug),source_hand_overlay_diagnostic_sha256=sha(debug),output=str(args.output),output_sha256=sha(args.output),operations=['raw-side-frame copy','explicit same-frame crop+resize','projected-contact markers','text and leaders'],contact_positions_from_pixels=False,physics_state_changed=False,source_photos_repainted=False,external_send_performed=False,limitations=['Color mask/rasterization boundary test is pixel-level validation, not exact camera recalibration.','Source hand outline diagnostic requires separate visual inspection; cube edge validation alone does not certify all projected surfaces.','Contact points belong to the PhysX collision model; it can differ slightly from original visual STL surfaces.','Projected contact points may be occluded in the rendered surface.'])
 out=args.output.with_suffix('.json');out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(dict(image=str(args.output),audit=str(out),maximum_projection_edge_error_px=float(max(abs(error)))),ensure_ascii=False))

if __name__=='__main__':main()
