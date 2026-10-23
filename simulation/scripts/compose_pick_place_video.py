"""Compose native views plus measured tabletop trajectory; no scene animation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
r=json.loads((a.run/'evaluation.json').read_text());archive=np.load(a.run/'trajectory.npz');d={k:archive[k] for k in archive.files};archive.close();assert r['success_per_env'][0], 'Do not present failed episode as success'
assert r['algorithm']=='goal_conditioned_residual_PPO_with_geometric_phase_controller'
fontcache={}
def font(n):
 if n not in fontcache:fontcache[n]=ImageFont.truetype('assets/fonts/NotoSansCJKsc-Regular.otf',n)
 return fontcache[n]
phases=['初始持镊','接近方块','下降对准方块','双指挤压夹紧','提起方块','搬运到目标','下放至桌面','松开镊尖','镊尖撤离','方块稳定留置']
readers=[imageio.get_reader(a.run/(n+'.mp4')) for n in ['overview','task']];frames=min(x.count_frames() for x in readers)
writer=imageio.get_writer(a.output,fps=30,codec='libx264',quality=8)
start=d['start'][0,:2];goal=d['goal'][0,:2];origin=d['origins'][0];block=d['block'][:,0,:3]-origin
low=np.array(r['task']['place_config']['xy_lower']);high=np.array(r['task']['place_config']['xy_upper'])
for f in range(frames):
 k=int(d['render_control_steps'][f]) if 'render_control_steps' in d else min(2*f,len(block)-1);time=(k+1)*float(d['dt']);phase=int(d['phase'][k,0]);canvas=Image.new('RGB',(1920,1152),'#f3f5f8');draw=ImageDraw.Draw(canvas)
 draw.text((30,14),'UR5e + XHand：夹起 → 搬运 → 放下并松开',font=font(36),fill='#17243a')
 draw.text((30,67),f'边界对角测试 | 当前阶段：{phases[phase]}   |   仿真时间 {time:.2f}s   |   原速播放',font=font(25),fill='#006b72')
 for j,reader in enumerate(readers):canvas.paste(Image.fromarray(reader.get_data(f)).resize((940,705)),(20+j*960,110))
 def xy(v):return (50+(v[0]-low[0])*800,1090-(v[1]-low[1])*800)
 draw.rectangle((50,850,274,1090),outline='#9bacbe',width=2)
 trail=block[:k+1,:2]
 if len(trail)>1:draw.line([xy(v) for v in trail],fill='#e87924',width=3)
 for v,color,label in [(start,'#e87924','起点'),(goal,'#008d67','目标')]:
  x,y=xy(v);draw.ellipse((x-6,y-6,x+6,y+6),fill=color);draw.text((x+7,y-17),label,font=font(17),fill=color)
 x,y=xy(block[k,:2]);draw.ellipse((x-5,y-5,x+5,y+5),fill='#243755')
 draw.text((360,850),f'起点 XY：({start[0]:.3f}, {start[1]:.3f}) m     目标 XY：({goal[0]:.3f}, {goal[1]:.3f}) m',font=font(28),fill='#17243a')
 draw.text((360,902),f'搬运距离 {np.linalg.norm(goal-start)*100:.1f} cm    方块离桌高度 {d["lift"][k,0]*1000:.1f} mm',font=font(28),fill='#17243a')
 draw.text((360,954),f'到目标的水平误差 {d["goal_error"][k,0]*1000:.1f} mm    镊尖双侧接触：'+('有' if d['both'][k,0] else '无'),font=font(28),fill='#17243a')
 status='已通过：松开、撤离后方块稳定留置' if d['success'][k,0] else '等待完整放置验收'
 draw.text((360,1006),status,font=font(30),fill='#008d67' if d['success'][k,0] else '#536179')
 draw.text((360,1061),'控制：分阶段运动控制 + 目标条件 PPO 修正；工具和方块均由真实接触运动。',font=font(24),fill='#536179')
 writer.append_data(np.asarray(canvas))
 if f==min(frames-1,max(1,int(frames*.6))):canvas.save(a.output.with_name(a.output.stem+'_cover.png'))
writer.close()
for x in readers:x.close()
check=imageio.get_reader(a.output);count=sum(1 for _ in check);check.close();assert count==frames
(a.output.with_suffix('.json')).write_text(json.dumps(dict(source_run=str(a.run),source_frames=frames,output_frames=count,playback_speed=1,source_simulation_control_stride=2,output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest()),indent=2))
print(a.output)
