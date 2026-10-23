"""Four simultaneous native boundary episodes, without repositioning objects."""
import argparse,json,hashlib
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.run/'evaluation.json').read_text());z=np.load(a.run/'trajectory.npz');d={k:z[k] for k in z.files};z.close()
assert r['episodes']==4 and all(r['success_per_env']), 'All four delivered boundary cases must have passed'
assert r['algorithm']=='goal_conditioned_residual_PPO_with_geometric_phase_controller'
fonts={}
def font(n):
 if n not in fonts:fonts[n]=ImageFont.truetype('assets/fonts/NotoSansCJKsc-Regular.otf',n)
 return fonts[n]
phases=['初始持镊','接近方块','下降对准','夹紧','提起','搬运','下放','松开','撤离','稳定留置']
readers=[imageio.get_reader(a.run/f'case{i}.mp4') for i in range(4)];frames=min(x.count_frames() for x in readers)
writer=imageio.get_writer(a.output,fps=30,codec='libx264',quality=8)
for f in range(frames):
 k=int(d['render_control_steps'][f]);time=(k+1)*float(d['dt']);im=Image.new('RGB',(1920,1728),'#f3f5f8');draw=ImageDraw.Draw(im)
 draw.text((30,15),'工作区边界测试：四组对角搬运，每组约 41 厘米',font=font(38),fill='#17243a')
 draw.text((30,65),f'四个仿真实例同时运行  |  仿真 {time:.2f}s  |  原速  |  绿色框为目标',font=font(25),fill='#006b72')
 for i,reader in enumerate(readers):
  x=(i%2)*960;y=(i//2)*800;s=d['start'][i];g=d['goal'][i];done=bool(d['success'][k,i])
  draw.text((x+20,y+110),f'{i+1}  起点 ({s[0]:.2f}, {s[1]:.2f}) → 目标 ({g[0]:.2f}, {g[1]:.2f}) m',font=font(24),fill='#17243a')
  im.paste(Image.fromarray(reader.get_data(f)).resize((940,705)),(x+20,y+145))
  label='已完成完整放置验收' if done else phases[int(d['phase'][k,i])]
  draw.text((x+20,y+855),f'{label}    水平误差 {d["goal_error"][k,i]*1000:.1f} mm',font=font(25),fill='#008d67' if done else '#536179')
 draw.text((30,1695),'分阶段运动控制 + PPO 修正；方块的夹起、搬运、落桌与释放均由仿真接触产生。',font=font(21),fill='#536179')
 writer.append_data(np.asarray(im))
 if f==frames-1:im.save(a.output.with_name(a.output.stem+'_cover.png'))
writer.close()
for x in readers:x.close()
check=imageio.get_reader(a.output);count=sum(1 for _ in check);check.close();assert count==frames
(a.output.with_suffix('.json')).write_text(json.dumps(dict(source_run=str(a.run),frames=frames,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),success_per_environment=r['success_per_env']),indent=2));print(a.output)
