"""Native two-environment catch/control comparison with measured telemetry."""
import argparse,json,hashlib
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.load(open(a.run/'report.json'));d=np.load(a.run/'timeseries.npz');cams=json.load(open(a.run/'camera_manifest.json'));summary=json.load(open(a.run/'catch_analysis.json'));fonts={}
def font(n):
 if n not in fonts:fonts[n]=ImageFont.truetype('assets/fonts/NotoSansCJKsc-Regular.otf',n)
 return fonts[n]
assert len(cams)==2 and not r['runtime_object_state_writes']
readers=[imageio.get_reader(a.run/(c['name']+'.mp4')) for c in cams];frames=min(v.count_frames() for v in readers);stride=round(1/(30*r['physics_dt_s']));writer=imageio.get_writer(a.output,fps=30,codec='libx264',quality=8);middle=d['role_support_z_N'][:,:,2];normal=d['role_normal_force_w'][:,:,2,2];t=d['time'];base=(t>.8)&(t<=1)
for f in range(frames):
 k=f*stride;time=float(t[k]);canvas=Image.new('RGB',(1920,1120),'#f3f5f8');draw=ImageDraw.Draw(canvas)
 draw.text((35,15),'中指下方备用承托：同样松开双指的对照测试',font=font(39),fill='#17243a')
 stage='正常夹持，中指留有间隙' if time<=1 else ('逐步减小双指夹力' if time<1.5 else ('保持松动，观察承托' if time<=3 else ('恢复双指夹力' if time<3.5 else '恢复后保持')))
 draw.text((35,68),f'{stage}  |  仿真 {time:.2f}s  |  0.5×慢放',font=font(24),fill='#006b72')
 for v,(c,reader) in enumerate(zip(cams,readers)):
  x=20+v*960;env=c.get('env',0);pixels=reader.get_data(f);canvas.paste(Image.fromarray(pixels).resize((940,705)),(x,140));label='中指在镊子下方' if v==0 else '对照：中指收回';draw.text((x,106),label,font=font(24),fill='#17243a');force=float(middle[k,env]);under=bool(d['middle_under_contact_count'][k,env]>0)
  status='正在向上承托' if under and force>.03 else '当前没有向上承托';color='#006b72' if under and force>.03 else '#536179'
  draw.text((x+20,855),status,font=font(32),fill=color)
  draw.text((x+20,910),f'中指向上净接触力  {force:.3f} N',font=font(29),fill=color)
  drop=float((d['tip_center_w'][base,env,2].mean()-d['tip_center_w'][k,env,2])*1000)
  draw.text((x+20,960),f'镊尖相对初始下降  {drop:.1f} mm',font=font(29),fill='#17243a')
  draw.text((x+20,1008),f'仅改变手指关节目标；中指没有粘接、工具没有外加托力。',font=font(22),fill='#536179')
 draw.text((35,1070),'这是预设松动扰动的物理诊断；夹方块的 PPO 策略另见配套视频。两侧相机方向相同。',font=font(25),fill='#536179')
 for _ in range(2):writer.append_data(np.asarray(canvas))
 if f==min(75,frames-1):canvas.save(a.output.with_name(a.output.stem+'_cover.png'))
writer.close()
for v in readers:v.close()
# Decode every output frame to verify successful encoding.
check=imageio.get_reader(a.output);count=sum(1 for _ in check);check.close();assert count==frames*2
manifest=dict(source_run=str(a.run),native_cameras=cams,source_frames=frames,output_frames=count,physics_frame_stride=stride,playback_speed=.5,scripted_diagnostic_not_RL=True,raw_state_writes=False,output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),analysis=summary)
a.output.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(a.output)
