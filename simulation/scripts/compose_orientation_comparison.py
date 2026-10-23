#!/usr/bin/env python3
"""Compose actual top/side renders with recorded orientation, plus human reference.

No pixel re-painting, simulated pose changes, or human 3D angle estimation.
Use --diagnostic-existing-views only for layout checks with old oblique cameras;
such outputs are prominently marked and are never called top/side views.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compose_rl_video import frame_alignment, checkpoint_lineage

ROOT=Path(__file__).resolve().parents[1]
BG,INK,MUTED,LINE,ACCENT='#f3f5f8','#17243a','#526078','#d2dae6','#006b72'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value,digits=1):
    return f'{float(value):.{digits}f}' if value is not None and np.isfinite(value) else '未记录'


def handle_axis(state):
    q=np.asarray(state[3:7],dtype=float);q/=np.linalg.norm(q)
    w,x,y,z=q
    return np.array([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])


def down_angle(axis):
    axis=np.asarray(axis,dtype=float)
    return float(np.degrees(np.arctan2(-axis[2],np.hypot(*axis[:2]))))


def check_metrics(records,diagnostic):
    errors=[]
    for i,r in enumerate(records):
        required=['palm_normal_w','tool_axis_w','tool_down_angle_deg','root_to_tip_axis_w','root_to_tip_down_angle_deg']
        missing=[k for k in required if k not in r]
        if missing:
            if diagnostic: continue
            raise ValueError(f'Trajectory row {i} lacks orientation fields: {missing}')
        for key in ['palm_normal_w','tool_axis_w','root_to_tip_axis_w']:
            value=np.asarray(r[key])
            if value.shape!=(3,) or not np.isfinite(value).all() or abs(np.linalg.norm(value)-1)>.001:
                raise ValueError(f'Invalid unit orientation vector {key} at row {i}')
        expected=handle_axis(r['tool'])
        axis_err=float(np.max(np.abs(expected-r['tool_axis_w'])))
        angle_err=abs(down_angle(expected)-r['tool_down_angle_deg'])
        tip_angle_err=abs(down_angle(r['root_to_tip_axis_w'])-r['root_to_tip_down_angle_deg'])
        if axis_err>1e-4 or angle_err>1e-3 or tip_angle_err>1e-3:
            raise ValueError(f'Orientation fields disagree with source state at row {i}')
        errors.append([axis_err,angle_err,tip_angle_err])
    return dict(checked_records=len(errors),maximum_errors=np.max(errors,axis=0).tolist() if errors else None,
                error_fields=['tool_axis_vs_root_quaternion','tool_angle_deg_vs_root_quaternion','tip_angle_deg_vs_recorded_tip_axis'],
                palm_normal_check='Unit/finite check only; camera compositor does not independently calibrate the robot palm surface.',
                missing_metrics_allowed_only_for_diagnostic=diagnostic)


class Layout:
    def __init__(self,font,task,evaluation,diagnostic=False,orthographic_top=False,video_view_pair=('top','side')):
        self.font_path=str(font);self.fonts={};self.task=task;self.eval=evaluation;self.diagnostic=diagnostic
        top_name='正交俯视相机' if orthographic_top else '真实俯视相机（透视投影）'
        self.view_names=['旧斜视相机 A（非俯视）','旧斜视相机 B（非侧视）'] if diagnostic else [top_name,'真实水平侧视相机（透视投影）']
        labels={'top':top_name,'side':'真实水平侧视相机（透视投影）',
                'closeup':'真实斜近景相机（透视投影）','overview':'真实斜视全景相机（透视投影）'}
        self.video_view_names=self.view_names if diagnostic else [labels[name] for name in video_view_pair]
        self.selected_success=(evaluation.get('episodes')==1 and evaluation.get('successes')==1 and not diagnostic)
    def font(self,size):
        if size not in self.fonts:self.fonts[size]=ImageFont.truetype(self.font_path,size)
        return self.fonts[size]
    def text(self,d,xy,s,size=24,color=INK):d.text(xy,s,font=self.font(size),fill=color)
    def fit(self,d,xy,s,width,size=24,color=INK):
        while size>12 and d.textlength(s,font=self.font(size))>width:size-=1
        self.text(d,xy,s,size,color)
    def photo(self,can,pixels,box):
        x,y,w,h=box;im=Image.fromarray(pixels) if isinstance(pixels,np.ndarray) else pixels.copy()
        im.thumbnail((w,h),Image.Resampling.LANCZOS)
        ImageDraw.Draw(can).rectangle((x,y,x+w,y+h),fill='#e7ebf1',outline=LINE)
        can.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
    def values(self,record):
        palm=record.get('palm_normal_w');palm_z=float(palm[2]) if palm is not None else None
        return [('仿真时间',fmt(record['time'],3)+' s'),('方块抬升',fmt(record.get('cube_lift_m',0)*1000)+' mm'),
                ('掌面法线 n_z',fmt(palm_z,3)),('手柄轴对桌面下倾角',fmt(record.get('tool_down_angle_deg'))+'°'),
                ('根部→双颚尖中点下倾角',fmt(record.get('root_to_tip_down_angle_deg'))+'°')]
    def physical_footer(self,r):
        fixture='固定夹具存在' if self.task.get('fixture_present') or r.get('fixture_active') else '任务期间无固定夹具'
        coll=self.task.get('robot_self_collisions_enabled')
        pairs=self.task.get('filtered_mechanical_pairs')
        n=len({tuple(sorted(Path(x).name for x in pair)) for pair in pairs}) if isinstance(pairs,list) else None
        collision='自碰撞开启' if coll is True else ('自碰撞关闭' if coll is False else '自碰撞未记录')
        if n is not None:collision+=f'（排除 {n} 组机械相邻 link 对）'
        reset='稳定持镊快照初始化' if self.task.get('reset_snapshot') else '预持镊初始化'
        forces=r.get('jaw_contact_N',[None,None])
        return f'{fixture} | {collision} | {reset} | 双颚接触力 {fmt(forces[0],3)} / {fmt(forces[1],3)} N'
    def video(self,a,b,r):
        can=Image.new('RGB',(1920,1080),BG);d=ImageDraw.Draw(can)
        title='格式预览：旧相机，不代表姿态已纠正' if self.diagnostic else ('PPO 持镊：掌面朝下 · 选取的成功回合' if self.selected_success else 'PPO 持镊：实际掌面朝向与工具倾角')
        self.text(d,(40,18),title,42)
        self.fit(d,(1310,30),'检查点 '+Path(str(self.eval.get('checkpoint','未记录'))).name,570,25)
        self.text(d,(40,85),'n_z < 0 表示掌面朝下；工具角度来自三维仿真状态，不从屏幕斜率推断。',27,MUTED)
        for image,x,name in [(a,40,self.video_view_names[0]),(b,968,self.video_view_names[1])]:
            self.text(d,(x,140),name,28);self.photo(can,image,(x,184,912,684))
        boxes=[(40,230),(288,270),(576,330),(924,440),(1382,498)]
        for (x,w),(label,value) in zip(boxes,self.values(r)):
            d.rounded_rectangle((x,888,x+w,986),radius=12,fill='white',outline=LINE)
            self.fit(d,(x+14,898),label,w-28,22,MUTED);self.fit(d,(x+14,935),value,w-28,34,ACCENT)
        self.fit(d,(40,1002),self.physical_footer(r),1840,22)
        note='所示为选取的成功回合，不代表总体成功率；原始画面仅缩放拼接，未改物理状态。' if self.selected_success else '原始相机画面仅缩放拼接；不改姿态、不重画接触。无真人三维真值，不据此宣称真人级自然。'
        self.fit(d,(40,1040),note,1840,22,MUTED)
        return np.asarray(can)
    def comparison(self,reference,a,b,r,ref_label,out):
        can=Image.new('RGB',(2704,1320),BG);d=ImageDraw.Draw(can)
        title='参考与旧相机：仅版式预览，未作姿态纠正' if self.diagnostic else '人类参考与机器人：用俯视、侧视检查持镊方向'
        self.text(d,(48,26),title,53)
        subtitle='参考为原始视频；仿真为选取的成功回合、同一时刻，不代表总体成功率；只作姿态对照，不是真人 3D 配准。' if self.selected_success else '参考为原始视频；仿真两幅画面来自同一回合、同一时刻。只作视角和姿态对照，不是真人 3D 配准。'
        self.fit(d,(48,104),subtitle,2608,30,MUTED)
        labels=[f'人类原始参考 | {ref_label}',self.view_names[0]+f' | 仿真 {r["time"]:.3f} s',self.view_names[1]+f' | 仿真 {r["time"]:.3f} s']
        for x,im,label in zip([48,928,1808],[reference,a,b],labels):
            self.fit(d,(x,170),label,848,30);self.photo(can,im,(x,225,848,636))
        for i,(label,value) in enumerate(self.values(r)):
            x=48+i*528;d.rounded_rectangle((x,895,x+496,1015),radius=15,fill='white',outline=LINE)
            self.fit(d,(x+20,911),label,456,27,MUTED);self.fit(d,(x+20,953),value,456,40,ACCENT)
        self.fit(d,(48,1040),self.physical_footer(r),2608,28)
        self.text(d,(48,1093),'参考 216–228 s 为近俯视、主要见手背／拇指侧；没有同步侧视，也没有可用的真实工具对桌面角度。',29,MUTED)
        self.text(d,(48,1141),'n_z 是机器人经几何标定的掌面外法线竖直分量；负值朝下，正值朝上。不是从真人图片拟合得到的数值。',29,MUTED)
        self.fit(d,(48,1190),'来源：PennMicro / University of Pennsylvania · Holding the instruments · KXDHa39Z-2s | 检查点 '+str(self.eval.get('checkpoint')),2608,26,MUTED)
        self.text(d,(48,1240),'真人图未裁切、未镜像；仿真图未改物理状态；三幅图仅按各自纵横比缩放，空白处为版面留边。',27,MUTED)
        can.save(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--output-prefix',default='orientation_comparison')
    p.add_argument('--font',type=Path,default=ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf')
    p.add_argument('--reference-image',type=Path)
    p.add_argument('--static-frame',type=int,default=0,help='Zero-based source video frame for static comparison')
    p.add_argument('--frame-times-json',type=Path)
    p.add_argument('--static-only',action='store_true')
    p.add_argument('--view-pair',nargs=2,choices=['top','side','closeup','overview'],default=['top','side'],
                   help='Actual camera pair for VIDEO only; static reference comparison always uses top and side')
    p.add_argument('--diagnostic-existing-views',action='store_true')
    p.add_argument('--max-mb',type=float,default=24.)
    args=p.parse_args();args.run=args.run.resolve()
    if Path(args.output_prefix).name!=args.output_prefix:p.error('output-prefix must be a basename')
    out=args.output_dir or args.run/'orientation_delivery';out.mkdir(parents=True,exist_ok=True)
    task=read(args.run/'task.json');evaluation=read(args.run/'evaluation.json');records=read(args.run/'trajectory.json')
    launch=read(args.run/'launch.json') if (args.run/'launch.json').exists() else {}
    orthographic_top=bool(launch.get('arguments',{}).get('orthographic_top')) and not args.diagnostic_existing_views
    checks=check_metrics(records,args.diagnostic_existing_views)
    evidence=read(ROOT/'references/human/landmarks.json')
    reference_record=min([f for f in evidence['frames'] if f.get('usable_for_retargeting') and f.get('hand_role')=='tweezer_hand'],key=lambda f:abs(f['time_s']-222))
    reference_path=args.reference_image or ROOT/reference_record['frame_path']
    reference=Image.open(reference_path).convert('RGB')
    ref_label=f'{reference_record["time_s"]:.3f} s / frame {reference_record["frame_index"]}' if args.reference_image is None else '自定义原图（时间未自动核验）'
    video_names=['overview','closeup'] if args.diagnostic_existing_views else args.view_pair
    static_names=['overview','closeup'] if args.diagnostic_existing_views else ['top','side']
    if len(set(video_names))!=2:p.error('view-pair must name two different cameras')
    names=list(dict.fromkeys(static_names+([] if args.static_only else video_names)))
    paths=[args.run/(name+'.mp4') for name in names]
    readers=[imageio.get_reader(path) for path in paths]
    cameras=dict(zip(names,readers))
    try:
        counts=[r.count_frames() for r in readers];rates=[float(r.get_meta_data()['fps']) for r in readers]
        if len(set(counts))!=1 or min(counts)<1 or max(rates)-min(rates)>1e-6:raise ValueError('Camera video counts/rates disagree')
        count,fps=counts[0],rates[0]
        indices,times,alignment=frame_alignment(args,task,records,count)
        if not 0<=args.static_frame<count:p.error('static-frame outside source video')
        if count>1 and not np.allclose(np.diff(times),1/fps,atol=1e-5):raise ValueError('Source timing disagrees with raw fps')
        layout=Layout(args.font,task,evaluation,args.diagnostic_existing_views,orthographic_top,video_names)
        still=out/(args.output_prefix+'.png');video=out/(args.output_prefix+'.mp4')
        cover=out/(args.output_prefix+'_cover.png')
        i=args.static_frame
        layout.comparison(reference,cameras[static_names[0]].get_data(i),cameras[static_names[1]].get_data(i),records[indices[i]],ref_label,still)
        if not args.static_only:
            Image.fromarray(layout.video(cameras[video_names[0]].get_data(i),cameras[video_names[1]].get_data(i),records[indices[i]])).save(cover)
            bitrate=min(8_000_000,int(args.max_mb*1e6*8*.82/(count/fps)))
            if bitrate<300000:raise ValueError('Video size budget too small')
            writer=imageio.get_writer(video,fps=fps,codec='libx264',quality=None,bitrate=str(bitrate),macro_block_size=1,pixelformat='yuv420p',ffmpeg_params=['-preset','medium','-movflags','+faststart','-threads','4'])
            try:
                for i,j in enumerate(indices):
                    writer.append_data(layout.video(cameras[video_names[0]].get_data(i),cameras[video_names[1]].get_data(i),records[j]))
            finally:writer.close()
            if video.stat().st_size>=args.max_mb*1e6:raise RuntimeError('Video exceeds size cap')
        checkpoint=Path(evaluation['checkpoint']) if evaluation.get('checkpoint') not in [None,'zero'] else None
        if checkpoint is not None and not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
        report=dict(run=str(args.run),source_trajectory_sha256=sha(args.run/'trajectory.json'),source_task_sha256=sha(args.run/'task.json'),source_evaluation_sha256=sha(args.run/'evaluation.json'),source_camera_videos=[dict(name=n,path=str(path),sha256=sha(path)) for n,path in zip(names,paths)],source_reference=dict(path=str(reference_path),sha256=sha(reference_path),label=ref_label,landmark_record_used_for_reference_selection=args.reference_image is None,video_sha256=evidence['video_sha256'],source_url=evidence['source_url']),diagnostic_only=args.diagnostic_existing_views,orientation_metrics_checks=checks,frame_alignment=alignment,source_frame_indices=indices,source_frame_times_s=times.tolist(),static_frame_index=args.static_frame,static_sim_time_s=float(times[args.static_frame]),static_image=dict(path=str(still),sha256=sha(still)),video=None if args.static_only else dict(path=str(video),sha256=sha(video),frames=count,fps=fps,duration_s=count/fps,bytes=video.stat().st_size),checkpoint=evaluation.get('checkpoint'),checkpoint_sha256_computed_at_composition=sha(checkpoint) if checkpoint and checkpoint.exists() else None,checkpoint_training_lineage=checkpoint_lineage(checkpoint),reset_snapshot=task.get('reset_snapshot'),reset_snapshot_recorded_sha256=task.get('reset_snapshot_sha256'),physics=task,operations=['aspect-preserving image resize','letterbox/paste','outside-image HUD'],human_image_crop=False,human_image_mirror=False,physics_state_edited=False,claims_pose_corrected=False,limitations=['Human source camera is uncalibrated and not synchronized with a side camera.','Reference angle is not measured; robot angles come from actual telemetry.','Camera directions do not make the rendered images orthographic.','Displayed orientation does not independently establish humanlike motion or successful physics.'])
        report['static_image']['view_pair']=static_names
        report['static_image']['view_labels']=layout.view_names
        report['selected_success_episode']=layout.selected_success
        report['selected_episode_is_not_independent_success_rate']=layout.selected_success
        if report['video'] is not None:
            report['video']['view_pair']=video_names
            report['video']['view_labels']=layout.video_view_names
            report['video']['cover']={'path':str(cover),'sha256':sha(cover)}
        report['camera_projection']={'top':'orthographic' if orthographic_top else 'perspective','side':'perspective','closeup':'perspective','overview':'perspective','source':'launch.arguments.orthographic_top','diagnostic_existing_views':args.diagnostic_existing_views}
        report['limitations'][2]='Top camera is orthographic; side camera is perspective. The human source remains uncalibrated.' if orthographic_top else 'Camera directions do not make the rendered images orthographic.'
        manifest=out/(args.output_prefix+'_manifest.json');manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(image=str(still),video=None if args.static_only else str(video),manifest=str(manifest)),ensure_ascii=False,indent=2))
    finally:
        for reader in readers:reader.close()


if __name__=='__main__':main()
