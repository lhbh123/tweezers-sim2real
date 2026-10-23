#!/usr/bin/env python3
"""Compose untouched human reference and synchronized actual tripod camera renders.

Image operations are only aspect-preserving resizing, letterboxing and HUD outside
camera images. Contact numbers are measured telemetry, never inferred from pixels.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from compose_rl_video import frame_alignment, checkpoint_lineage

ROOT=Path(__file__).resolve().parents[1]
BG,INK,MUTED,LINE,TEAL,AMBER='#f3f5f8','#17243a','#536179','#d2dae6','#006b72','#986211'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(v):return v is not None and np.isfinite(v)
def number(v,n=3):return f'{v:.{n}f}' if finite(v) else '未记录'


def selected_environment_evidence(evaluation):
    """Only environment 0 is present in these trajectory/camera recordings."""
    episodes=evaluation.get('episodes')
    flags=evaluation.get('success_per_env')
    if flags is not None:
        if type(episodes) is not int or episodes<1 or not isinstance(flags,list) or len(flags)!=episodes:
            raise ValueError('success_per_env must contain exactly episodes entries')
        if any(type(value) is not bool for value in flags):
            raise ValueError('success_per_env entries must be explicit JSON booleans')
        if evaluation.get('successes') is not None and evaluation['successes']!=sum(flags):
            raise ValueError('success_per_env disagrees with aggregate successes')
        success=flags[0];source='explicit success_per_env[0]'
    else:
        success=episodes==1 and evaluation.get('successes')==1
        source='legacy single-episode count' if episodes==1 else 'unknown: population count cannot identify recorded env0'
    return dict(recorded_environment_index=0,population_episodes=episodes,
                selected_success_episode=bool(success),success_evidence=source)


def controller_provenance(evaluation):
    checkpoint=evaluation.get('checkpoint')
    if checkpoint=='zero':return dict(label='控制：零动作',algorithm='constant_zero')
    cp=Path(checkpoint) if checkpoint else None
    if cp is not None and not cp.is_absolute():cp=ROOT/cp
    if cp is None:return dict(label='策略类型未记录',algorithm='unknown')
    launch=cp.parent/'launch.json';report=cp.parent/'report.json'
    metadata=read(report).get('metadata',{}) if report.exists() else {}
    if metadata.get('algorithm')=='behavior_cloning_initializer':
        return dict(label='离线 BC 策略（未作 PPO）',algorithm='behavior_cloning_initializer',metadata_file=str(report),metadata_sha256=sha(report))
    training=read(launch) if launch.exists() else {};initial=training.get('arguments',{}).get('initialize_policy')
    token=cp.stem.removeprefix('iteration_');iteration=int(token) if token.isdecimal() else None
    if initial:
        path=Path(initial);path=path if path.is_absolute() else ROOT/path
        return dict(label='BC 初始化 + PPO'+(f'（{iteration} 轮）' if iteration is not None else ''),
                    algorithm='PPO_with_recorded_policy_initialization',initialization_checkpoint=str(path),
                    initialization_sha256=sha(path) if path.exists() else None,
                    training_launch=str(launch),training_launch_sha256=sha(launch),selected_iteration=iteration)
    return dict(label='策略 / '+cp.name,algorithm='unspecified_policy',checkpoint=str(cp))


def temporal_scope(task):
    key=task.get('grasp_support_temporal_aggregation')
    mean=key=='mean_over_every_physics_substep_in_control_interval'
    dt=task.get('physics_dt');decimation=task.get('decimation')
    ms=1000*float(dt)*int(decimation) if dt is not None and decimation is not None else None
    return dict(recorded_value=key,is_control_interval_mean=mean,
                control_interval_ms=ms,
                label=(number(ms,1)+' ms 控制周期均值') if mean else '记录值（未标注时均）',
                caveat='Interval means do not establish uninterrupted contact at every physics substep.')


def web_source_scope(run):
    """Distinguish older all-tool ROI sums from newer Handle-only telemetry."""
    source=run/'grasp_contacts.py';scene=run/'rl_scene.py'
    if not source.exists():return dict(scope='unknown',label='虎口 ROI 支撑（范围未知）')
    text=source.read_text();tree=ast.parse(text)
    cls=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TripodContactReader'),None)
    if cls is None:return dict(scope='unknown',label='虎口 ROI 支撑（范围未知）')
    groups=next((ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='groups' for t in n.targets)),None)
    if groups!=['thumb','index','middle','wrist_palm','other']:raise ValueError('Archived contact group order unknown; cannot map thumb/index/middle safely')
    if 'self.web_tool_body' not in text and 'if with_web and self.web_region:' in text:
        return dict(scope='all_tool_bodies',label='全工具–虎口 ROI 支撑',source=str(source),sha256=sha(source))
    constructor=next((n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'),None)
    default=None
    if constructor:
        defaults=dict(zip([n.arg for n in constructor.args.args][-len(constructor.args.defaults):],constructor.args.defaults))
        if isinstance(defaults.get('web_tool_body'),ast.Constant):default=defaults['web_tool_body'].value
    calls=[n for n in ast.walk(ast.parse(scene.read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='TripodContactReader'] if scene.exists() else []
    if len(calls)==1:
        override=next((k.value for k in calls[0].keywords if k.arg=='web_tool_body'),None)
        if override is not None:default=override.value if isinstance(override,ast.Constant) else None
    else:default=None
    if default and 'self.web_tool_body' in text:
        scope=str(default).lower();return dict(scope=scope,label=scope.title()+'–虎口 ROI 支撑',source=str(source),sha256=sha(source),scene_sha256=sha(scene))
    return dict(scope='unknown',label='虎口 ROI 支撑（范围未知）',source=str(source),sha256=sha(source))


def contact_values(record):
    s=record.get('grasp_support',{})
    strength=s.get('role_contact_strength_N')
    normal=s.get('role_normal_force_w');friction=s.get('role_friction_force_w')
    support=s.get('role_support_z_N')
    for key,value,shape in [('strength',strength,(5,)),('normal',normal,(5,3)),('friction',friction,(5,3)),('support',support,(5,))]:
        if value is not None and np.asarray(value).shape!=shape:raise ValueError(f'Unexpected contact {key} shape')
    middle=float(support[2]) if support is not None else None
    if normal is not None and friction is not None:
        derived=(np.asarray(normal,dtype=float)+np.asarray(friction,dtype=float))[:,2]
        if support is not None and not np.allclose(derived,support,rtol=1e-4,atol=1e-6):
            raise ValueError('Recorded grouped support disagrees with normal+friction force')
        middle=float(derived[2])
    wn=s.get('web_normal_force_w');wf=s.get('web_friction_force_w')
    web=float(wn[2]+wf[2]) if wn is not None and wf is not None else s.get('web_support_z_N')
    if web is not None and s.get('web_support_z_N') is not None and abs(web-s['web_support_z_N'])>1e-6:
        raise ValueError('Recorded web support disagrees with normal+friction force')
    values=dict(strength=[float(v) for v in strength[:3]] if strength is not None else [None]*3,
                middle_up_N=middle,web_up_N=web,web_contact_count=s.get('web_contact_count'),
                middle_under_contact_count=s.get('middle_under_contact_count'))
    if any(v is not None and not np.isfinite(v) for v in values['strength']+[middle,web]):
        raise ValueError('Nonfinite contact telemetry must not be shown as healthy data')
    return values


def camera_label(name,manifest,launch):
    c=next((x for x in manifest if x['name']==name),None)
    base={'closeup':'近景','top':'top 相机','side':'侧相机'}[name]
    if c is None:return base+'（原始视角；方向未核验）'
    v=np.asarray(c['eye_world'])-np.asarray(c['target_world']);v/=np.linalg.norm(v)
    if abs(v[2])>.999:base='俯视' if v[2]>0 else '仰视'
    elif abs(v[2])<.005:base='水平侧视'
    else:base='斜近景' if name=='closeup' else '斜侧视'
    ortho='orthographic_width_m' in c.get('custom_settings',{}) or (name=='top' and launch.get('arguments',{}).get('orthographic_top'))
    return ('正交' if ortho else '透视')+base


class Layout:
    def __init__(self,font,task,evaluation,labels,reference,ref_label,web_scope,purpose='task'):
        self.font_path=str(font);self.fonts={};self.task=task;self.ev=evaluation
        self.labels=labels;self.reference=reference;self.ref_label=ref_label
        self.web_scope=web_scope
        self.temporal=temporal_scope(task)
        self.purpose=purpose
        self.selection=selected_environment_evidence(evaluation)
        self.selected_success=self.selection['selected_success_episode']
        self.controller=controller_provenance(evaluation)
    def font(self,size):
        if size not in self.fonts:self.fonts[size]=ImageFont.truetype(self.font_path,size)
        return self.fonts[size]
    def text(self,d,xy,t,size=24,color=INK,width=None):
        if width:
            while size>14 and d.textlength(t,font=self.font(size))>width:size-=1
        d.text(xy,t,font=self.font(size),fill=color)
    def photo(self,canvas,pixels,box):
        x,y,w,h=box;im=pixels.copy() if isinstance(pixels,Image.Image) else Image.fromarray(pixels)
        im.thumbnail((w,h),Image.Resampling.LANCZOS)
        ImageDraw.Draw(canvas).rectangle((x,y,x+w,y+h),fill='#e6eaf0',outline=LINE)
        canvas.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
    def frame(self,images,r,speed):
        cv=Image.new('RGB',(1920,1640),BG);d=ImageDraw.Draw(cv)
        status='握持验证 · 未演示夹起方块' if self.purpose=='hold-validation' else ('选取成功回合，不代表总体成功率' if self.selected_success else '诊断回合 · 未确认任务成功')
        self.text(d,(40,18),'三指持镊：自然握持验证' if self.purpose=='hold-validation' else '三指持镊：真实参考与实际仿真',42)
        self.text(d,(40,76),status+'；接触与支撑为'+self.temporal['label']+'。',26,TEAL if self.selected_success else AMBER,width=1180)
        checkpoint=self.ev.get('checkpoint','未记录')
        controller=self.controller['label']
        self.text(d,(1260,28),controller,25,width=620)
        self.text(d,(1260,73),f'播放 {speed:g}×  |  仿真 {r["time"]:.3f} s',25,MUTED,width=620)
        items=[(self.reference,'人类原始参考 · '+self.ref_label,40,130),
               (images['closeup'],self.labels['closeup'],968,130),
               (images['top'],self.labels['top'],40,750),
               (images['side'],self.labels['side'],968,750)]
        for pixels,label,x,y in items:
            self.text(d,(x,y),label,27,width=912);self.photo(cv,pixels,(x,y+42,912,556))
        m=contact_values(r);lift=r.get('cube_lift_m')
        cards=[(40,300,'方块抬升',number(None if lift is None else lift*1000,1)+' mm'),
               (356,640,'拇 / 食 / 中接触强度',' / '.join(number(v) for v in m['strength'])+' N'),
               (1012,410,'中指向上支撑',number(m['middle_up_N'])+' N'),
               (1438,442,self.web_scope['label'],number(m['web_up_N'])+' N')]
        for x,w,label,value in cards:
            d.rounded_rectangle((x,1368,x+w,1470),12,fill='white',outline=LINE)
            self.text(d,(x+14,1378),label,23,MUTED,width=w-28);self.text(d,(x+14,1414),value,31,TEAL,width=w-28)
        axis=r.get('web_side_axis_w');webz=float(axis[2]) if axis is not None else None
        fixture='夹具启用' if self.task.get('fixture_present') or r.get('fixture_active') else ('任务期间无夹具' if self.task.get('fixture_present') is False else '夹具状态未记录')
        init='持镊快照初始化' if self.task.get('reset_snapshot') else '预持镊初始化'
        self.text(d,(40,1485),f'虎口代理轴朝上分量 {number(webz)}  |  工具下倾 {number(r.get("tool_down_angle_deg"),1)}°  |  {fixture}  |  {init}',24,width=1840)
        self.text(d,(40,1525),'接触强度：各工具/手部 link 法向合力模长之和。支撑：工具受力的竖直分量（含摩擦），向上为正。',21,MUTED,width=1840)
        count=m['web_contact_count'];countstr=number(count,3) if self.temporal['is_control_interval_mean'] else (str(count) if count is not None else '未记录')
        count_label='控制周期平均法向接触点数' if self.temporal['is_control_interval_mean'] else '法向接触点数'
        caveat='周期均值不代表每个物理步都连续接触。' if self.temporal['is_control_interval_mean'] else '是否稳定搭靠须另作连续受力与无穿模审计。'
        self.text(d,(40,1560),f'虎口 ROI {count_label} {countstr}；ROI 为机器人几何代理。'+caveat,22,MUTED,width=1840)
        footer='本视频仅验证握持，未演示夹起方块，不代表完整任务已完成。' if self.purpose=='hold-validation' else ('本视频仅展示选中的单回合，不能作为总体成功率。' if self.selected_success else '此诊断视频不作为成功演示。')
        self.text(d,(40,1598),footer+' 原图未裁切或镜像；仿真仅缩放拼接与外部 HUD，未改物理姿态。',22,MUTED,width=1840)
        return np.asarray(cv)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output-dir',type=Path);p.add_argument('--output-prefix',default='tripod_comparison')
    p.add_argument('--font',type=Path,default=ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf')
    p.add_argument('--frame-times-json',type=Path);p.add_argument('--static-frame',type=int,default=0)
    p.add_argument('--playback-speed',type=float,default=1.);p.add_argument('--max-mb',type=float,default=24.)
    p.add_argument('--static-only',action='store_true');p.add_argument('--require-success',action='store_true')
    p.add_argument('--purpose',choices=['task','hold-validation'],default='task',help='Hold validation is explicitly labelled as not a block-lifting success')
    args=p.parse_args();args.run=args.run.resolve()
    if Path(args.output_prefix).name!=args.output_prefix:p.error('output-prefix must be a basename')
    if not 0<args.playback_speed<=8 or args.max_mb<=0:p.error('speed in (0,8], size cap positive')
    out=(args.output_dir or args.run/'tripod_delivery').resolve();out.mkdir(parents=True,exist_ok=True)
    task,ev,launch,records=[read(args.run/n) for n in ['task.json','evaluation.json','launch.json','trajectory.json']]
    if args.purpose=='hold-validation' and task.get('mode')!='hold':p.error('hold-validation requires recorded task.mode=hold')
    try:selection=selected_environment_evidence(ev)
    except ValueError as exc:p.error(str(exc))
    if args.require_success and not selection['selected_success_episode']:
        p.error('Recorded environment 0 is not explicitly successful; multi-env runs require aligned success_per_env with env0=True')
    checks=[contact_values(r) for r in records]
    manifest_path=args.run/'camera_manifest.json';cam_manifest=read(manifest_path) if manifest_path.exists() else []
    labels={n:camera_label(n,cam_manifest,launch) for n in ['closeup','top','side']}
    evidence=read(ROOT/'references/human/landmarks.json')
    ref=min([f for f in evidence['frames'] if f.get('usable_for_retargeting') and f.get('hand_role')=='tweezer_hand'],key=lambda f:abs(f['time_s']-222))
    reference_path=ROOT/ref['frame_path'];reference=Image.open(reference_path).convert('RGB')
    scope=web_source_scope(args.run)
    layout=Layout(args.font,task,ev,labels,reference,f'{ref["time_s"]:.3f} s / frame {ref["frame_index"]}',scope,args.purpose)
    names=['closeup','top','side'];paths={n:args.run/(n+'.mp4') for n in names};readers={n:imageio.get_reader(str(paths[n])) for n in names}
    try:
        counts=[r.count_frames() for r in readers.values()];rates=[float(r.get_meta_data()['fps']) for r in readers.values()]
        if len(set(counts))!=1 or counts[0]<1 or max(rates)-min(rates)>1e-6:raise ValueError('Source camera counts/rates disagree')
        count,fps=counts[0],rates[0];indices,times,alignment=frame_alignment(args,task,records,count)
        if count>1 and not np.allclose(np.diff(times),1/fps,atol=1e-5):raise ValueError('Source timestamps disagree with raw video fps')
        if not 0<=args.static_frame<count:p.error('static-frame outside source')
        still=out/(args.output_prefix+'.png');video=out/(args.output_prefix+'.mp4');cover=out/(args.output_prefix+'_cover.png')
        def compose(i):return layout.frame({n:readers[n].get_data(i) for n in names},records[indices[i]],args.playback_speed)
        Image.fromarray(compose(args.static_frame)).save(still)
        cover_i=max(range(count),key=lambda i:float(records[indices[i]].get('cube_lift_m',-float('inf'))))
        Image.fromarray(compose(cover_i)).save(cover)
        output_fps=fps*args.playback_speed;duration=count/output_fps;decoded=None
        if not args.static_only:
            bitrate=min(8_000_000,int(args.max_mb*1e6*8*.82/duration))
            if bitrate<300000:raise ValueError('Size budget too small')
            writer=imageio.get_writer(str(video),fps=output_fps,codec='libx264',quality=None,bitrate=str(bitrate),macro_block_size=1,pixelformat='yuv420p',ffmpeg_params=['-preset','medium','-movflags','+faststart','-threads','4'])
            try:
                for i in range(count):writer.append_data(compose(i))
            finally:writer.close()
            if video.stat().st_size>=args.max_mb*1e6:raise RuntimeError('Video exceeds size cap')
            verify=imageio.get_reader(str(video))
            try:
                decoded=sum(1 for frame in verify)
                if decoded!=count:raise RuntimeError('Full decoded frame count mismatch')
            finally:verify.close()
        checkpoint=ev.get('checkpoint');cp=Path(checkpoint) if checkpoint not in [None,'zero'] else None
        if cp and not cp.is_absolute():cp=ROOT/cp
        report=dict(run=str(args.run),script_sha256=sha(__file__),source_files={n:dict(path=str(args.run/n),sha256=sha(args.run/n)) for n in ['task.json','evaluation.json','launch.json','trajectory.json']},source_cameras={n:dict(path=str(paths[n]),sha256=sha(paths[n]),label=labels[n]) for n in names},camera_manifest=dict(path=str(manifest_path),sha256=sha(manifest_path)) if manifest_path.exists() else None,source_reference=dict(path=str(reference_path),sha256=sha(reference_path),source_url=evidence['source_url'],video_sha256=evidence['video_sha256'],time_s=ref['time_s'],frame_index=ref['frame_index'],hand_role='camera-left tweezer hand',no_human_3d_ground_truth=True),checkpoint=checkpoint,checkpoint_sha256_at_composition=sha(cp) if cp and cp.exists() else None,checkpoint_lineage=checkpoint_lineage(cp),reset_snapshot=task.get('reset_snapshot'),reset_snapshot_recorded_sha256=task.get('reset_snapshot_sha256'),selected_success_episode=layout.selected_success,recorded_environment_index=0,population_episodes=ev.get('episodes'),success_selection_evidence=layout.selection,controller_provenance=layout.controller,media_purpose=args.purpose,complete_lifting_task_claimed=False if args.purpose=='hold-validation' else layout.selected_success,reported_evaluation=ev,frame_alignment=alignment,source_trajectory_indices=indices,frame_simulation_times_s=times.tolist(),playback_speed=args.playback_speed,contact_metric_definitions=dict(groups=['thumb','index','middle','wrist_palm','other'],strength='Sum over tool body / robot body pairs of norm(normal resultant force); not sum of point force magnitudes.',support='World-Z component of normal + friction force acting ON tool, positive upward.',web=dict(scope=scope,definition='Measured contacts on merged palm/wrist body inside configured geometric web ROI; not all wrist contacts.'),source_fields_checked='All recorded group support values independently compared with normal+friction; web support derived directly from saved vectors.',human_inference='No force, precise contact location or 3D orientation is inferred from human pixels.'),contact_audited_rows=len(checks),contact_temporal_aggregation=temporal_scope(task),source_contact_reader_sha256=sha(args.run/'grasp_contacts.py') if (args.run/'grasp_contacts.py').exists() else None,still=dict(path=str(still),sha256=sha(still),frame=args.static_frame),cover=dict(path=str(cover),sha256=sha(cover),frame=cover_i),video=None if args.static_only else dict(path=str(video),sha256=sha(video),bytes=video.stat().st_size,frames=count,fully_decoded_frames=decoded,fps=output_fps,duration_s=duration,resolution=[1920,1640]),operations=['aspect-preserving resize','letterbox/paste','outside-image HUD'],raw_images_unchanged=True,physics_states_changed=False,success_rate_claimed_from_selected_episode=False,external_send_performed=False,limitations=['Human camera is uncalibrated; matching panels does not register 3D pose.','Static reference middle contact is occluded.','Neither sampled nor control-interval mean support proves uninterrupted substep contacts or long-term stability.','Tail/web acceptance additionally requires independent geometry and dynamics checks.'])
        dest=out/(args.output_prefix+'_manifest.json');dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(image=str(still),video=None if args.static_only else str(video),manifest=str(dest)),ensure_ascii=False,indent=2))
    finally:
        for r in readers.values():r.close()

if __name__=='__main__':main()
