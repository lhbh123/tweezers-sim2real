#!/usr/bin/env python3
"""Plot recorded human landmarks against untouched reference frames.

No detector is rerun, no points are corrected, and no reference photo is edited
on disk. The left photo panels are pixel-identical copies of the decoded JPEGs.
Run with .venv-human/bin/python scripts/visualize_human_reference.py.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CHAINS = {'thumb': [0,1,2,3,4], 'index': [0,5,6,7,8],
          'middle': [0,9,10,11,12], 'ring': [0,13,14,15,16], 'pinky': [0,17,18,19,20]}
COLORS = {'thumb': '#f36d5c', 'index': '#e7ad00', 'middle': '#179c71',
          'ring': '#218cda', 'pinky': '#ad58ce'}
NAMES = {'thumb': '拇指 1–4', 'index': '食指 5–8', 'middle': '中指 9–12',
         'ring': '无名指 13–16', 'pinky': '小指 17–20'}
BG, INK, MUTED, LINE = '#f3f5f8', '#17243a', '#526078', '#d2dae6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bend(points, ids=(6,7,8)):
    a,b,c = np.asarray(points)[list(ids)]
    u,v = b-a,c-b
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)), -1, 1))))


class Painter:
    def __init__(self, font):
        self.font_path = str(font)
        self.fonts = {}

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_path, size)
        return self.fonts[size]

    def text(self, draw, xy, text, size=30, fill=INK, **kwargs):
        draw.text(xy, text, font=self.font(size), fill=fill, **kwargs)

    def legend(self, draw, y, start=48, spacing=450):
        for index,name in enumerate(CHAINS):
            x = start+index*spacing
            draw.line((x,y+22,x+48,y+22), fill=COLORS[name], width=9)
            self.text(draw, (x+64,y), NAMES[name], 30)


def overlay(photo, xy, painter):
    result = photo.copy()
    draw = ImageDraw.Draw(result)
    for name, ids in CHAINS.items():
        coords = [tuple(xy[i]) for i in ids]
        draw.line(coords, fill='#131a24', width=9)
        draw.line(coords, fill=COLORS[name], width=5)
    offsets = {4:(12,-25),8:(-31,10),11:(13,9),12:(14,0),15:(-35,-5),
               16:(12,-18),18:(-34,1),19:(-34,-18),20:(13,-10)}
    for index,(x,y) in enumerate(xy):
        name = next((n for n,ids in CHAINS.items() if index in ids[1:]), None)
        draw.ellipse((x-7,y-7,x+7,y+7), fill=COLORS[name] if name else '#ffffff', outline='#142234', width=2)
        dx,dy = offsets.get(index,(10,-12))
        painter.text(draw,(x+dx,y+dy),str(index),21,fill='white',stroke_width=2,stroke_fill='#17243a')
    return result


def comparison(records, photos, data, painter, out):
    width,height = 2704,2926
    canvas = Image.new('RGB',(width,height),BG)
    draw = ImageDraw.Draw(canvas)
    painter.text(draw,(48,25),'人类持镊姿态：原始视频与 21 点估计',64)
    painter.text(draw,(48,112),'同帧一一对照 · 仅画面左侧持镊手 · 模型推断，不是真值（model inferred, not ground truth）',33,MUTED)
    painter.legend(draw,166,spacing=510)
    painter.text(draw,(48,232),'A  原始帧：照片像素保持不变',37)
    painter.text(draw,(1376,232),'B  相同原始帧 + 21 点投影骨架',37)
    panel_positions=[]
    for row,(record,photo) in enumerate(zip(records,photos)):
        y = 297+row*806
        label=f'{record["time_s"]:.3f} s  |  frame {record["frame_index"]}  |  1280 × 720'
        painter.text(draw,(48,y),label,29,MUTED)
        painter.text(draw,(1376,y),'图像坐标：u = x_norm × 1280，v = y_norm × 720',29,MUTED)
        iy=y+52
        canvas.paste(photo,(48,iy))
        xy=np.asarray(record['landmarks_normalized'])[:,:2]*[photo.width,photo.height]
        canvas.paste(overlay(photo,xy,painter),(1376,iy))
        for x in [48,1376]:
            draw.rectangle((x-1,iy-1,x+1280,iy+720),outline=LINE,width=1)
        panel_positions.append((48,iy,48+1280,iy+720))
    y=2736
    painter.text(draw,(48,y),'0 为腕点；画面右侧拿笔的手未纳入。手指和工具相互遮挡，遮挡点同样是模型预测。',31)
    painter.text(draw,(48,y+49),'没有逐点定位/遮挡置信度；模型的 handedness 分数只代表左右手分类，不代表关键点精度。',31,MUTED)
    painter.text(draw,(48,y+102),'来源：PennMicro / University of Pennsylvania · Holding the instruments · YouTube KXDHa39Z-2s',27,MUTED)
    canvas.save(out)
    decoded=Image.open(out).convert('RGB')
    assert all(np.array_equal(np.asarray(decoded.crop(box)),np.asarray(photo)) for box,photo in zip(panel_positions,photos))
    return dict(size_px=list(canvas.size),raw_photo_panels_pixel_identical=True,
                raw_photo_panel_boxes_xyxy=panel_positions,photo_crop_or_resize=False)


def skeleton_plot(points, font, world=False):
    fig=plt.figure(figsize=(7.15,6.35),dpi=160,facecolor=BG)
    ax=fig.add_subplot(111,projection='3d' if world else None,proj_type='ortho') if world else fig.add_subplot(111)
    ax.set_facecolor(BG)
    pts=np.asarray(points)*1000 if world else np.asarray(points)
    for name,ids in CHAINS.items():
        p=pts[ids]
        if world:
            ax.plot(*p.T,color=COLORS[name],linewidth=3.4,marker='o',markersize=5)
        else:
            ax.plot(*p.T,color=COLORS[name],linewidth=3.4,marker='o',markersize=6)
    for index in ([0,4,6,7,8,12,16,20] if world else range(21)):
        p=pts[index]
        if world:
            ax.text(*p,str(index),fontsize=13,color=INK)
        else:
            offsets={6:(8,-18),7:(8,-18),8:(-18,-22),11:(13,5),15:(-17,1),18:(-19,3),19:(-15,-8)}
            off=offsets.get(index,(9,-7))
            t=ax.annotate(str(index),p,xytext=off,textcoords='offset points',fontsize=12,color=INK)
            t.set_path_effects([patheffects.withStroke(linewidth=3,foreground=BG)])
    ids=[6,7,8]
    if world:
        ax.scatter(*pts[0],s=60,facecolors='white',edgecolors=INK,linewidths=1.2,depthshade=False)
        ax.scatter(*pts[ids].T,s=130,facecolors=COLORS['index'],edgecolors=INK,linewidths=1.4,depthshade=False)
        ax.set_xlabel('模型 x / mm',fontproperties=font,labelpad=12)
        ax.set_ylabel('模型 y / mm',fontproperties=font,labelpad=12)
        ax.set_zlabel('模型 z / mm',fontproperties=font,labelpad=12)
        bounds=np.ptp(pts,axis=0)
        ax.set_box_aspect(bounds)
        ax.view_init(elev=22,azim=-62)
        ax.tick_params(labelsize=11,pad=2)
        fig.subplots_adjust(left=0.,right=.93,bottom=.05,top=.99)
    else:
        ax.scatter(*pts[0],s=65,facecolors='white',edgecolors=INK,linewidths=1.2,zorder=4)
        ax.scatter(*pts[ids].T,s=150,facecolors=COLORS['index'],edgecolors=INK,linewidths=1.4,zorder=3)
        ax.set_xlim(pts[:,0].min()-48,pts[:,0].max()+55)
        ax.set_ylim(pts[:,1].max()+48,pts[:,1].min()-40)
        ax.set_aspect('equal')
        ax.set_xlabel('图像 u / px（向右）',fontproperties=font,labelpad=12)
        ax.set_ylabel('图像 v / px（向下）',fontproperties=font,labelpad=12)
        ax.grid(color=LINE,alpha=.7,linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=12)
        fig.subplots_adjust(left=.17,right=.94,bottom=.12,top=.99)
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=160,facecolor=BG);plt.close(fig);stream.seek(0)
    return Image.open(stream).convert('RGB')


def limits(record,painter,out):
    xy=np.asarray(record['landmarks_pixel_xy'])
    world=np.asarray(record['landmarks_world_m'])
    angle2,angle3=bend(xy),bend(world)
    font_manager.fontManager.addfont(painter.font_path)
    prop=font_manager.FontProperties(fname=painter.font_path,size=15)
    plt.rcParams.update({'font.family':prop.get_name(),'axes.unicode_minus':False})
    canvas=Image.new('RGB',(2464,1840),BG);draw=ImageDraw.Draw(canvas)
    painter.text(draw,(48,28),'同一帧的 2D 投影与单目 3D：估计在哪里不可靠',54)
    painter.text(draw,(48,110),f'{record["time_s"]:.3f} s · frame {record["frame_index"]} · 画面左侧持镊手 · 全部关键点来自原 landmarks.json',31,MUTED)
    painter.legend(draw,165,spacing=465)
    painter.text(draw,(48,230),'A  2D 投影骨架（原图像素坐标）',37)
    painter.text(draw,(1272,230),'B  MediaPipe world3D（模型坐标）',37)
    painter.text(draw,(48,290),'只作弱姿态线索，不是解剖关节角',28,MUTED)
    painter.text(draw,(1272,290),'深度与尺度未经标定，未用作强监督',28,'#a24b29')
    canvas.paste(skeleton_plot(xy,prop),(48,345))
    canvas.paste(skeleton_plot(world,prop,world=True),(1272,345))
    y=1390
    draw.rounded_rectangle((48,y,2416,y+180),radius=18,fill='#fff3d9',outline='#e5ca8c',width=2)
    painter.text(draw,(78,y+16),'食指末节折叠疑点：6 → 7 → 8（7 为 DIP；8 为指尖）',35)
    painter.text(draw,(78,y+78),f'相邻线段转角：2D 投影 {angle2:.1f}°    /    world3D 输出 {angle3:.1f}°',44,'#7b4b00')
    painter.text(draw,(48,1602),'该差异提示单目形状/深度不一致；两者都不是真值，差值不能当作三维估计误差。',31)
    painter.text(draw,(48,1653),'对照原始帧人工复核：world3D 的食指折返不可信；准确 3D 姿态仍需标定、多视角或测量。',31,MUTED)
    painter.text(draw,(48,1704),'本项目保留原始预测，只取 2D 弯曲代理作为弱先验；遮挡关节、相机深度和真实角度仍不确定。',30,MUTED)
    painter.text(draw,(48,1762),'来源：PennMicro · KXDHa39Z-2s；模型推断，不是真值（model inferred, not ground truth）。',27,MUTED)
    canvas.save(out)
    return dict(size_px=list(canvas.size),frame_index=record['frame_index'],time_s=record['time_s'],
                index_DIP_segment_bend_2d_deg=angle2,index_DIP_segment_bend_world3d_deg=angle3,
                angle_definition='angle between (p7-p6) and (p8-p7), 0 degrees means straight; not anatomical ground truth',
                world_display='Original model x/y/z multiplied by 1000 for mm; orthographic display, no fitted camera or landmark correction.',
                world_plot_view=dict(elevation_deg=22,azimuth_deg=-62),
                difference_is_not_estimation_error=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--landmarks',type=Path,default=ROOT/'references/human/landmarks.json')
    parser.add_argument('--font',type=Path,default=ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--audit',type=Path,default=ROOT/'reports/human_visualization_audit.json')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);args.audit.parent.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.landmarks.read_text());fps=data['source_fps'];resolution=data['resolution']
    raw_path=ROOT/'references/human/landmarks_mediapipe_raw.json';raw=json.loads(raw_path.read_text())
    source=ROOT/data['video_path'];source_sha=sha(source)
    assert source_sha==data['video_sha256']
    chosen=[min([f for f in data['frames'] if f.get('usable_for_retargeting') and f.get('hand_role')=='tweezer_hand'],key=lambda f:abs(f['time_s']-t)) for t in [216,222,228]]
    photos=[];audit_frames=[]
    reader=imageio.get_reader(source)
    try:
        for target,f in zip([216,222,228],chosen):
            path=ROOT/f['frame_path'];photo=Image.open(path).convert('RGB');xy=np.asarray(f['landmarks_normalized'])[:,:2]*resolution
            assert list(photo.size)==resolution==[1280,720]
            assert np.asarray(f['landmarks_normalized']).shape==(21,3) and np.asarray(f['landmarks_world_m']).shape==(21,3)
            assert np.isfinite(xy).all() and np.isfinite(f['landmarks_world_m']).all()
            assert abs(f['time_s']-target)<.03 and abs(f['frame_index']/fps-f['time_s'])<1e-9
            assert np.allclose(xy,f['landmarks_pixel_xy'],rtol=0,atol=1e-9)
            prior=next(r for r in raw['frames'] if r['frame_index']==f['frame_index'] and r.get('hand_index')==f['hand_index'])
            assert prior['landmarks_normalized']==f['landmarks_normalized'] and prior['landmarks_world_m']==f['landmarks_world_m']
            video_frame=reader.get_data(f['frame_index'])[:,:,:3]
            encoded=io.BytesIO();Image.fromarray(video_frame).save(encoded,format='JPEG',quality=95)
            same=hashlib.sha256(encoded.getvalue()).hexdigest()==sha(path)
            # Same decoder/PIL recipe proves the image was taken from this exact
            # indexed source frame, not a similarly named previous extraction.
            assert same, f'Saved JPEG does not match indexed source frame: {path}'
            photos.append(photo)
            audit_frames.append(dict(requested_time_s=target,time_s=f['time_s'],frame_index=f['frame_index'],hand_index=f['hand_index'],hand_role=f['hand_role'],frame_path=str(path),frame_sha256=sha(path),image_size_px=list(photo.size),normalized_to_pixel_max_abs_error=float(np.max(np.abs(xy-f['landmarks_pixel_xy']))),time_index_error_s=abs(f['time_s']-f['frame_index']/fps),reencoded_video_frame_jpeg_matches_source=True,raw_mediapipe_landmarks_unchanged=True,handedness_score=f['confidence'],handedness_score_semantics=f['confidence_kind'],landmark_localization_confidence=f.get('landmark_localization_confidence'),occlusion_confidence=f.get('occlusion_confidence'),index_DIP_segment_bend_2d_deg=bend(xy),index_DIP_segment_bend_world3d_deg=bend(f['landmarks_world_m']),source_occlusion_notes=f['occlusion_notes']))
    finally:
        reader.close()
    painter=Painter(args.font)
    compare_file=args.output_dir/'human_reference_comparison.png';limits_file=args.output_dir/'human_pose_estimation_limits.png'
    comparison_meta=comparison(chosen,photos,data,painter,compare_file)
    limits_meta=limits(chosen[1],painter,limits_file)
    assert all(sha(ROOT/f['frame_path'])==a['frame_sha256'] for f,a in zip(chosen,audit_frames))
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),script=str(Path(__file__).resolve()),script_sha256=sha(__file__),source_landmarks=str(args.landmarks),source_landmarks_sha256=sha(args.landmarks),raw_landmarks_sha256=sha(raw_path),source_video=str(source),source_video_sha256=source_sha,source_url=data['source_url'],video_url='https://www.youtube.com/watch?v='+data['video_id'],video_id=data['video_id'],source_fps=fps,source_resolution=resolution,method=data['frames'][5]['method'],model_sha256=data['model_sha256'],selected_frames=audit_frames,finger_colors=COLORS,landmark_chains=CHAINS,font=dict(path=str(args.font),sha256=sha(args.font),source_url='https://github.com/notofonts/noto-cjk/blob/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf'),outputs=[dict(path=str(compare_file),sha256=sha(compare_file),**comparison_meta),dict(path=str(limits_file),sha256=sha(limits_file),**limits_meta)],all_raw_inputs_preserved=True,raw_photos_modified=False,landmarks_modified=False,detector_rerun=False,external_send_performed=False,limitations=['Only the camera-left tweezer hand from reviewed positive 216-228 s interval is included.','No calibrated per-landmark or occlusion confidence is available; handedness scores are not localization accuracy.','Image-space bend is a projection proxy, not measured anatomical flexion.','World coordinates are monocular model inference with unreliable depth and scale; not used as strong supervision.','2D-vs-3D bend difference alone is not a metric 3D reconstruction error.','Displayed figures do not imply confidence-weighted retargeting.'])
    args.audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(comparison=str(compare_file),limits=str(limits_file),audit=str(args.audit)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
