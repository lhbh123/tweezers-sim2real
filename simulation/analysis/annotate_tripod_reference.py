"""Conservative annotations of existing human video; no generated anatomy.

Manual 2-D regions identify visible structures and an explicitly uncertain
middle-support candidate. They are not contact-force or calibrated 3-D data.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
FONT=ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'
BG='#f4f6f9';INK='#17243a';MUTED='#526078';UNCERTAIN='#b36b16'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def font(size):return ImageFont.truetype(str(FONT),size)


def wrapped(draw,xy,text,width,size=25,fill=INK,line_gap=8):
    x,y=xy;line=''
    for c in text:
        if c=='\n' or draw.textlength(line+c,font=font(size))>width:
            draw.text((x,y),line,font=font(size),fill=fill);y+=size+line_gap;line='' if c=='\n' else c
        else:line+=c
    if line:draw.text((x,y),line,font=font(size),fill=fill);y+=size+line_gap
    return y


def dashed_ellipse(draw,box,color,width=3):
    for start in range(0,360,24):draw.arc(box,start,start+12,fill=color,width=width)


def annotated(image):
    result=image.copy();d=ImageDraw.Draw(result)
    annotations=[
        dict(id=1,label='拇指尺侧／指腹侧缘',anchor=[700,415],marker=[803,326],color='#b63657',
             evidence='visible structure; close apposition to upper image-space arm',certainty='direct image evidence; microscopic contact not measured'),
        dict(id=2,label='食指邻近另一臂',anchor=[678,491],marker=[817,566],color='#1767a4',
             evidence='curled index finger below the lower image-space arm',certainty='visible structure and image-space apposition'),
        dict(id=3,label='中指承托候选区（遮挡，推测）',anchor=[724,459],marker=[844,398],color=UNCERTAIN,
             evidence='overlapping finger tissue behind the gripping pair; exact support surface occluded',certainty='hypothesis; not a measured middle contact'),
        dict(id=4,label='虎口开口区域',anchor=[571,397],marker=[505,299],color='#257a65',
             evidence='visible thumb-index web opening adjacent to proximal tool segment',certainty='2-D region only; 3-D outward normal unknown'),
        dict(id=5,label='镊尾／连接端',anchor=[281,418],marker=[192,476],color='#7252a0',
             evidence='visible proximal joined end extending to the image left',certainty='visible tool part; web contact gap unknown')]
    for a in annotations:
        x,y=a['anchor'];mx,my=a['marker'];color=a['color']
        if a['id']==3:dashed_ellipse(d,(681,425,758,488),color,4)
        elif a['id']==4:d.ellipse((524,354,632,429),outline=color,width=3)
        else:d.ellipse((x-15,y-15,x+15,y+15),outline=color,width=3)
        d.line([(x,y),(mx,my)],fill=color,width=3)
        d.ellipse((mx-23,my-23,mx+23,my+23),fill='white',outline=color,width=3)
        d.text((mx,my-2),str(a['id']),anchor='mm',font=font(29),fill=color)
    return result,annotations


def main():
    source=ROOT/'references/human/landmarks.json';data=json.loads(source.read_text())
    records=[r for r in data['frames'] if r.get('usable_for_retargeting') and r.get('hand_role')=='tweezer_hand' and 216<=r['time_s']<=228.1]
    assert len(records)==7
    video=ROOT/data['video_path'];reader=imageio.get_reader(video);evidence=[];images={}
    try:
        for r in records:
            path=ROOT/r['frame_path'];im=Image.open(path).convert('RGB');assert im.size==(1280,720)
            assert abs(r['frame_index']/data['source_fps']-r['time_s'])<1e-8
            raw=reader.get_data(r['frame_index'])[:,:,:3];buffer=io.BytesIO();Image.fromarray(raw).save(buffer,format='JPEG',quality=95)
            assert hashlib.sha256(buffer.getvalue()).hexdigest()==sha(path),'Saved keyframe does not match source video index'
            images[round(r['time_s'])]=im
            evidence.append(dict(time_s=r['time_s'],frame_index=r['frame_index'],path=str(path.relative_to(ROOT)),
                                 sha256=sha(path),source_video_redecode_matches_jpeg=True,resolution=[1280,720]))
    finally:reader.close()
    selected=next(r for r in records if abs(r['time_s']-222)<.03)
    annotation,regions=annotated(images[222])
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    canvas=Image.new('RGB',(1920,1390),BG);d=ImageDraw.Draw(canvas)
    d.text((44,20),'人类持镊参考：可见握持关系与遮挡边界',font=font(45),fill=INK)
    d.text((46,85),'222.013 s · 画面左侧持镊手 · 人工二维标注，非三维真值；③虚线区为推测',font=font(27),fill=MUTED)
    canvas.paste(annotation,(44,142))
    descriptions=[
        ('1  拇指尺侧／指腹侧缘','靠近画面上方镊臂。216 s 原字幕将接触部位纠正为拇指尺侧。','#b63657'),
        ('2  食指','卷曲于图像下方，邻近另一镊臂；拇食形成相对夹持。','#1767a4'),
        ('3  中指承托候选区','位于握持后方且被遮挡。下托位置、接触法向和受力均未直接测得。',UNCERTAIN),
        ('4  虎口开口','局部开口可见；不能据单张照片确定世界竖直方向或开口法线。','#257a65'),
        ('5  镊尾／连接端','向画面左侧、虎口一侧延伸。是否贴靠虎口、实际间隙未知。','#7252a0')]
    y=147
    for title,body,color in descriptions:
        d.text((1360,y),title,font=font(27),fill=color);y+=43
        y=wrapped(d,(1360,y),body,510,24,MUTED,7)+19
    wrapped(d,(1360,760),'图像“上／下”不等于世界“上／下”。右侧拿笔手未纳入分析。',510,25,INK)
    d.text((44,891),'连续片段交叉核对：下列均为原始帧，仅等比缩小',font=font(29),fill=INK)
    for x,t in [(44,216),(672,222),(1300,228)]:
        r=next(r for r in records if abs(r['time_s']-t)<.03)
        im=images[t].copy();im.thumbnail((576,324));canvas.paste(im,(x,970))
        d.text((x,936),f'{r["time_s"]:.3f} s / frame {r["frame_index"]}',font=font(22),fill=MUTED)
    wrapped(d,(44,1313),'设计要求来自用户：虎口朝上、拇食捏持、中指下托、镊尾靠虎口上方。参考图支持拓扑判断，不提供精确三维角度或接触力。',1830,25,MUTED)
    figure=out/'tripod_reference_annotated.png';canvas.save(figure)
    panel=Image.new('RGB',(1280,904),BG);pd=ImageDraw.Draw(panel)
    pd.text((22,10),'人类参考 · 222.013 s · 人工二维区域标注',font=font(32),fill=INK)
    pd.text((22,55),'③中指承托区被遮挡，仅为候选；不是三维接触真值',font=font(24),fill=UNCERTAIN)
    panel.paste(annotation,(0,98))
    pd.text((20,830),'①拇指尺侧  ②食指  ③中指候选区（推测）  ④虎口开口  ⑤镊尾',font=font(25),fill=INK)
    pd.text((20,869),'原始帧保留；仅添加编号、引线和区域轮廓；未补画手指或接触。',font=font(22),fill=MUTED)
    panel_path=out/'tripod_reference_left_panel.png';panel.save(panel_path)
    config_path=ROOT/'configs/human_grasp_distal_suspend30.json';cfg=json.loads(config_path.read_text())
    report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Internal reference topology constraints for a new tripod/web-space grip; not an assessment that the old controller satisfies the new user request',
        source=dict(url=data['source_url'],video_id=data['video_id'],video_path=str(video.relative_to(ROOT)),video_sha256=sha(video),
                    landmarks_path=str(source.relative_to(ROOT)),landmarks_sha256=sha(source),frames=evidence,
                    analyzed_hand='camera-left tweezer-holding hand; other hand holding pen excluded'),
        annotation=dict(method='manual interpretation of actual video frames; pixel regions are approximate anatomical locations, not measured contact points',
                        selected_frame_index=selected['frame_index'],selected_time_s=selected['time_s'],pixel_coordinate_origin='top left, x right, y down',
                        regions=regions,occluded_candidate_bounds_xyxy=[681,425,758,488],
                        no_generated_anatomy=True,no_modified_landmarks=True,no_source_overwrite=True),
        directly_observable=[
            dict(id='O1',statement='Thumb and curled index are adjacent to opposite image-space tool arms; opening narrows across the selected interval.',evidence_level='visible image topology, no measured normal forces'),
            dict(id='O2',statement='The source 216.007 s frame explicitly corrects the thumb contact description to ULNAR aspect of thumb.',evidence_level='visible source subtitle'),
            dict(id='O3',statement='A thumb-index web opening is visible adjacent to the proximal tool segment.',evidence_level='visible 2-D region'),
            dict(id='O4',statement='The joined proximal tail extends to image left; pointed jaws extend to image right.',evidence_level='visible tool geometry'),
            dict(id='O5',statement='Fingers behind the active pinch overlap and occlude the candidate middle support region.',evidence_level='visible occlusion')],
        uncertain=[
            'Exact middle-finger contact point, whether sustained normal support occurs, and support force direction are not visible.',
            'The exact tail-to-web separation/contact, and which skin surface supports the proximal segment, are not resolved.',
            'A single uncalibrated overhead/oblique view does not establish the world vertical direction of the local web opening or a numeric tool/table angle.',
            'MediaPipe 3-D has previously observed depth-fold artifacts, and handedness scores are not landmark/contact confidence.',
            'Image-space upper/lower arm must not be relabeled as world-above/world-below.'],
        user_design_requirements=[
            dict(id='U1',requirement='虎口朝上',interpretation='Expose the local first-web-space opening toward world up, separately from the whole-palm normal.',origin='explicit user requirement; not numeric 3-D recovered from the video'),
            dict(id='U2',requirement='拇指与食指捏持',interpretation='Maintain opposed contacts on the two tweezer arms; favor thumb ulnar/pad-side surface rather than the nail.',origin='user requirement and visible reference topology'),
            dict(id='U3',requirement='中指从下方承托',interpretation='Establish and measure a third support contact beneath the local tool segment.',origin='explicit user requirement; exact human contact remains occluded'),
            dict(id='U4',requirement='镊尾靠虎口上方',interpretation='Locate the joined proximal end near the first-web region and above its selected support anchor.',origin='explicit user requirement; exact human gap unmeasured')],
        current_two_finger_configuration=dict(path=str(config_path.relative_to(ROOT)),sha256=sha(config_path),
            active_contact_targets=cfg['contact_targets'],middle_closure_delta={k:v for k,v in cfg['closure_delta'].items() if 'mid_' in k},
            initial_middle_contact_claimed=cfg['geometry_validation'].get('initial_middle_contact_claimed'),
            misses=['Only thumb/index contain explicit local contact points; middle is merely a relaxed candidate.',
                    'Both middle closure deltas are zero; no learned middle-finger support action is defined by this synergy.',
                    'No first-web anchor, opening direction or tail/web positional objective is present.',
                    'Palm-normal-down gating does not encode web-space-up.',
                    'Existing successful block lift/contact metrics do not prove finger-specific tripod support.'],
            caution='The configuration proves missing intentional support constraints, not that incidental middle/tool contact is impossible in every dynamic frame.'),
        proposed_robot_acceptance_metrics=[
            dict(name='web_opening_orientation',definition='Calibrate a local web opening direction from the actual thumb/index base geometry and labeled web region; transform it through FK and report its dot product with world up.',threshold='Set after asset calibration; do not substitute palm_normal_z or a monocular human world vector.',status='engineering proposal, not human ground truth'),
            dict(name='opposed_pinch',definition='Identify thumb/tool and index/tool contact pairs separately; report contact locations and signed normals/forces on opposite arms over the valid task interval.',threshold='Use declared simulator contact/noise thresholds and durations; do not infer contact from visual overlap alone.',status='engineering proposal'),
            dict(name='middle_under_support',definition='Locate middle pad below the corresponding tool segment in world z and measure sustained middle-to-tool support force; audit actual source-mesh penetration.',threshold='Report gap, force vector and contact duty cycle; use a declared force threshold, not a rendered-point proxy.',status='engineering proposal'),
            dict(name='tail_web_relation',definition='Report tail-to-web support-region distance, signed height difference, and proximal/distal ordering relative to the pinch.',threshold='Define mesh anchors and an allowable band from robot geometry; do not treat selected mm thresholds as measured human values.',status='engineering proposal'),
            dict(name='camera_acceptance',definition='Use a view that exposes thumb/index/web and a second real side view showing the middle under-support; keep tool tail and both tips in frame.',threshold='Actual rendered pixels plus contact/FK evidence; saved USD projection alone has shown a mismatch.',status='engineering proposal')],
        clarification_of_prior_viewpoint_summary='The reference shows dorsal wrist/thumb surfaces together with an exposed first-web opening. A single whole-palm up/down label discards local contact topology and is insufficient for this new request.',
        outputs=[dict(path=str(figure.relative_to(ROOT)),sha256=sha(figure)),dict(path=str(panel_path.relative_to(ROOT)),sha256=sha(panel_path))],
        script_sha256=sha(__file__),font_sha256=sha(FONT),external_send_performed=False)
    target=ROOT/'reports/tripod_reference_constraints.json';target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(dict(report=str(target),images=[str(figure),str(panel_path)],verified_video_frames=len(evidence)),ensure_ascii=False))


if __name__=='__main__':main()
