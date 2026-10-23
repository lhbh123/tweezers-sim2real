#!/usr/bin/env python3
"""Independent provenance/proxy audit and an internal-only visual review sheet.

This script reads the human evidence and the retargeted configuration. It never
modifies training, simulation, raw keypoints, or the configuration being audited.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / 'references/human'
REPORT = ROOT / 'reports'
CHAINS = {'thumb': [0,1,2,3,4], 'index': [0,5,6,7,8],
          'middle': [0,9,10,11,12], 'ring': [0,13,14,15,16], 'pinky': [0,17,18,19,20]}
COLORS = [(255,115,80), (255,220,70), (90,235,130), (75,180,255), (220,130,255)]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(path.read_text())

def proxy_angles(points):
    angles = {}
    for finger, ids in CHAINS.items():
        angles[finger] = []
        for a,b,c in zip(ids, ids[1:], ids[2:]):
            u,v = points[b]-points[a],points[c]-points[b]
            angles[finger].append(float(np.arccos(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)), -1, 1))))
    return angles

def font(n, bold=False):
    return ImageFont.truetype(FONT + ('-Bold' if bold else '') + '.ttf', n)

def label(draw, xy, text, size=20, color=(225,235,245), bold=False):
    draw.text(xy, text, font=font(size,bold), fill=color)

def paragraph(draw, xy, text, width=56, size=21, color=(185,202,220), gap=29):
    y=xy[1]
    for line in textwrap.wrap(text,width=width):
        label(draw,(xy[0],y),line,size,color);y+=gap
    return y

def visual_panel(canvas, record, x, y, title, caption, accepted, roi=False, overlay=False):
    draw=ImageDraw.Draw(canvas);w=648;h=510
    accent=(76,220,180) if accepted else (255,135,120)
    draw.rounded_rectangle((x,y,x+w,y+h),radius=14,fill=(19,29,42),outline=(49,66,84),width=2)
    label(draw,(x+18,y+16),title,23,accent,True)
    im=Image.open(ROOT/record['frame_path']).convert('RGB')
    # ROI is disclosed explicitly; no pixel content or keypoint coordinate is altered.
    crop=(300,80,880,660) if roi else (0,0,im.width,im.height)
    im=im.crop(crop);scale=min(612/im.width,362/im.height)
    im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
    ox=x+18+(612-im.width)//2;oy=y+57+(362-im.height)//2
    canvas.paste(im,(ox,oy));draw=ImageDraw.Draw(canvas)
    if overlay:
        points=np.asarray(record['landmarks_pixel_xy'])
        pts=[(ox+(p[0]-crop[0])*scale,oy+(p[1]-crop[1])*scale) for p in points]
        for ids,color in zip(CHAINS.values(),COLORS):
            draw.line([pts[i] for i in ids],fill=color,width=2)
            for i in ids:
                a,b=pts[i]
                if not(ox<=a<=ox+im.width and oy<=b<=oy+im.height):continue
                draw.ellipse((a-3,b-3,a+3,b+3),fill=color,outline=(0,0,0))
                dx=6 if i%2 else -13;dy=-10 if i%3 else 4
                draw.text((a+dx,b+dy),str(i),font=font(11,True),fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0))
    paragraph(draw,(x+18,y+431),caption,width=57,size=18,gap=24)

def make_collage(records, output):
    near=lambda t:min([r for r in records if r.get('detected')],key=lambda r:abs(r['time_s']-t))
    canvas=Image.new('RGB',(2040,1350),(9,16,24));d=ImageDraw.Draw(canvas)
    label(d,(24,20),'HUMAN TWEEZER REFERENCE | INTERNAL RESEARCH REVIEW',35,bold=True)
    label(d,(24,73),'PennMicro video KXDHa39Z-2s | actual decoded frames | 21-point estimates, not measured 3D pose',23,(150,175,200))
    visual_panel(canvas,near(216),24,125,'ACCEPT  216.007 s | source frame',
        'Source corrects its spoken wording: ULNAR aspect of thumb. Original full frame; other hand holds a pen.',True)
    visual_panel(canvas,near(220),696,125,'ACCEPT  220.011 s | hand ROI + 21 points',
        'Correct grip context. Thumb/index image placement is a weak prior; hidden joints and depth remain uncertain.',True,True,True)
    visual_panel(canvas,near(228),1368,125,'ACCEPT  228.019 s | hand ROI + 21 points',
        '7 accepted frames total: 216-228 s. Middle-finger support is taught; do not infer robot contact force from pixels.',True,True,True)
    visual_panel(canvas,near(234),24,660,'EXCLUDE  233.984 s | counterexample',
        'Instructor is explaining an incorrect pad-to-pad grip. High handedness score does not make this a positive example.',False)
    visual_panel(canvas,near(240),696,660,'EXCLUDE  239.990 s | ROI + 21 points',
        'Do not average this counterexample into the positive prior. Original keypoints retained only for audit.',False,True,True)
    d=ImageDraw.Draw(canvas);x=1368;y=660
    d.rounded_rectangle((x,y,x+648,y+510),radius=14,fill=(19,29,42),outline=(49,66,84),width=2)
    label(d,(x+18,y+18),'QUALITY / INTERPRETATION',25,(255,200,110),True)
    text=[
      '2D: image-projected bend proxies only. Camera viewpoint and occlusion limit anatomical interpretation.',
      '3D: reject strong supervision. Index DIP estimates fold to 110.8-157.3 deg, versus 27.8-34.6 deg in the image.',
      'Confidence is LEFT/RIGHT classification, not point accuracy. No calibrated per-point occlusion score is available.',
      'Robot precision_v2 starts with two geometric contacts. Middle support must be demonstrated dynamically.'
    ]
    yy=y+66
    for t in text:
        yy=paragraph(d,(x+18,yy),t,width=46,size=20,gap=27)+19
    d=ImageDraw.Draw(canvas)
    label(d,(24,1200),'ANNOTATION LEGEND',20,(150,175,200),True)
    for i,(finger,color) in enumerate(zip(CHAINS,COLORS)):
        label(d,(330+i*220,1200),finger.upper(),20,color,True)
    paragraph(d,(24,1243),'ROI boxes are (x0,y0,x1,y1)=(300,80,880,660) in the original 1280x720 frame. Images are local research references only; no source video or collage has been sent externally.',width=153,size=21,gap=29)
    canvas.save(output)

def main():
    REPORT.mkdir(exist_ok=True)
    data=load(HUMAN/'landmarks.json');manifest=load(HUMAN/'reference_manifest.json')
    config_path=ROOT/'configs/human_grasp_precision_v2.json';config=load(config_path)
    evidence=config['human_evidence'];frames=data['frames']
    positive=[r for r in frames if r.get('usable_for_retargeting') and r.get('hand_role')=='tweezer_hand']
    checks=[]
    def check(name, passed, details):
        checks.append({'check':name,'passed':bool(passed),'details':details})
    actual_sha=sha(HUMAN/'landmarks.json')
    check('landmark_payload_hash',actual_sha==evidence['landmarks_sha256'],actual_sha)
    for source in manifest['sources']:
        path=ROOT/source['local_video'];actual=sha(path)
        check('video_hash_'+source['video_id'],actual==source['video_sha256'],{'bytes':path.stat().st_size,'sha256':actual})
    check('config_video_hash',evidence['video_sha256']==data['video_sha256'],evidence['video_sha256'])
    check('model_hash',sha(HUMAN/'models/hand_landmarker.task')==manifest['estimation']['model_sha256'],manifest['estimation']['model_sha256'])
    expected=[r['frame_index'] for r in positive]
    check('selected_positive_frames',evidence['selected_frame_indices']==expected,expected)
    check('selected_count',evidence['selected_frame_count']==len(positive)==7,len(positive))
    check('counterexamples_excluded',all(not r.get('usable_for_retargeting') for r in frames if 233<=r['time_s']<=241),'233.984-239.990 s excluded')
    check('strong_world_supervision_disabled',all(not r['world_3d_used_for_strong_supervision'] for r in evidence['frame_diagnostics']),'All 7 config diagnostics mark false')
    check('frame_paths_exist',all((ROOT/r[k]).is_file() for r in positive for k in ['frame_path','annotated_frame_path']),'7 originals and 7 annotated frames')
    # Re-decode source frame indices independently to check that stored JPEGs
    # correspond to actual source video, allowing normal lossy JPEG differences.
    cap=cv2.VideoCapture(str(ROOT/data['video_path']));jpeg_errors=[]
    for r in positive:
        cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame_index']);ok,bgr=cap.read()
        if not ok:raise RuntimeError('Failed to decode source frame')
        raw=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(float)
        jpg=np.asarray(Image.open(ROOT/r['frame_path']).convert('RGB')).astype(float)
        jpeg_errors.append(float(np.mean(np.abs(raw-jpg))))
    cap.release()
    check('keyframes_match_source_video',max(jpeg_errors)<5,{'mean_absolute_RGB_error_0_255':jpeg_errors,'threshold':5,'reason':'Allowance for JPEG compression; not a pose-error metric'})
    # Recompute proxies and the disclosed robot mapping without calling the
    # implementation under audit. No MediaPipe world depth enters this mapping.
    width,height=data['resolution'];raw_priors=[];bounded_priors=[];angle_error=0;clip_counts={n:0 for n in config['q_reference']}
    diag_by_time={r['time_s']:r for r in evidence['frame_diagnostics']}
    for r in positive:
        xy=np.asarray(r['landmarks_normalized'])[:,:2]*[width,height];angles=proxy_angles(xy);q={}
        thumb_names=['right_hand_thumb_bend_joint','right_hand_thumb_rota_joint1','right_hand_thumb_rota_joint2']
        for n,a,g in zip(thumb_names,angles['thumb'],[.35,1.,.5]):q[n]=.25*a+.75*g
        q['right_hand_index_bend_joint']=0.
        for name in ['index','middle','ring','pinky']:
            stem='mid' if name=='middle' else name
            q[f'right_hand_{stem}_joint1']=angles[name][0]
            q[f'right_hand_{stem}_joint2']=angles[name][1]+.5*angles[name][2]
        bounded={n:float(np.clip(v,*config['soft_bounds'][n])) for n,v in q.items()}
        for n in q:clip_counts[n]+=int(abs(q[n]-bounded[n])>1e-12)
        raw_priors.append(q);bounded_priors.append(bounded)
        diagnostic=diag_by_time[r['time_s']]['measured_finger_angles']
        for name,values in angles.items():
            actual=diagnostic['mid' if name=='middle' else name]
            for value,key in zip(values,['mcp_flexion_rad','pip_flexion_rad','dip_flexion_rad']):angle_error=max(angle_error,abs(value-actual[key]))
    medians={n:float(np.median([q[n] for q in bounded_priors])) for n in config['q_reference']}
    stds={n:float(np.std([q[n] for q in bounded_priors])) for n in config['q_reference']}
    prior_error=max(abs(medians[n]-config['human_joint_prior'][n]) for n in medians)
    std_error=max(abs(stds[n]-config['human_joint_prior_std'][n]) for n in medians)
    check('pixel_proxy_angles_recomputed',angle_error<1e-12,{'max_abs_error_rad':angle_error})
    check('bounded_prior_medians_recomputed',prior_error<1e-12,{'max_abs_error_rad':prior_error})
    check('bounded_prior_std_recomputed',std_error<1e-12,{'max_abs_error_rad':std_error})
    check('q_closed_consistency',max(abs(config['q_closed'][n]-config['q_reference'][n]-config['closure_delta'][n]) for n in medians)<1e-12,'q_closed = q_reference + closure_delta')
    qmargin={n:float(min(q-lo,hi-q)/(hi-lo)) for n,q in config['q_reference'].items() for lo,hi in [config['soft_bounds'][n]]}
    delta={n:float(config['q_reference'][n]-medians[n]) for n in medians}
    rms=float(np.sqrt(np.mean(np.square(list(delta.values())))))
    check('refinement_delta_consistency',max(abs(delta[n]-config['joint_refinement_delta'][n]) for n in medians)<1e-12,'Geometric refinement separated from human-image proxy')
    R=np.asarray(config['T_hand_tool'])[:3,:3]
    check('tool_rotation_valid',np.linalg.norm(R.T@R-np.eye(3))<1e-10 and abs(np.linalg.det(R)-1)<1e-10,{'determinant':float(np.linalg.det(R))})
    findings=[
      {'severity':'info','topic':'provenance','finding':'Source, video/model hashes, selected correct-grip frames, 2D proxy mapping and disclosed prior blending are independently reproducible.'},
      {'severity':'important','topic':'third_contact','finding':'The source teaches a three-digit grip. precision_v2 verifies only initial thumb/index geometric contacts, explicitly not middle support. This is a two-contact initialization inspired by the video until dynamic third-contact evidence exists.'},
      {'severity':'important','topic':'weak_depth','finding':'MediaPipe world depth is rejected as strong supervision. Its canonicalized 3D points remain in diagnostics only; they must not be reinterpreted later as a target 3D trajectory.'},
      {'severity':'clarification','topic':'terminology','finding':'method="measured human landmark prior" and measured_finger_angles can overstate accuracy. Prefer "image-estimated 2D bend proxies + disclosed robot regularization + geometric contact fitting".'},
      {'severity':'clarification','topic':'weights','finding':'Ring/pinky prior_weights=1.5 exceed active-digit weights=.6 despite weak visual evidence. Treat them as hand-authored posture-stability weights, not confidence or human-imitation accuracy weights.'},
      {'severity':'clarification','topic':'clipping','finding':'Joint-prior standard deviations are computed after comfort-bound clipping. Ring/pinky joint2 std=0 reflects all 7 values saturating a bound, not certainty about human joints.'},
      {'severity':'clarification','topic':'thumb_contact','finding':'Video explicitly corrects thumb contact to its ulnar aspect; the fitter describes a robot thumb-pad/lateral-index contact. Robot surface coordinates alone do not prove anatomical equivalence; compare matched-view contact location before claiming exact imitation.'},
      {'severity':'clarification','topic':'authored_geometry','finding':'Tool-to-hand transform and 35 mm longitudinal contact offset come from geometry fitting/candidate design, not calibrated measurement of the human video.'},
      {'severity':'limitation','topic':'motion','finding':'Seven nearly static frames are posture evidence, not a full acquisition, squeeze-force or dynamic trajectory demonstration.'},
    ]
    metrics=[
      {'name':'matched_view_2d_landmark_error','definition':'Weighted median/mean image-point distance after one scale+rotation+translation alignment, divided by palm length; thumb/index first, occluded fingers reported separately. Keep the same correspondence and one fixed virtual camera per comparison.','requires':'Rendered robot joint landmarks and a documented source-comparable camera.','interpretation':'View-dependent silhouette/posture similarity only; no 3D anatomical accuracy claim.'},
      {'name':'projected_bend_proxy_error','definition':'Per-finger absolute difference of projected adjacent-segment bend angles, in degrees. Report all joints and robust aggregate; map robot coupled joints explicitly.','requires':'Robot projected chain landmarks; raw human 2D proxy distribution.','interpretation':'Perspective/occlusion can change it. Do not compare robot intrinsic joint values directly against human projected angles.'},
      {'name':'contact_layout_and_support_duty','definition':'Fraction of post-reset lift/hold steps with thumb and index contact on opposing arms; separately fraction with measured middle-finger tool support above a documented noise threshold.','requires':'Per-finger per-arm contact forces and reset-fixture flag.','interpretation':'Tests the source-inspired three-digit mechanism; no initial middle contact is currently certified.'},
      {'name':'active_grasp_contact_position','definition':'Contact centroid position along each finger/tool normalized by corresponding link/tool length, plus opposed-normal dot product and longitudinal offset.','requires':'Dynamic contact points/normals; matched-view qualitative human contact region.','interpretation':'Contact regions can be compared qualitatively; do not claim human millimetre accuracy from this uncalibrated video.'},
      {'name':'comfort_margin_and_noncontact_penetration','definition':'Minimum normalized distance to configured joint comfort bounds; count/time of noncontact-finger tool penetration or self-collision; report actual PhysX contact checks separately from vertex sampling.','requires':'Robot joint trajectory and contact/collision queries.','interpretation':'Robot comfort/plausibility measure, not proof of human naturalness. precision_v2 index_joint2 starts at its configured upper comfort bound.'},
      {'name':'tool_slip_and_grip_force','definition':'Post-reset tool displacement/rotation relative to the hand, tool drops, force median/95th percentile, and contact switching frequency during steady hold.','requires':'Hand/tool poses, finger forces and timestamps.','interpretation':'Measures stable economical manipulation; the human video supplies no force ground truth.'},
      {'name':'motion_smoothness','definition':'Joint speed/acceleration/jerk RMS and 95th percentile, and command oscillation spectral energy during steady hold, using one fixed sampling/filter procedure.','requires':'Robot trajectory sampled fast enough for dynamics.','interpretation':'Report per task phase; seven human keyframes cannot calibrate a human motion-spectrum target.'},
      {'name':'provenance_separated_pose_deviation','definition':'Report q_reference minus the mapped 2D prior separately from q_policy minus q_reference; distinguish geometric fitting, authored regularization, and learned adaptation.','requires':'Current config and robot policy trajectory.','interpretation':'A smaller difference from a noisy prior is not automatically more natural; show alongside contact success and visual scores.'},
    ]
    collage=HUMAN/'human_reference_audit_collage.png';make_collage(frames,collage)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only independent evidence/config audit; no RL or simulation code modified; collage internal only',
      'config_path':str(config_path.relative_to(ROOT)),'config_sha256':sha(config_path),'reference_manifest':'references/human/reference_manifest.json',
      'collage':str(collage.relative_to(ROOT)),'checks':checks,'all_checks_passed':all(c['passed'] for c in checks),
      'findings':findings,'quantitative_config_observations':{'prior_clip_counts_out_of_7':clip_counts,'joint_refinement_delta_rad':delta,'refinement_rms_rad':rms,'refinement_rms_deg':float(np.rad2deg(rms)),'normalized_comfort_margins':qmargin,'reference_quality_from_config_not_independently_dynamic_verified':config['reference_quality']},
      'recommended_metrics':metrics,'overall_assessment':'Evidence chain is accurate and reproducible for an image-derived weak posture prior. Naturalness and human-equivalent three-point support are not established solely by this configuration; they require matched-view visual and post-reset dynamic checks.'}
    (REPORT/'human_reference_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'all_checks_passed':report['all_checks_passed'],'check_count':len(checks),'collage':str(collage),'report':str(REPORT/'human_reference_audit.json'),'rms_refinement_deg':float(np.rad2deg(rms))},indent=2))

if __name__=='__main__':main()
