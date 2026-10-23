#!/usr/bin/env python3
"""Project actual same-substep broad-face contacts/normals onto a real camera frame.

No contacts are guessed from pixels. The final physical substep must contain
qualified thumb and index contacts at once; earlier substeps cannot be overlaid
on a later rendered state. Failed cube-projection checks refuse annotation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from annotate_actual_tripod_contacts import camera_from_usd,project,rotation,sha,arrow
from annotate_successful_tripod_contacts import frame_image,cube_check


def classify(contact,body,task):
    p=np.asarray(contact['point_world'],float);force=np.asarray(contact['force_on_tool_N'],float)
    if p.shape!=(3,) or force.shape!=(3,) or not np.isfinite(np.r_[p,force]).all():
        raise ValueError('Invalid raw contact')
    R=rotation(body[3:7]);local=(p-np.array(body[:3]))@R;flocal=force@R
    sign=-1 if contact['tool']=='left' else 1
    geo=task['tool_geometry'];criteria=task['face_pinch_criteria']
    tip=geo.get('tip_length_m',.015);blade=geo['length_m']-tip;is_tip=local[2]>blade
    width=min(.005,geo['strip_width_m']) if is_tip else geo['strip_width_m']
    z0,z1=(blade,geo['length_m']) if is_tip else (0.,blade)
    margin=width/2-abs(local[1]);axial=min(local[2]-z0,z1-local[2])
    plane=abs(sign*local[0]-geo['strip_thickness_m']/2)
    inward=-sign*flocal[0];alignment=inward/max(np.linalg.norm(flocal),1e-12)
    qualified=bool(margin>=criteria['width_margin_m'] and axial>=criteria['axial_margin_m']
        and plane<=criteria['face_tolerance_m'] and alignment>=criteria['min_inward_cos']
        and inward>1e-8 and (not is_tip or criteria['allow_tip_contact']))
    if not np.allclose(local,contact['point_jaw_local_m'],atol=8e-7,rtol=1e-5):
        raise ValueError('Contact local point does not match actual jaw body pose')
    if not np.allclose(flocal,contact['force_jaw_local_N'],atol=2e-6,rtol=1e-4):
        raise ValueError('Contact local force does not match actual jaw orientation')
    if qualified!=contact['broad_face_qualified']:
        raise ValueError('Raw qualification disagrees with independently recomputed geometry/normal')
    return dict(qualified=qualified,width_margin_m=float(margin),axial_margin_m=float(axial),
        outer_plane_error_m=float(plane),inward_alignment_cos=float(alignment),
        inward_force_N=float(inward),point_jaw_local_m=local.tolist(),
        outer_face_normal_world=(R@np.array([sign,0,0])).tolist(),
        actual_normal_force_world_N=force.tolist())


def choose(record,task):
    raw=record['grasp_contact_points'];substeps=record.get('grasp_contact_substeps')
    if not substeps or len(substeps)!=task['decimation']:
        raise ValueError('Complete physical-substep records required')
    final=substeps[-1]
    for body in ['tool','left','right']:
        if not np.allclose(record[body],final[body],atol=1e-7,rtol=1e-7):
            raise ValueError('Rendered final state differs from final physical substep')
    if raw!=final['grasp_contact_points']:
        raise ValueError('Final raw contact record differs from final-substep evidence')
    selected=[]
    for role in ['thumb','index']:
        candidates=[]
        for c in raw['normal_contacts']:
            if c['role']!=role or c['tool']!=task['face_pinch_assignment'][role]:continue
            computed=classify(c,record[c['tool']],task)
            if computed['qualified']:candidates.append(dict(c,independent_face_check=computed))
        if not candidates:return None
        selected.append(max(candidates,key=lambda c:c['independent_face_check']['inward_force_N']))
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--camera',default='side')
    p.add_argument('--first-frame',type=int,default=0);p.add_argument('--last-frame',type=int)
    p.add_argument('--min-lift-mm',type=float,default=0.)
    p.add_argument('--require-success',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run.resolve()
    task=json.loads((run/'task.json').read_text());records=json.loads((run/'trajectory.json').read_text())
    ev=json.loads((run/'evaluation.json').read_text());flags=ev.get('success_per_env')
    selected_success=(flags[0] if isinstance(flags,list) and len(flags)==ev['episodes']
                      and all(type(v) is bool for v in flags) and sum(flags)==ev['successes']
                      else ev.get('episodes')==1 and ev.get('successes')==1)
    if args.require_success and not selected_success:p.error('Recorded env0 is not explicitly successful')
    if task.get('require_face_pinch') is not True or task.get('require_tripod_support'):
        p.error('Require revised face-pinch task; old continuous-support mode not accepted')
    geo=task['tool_geometry'];extension=geo.get('handle_extension')
    if extension and any(abs(extension[k][j])>1e-8 for k in ['start_m','end_m'] for j in [0,1]):
        p.error('Bent-tail geometry cannot serve as revised straight-tail demonstration')
    capture='step % max(1, round(1/(30*env.dt))) == 0'
    if capture not in (run/'train_rl.py').read_text():p.error('Unknown capture cadence')
    dt=task['physics_dt']*task['decimation'];stride=max(1,round(1/(30*dt)))
    indices=list(range(0,len(records),stride));last=len(indices)-1 if args.last_frame is None else args.last_frame
    if not 0<=args.first_frame<=last<len(indices):p.error('Requested frame range outside recorded cadence')
    camera=camera_from_usd(run,args.camera)
    selected=None;rejected=[]
    candidates=sorted(range(args.first_frame,last+1),key=lambda i:records[indices[i]]['cube_lift_m'],reverse=True)
    for frame in candidates:
        row=records[indices[frame]]
        if row['cube_lift_m']*1000<args.min_lift_mm or not row.get('face_pinch_acceptable'):continue
        contacts=choose(row,task)
        if contacts is None:continue
        raw_path,raw=frame_image(run,args.camera,frame)
        if raw.size!=(960,720):raise ValueError('Projection helper requires actual960x720 raw cameras')
        try:projection,cube_uv=cube_check(raw,row,camera)
        except ValueError as exc:
            rejected.append(dict(frame=frame,reason=str(exc)));continue
        selected=contacts;break
    if selected is None:raise RuntimeError('No same-final-substep two-face frame with validated cube projection; no annotation produced')
    points=np.array([c['point_world'] for c in selected]);uv=project(points,camera)
    arrows=[]
    for c,pixel in zip(selected,uv):
        check=c['independent_face_check'];normal=np.array(check['outer_face_normal_world'])
        force=np.array(c['force_on_tool_N']);force/=np.linalg.norm(force)
        pair=project([np.array(c['point_world'])+.012*normal,np.array(c['point_world'])+.012*force],camera)
        c['pixel_xy']=pixel.tolist();arrows.append(pair)
    allpoints=np.concatenate([uv,*arrows])
    lo=np.maximum(np.floor(allpoints.min(0)-[100,125]).astype(int),[0,0])
    hi=np.minimum(np.ceil(allpoints.max(0)+[100,125]).astype(int),raw.size)
    crop=tuple(np.r_[lo,hi]);scale=min(1140/(hi[0]-lo[0]),855/(hi[1]-lo[1]))
    size=tuple(np.rint((hi-lo)*scale).astype(int));offset=np.array([1220+(1140-size[0])//2,195+(855-size[1])//2])
    font=lambda n:ImageFont.truetype(str(ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'),n)
    canvas=Image.new('RGB',(2400,1540),'#f4f6f9');draw=ImageDraw.Draw(canvas)
    draw.text((40,20),'双指宽面夹持：真实接触点与表面法向投影',font=font(46),fill='#17243a')
    status='选中成功回合' if selected_success else '诊断回合（不作成功声明）'
    draw.text((40,84),f'{status} · env0 · t={row["time"]:.6f}s · 抬升 {row["cube_lift_m"]*1000:.1f}mm',font=font(30),fill='#006b72')
    draw.text((40,145),'未加注的实际相机原帧 · '+args.camera,font=font(29),fill='#17243a')
    draw.text((1220,145),'同帧裁切放大；投影点可能被手指／镊臂遮挡',font=font(29),fill='#17243a')
    canvas.paste(raw.resize((1140,855),Image.Resampling.LANCZOS),(40,195))
    canvas.paste(raw.crop(crop).resize(size,Image.Resampling.LANCZOS),tuple(offset))
    colors={'thumb':'#b22d52','index':'#2366a9'}
    for c,pair in zip(selected,arrows):
        point=(np.array(c['pixel_xy'])-lo)*scale+offset
        role=c['role'];color=colors[role];targets=(pair-lo)*scale+offset
        for target,col in zip(targets,['#b17b0a','#00805d']):
            if np.linalg.norm(target-point)>3:arrow(draw,tuple(point),tuple(target),col,width=5)
        draw.ellipse((point[0]-10,point[1]-10,point[0]+10,point[1]+10),outline=color,width=4)
        label=(1240,218) if role=='thumb' else (2040,950)
        text=('拇指' if role=='thumb' else '食指')+' → '+('左' if c['tool']=='left' else '右')+'臂外宽面'
        arrow(draw,(label[0]+100,label[1]+40),tuple(point),color,width=3)
        draw.text(label,text,font=font(31),fill=color)
    text_y=1080
    for c in selected:
        m=c['independent_face_check'];role='拇指' if c['role']=='thumb' else '食指'
        draw.text((40,text_y),f'{role}：Y边缘余量 {m["width_margin_m"]*1000:.2f}mm；外宽面误差 {m["outer_plane_error_m"]*1000:.3f}mm；内压力 {m["inward_force_N"]:.3f}N；法向对齐 cos={m["inward_alignment_cos"]:.3f}',font=font(28),fill=colors[c['role']]);text_y+=58
    lines=[
        '金色箭头=实际镊臂外宽面法线；绿色箭头=该点实测法向力方向（均固定12mm示意，不按力大小缩放）。',
        '两点均来自相机拍摄时同一末物理子步；没有用早先子步点覆盖后来的手姿。中指／虎口仅备用，不要求持续接触。',
        f'相机按存档 USD 投影；实际方块像素边界最大误差 {projection["max_abs_edge_error_px"]:.2f}px。局部表面可能遮挡，投影不表示可见。',
        '接触点属于物理碰撞模型；是否落在真实源 STL 指腹仍需独立几何审计。此图没有补画隐藏手指或工具。',
        '这是单帧物理证据，不表示每个物理子步都连续双面接触；视频选中回合不等于总体成功率。']
    for line in lines:draw.text((40,text_y),line,font=font(26),fill='#536179');text_y+=58
    args.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(args.output)
    audit=dict(run=str(run),script_sha256=sha(__file__),source_trajectory_sha256=sha(run/'trajectory.json'),
        source_task_sha256=sha(run/'task.json'),source_camera_USD_sha256=camera['source_sha256'],
        raw_video=str(run/(args.camera+'.mp4')),raw_video_sha256=sha(run/(args.camera+'.mp4')),
        raw_frame=str(raw_path),raw_frame_sha256=sha(raw_path),frame=frame,trajectory_index=indices[frame],
        simulation_time_s=row['time'],same_final_physics_substep=True,selected_contacts=selected,
        outward_normal_and_force_arrow_length_m=.012,arrow_length_encodes_force_magnitude=False,
        projection_check=projection,rejected_projection_candidates=rejected,crop_source_xyxy=[int(v) for v in crop],
        source_recorded_env0_success=selected_success,population_episodes=ev['episodes'],population_successes=ev['successes'],
        middle_web_continuity_required=False,source_STL_finger_pad_geometry_verified_by_this_script=False,
        output=str(args.output),output_sha256=sha(args.output),external_send_performed=False,
        limitations=['Camera helper assumes960x720 perspective camera; actual cube check must pass.',
            'Occlusion is not removed or guessed; contact projection can lie behind visible surfaces.',
            'Broad-face evidence uses actual raw collision points, not human reference inference.'])
    args.output.with_suffix('.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'image':str(args.output),'frame':frame,'projection_max_error_px':projection['max_abs_edge_error_px']},ensure_ascii=False))


if __name__=='__main__':main()
