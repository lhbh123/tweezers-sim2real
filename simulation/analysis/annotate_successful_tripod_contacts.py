#!/usr/bin/env python3
"""Annotate a real successful-policy frame using same-substep contact positions.

This does not change the earlier zero-action hold annotation or any raw image.
"""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'));sys.path.insert(0,str(ROOT/'src'))
from annotate_actual_tripod_contacts import camera_from_usd,project,rotation,sha,arrow
from kinematics import Kinematics,ASSET,stl_triangles


def choose_contacts(row):
    raw=row['grasp_contact_points'];selected=[]
    for role in ['thumb','index','middle','web']:
        candidates=[]
        for c in raw['normal_contacts']:
            match=(c['role']==role) if role!='web' else c['tool']=='handle' and c.get('within_web_region')
            if match and np.linalg.norm(c['force_on_tool_N'])>1e-6:
                if role=='middle' and c.get('radial_offset_world_z_m',0)>=0:continue
                candidates.append(c)
        if not candidates:return None
        selected.append(dict(label=role,**max(candidates,key=lambda c:np.linalg.norm(c['force_on_tool_N']))))
    # Verify upward resultant in this exact substep separately from the HUD means.
    forces={'middle':np.zeros(3),'web':np.zeros(3)}
    for c in raw['normal_contacts']+raw.get('friction_anchors',[]):
        if c['role']=='middle':forces['middle']+=c['force_on_tool_N']
        if c['tool']=='handle' and c.get('within_web_region'):forces['web']+=c['force_on_tool_N']
    return selected,{k:v.tolist() for k,v in forces.items()}


def frame_image(run,name,index):
    out=run/'visual_review'/f'{name}_{index:04d}.png';out.parent.mkdir(exist_ok=True)
    if not out.exists():
        code='import sys,imageio.v2 as i;from PIL import Image;r=i.get_reader(sys.argv[1]);Image.fromarray(r.get_data(int(sys.argv[2]))).save(sys.argv[3]);r.close()'
        subprocess.run([str(ROOT/'.venv-human/bin/python'),'-c',code,str(run/(name+'.mp4')),str(index),str(out)],check=True)
    return out,Image.open(out).convert('RGB')


def cube_check(image,row,camera):
    cube=np.asarray(row['cube']);v=np.array([[x,y,z] for x in [-.006,.006] for y in [-.006,.006] for z in [-.006,.006]])
    uv=project(v@rotation(cube[3:7]).T+cube[:3],camera);expected=np.r_[uv.min(0),uv.max(0)]
    # Other parallel environments appear in the background. Restrict color
    # validation to the independently projected cube bbox plus 8px padding.
    roi=np.r_[np.floor(expected[:2]-8),np.ceil(expected[2:]+8)].astype(int)
    roi[:2]=np.maximum(roi[:2],0);roi[2:]=np.minimum(roi[2:],image.size)
    a=np.array(image)[roi[1]:roi[3],roi[0]:roi[2]].astype(float);r,g,b=a.transpose(2,0,1)
    y,x=np.nonzero((r>140)&(r-g>25)&(r-b>70)&(g>b*1.15))
    if len(x)<20:raise ValueError('Too few cube pixels in projected validation ROI')
    observed=np.array([x.min()+roi[0],y.min()+roi[1],x.max()+roi[0],y.max()+roi[1]])
    error=observed-expected
    if max(abs(error))>3:raise ValueError(f'Cube projection mismatch {error}; no annotation allowed')
    return dict(expected_bbox=expected.tolist(),observed_color_bbox=observed.tolist(),
                color_check_roi_xyxy=roi.tolist(),edge_error_px=error.tolist(),max_abs_edge_error_px=float(max(abs(error)))),uv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/ur5e_xhand_tripod_rl_contact_points.png')
    p.add_argument('--first-frame',type=int,default=130);p.add_argument('--last-frame',type=int,default=144)
    args=p.parse_args();run=args.run.resolve();task=json.loads((run/'task.json').read_text());ev=json.loads((run/'evaluation.json').read_text())
    flags=ev.get('success_per_env');assert isinstance(flags,list) and len(flags)==ev['episodes'] and flags[0] is True
    assert sum(flags)==ev['successes']
    validation=json.loads((run/'rl_validation.json').read_text());assert validation['success'] is True
    rows=json.loads((run/'trajectory.json').read_text());dt=task['physics_dt']*task['decimation'];stride=round(1/(30*dt))
    if 'step % max(1, round(1/(30*env.dt))) == 0' not in (run/'train_rl.py').read_text():raise ValueError('Unverified frame cadence')
    assert stride==2 and np.isclose(dt,1/60)
    camera=camera_from_usd(run,'side');choices=[];rejected=[]
    for frame in range(args.first_frame,min(args.last_frame+1,(len(rows)+1)//stride)):
        row=rows[frame*stride]
        if row['cube_lift_m']<.9*task['target_lift'] or not row['grasp_support_acceptable']:continue
        contacts=choose_contacts(row)
        if contacts is None:continue
        selected,forces=contacts
        if min(forces['middle'][2],forces['web'][2])<=task.get('support_vertical_threshold_N',.001):continue
        choices.append((row['cube_lift_m'],frame,selected,forces))
    if not choices:raise ValueError('No valid same-substep four-role successful-lift contact frame; refusing to guess')
    for _,frame,selected,forces in sorted(choices,reverse=True):
        row=rows[frame*stride];raw_path,raw=frame_image(run,'side',frame)
        try:projection,cube_uv=cube_check(raw,row,camera)
        except ValueError as exc:rejected.append(dict(frame=frame,reason=str(exc)));continue
        break
    else:raise ValueError(f'No camera-validated candidate: {rejected}')
    uv=project([c['point_world'] for c in selected],camera)
    for c,point in zip(selected,uv):c['pixel_xy']=point.tolist();c['visibility']='Physical collider-contact projection; may be hidden behind visual surfaces.'
    text=(run/'final_scene.usda').read_text();m=re.search(r'def Xform "env_0"\s*\{.*?double3 xformOp:translate = \(([^)]+)\)',text,re.S)
    origin=np.fromstring(m.group(1),sep=',');kin=Kinematics();fk=kin.fk(dict(zip(task['joint_names'],row['joint_pos'])))
    mask=Image.new('L',raw.size);md=ImageDraw.Draw(mask);all_uv=[cube_uv,uv]
    for link in kin.root.findall('link'):
        name=link.get('name')
        if not name.startswith('right_hand_'):continue
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            tri=stl_triangles(ASSET/mesh.get('filename'));scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            o=visual.find('origin');attrs={} if o is None else o.attrib
            R=Rotation.from_euler('xyz',np.fromstring(attrs.get('rpy','0 0 0'),sep=' ')).as_matrix();t=np.fromstring(attrs.get('xyz','0 0 0'),sep=' ')
            tri=(tri*scale)@R.T+t;points=tri@fk[name][:3,:3].T+fk[name][:3,3]+origin+[0,0,.72]
            projected=project(points.reshape(-1,3),camera);all_uv.append(projected)
            for triangle in projected.reshape(-1,3,2):md.polygon([tuple(p) for p in triangle],fill=255)
    all_uv=np.concatenate(all_uv);lo=np.floor(all_uv.min(0)-[35,30]).astype(int);hi=np.ceil(all_uv.max(0)+[35,30]).astype(int)
    lo=np.maximum(lo,[0,0]);hi=np.minimum(hi,raw.size);crop=tuple(np.r_[lo,hi])
    edge=np.asarray(mask).astype(int)-np.asarray(mask.filter(ImageFilter.MinFilter(3))).astype(int);debug=np.array(raw);debug[edge>0]=[0,190,90]
    diagnostic=run/'visual_review'/f'success_contact_source_hand_projection_{frame:04d}.png';Image.fromarray(debug).save(diagnostic)
    font=lambda n:ImageFont.truetype(str(ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf'),n)
    canvas=Image.new('RGB',(2400,1530),'#f4f6f9');d=ImageDraw.Draw(canvas)
    d.text((40,20),'成功抬升回合：真实四角色接触点投影',font=font(46),fill='#17243a')
    d.text((40,83),f'BC 初始化 + PPO 25 轮 · 选中 env0 · t={row["time"]:.6f}s · 方块抬升 {1000*row["cube_lift_m"]:.1f}mm',font=font(30),fill='#006b72')
    d.text((40,145),'原始水平侧视帧（背景为其他并行环境）',font=font(29),fill='#17243a')
    d.text((1220,145),'同帧动态裁切放大 + 日志坐标投影',font=font(29),fill='#17243a')
    canvas.paste(raw.resize((1140,855),Image.Resampling.LANCZOS),(40,195))
    scale=min(1140/(hi[0]-lo[0]),855/(hi[1]-lo[1]));size=tuple(np.rint((hi-lo)*scale).astype(int));offset=np.array([1220+(1140-size[0])//2,195+(855-size[1])//2])
    canvas.paste(raw.crop(crop).resize(size,Image.Resampling.LANCZOS),tuple(offset))
    colors={'thumb':'#b22d52','index':'#2366a9','middle':'#ad741c','web':'#7753af'}
    texts={'thumb':'拇指','index':'食指','middle':'中指承托接触','web':'镊尾—虎口 ROI'}
    # Only label layout is chosen on the page. Every marker uses source physics coordinates.
    for c in selected:
        role=c['label'];point=(np.asarray(c['pixel_xy'])-lo)*scale+offset
        if role=='web':label=(1240,212);start=(1500,254)
        elif role=='index':label=(1240,max(360,min(720,int(point[1]-70))));start=(1350,label[1]+42)
        elif role=='thumb':label=(2200,max(320,min(650,int(point[1]-150))));start=(2195,label[1]+45)
        else:label=(2040,min(970,int(point[1]+150)));start=(2040,label[1]+30)
        color=colors[role];arrow(d,start,tuple(point),color);d.ellipse((point[0]-9,point[1]-9,point[0]+9,point[1]+9),outline=color,width=4)
        bbox=d.textbbox(label,texts[role],font=font(31));d.rounded_rectangle((bbox[0]-7,bbox[1]-4,bbox[2]+7,bbox[3]+4),6,fill='#f4f6f9');d.text(label,texts[role],font=font(31),fill=color)
    g=row['grasp_support'];middle=g['role_support_z_N'][2];web=g['web_normal_force_w'][2]+g['web_friction_force_w'][2]
    lines=[f'16.7ms 控制周期平均净向上支撑：中指 {middle:.3f}N；镊尾—虎口 {web:.3f}N。',
           '四个圆圈来自同一物理子步的实际接触点；箭头只指位置，不代表力方向，接触面可能被遮挡。',
           f'相机参数直接取存档 USD；用实际方块姿态校验像素边缘，最大差 {projection["max_abs_edge_error_px"]:.2f}px，未拟合校准。',
           '右图只对原帧裁切、缩放和加注；未补画手、镊子、方块或遮挡部分。',
           '这是选中的成功回合，不代表总体成功率；单帧四角色接触不代表每个物理子步都持续四点接触。',
           '几何与受力日志用于说明机器人接触；人类参考图不提供精确三维姿态或接触力真值。']
    for i,line in enumerate(lines):d.text((40,1080+i*64),line,font=font(28),fill='#536179')
    args.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(args.output)
    report=dict(run=str(run),recorded_environment_index=0,population_episodes=ev['episodes'],population_successes=ev['successes'],
                selected_env_success=flags[0],checkpoint=ev['checkpoint'],selected_frame=frame,trajectory_row=frame*stride,
                physics_time_s=row['time'],cube_lift_m=row['cube_lift_m'],same_final_physics_substep=True,
                candidate_range_frames=[args.first_frame,args.last_frame],rejected_projection_candidates=rejected,
                selected_raw_contact_points=selected,selected_substep_net_forces_on_tool_N=forces,
                control_period_mean_upward_support_N=dict(middle=middle,web=web),projection_check=projection,
                environment_origin_from_USD_m=origin.tolist(),camera=dict(name='side',eye=camera['eye'].tolist(),R=camera['R'].tolist(),fx=camera['fx'],fy=camera['fy']),
                source_files={str(run/n):sha(run/n) for n in ['trajectory.json','evaluation.json','task.json','rl_validation.json','final_scene.usda','side.mp4']},
                raw_frame=str(raw_path),raw_frame_sha256=sha(raw_path),crop_source_xyxy=list(map(int,crop)),display_scale=float(scale),
                hand_projection_diagnostic=str(diagnostic),output=str(args.output),output_sha256=sha(args.output),
                new_script_sha256=sha(__file__),projection_helper_sha256=sha(ROOT/'analysis/annotate_actual_tripod_contacts.py'),
                physical_state_changed=False,photos_repainted=False,contacts_guessed_from_pixels=False,
                raw_images_preserved=True,external_send_performed=False,
                limitations=['Contacts belong to the physics collision model, not necessarily the exact visible surface.',
                             'Markers can project to visually hidden surfaces; no occluded pixels are reconstructed.',
                             'Contact strength and instantaneous support here do not prove uninterrupted four-point contact over the whole episode.'])
    args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),frame=frame,time=row['time'],lift_m=row['cube_lift_m'],projection=projection),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
