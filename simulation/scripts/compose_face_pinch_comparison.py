#!/usr/bin/env python3
"""Compose real straight-tail broad-face-pinch renders with factual telemetry.

This is separate from the earlier four-role-support compositor. It does not
relabel old bent-tail or continuous-tripod episodes as the revised task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compose_rl_video import frame_alignment, checkpoint_lineage
from compose_tripod_comparison import selected_environment_evidence, controller_provenance, temporal_scope

ROOT = Path(__file__).resolve().parents[1]
BG, INK, MUTED, TEAL, AMBER = '#f3f5f8', '#17243a', '#536179', '#006b72', '#986211'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{label}: expected finite {shape}, got {a.shape}')
    return a


def validate_task(task):
    if task.get('require_face_pinch') is not True:
        raise ValueError('This compositor requires an explicitly recorded face-pinch task')
    if task.get('require_tripod_support'):
        raise ValueError('Continuous tripod support is not the revised task')
    assignment = task.get('face_pinch_assignment')
    if not isinstance(assignment, dict) or set(assignment) != {'thumb', 'index'} or set(assignment.values()) != {'left', 'right'}:
        raise ValueError('Thumb/index must be assigned to distinct recorded jaws')
    geometry = task.get('tool_geometry', {})
    if not all(float(geometry.get(k, 0)) > 0 for k in ['strip_thickness_m', 'strip_width_m', 'length_m']):
        raise ValueError('Recorded tool dimensions are required')
    extension = geometry.get('handle_extension')
    if extension:
        start = array(extension.get('start_m'), (3,), 'tail start')
        end = array(extension.get('end_m'), (3,), 'tail end')
        if np.max(np.abs(start[:2])) > 1e-8 or np.max(np.abs(end[:2])) > 1e-8:
            raise ValueError('Bent/off-axis tail must not be presented as straight-tail media')
    return dict(assignment=assignment, tail_record='No off-axis extension in recorded task geometry',
                actual_render_shape_requires_visual_review=True,
                optional_support_requirement=task.get('middle_web_support_requirement'))


def face_values(record, task):
    """Check displayed gate against actual logged force/count assignment."""
    values = record.get('broad_face_contacts', {})
    force = array(values.get('broad_face_inward_force_N'), (2, 2), 'qualified face force')
    count = array(values.get('broad_face_contact_count'), (2, 2), 'qualified face count')
    torque = array(values.get('broad_face_closing_torque_Nm'), (2, 2), 'qualified contact moment')
    if np.any(force < -1e-8) or np.any(count < 0):
        raise ValueError('Qualified inward force/count cannot be negative')
    assignment = task['face_pinch_assignment']
    pairs = [(('left', 'right').index(assignment[role]), i) for i, role in enumerate(['thumb', 'index'])]
    selected = np.array([force[j, r] for j, r in pairs])
    selected_count = np.array([count[j, r] for j, r in pairs])
    recorded = array(record.get('face_pinch_inward_force_N'), (2,), 'selected thumb/index force')
    if not np.allclose(selected, recorded, rtol=1e-5, atol=1e-7):
        raise ValueError('Selected face forces disagree with per-jaw/role matrix')
    acceptable = bool(np.all(selected > float(task['grasp_contact_threshold_N'])) and np.all(selected_count > 0))
    if type(record.get('face_pinch_acceptable')) is not bool or record['face_pinch_acceptable'] != acceptable:
        raise ValueError('Recorded face gate disagrees with qualified force/count')
    support = record.get('grasp_support', {})
    strengths = support.get('role_contact_strength_N')
    middle = None if strengths is None else float(array(strengths, (5,), 'role contact strength')[2])
    web = support.get('web_contact_count')
    if web is not None and (not np.isfinite(web) or web < 0):
        raise ValueError('Invalid optional web count')
    return dict(force_N=selected.tolist(), count=selected_count.tolist(),
                contact_torque_Nm=[float(torque[j, r]) for j, r in pairs],
                acceptable=acceptable, optional_middle_normal_strength_N=middle,
                optional_web_contact_count=web)


def camera_label(name, cameras):
    camera = next((c for c in cameras if c['name'] == name), None)
    if camera is None:
        return name + ' 原始相机（方向未核验）'
    direction = array(camera['eye_world'], (3,), 'camera eye') - array(camera['target_world'], (3,), 'camera target')
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError('Coincident camera eye and target')
    z = direction[2] / norm
    if z > .999:
        view = '俯视'
    elif z < -.999:
        view = '仰视'
    elif abs(z) < .005:
        view = '水平侧视'
    else:
        view = '斜近景' if name == 'closeup' else '斜视'
    ortho = 'orthographic_width_m' in camera.get('custom_settings', {})
    return ('正交' if ortho else '透视') + view


class Layout:
    def __init__(self, task, evaluation, labels, reference, ref_label, font):
        self.task, self.ev, self.labels = task, evaluation, labels
        self.ref, self.ref_label, self.font_path = reference, ref_label, str(font)
        self.fonts = {}
        self.selection = selected_environment_evidence(evaluation)
        self.controller = controller_provenance(evaluation)
        self.temporal = temporal_scope(task)

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_path, size)
        return self.fonts[size]

    def text(self, draw, pos, text, size=25, color=INK, width=None):
        while width and size > 15 and draw.textlength(text, font=self.font(size)) > width:
            size -= 1
        draw.text(pos, text, font=self.font(size), fill=color)

    def photo(self, canvas, pixels, box):
        x, y, w, h = box
        image = pixels.copy() if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
        factor = min(w/image.width, h/image.height)
        image = image.resize((round(image.width*factor), round(image.height*factor)), Image.Resampling.LANCZOS)
        canvas.paste(image, (x+(w-image.width)//2, y+(h-image.height)//2))

    def frame(self, images, record, speed):
        canvas = Image.new('RGB', (1920, 1280), BG)
        draw = ImageDraw.Draw(canvas)
        values = face_values(record, self.task)
        self.text(draw, (40, 20), '中指下方备用承托：PPO 夹取' if self.task.get('backup_support_geometry') else '直尾镊子：双指外侧宽面夹持', 42)
        status = '选取成功回合，不代表总体成功率' if self.selection['selected_success_episode'] else '诊断回合：未确认任务成功'
        self.text(draw, (40, 78), status, 25, TEAL if self.selection['selected_success_episode'] else AMBER)
        self.text(draw, (1050, 25), self.controller['label'], 26, width=830)
        self.text(draw, (1050, 75), f'仿真 {record["time"]:.3f} s  |  播放 {speed:g}×  |  记录 env0', 25, MUTED)
        for i, (name, pixels) in enumerate(images.items()):
            x = 40 + i*940
            self.text(draw, (x, 127), self.labels[name], 26)
            self.photo(canvas, pixels, (x, 168, 900, 675))
        self.text(draw, (40, 873), 'Penn 原始参考 · ' + self.ref_label, 23, width=510)
        self.photo(canvas, self.ref, (40, 918, 500, 281))
        self.text(draw, (580, 872), f'方块抬升  {float(record["cube_lift_m"])*1000:.1f} mm', 31, TEAL)
        self.text(draw, (1120, 872), '双侧宽面判据：' + ('满足' if values['acceptable'] else '暂未满足'), 29, TEAL if values['acceptable'] else AMBER)
        self.text(draw, (580, 929), '合格宽面法向内压力  拇指 / 食指', 26)
        self.text(draw, (1260, 929), ' / '.join(f'{v:.3f}' for v in values['force_N'])+' N', 29, TEAL)
        self.text(draw, (580, 978), self.temporal['label']+'；计入宽面内部、朝内的法向接触。', 23, MUTED, 1290)
        middle = values['optional_middle_normal_strength_N']
        web = values['optional_web_contact_count']
        present = lambda value: '未记录' if value is None else ('有接触' if value > 0 else '当前无接触')
        self.text(draw, (580, 1023), f'备用承托：中指 {present(middle)}  |  虎口 {present(web)}；不要求持续接触。', 25, MUTED, 1290)
        web_tilt=float(np.rad2deg(np.arccos(np.clip(record['web_side_axis_w'][2],-1,1))))
        self.text(draw, (580, 1070), f'虎口相对朝上倾斜 {web_tilt:.1f}°；拇指、食指分别挤压两片镊臂的外侧宽面。', 23, MUTED, 1290)
        fixture = '任务期间无夹具' if self.task.get('fixture_present') is False and not record.get('fixture_active') else '夹具状态须查原始记录'
        init = '预持镊快照初始化' if self.task.get('reset_snapshot') else '预持镊初始化'
        self.text(draw, (580, 1116), f'直尾镊子 {self.task["tool_geometry"]["length_m"]*1000:.0f} mm  |  当前倾角 {record["tool_down_angle_deg"]:.1f}°  |  '+init, 23, MUTED, 1290)
        self.text(draw, (580, 1161), '参考图用于握法对照；单目图像不能确定精确三维深度。', 23, MUTED, 1290)
        self.text(draw, (40, 1232), 'UR5e + XHand · 实际物理仿真相机画面 · 工具与方块没有固定连接；本回合从预持镊状态开始。', 23, MUTED, 1840)
        return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--output-prefix', default='straight_tail_face_pinch_comparison')
    parser.add_argument('--views', nargs=2, default=['closeup','side'])
    parser.add_argument('--font', type=Path, default=ROOT/'assets/fonts/NotoSansCJKsc-Regular.otf')
    parser.add_argument('--frame-times-json', type=Path)
    parser.add_argument('--static-frame', type=int, default=0)
    parser.add_argument('--playback-speed', type=float, default=1.)
    parser.add_argument('--max-mb', type=float, default=24.)
    parser.add_argument('--static-only', action='store_true')
    parser.add_argument('--require-success', action='store_true')
    args = parser.parse_args()
    args.run = args.run.resolve()
    if Path(args.output_prefix).name != args.output_prefix or any(Path(n).name != n for n in args.views):
        parser.error('Output prefix and camera names must be basenames')
    if len(set(args.views)) != 2 or not 0 < args.playback_speed <= 8 or args.max_mb <= 0:
        parser.error('Two distinct camera names, positive size budget and speed in (0,8] required')
    task, evaluation, records = [read(args.run/name) for name in ['task.json','evaluation.json','trajectory.json']]
    task_check = validate_task(task)
    selection = selected_environment_evidence(evaluation)
    if args.require_success and not selection['selected_success_episode']:
        parser.error('Recorded environment 0 is not explicitly successful')
    for record in records:
        face_values(record, task)
        if not np.isfinite(record.get('cube_lift_m', np.nan)):
            raise ValueError('Nonfinite cube lift cannot be displayed as healthy telemetry')
    cameras = read(args.run/'camera_manifest.json')
    labels = {name: camera_label(name, cameras) for name in args.views}
    evidence = read(ROOT/'references/human/broad_face_review/extracted_frames.json')
    ref = next(r for r in evidence['frames'] if r['frame_index'] == 5323)
    reference_path = ROOT/ref['path']
    if sha(reference_path) != ref['sha256']:
        raise ValueError('Human reference hash changed')
    layout = Layout(task, evaluation, labels, Image.open(reference_path).convert('RGB'),
                    f'{ref["time_s"]:.3f} s', args.font)
    out = (args.output_dir or args.run/'face_pinch_delivery').resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: args.run/(name+'.mp4') for name in args.views}
    readers = {name: imageio.get_reader(str(path)) for name, path in paths.items()}
    try:
        counts = [reader.count_frames() for reader in readers.values()]
        rates = [float(reader.get_meta_data()['fps']) for reader in readers.values()]
        if len(set(counts)) != 1 or counts[0] < 1 or max(rates)-min(rates) > 1e-6:
            raise ValueError('Raw camera frame counts or fps disagree')
        count, fps = counts[0], rates[0]
        indices, times, alignment = frame_alignment(args, task, records, count)
        if count > 1 and not np.allclose(np.diff(times), 1/fps, atol=1e-5):
            raise ValueError('Source timestamps disagree with source video rate')
        if not 0 <= args.static_frame < count:
            parser.error('static-frame outside source video')
        compose = lambda i: layout.frame({n:r.get_data(i) for n,r in readers.items()}, records[indices[i]], args.playback_speed)
        still, cover, video = [out/(args.output_prefix+suffix) for suffix in ['.png','_cover.png','.mp4']]
        cover_i = max(range(count), key=lambda i: records[indices[i]]['cube_lift_m'])
        Image.fromarray(compose(args.static_frame)).save(still)
        Image.fromarray(compose(cover_i)).save(cover)
        output_rate = fps*args.playback_speed
        decoded = None
        if not args.static_only:
            bitrate = min(8_000_000, int(args.max_mb*1e6*8*.82/(count/output_rate)))
            if bitrate < 300000:
                raise ValueError('Requested video size budget too small')
            writer = imageio.get_writer(str(video), fps=output_rate, codec='libx264', quality=None,
                bitrate=str(bitrate), macro_block_size=1, pixelformat='yuv420p',
                ffmpeg_params=['-preset','medium','-movflags','+faststart','-threads','4'])
            try:
                for i in range(count):
                    writer.append_data(compose(i))
            finally:
                writer.close()
            if video.stat().st_size >= args.max_mb*1e6:
                raise RuntimeError('Encoded video exceeds size cap')
            verify = imageio.get_reader(str(video))
            try:
                decoded = sum(1 for _ in verify)
                if decoded != count:
                    raise RuntimeError('Full output decode count mismatch')
            finally:
                verify.close()
        checkpoint = evaluation.get('checkpoint')
        cp = Path(checkpoint) if checkpoint not in [None,'zero'] else None
        if cp is not None and not cp.is_absolute():
            cp = ROOT/cp
        sources = ['task.json','evaluation.json','trajectory.json','camera_manifest.json']
        sources += [n for n in ['launch.json','grasp_contacts.py','rl_task.py','rl_scene.py'] if (args.run/n).exists()]
        manifest = dict(schema_version=1, run=str(args.run), script_sha256=sha(__file__),
            source_files={n:dict(path=str(args.run/n),sha256=sha(args.run/n)) for n in sources},
            source_cameras={n:dict(path=str(p),sha256=sha(p),label=labels[n]) for n,p in paths.items()},
            human_reference=ref, human_video_sha256=sha(ROOT/'references/human/raw/penn_KXDHa39Z-2s.mp4'),
            human_source_url='https://www.med.upenn.edu/microsurgery/videos',
            task_geometry_check=task_check, success_selection_evidence=selection,
            recorded_environment_index=0, population_episodes=evaluation['episodes'], reported_evaluation=evaluation,
            checkpoint=checkpoint, checkpoint_sha256=sha(cp) if cp and cp.exists() else None,
            checkpoint_lineage=checkpoint_lineage(cp), controller_provenance=layout.controller,
            reset_snapshot=task.get('reset_snapshot'), reset_snapshot_sha256=task.get('reset_snapshot_sha256'),
            face_gate_recomputed_from_qualified_matrix_rows=len(records), face_criteria=task.get('face_pinch_criteria'),
            contact_temporal_aggregation=layout.temporal,
            contact_metric_scope='Qualified NORMAL force on each tool jaw, along inward broad-face direction. Friction cannot satisfy this displayed force. Continuous middle/web support is neither required nor claimed.',
            frame_alignment=alignment, source_trajectory_indices=indices, frame_simulation_times_s=times.tolist(),
            still=dict(path=str(still),sha256=sha(still),frame=args.static_frame),
            cover=dict(path=str(cover),sha256=sha(cover),frame=cover_i),
            video=None if args.static_only else dict(path=str(video),sha256=sha(video),bytes=video.stat().st_size,
                frames=count,fully_decoded_frames=decoded,fps=output_rate,duration_s=count/output_rate,resolution=[1920,1280]),
            operations=['aspect-preserving resize','letterbox','external HUD'],physics_states_changed=False,
            success_rate_claimed_from_selected_episode=False, external_send_performed=False,
            limitations=['Displayed gate is recomputed from logged qualified force/count; independent raw point and source pad geometry audit remains separate.',
                'Task metadata tail shape check requires final actual USD and visual confirmation.',
                'Human reference is uncalibrated; exact 3D contact normals are not inferred.',
                'Control-period means do not prove uninterrupted contact at every physics substep.'])
        dest = out/(args.output_prefix+'_manifest.json')
        dest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'image':str(still),'video':None if args.static_only else str(video),'manifest':str(dest)},ensure_ascii=False))
    finally:
        for reader in readers.values():
            reader.close()


if __name__ == '__main__':
    main()
