#!/usr/bin/env python3
"""Compose two untouched RL camera streams with recorded state and assumptions.

Uses trajectory.json, task.json, evaluation.json, launch.json, and the run's
archived train_rl.py capture cadence. Can instead accept explicit frame times.
Every source camera frame is preserved once; only aspect-preserving resize and
outside-image HUD are applied. No success reenactment or phase labels are added.
"""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from compose_video import Compositor, BG, PANEL, BORDER, WHITE, MUTED, CYAN, AMBER


def read(path, required=True):
    if not path.exists() and not required:
        return {}
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, precision=1, suffix=''):
    try:
        value = float(value)
        return f'{value:.{precision}f}{suffix}' if math.isfinite(value) else '--'
    except (TypeError, ValueError):
        return '--'


def checkpoint_lineage(checkpoint_path):
    """Record inherited iterations separately from updates in this branch."""
    if checkpoint_path is None:
        return None
    launch_path = checkpoint_path.parent/'launch.json'
    source = read(launch_path, required=False)
    resume = source.get('arguments', {}).get('resume')
    resume_path = Path(resume) if resume else None
    if resume_path is not None and not resume_path.is_absolute():
        resume_path = Path(__file__).resolve().parents[1]/resume_path
    def iteration(path):
        token = path.stem.removeprefix('iteration_') if path is not None else ''
        return int(token) if token.isdecimal() else None
    selected, inherited = iteration(checkpoint_path), iteration(resume_path)
    return dict(source_training_launch=str(launch_path),
                source_training_launch_sha256=digest(launch_path) if launch_path.exists() else None,
                resume_checkpoint=resume,
                resume_checkpoint_sha256_computed_at_composition=digest(resume_path) if resume_path and resume_path.exists() else None,
                selected_checkpoint_iteration=selected, inherited_iteration=inherited,
                additional_updates_to_selected_checkpoint=selected-inherited if selected is not None and inherited is not None else None,
                caveat='Iteration labels include inherited updates; checkpoint hashes computed at composition time.')


class RLCompositor(Compositor):
    def __init__(self, width, task, launch, evaluation, duration, speed, closeup_only=False,
                 title='UR5e + XHand | PPO tweezer lift'):
        super().__init__(width, title,
                         task.get('fixture_present'), 'passive spring jaws', duration, 0.,
                         closeup_only=closeup_only, playback_speed=speed)
        self.task, self.args, self.evaluation = task, launch.get('arguments', {}), evaluation
        pose = Path(str(self.args.get('pose', ''))).stem
        self.pose_label = ('DISTAL THUMB-INDEX GRASP' if 'distal' in pose else
                           'PROXIMAL THUMB-INDEX GRASP' if 'proximal' in pose else 'VIDEO-PRIOR HAND POSE')

    def frame(self, overview, closeup, record, video_time):
        canvas = Image.new('RGB', self.size, BG)
        draw = ImageDraw.Draw(canvas)
        self.label(draw, 'LEARNED PPO POLICY | ' + self.pose_label, 40, 18, 18, CYAN, True)
        self.label(draw, self.title, 40, 48, 37, WHITE, True)
        checkpoint = Path(str(self.evaluation.get('checkpoint', 'UNRECORDED'))).name
        self.fit_label(draw, 'CHECKPOINT  ' + checkpoint, 1310, 24, 570, 21, WHITE, True)
        self.label(draw, f'PLAYBACK {self.playback_speed:g}x', 1630, 59, 19, MUTED, True)
        selfcoll = self.task.get('robot_self_collisions_enabled')
        filters = next((self.task[k] for k in ['filtered_mechanical_pairs', 'robot_self_collision_excluded_pairs', 'robot_filtered_collision_pairs', 'self_collision_filtered_pairs'] if isinstance(self.task.get(k), list)), None)
        unique_filters = {tuple(sorted(Path(link).name for link in pair)) for pair in filters} if filters is not None else None
        if selfcoll is False:
            status, color = 'DIAGNOSTIC RUN: ROBOT SELF-COLLISIONS DISABLED', AMBER
        elif selfcoll is True:
            suffix = f' | {len(unique_filters)} EXCLUDED LINK PAIRS / ENV' if unique_filters is not None else (' | MECHANICAL-PAIR FILTER ON; PAIR LIST NOT RECORDED' if self.task.get('filtered_mechanical_self_pairs') else ' | FILTER DETAILS NOT RECORDED')
            status, color = 'ROBOT SELF-COLLISIONS ENABLED' + suffix, CYAN if filters is not None else AMBER
        else:
            status, color = 'ROBOT SELF-COLLISION SETTING NOT RECORDED', AMBER
        self.rect(draw, (40, 100, 1880, 140), PANEL, BORDER, radius=8)
        self.fit_label(draw, status, 55, 108, 1810, 21, color, True)
        fixture = self.task.get('fixture_present') is True or record.get('fixture_active') is True
        fixture_text = 'HANDLE FIXTURE ACTIVE' if fixture else ('NO HANDLE FIXTURE' if self.task.get('fixture_present') is False else 'FIXTURE SETTING UNKNOWN')
        self.fit_label(draw, f'{fixture_text}  |  PASSIVE SPRING JAWS  |  FINGER FRICTION {number(self.args.get("friction"), 2)}  |  SPRING {number(self.args.get("spring"), 2)} N m/rad',
                       40, 150, 1840, 20, AMBER if fixture else MUTED)
        if self.closeup_only:
            self.label(draw, 'HAND + TWEEZERS / CLOSEUP', 40, 185, 18, MUTED, True)
            views = [(closeup, 40, 1840, 1380)]
        else:
            self.label(draw, '01 / ARM + HAND', 40, 185, 18, MUTED, True)
            self.label(draw, '02 / HAND + TWEEZERS', 968, 185, 18, MUTED, True)
            views = [(overview, 40, 912, 684), (closeup, 968, 912, 684)]
        for pixels, x, width, height in views:
            draw.rectangle((*self.xy(x-1, 215), *self.xy(x+width+1, 217+height)), fill=BORDER)
            self.camera(canvas, pixels, (x, 216, width, height))
        shift = 696 if self.closeup_only else 0
        for x0, x1 in [(40, 390), (410, 760), (780, 1460), (1480, 1880)]:
            self.rect(draw, (x0, 918+shift, x1, 1017+shift), PANEL, BORDER)
        self.label(draw, 'SIMULATION TIME', 58, 932+shift, 15, MUTED, True)
        self.label(draw, number(record.get('time'), 3, ' s'), 58, 960+shift, 31, WHITE, True, True)
        self.label(draw, 'MEASURED BLOCK LIFT', 428, 932+shift, 15, MUTED, True)
        lift = record.get('cube_lift_m')
        self.label(draw, number(None if lift is None else 1000*lift, 1, ' mm'), 428, 960+shift, 31, CYAN, True, True)
        self.label(draw, 'JAW CONTACT RESULTANT MAGNITUDES', 798, 932+shift, 15, MUTED, True)
        forces = record.get('jaw_contact_N', [None, None])
        self.fit_label(draw, f'L  {number(forces[0], 3, " N")}       R  {number(forces[1], 3, " N")}', 798, 964+shift, 640, 28, WHITE, True)
        self.label(draw, 'TOOL RETENTION ERROR', 1498, 932+shift, 15, MUTED, True)
        grip = record.get('tool_grip_error_m')
        self.label(draw, number(None if grip is None else 1000*grip, 2, ' mm'), 1498, 960+shift, 31, WHITE, True, True)
        # These disclosures describe the known RL task initialization and model.
        clearance = self.task.get('initial_tip_clearance')
        gap = number(None if clearance is None else 1000*clearance, 1, ' mm')
        initialization = (f'STABLE FREE-GRASP SNAPSHOT RESET  |  PRE-HELD, NOMINAL TIP-TO-TABLE CLEARANCE {gap}'
                          if self.task.get('reset_snapshot') else
                          f'STARTS PRE-HELD NEAR BLOCK (CLEARANCE {gap})')
        self.fit_label(draw, initialization + '  |  RAW CAMERAS + TELEMETRY',
                       40, 1033+shift, 1840, 16, MUTED)
        progress = float(np.clip(video_time/max(self.duration, 1e-9), 0, 1))
        draw.rectangle((*self.xy(40, 1065+shift), *self.xy(1880, 1070+shift)), fill=BORDER)
        draw.rectangle((*self.xy(40, 1065+shift), *self.xy(40+1840*progress, 1070+shift)), fill=CYAN)
        return np.asarray(canvas)


def frame_alignment(args, task, records, count):
    times = np.array([f['time'] for f in records], dtype=float)
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('Trajectory times must be finite and strictly increasing')
    frame_file = args.frame_times_json or args.run/'video_frame_times.json'
    if frame_file.exists():
        recorded = read(frame_file)
        frame_times = np.asarray(recorded if isinstance(recorded, list) else recorded['times'], dtype=float)
        if len(frame_times) != count:
            raise ValueError('Video frame-times count does not match source video')
        indices = [bisect_right(times, t+1e-9)-1 for t in frame_times]
        if not np.isfinite(frame_times).all() or any(i < 0 for i in indices) or frame_times[-1] > times[-1]+1e-9:
            raise ValueError('Frame times outside trajectory range')
        method = {'source': str(frame_file), 'sha256': digest(frame_file), 'method': 'explicit video frame timestamps; last recorded observation'}
    else:
        source = args.run/'train_rl.py'
        capture = 'step % max(1, round(1/(30*env.dt))) == 0'
        if not source.exists() or capture not in source.read_text():
            raise ValueError('Unknown capture cadence: supply --frame-times-json')
        dt = float(task['physics_dt'])*int(task['decimation'])
        stride = max(1, round(1/(30*dt)))
        indices = list(range(0, len(records), stride))
        if len(indices) != count:
            raise ValueError(f'Capture cadence predicts {len(indices)} frames, source has {count}; cannot safely synchronize')
        frame_times = times[indices]
        method = {'source': str(source), 'sha256': digest(source), 'method': 'archived capture-loop cadence, post-step trajectory', 'control_dt_s': dt, 'render_stride_control_steps': stride}
    return indices, frame_times, method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--output-name', default='ppo_diagnostic.mp4')
    parser.add_argument('--frame-times-json', type=Path)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--max-mb', type=float, default=24.)
    parser.add_argument('--playback-speed', type=float, default=1.)
    parser.add_argument('--closeup-only', action='store_true')
    parser.add_argument('--title', default='UR5e + XHand | PPO tweezer lift')
    args = parser.parse_args()
    if args.width < 960 or args.width % 32 or args.max_mb <= 0 or not 0 < args.playback_speed <= 8:
        parser.error('width >=960 divisible by32; max-mb >0; speed in(0,8] required')
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith('.mp4'):
        parser.error('output-name must be an mp4 basename')
    args.run = args.run.resolve()
    output_dir = args.output_dir or args.run/'preview_delivery'
    output_dir.mkdir(parents=True, exist_ok=True)
    task, launch, evaluation = [read(args.run/n) for n in ['task.json', 'launch.json', 'evaluation.json']]
    records = read(args.run/'trajectory.json')
    paths = [args.run/n for n in (['closeup.mp4'] if args.closeup_only else ['overview.mp4', 'closeup.mp4'])]
    readers = [imageio.get_reader(str(p)) for p in paths]
    try:
        counts = [r.count_frames() for r in readers]
        rates = [float(r.get_meta_data()['fps']) for r in readers]
        if len(set(counts)) != 1 or counts[0] < 1 or max(rates)-min(rates) > 1e-6:
            raise ValueError('Source cameras have different counts/rates or no frames')
        count, fps = counts[0], rates[0]
        indices, frame_times, alignment = frame_alignment(args, task, records, count)
        out_fps, duration = fps*args.playback_speed, count/(fps*args.playback_speed)
        if count > 1 and not np.allclose(np.diff(frame_times), 1/fps, atol=1e-5):
            raise ValueError('Recorded simulation cadence differs from source playback fps; explicit retiming is required')
        renderer = RLCompositor(args.width, task, launch, evaluation, duration, args.playback_speed,
                                args.closeup_only, title=args.title)
        limit = int(args.max_mb*1e6)
        bitrate = min(8_000_000, int(limit*8*.82/duration))
        if bitrate < 300000:
            raise ValueError('Requested video size budget is too low')
        output = output_dir/args.output_name
        cover = output_dir/(output.stem+'_cover.png')
        cover_index = max(range(count), key=lambda i: float(records[indices[i]].get('cube_lift_m', -math.inf)))
        writer = imageio.get_writer(str(output), fps=out_fps, codec='libx264', quality=None,
                                   bitrate=str(bitrate), macro_block_size=1, pixelformat='yuv420p',
                                   ffmpeg_params=['-preset', 'medium', '-maxrate', str(bitrate), '-bufsize', str(bitrate), '-movflags', '+faststart', '-threads', '4'])
        try:
            for i, source_index in enumerate(indices):
                composed = renderer.frame(None if args.closeup_only else readers[0].get_data(i),
                                          readers[-1].get_data(i), records[source_index], i/out_fps)
                writer.append_data(composed)
                if i == cover_index:
                    Image.fromarray(composed).save(cover)
                if i % max(1, int(fps*2)) == 0:
                    print(f'Composed {i+1}/{count}', flush=True)
        finally:
            writer.close()
        if output.stat().st_size >= limit:
            raise RuntimeError('Encoded video exceeds requested cap')
        checkpoint_ref = evaluation.get('checkpoint')
        checkpoint_path = Path(checkpoint_ref) if checkpoint_ref and checkpoint_ref != 'zero' else None
        if checkpoint_path is not None and not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[1]/checkpoint_path
        manifest = dict(video=str(output), cover=str(cover), bytes=output.stat().st_size,
                        source_videos=[dict(path=str(p), sha256=digest(p)) for p in paths],
                        trajectory_sha256=digest(args.run/'trajectory.json'),
                        source_launch_sha256=digest(args.run/'launch.json'),
                        source_task_sha256=digest(args.run/'task.json'),
                        source_evaluation_sha256=digest(args.run/'evaluation.json'),
                        runtime_pose_sha256=digest(args.run/'runtime_pose.json') if (args.run/'runtime_pose.json').exists() else None,
                        frames=count, fps=out_fps, playback_speed=args.playback_speed,
                        view_mode='closeup_only' if args.closeup_only else 'side_by_side',
                        title=args.title, pose_label=renderer.pose_label, source_pose=launch['arguments'].get('pose'),
                        deterministic_policy_evaluation=True,
                        duration_s=duration, resolution=list(renderer.size), alignment=alignment,
                        source_trajectory_indices=indices, frame_simulation_times_s=frame_times.tolist(),
                        cover_frame_index=cover_index, checkpoint=evaluation.get('checkpoint'),
                        checkpoint_sha256_computed_at_composition=digest(checkpoint_path) if checkpoint_path is not None and checkpoint_path.is_file() else None,
                        checkpoint_training_lineage=checkpoint_lineage(checkpoint_path),
                        checkpoint_hash_caveat='Computed at composition time; not a stored evaluation-launch-time digest.',
                        physics_assumptions=dict(fixture_present=task.get('fixture_present'),
                                                 robot_self_collisions_enabled=task.get('robot_self_collisions_enabled'),
                                                 filtered_mechanical_self_pairs=task.get('filtered_mechanical_self_pairs'),
                                                 filtered_mechanical_pairs=task.get('filtered_mechanical_pairs'),
                                                 spring_Nm_rad=launch['arguments'].get('spring'),
                                                 finger_friction=launch['arguments'].get('friction'),
                                                 preheld_initialization=True,
                                                 initialization_method='stable free-grasp snapshot reset' if task.get('reset_snapshot') else 'authored pre-held initialization',
                                                 reset_snapshot=task.get('reset_snapshot'),
                                                 reset_snapshot_recorded_sha256=task.get('reset_snapshot_sha256'),
                                                 initial_tip_clearance_semantics='nominal tip-to-table clearance; not the measured tool-to-block gap',
                                                 initial_tip_clearance_m=task.get('initial_tip_clearance')),
                        telemetry_fields=['time', 'cube_lift_m', 'jaw_contact_N', 'tool_grip_error_m'],
                        image_operations=['aspect-preserving resize', 'outside-image HUD'],
                        no_physics_images_repainted=True, reported_evaluation=evaluation)
        (output_dir/(output.stem+'_manifest.json')).write_text(json.dumps(manifest, indent=2))
        print(json.dumps({'video':str(output), 'cover':str(cover), 'size_mb':output.stat().st_size/1e6}))
    finally:
        for reader in readers:
            reader.close()


if __name__ == '__main__':
    main()
