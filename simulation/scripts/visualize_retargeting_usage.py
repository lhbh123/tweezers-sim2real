#!/usr/bin/env python3
"""Illustrate the saved human-prior -> robot-reference -> actual-policy data.

All images are existing decoded source frames. Robot values use the archived
runtime pose and measured trajectory; the figure does not synthesize a pose.
Run with the project's isaac-python (imageio + matplotlib are available there).
"""
from pathlib import Path
import hashlib
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/rl_distal_open250_render_final'
FONT = ROOT / 'assets/fonts/NotoSansCJKsc-Regular.otf'
OUT = ROOT / 'outputs/human_to_xhand_usage.png'
REPORT = ROOT / 'reports/human_to_xhand_usage.json'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 13,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False})
    payload = read(ROOT / 'references/human/landmarks.json')
    human = min((f for f in payload['frames'] if f.get('usable_for_retargeting')),
                key=lambda f: abs(f['time_s'] - 222))
    human_path = ROOT / human['frame_path']
    human_crop = [240, 100, 940, 650]
    human_image = Image.open(human_path).convert('RGB').crop(human_crop)
    video_path = RUN / 'closeup.mp4'
    video_index = 75
    video = imageio.get_reader(video_path)
    robot_image = Image.fromarray(video.get_data(video_index))
    video.close()
    robot_crop = [410, 235, 630, 495]
    robot_image = robot_image.crop(robot_crop)
    trajectory = read(RUN / 'trajectory.json')
    task = read(RUN / 'task.json')
    pose = read(RUN / 'runtime_pose.json')
    dt = task['physics_dt'] * task['decimation']
    cadence = max(1, round(1 / (30 * dt)))
    trajectory_index = video_index * cadence
    record = trajectory[trajectory_index]
    actual = dict(zip(task['joint_names'], record['joint_pos']))
    names = list(pose['q_reference'])
    prior = np.array([pose['human_joint_prior'][n] for n in names])
    reference = np.array([pose['q_reference'][n] for n in names])
    measured = np.array([actual[n] for n in names])
    assert len(names) == 12 and np.isfinite([prior, reference, measured]).all()
    assert abs(record['time'] - (trajectory_index + 1) * dt) < 1e-8

    fig = plt.figure(figsize=(16, 13), facecolor='#f7f9fc')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=.38, wspace=.12,
                         left=.065, right=.975, bottom=.155, top=.88)
    fig.suptitle('人手估计如何进入当前 PPO：先生成静态握姿，再学习腕部移动与闭合量',
                 fontsize=21, fontweight='bold', y=.975)
    fig.text(.5, .936, '7 帧二维弯曲代理 → 逐关节中位数 → 接触几何修正 → 固定参考 q_ref → PPO',
             ha='center', fontsize=16, color='#2a596a')
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(human_image); ax.axis('off')
    ax.set_title(f'人类参考原图裁切 · Penn {human["time_s"]:.3f} s', fontsize=17, pad=12)
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(robot_image); ax.axis('off')
    ax.set_title(f'实际仿真帧裁切 · PPO250 · {record["time"]:.3f} s', fontsize=17, pad=12)
    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor('white')
    x = np.arange(len(names)); w = .24
    ax.bar(x-w, np.rad2deg(prior), w, color='#697ba5', label='初步映射先验（含遮挡正则与限位裁剪）')
    ax.bar(x, np.rad2deg(reference), w, color='#d99525', label='接触修正后的固定参考 q_ref')
    ax.bar(x+w, np.rad2deg(measured), w, color='#189b96', label='上方仿真帧的实际关节角')
    labels = ['拇指 1','拇指 2','拇指 3','食指侧摆','食指 1','食指 2',
              '中指 1','中指 2','无名指 1','无名指 2','小指 1','小指 2']
    ax.set_xticks(x, labels, rotation=25, ha='right')
    ax.set_ylabel('XHand 关节角（度）')
    ax.set_ylim(-12, 100); ax.set_xlim(-.65, 11.65)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.17), ncol=3, fontsize=11.5, frameon=False)
    ax.set_title('机器人关节空间中的三阶段对照（这些不是人体解剖关节角）', fontsize=17, pad=60)
    fig.text(.065, .075,
        '闭合控制：q_target = clip(q_ref + c·Δq)，c = clip(0.6 + 0.9·a4, −0.2, 2)，再做 0.15 系数平滑。\n'
        '当前 Δq 只有拇指和食指非零；中指、无名指、小指目标固定。视频 21 点不直接输入 PPO。',
        fontsize=13, linespacing=1.65)
    fig.text(.065, .025,
        '比较边界：人和机器人视角不同，仅作定性对照；不能据此计算像素对齐误差。遮挡手指与单目深度不是动捕真值。',
        fontsize=12, color='#6b4c35')
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=fig.get_facecolor()); plt.close(fig)

    closure = np.array([pose['closure_delta'][n] for n in names])
    report = dict(
        figure=str(OUT.relative_to(ROOT)), human_time_s=human['time_s'],
        human_frame_index=human['frame_index'], human_frame=str(human_path.relative_to(ROOT)),
        source_human_frame_sha256=digest(human_path), human_crop_xyxy=human_crop,
        robot_run=str(RUN.relative_to(ROOT)), robot_video_frame_index=video_index,
        robot_video_sha256=digest(video_path), robot_crop_xyxy=robot_crop,
        trajectory_index=trajectory_index, robot_time_s=record['time'],
        joint_names=names, mapped_prior_rad=prior.tolist(), reference_rad=reference.tolist(),
        actual_rad=measured.tolist(), closure_delta_rad=closure.tolist(),
        reference_minus_prior_rms_deg=float(np.rad2deg(np.sqrt(np.mean((reference-prior)**2)))),
        frame_actual_minus_reference_rms_deg=float(np.rad2deg(np.sqrt(np.mean((measured-reference)**2)))),
        raw_keypoints_input_to_policy=False, per_frame_human_motion_tracking=False,
        policy_action_size=4, independently_learned_hand_controls=1,
        fixed_target_fingers=['middle', 'ring', 'little'],
        prior_weights_field_consumed=False,
        notes=['Prior is already mapped, regularized and clipped, not raw measured human joint angles.',
               'Reference-vs-prior distance is geometric modification size, not measured human-pose reconstruction error.',
               'Photos cropped/resized only; no generated or retouched human/robot content.'],
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [
            ROOT/'references/human/landmarks.json', RUN/'runtime_pose.json',
            RUN/'trajectory.json', RUN/'task.json', ROOT/'src/human_retarget.py', ROOT/'src/rl_task.py']})
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'figure':str(OUT), 'report':str(REPORT),
                      'reference_minus_prior_rms_deg':report['reference_minus_prior_rms_deg']},ensure_ascii=False))


if __name__ == '__main__':
    main()
