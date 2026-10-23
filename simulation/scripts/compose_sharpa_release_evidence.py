#!/usr/bin/env python3
"""Compare actual opposite-camera frames during support, release and recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    evaluation = json.loads((run / "evaluation.json").read_text())
    validation = json.loads((run / "tactile_validation.json").read_text())
    if not evaluation.get("success_per_env") or not all(evaluation["success_per_env"]) or not validation.get("valid"):
        raise ValueError("Requires a successful task and valid tactile report")
    trajectory = np.load(run / "trajectory.npz", allow_pickle=False)
    tactile = np.load(run / "tactile.npz", allow_pickle=False)
    steps = trajectory["render_control_steps"].astype(int)
    phases = trajectory["phase"][steps].reshape(len(steps), -1)[:, 0]
    joint = evaluation["task"]["joint_names"].index("right_middle_MCP_FE")
    finger = tactile["finger_names"].tolist().index("middle")
    stages = [(6, "松开前 · 中指承托"), (7, "松开阶段 · 中指退让"), (8, "离块后 · 中指恢复")]
    indexes = []
    for phase, _ in stages:
        candidates = np.flatnonzero(phases == phase)
        if len(candidates) == 0:
            raise ValueError(f"No native frame in phase {phase}")
        indexes.append(int(candidates[-1]))
    canvas = Image.new("RGB", (2560, 1090), (10, 17, 28))
    draw = ImageDraw.Draw(canvas)
    fonts = {}

    def label(xy, text, size=30, colour=(232, 240, 250)):
        if size not in fonts:
            fonts[size] = ImageFont.truetype(str(ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"), size)
        draw.text(xy, text, font=fonts[size], fill=colour)

    label((40, 20), "W01：中指承托、释放退让与恢复", 52)
    label((42, 94), "同一次成功实验的真实对侧相机帧；数值来自对应时刻的关节状态和 PhysX 接触力。", 30, (66, 212, 201))
    provenance = []
    video = run / "support.mp4"
    with imageio.get_reader(video) as reader:
        count = reader.count_frames()
        if count != len(steps):
            raise ValueError("Video/control timestamps do not match")
        for column, ((phase, title), index) in enumerate(zip(stages, indexes)):
            control = int(steps[index])
            pixels = reader.get_data(index)
            if pixels.shape[:2] != (720, 960):
                raise ValueError("Expected native 960 x 720 camera")
            q = float(trajectory["q"][control, 0, joint])
            blend = float(trajectory["release_middle_blend"][control, 0])
            upward = -float(tactile["normal_force_w_N"][index, finger, 2] + tactile["friction_force_w_N"][index, finger, 2])
            time = float(tactile["frame_time_s"][index])
            jaw = float(trajectory["forces"][control, 0].max())
            x = 40 + 840 * column
            label((x, 155), title, 35)
            canvas.paste(Image.fromarray(pixels).resize((800, 600), Image.Resampling.LANCZOS), (x, 211))
            label((x, 830), f"t = {time:.2f} s  ·  中指关节实测 {q:.4f} rad", 27, (184, 201, 220))
            shown_upward = 0. if abs(upward) < .0005 else upward
            label((x, 872), f"中指对镊子竖直合力 {shown_upward:+.3f} N", 27, (184, 201, 220))
            label((x, 914), f"夹口—方块最大接触力 {jaw:.4f} N", 25, (151, 172, 196))
            provenance.append({"phase": phase, "title": title, "frame_index": index,
                "control_step": control, "simulation_time_s": time,
                "raw_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "middle_MCP_FE_actual_rad": q, "middle_release_command_blend": blend,
                "middle_force_on_tool_up_N_video_interval_mean": upward,
                "jaw_block_max_contact_force_N_control_step": jaw})
    label((42, 978), "中指侧边承托；表面触觉图为近似投影，不代表中央皮肤或硬件传感器读数。", 29, (255, 180, 78))
    label((42, 1031), "中指力为录像区间均值，正值向上；夹口力与关节角为该帧对应控制步读数。画面仅等比缩放。", 25, (151, 172, 196))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    report = {"run": str(run), "task_success_per_env": evaluation["success_per_env"],
        "source_video": str(video), "source_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "source_tactile_sha256": hashlib.sha256((run / "tactile.npz").read_bytes()).hexdigest(),
        "image": str(args.output.resolve()), "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "size": list(canvas.size), "source_frames": provenance,
        "operations": ["proportional resize", "layout", "captions outside camera frames"]}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"image": str(args.output), "sha256": report["image_sha256"], "bytes": args.output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
