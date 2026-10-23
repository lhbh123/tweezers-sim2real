#!/usr/bin/env python3
"""Display three unaltered native W01 camera frames with explicit provenance."""
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
    parser.add_argument("old_run", type=Path)
    parser.add_argument("new_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-frame", type=int, default=0)
    parser.add_argument("--new-frame", type=int, default=-1)
    parser.add_argument("--middle-support-up-N", type=float,
                        help="Measured final-one-second mean force on the tool, not force on the finger")
    parser.add_argument("--validated-task-run", type=Path,
                        help="Use a successful, tactile-validated task lift frame as panel three; panel two is the cold opposite view")
    args = parser.parse_args()
    canvas = Image.new("RGB", (2560, 1060), (10, 17, 28))
    draw = ImageDraw.Draw(canvas)
    fonts = {}

    def label(position, text, size=32, colour=(232, 240, 250)):
        if size not in fonts:
            fonts[size] = ImageFont.truetype(str(ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"), size)
        draw.text(position, text, font=fonts[size], fill=colour)

    label((40, 18), "W01 握姿调整：真实仿真画面对比", 54)
    supporting = args.middle_support_up_N is not None
    subtitle = ("拇指与食指保持宽面捏持；已测得中指向上承托，无名指与小指屈曲内收。"
                if supporting else "拇指与食指继续捏持宽面；中指移至下方备用位置，无名指与小指屈曲内收。")
    label((42, 90), subtitle, 31, (66, 212, 201))
    views = [
        (args.old_run / "task.mp4", args.old_frame, "旧 W01 · 原主视角", "中指、无名指、小指伸直外展"),
        (args.new_run / "task.mp4", args.new_frame, "新 W01 · 主视角",
         f"中指净向上 {args.middle_support_up_N:+.3f} N（末 1 秒均值）" if supporting else "完整掌部入画；闲指屈曲收拢"),
        (args.new_run / "support.mp4", args.new_frame, "新 W01 · 对侧视角",
         "中指侧边接触；触觉为近似投影" if supporting else "观察中指下方空间与屈曲指形"),
    ]
    task_evidence = None
    if args.validated_task_run is not None:
        task = args.validated_task_run
        evaluation = json.loads((task / "evaluation.json").read_text())
        tactile_validation = json.loads((task / "tactile_validation.json").read_text())
        if not evaluation.get("success_per_env") or not all(evaluation["success_per_env"]) or not tactile_validation.get("valid"):
            raise ValueError("Final comparison requires successful evaluation and valid tactile report")
        trajectory = np.load(task / "trajectory.npz", allow_pickle=False)
        steps = trajectory["render_control_steps"].astype(int)
        lifts = trajectory["lift"][steps].reshape(len(steps), -1)[:, 0]
        peak = int(np.argmax(lifts))
        views[1] = (args.new_run / "support.mp4", args.new_frame, "新 W01 · 冷起对侧视角",
                    f"中指净向上 {args.middle_support_up_N:+.3f} N（末 1 秒）" if supporting else "中指下方承托位；闲指屈曲收拢")
        views[2] = (task / "task.mp4", peak, "新 W01 · 完整任务举升帧",
                    f"方块提升 {float(lifts[peak]) * 1000:.1f} mm；中指侧边承托")
        task_evidence = {
            "run": str(task.resolve()), "success_per_env": evaluation["success_per_env"],
            "max_lift_m": evaluation["max_lift_m"], "final_xy_error_m": evaluation["final_xy_error_m"],
            "selected_peak_lift_frame": peak,
            "evaluation_sha256": hashlib.sha256((task / "evaluation.json").read_bytes()).hexdigest(),
            "tactile_validation_sha256": hashlib.sha256((task / "tactile_validation.json").read_bytes()).hexdigest(),
        }
    provenance = []
    for index, (video, requested, title, caption) in enumerate(views):
        with imageio.get_reader(video) as reader:
            count = reader.count_frames()
            frame_index = requested if requested >= 0 else count + requested
            if not 0 <= frame_index < count:
                raise ValueError(f"Frame {requested} outside {video}: {count} frames")
            pixels = reader.get_data(frame_index)
        picture = Image.fromarray(pixels)
        if picture.size != (960, 720):
            raise ValueError(f"Unexpected native camera dimensions: {picture.size}")
        x = 40 + index * 840
        label((x, 148), title, 37)
        canvas.paste(picture.resize((800, 600), Image.Resampling.LANCZOS), (x, 208))
        label((x, 828), caption, 28, (184, 201, 220))
        label((x, 871), f"原生相机帧 {frame_index} / {count - 1}", 23, (151, 172, 196))
        provenance.append({
            "video": str(video.resolve()), "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "frame_index": frame_index, "video_frames": count,
            "raw_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "native_size": list(picture.size), "display_size": [800, 600],
        })
    evidence_caption = (f"中图为 2 秒冷起对侧帧；右图来自已完成的夹取放置实验，落点误差 {evaluation['final_xy_error_m'][0] * 1000:.2f} mm。"
                        if task_evidence else "新图为 2 秒冷重置持镊预览；完整夹取与放置实验正在验证。")
    label((42, 928), evidence_caption, 32, (255, 180, 78))
    footer = ("原生相机帧仅按比例缩放，未改变姿势；触觉为刚体接触力的表面近似投影，不是硬件皮肤读数。"
              if supporting else "原生相机画面仅按比例缩放拼排，未改变姿势；机位不同，接触与承托以物理测量为准。")
    label((42, 985), footer, 25, (151, 172, 196))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    purpose = ("W01 posture comparison with a cold-reset opposite view and successful complete-task lift frame"
               if task_evidence else "W01 posture comparison; new frames are a cold-reset holding preview, not task-success evidence")
    report = {"purpose": purpose,
              "image": str(args.output.resolve()), "size": list(canvas.size),
              "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "source_frames": provenance, "middle_support_net_up_final_1s_mean_N": args.middle_support_up_N,
              "operations": ["proportional resize", "layout", "captions outside camera frames"]}
    if task_evidence:
        report["task_evidence"] = task_evidence
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"image": str(args.output), "sha256": report["image_sha256"], "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
