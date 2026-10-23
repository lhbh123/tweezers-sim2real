#!/usr/bin/env python3
"""Compose unchanged simulation camera images and synchronized W01 contact data.

Usage: .venv-human/bin/python scripts/compose_sharpa_video.py RUN --output FILE.mp4
Inputs: overview.mp4, task.mp4 (or closeup.mp4), trajectory.npz, tactile.npz.
The tactile archive is produced by src/sharpa_tactile.py at every physics step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FINGERS_ZH = ("拇指", "食指", "中指", "无名指", "小指")
PHASES = ("初始持镊", "接近方块", "下降对准", "挤压夹紧", "提起方块",
          "搬运至目标", "下降放置", "松开镊尖", "撤离镊尖", "稳定留置")
BG = (10, 17, 28)
PANEL = (18, 29, 44)
WHITE = (232, 240, 250)
MUTED = (151, 172, 196)
CYAN = (66, 212, 201)
AMBER = (255, 180, 78)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def heatmap(values, valid, maximum):
    """Linear colour encoding; nearest-neighbour image resizing preserves bins."""
    stops = np.asarray([[24, 44, 67], [23, 110, 145], [48, 196, 157],
                        [255, 202, 79], [244, 75, 65]], np.float64)
    level = np.clip(values / max(float(maximum), 1e-12), 0, 1) * (len(stops) - 1)
    index = np.minimum(level.astype(int), len(stops) - 2)
    fraction = level - index
    rgb = stops[index] * (1 - fraction[..., None]) + stops[index + 1] * fraction[..., None]
    rgb[~valid] = (8, 16, 27)
    return Image.fromarray(np.round(rgb).astype(np.uint8))


def contact_panel(values, valid, maximum, size=156):
    """Enlarge location glyphs for legibility; glyph area has no physical meaning."""
    bins = heatmap(values, valid, maximum)
    picture = bins.resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(picture)
    colours = np.asarray(bins)
    height, width = values.shape
    for row, col in np.argwhere((values > 1e-8) & valid):
        x, y = (col + .5) * size / width, (row + .5) * size / height
        draw.ellipse((x - 5, y - 5, x + 5, y + 5),
                     fill=tuple(colours[row, col]), outline=(167, 210, 223), width=1)
    return picture


class Composer:
    def __init__(self, tactile, trajectory, evaluation, title, contact_note=""):
        self.tactile, self.trajectory, self.evaluation, self.title = tactile, trajectory, evaluation, title
        self.contact_note = contact_note
        self.fonts = {}
        self.limits = {key: float(tactile[key + "_map_N"].max()) for key in ("normal", "friction")}
        self.force_limit = max(float(tactile["normal_load_N"].max()), float(tactile["friction_load_N"].max()), .01)

    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(str(ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"), size)
        return self.fonts[size]

    def label(self, draw, pos, text, size=23, color=WHITE):
        draw.text(pos, str(text), font=self.font(size), fill=color)

    @staticmethod
    def camera(canvas, pixels, box):
        picture = Image.fromarray(np.asarray(pixels, np.uint8)[:, :, :3])
        x, y, width, height = box
        ratio = min(width / picture.width, height / picture.height)
        picture = picture.resize((round(picture.width * ratio), round(picture.height * ratio)), Image.Resampling.LANCZOS)
        canvas.paste(picture, (x + (width - picture.width) // 2, y + (height - picture.height) // 2))

    def frame(self, camera_frames, index, control_step):
        tactile, trajectory = self.tactile, self.trajectory
        canvas = Image.new("RGB", (1920, 1280), BG)
        draw = ImageDraw.Draw(canvas)
        time = float(tactile["frame_time_s"][index])
        self.label(draw, (30, 15), self.title, 39)
        self.label(draw, (32, 68), "W01 · 22 自由度  |  仿真接触力 + 指尖表面近似投影", 23, CYAN)
        if self.contact_note:
            self.label(draw, (940, 76), self.contact_note, 20, AMBER)
        phase = int(np.asarray(trajectory.get("phase", [[-1]]))[control_step].reshape(-1)[0]) if "phase" in trajectory else -1
        phase_text = PHASES[phase] if 0 <= phase < len(PHASES) else "未记录阶段"
        self.label(draw, (1250, 25), f"{time:6.2f} s  |  {phase_text}", 30)
        self.label(draw, (32, 112), "环境视图", 20, MUTED)
        self.label(draw, (978, 112), "操作近景", 20, MUTED)
        for camera_index, pixels in enumerate(camera_frames):
            box = (30 + camera_index * 950, 145, 920, 510)
            draw.rectangle((box[0], box[1], box[0] + box[2], box[1] + box[3]), fill=(0, 0, 0))
            self.camera(canvas, pixels, box)
        if len(camera_frames) == 1:
            self.camera(canvas, camera_frames[0], (980, 145, 920, 510))
        metrics = []
        for key, label, factor, unit in (("lift", "方块提升", 1000., "mm"), ("goal_error", "目标误差", 1000., "mm")):
            if key in trajectory:
                value = float(np.asarray(trajectory[key][control_step]).reshape(-1)[0])
                metrics.append(f"{label} {value * factor:.1f} {unit}")
        if "both" in trajectory:
            bilateral = bool(np.asarray(trajectory["both"][control_step]).reshape(-1)[0])
            metrics.append("镊尖双侧接触：" + ("有" if bilateral else "无"))
        success = bool(np.asarray(trajectory["success"][control_step]).reshape(-1)[0]) if "success" in trajectory else False
        metrics.append("任务验收：已通过" if success else "任务验收：尚未通过")
        self.label(draw, (32, 673), "    |    ".join(metrics), 26, CYAN if success else WHITE)
        self.label(draw, (32, 717), "五指触觉：每格力 [N/bin] · 帧区间均值 · 接触点放大标记（非接触面积）", 23, MUTED)
        for finger, label in enumerate(FINGERS_ZH):
            x, y = 30 + finger * 378, 760
            draw.rounded_rectangle((x, y, x + 360, 1205), 12, fill=PANEL)
            self.label(draw, (x + 15, y + 10), label, 28)
            for column, (key, name, color) in enumerate((("normal", "法向", CYAN), ("friction", "切向", AMBER))):
                xx = x + 16 + column * 170
                force = float(tactile[key + "_load_N"][index, finger])
                self.label(draw, (xx, y + 58), f"{name} {force:.3f} N", 19, color)
                visual = contact_panel(tactile[key + "_map_N"][index, finger], tactile["surface_valid"][finger], self.limits[key])
                canvas.paste(visual, (xx, y + 90))
                limit = self.limits[key]
                self.label(draw, (xx, y + 251), f"色标 0–{limit:.3g} N", 16, MUTED)
            peak = tactile["normal_peak_load_N"][index, finger]
            outside = tactile["normal_unmapped_load_N"][index, finger]
            self.label(draw, (x + 16, y + 279), f"法向峰值 {peak:.3f} N  |  图外 {outside:.3f} N", 17, MUTED)
            # Same global axes for every finger and every frame. Full-trial time
            # span stays fixed, but only observations up to the current time show.
            chart = (x + 20, y + 327, x + 340, y + 420)
            draw.line([(chart[0], chart[1]), (chart[0], chart[3]), (chart[2], chart[3])], fill=(76, 96, 119), width=1)
            timestamps = tactile["frame_time_s"][:index + 1]
            total_time = max(float(tactile["frame_time_s"][-1]), 1e-6)
            for key, color in (("normal", CYAN), ("friction", AMBER)):
                values = tactile[key + "_load_N"][:index + 1, finger]
                points = [(chart[0] + float(t) / total_time * (chart[2] - chart[0]),
                           chart[3] - float(v) / self.force_limit * (chart[3] - chart[1])) for t, v in zip(timestamps, values)]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
            self.label(draw, (x + 20, y + 303), f"力随时间  0–{self.force_limit:.2g} N", 15, MUTED)
        self.label(draw, (32, 1224), "PhysX 法向接触点 / 摩擦锚点，不含扭矩；表面图为近似力分箱，非 TacMap 形变或实物电子皮肤读数。", 21, MUTED)
        return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="append", type=Path, help="Override camera input (at most two)")
    parser.add_argument("--title", default="SharpaWave W01：使用镊子夹取并放置方块")
    parser.add_argument("--contact-note", default="", help="Short contact interpretation note above camera views")
    parser.add_argument("--extra-keyframes", type=int, nargs="*", default=[],
                        help="Additional native video frame indices to save as evidence images")
    parser.add_argument("--max-mb", type=float, default=23.)
    parser.add_argument("--fps", type=float, default=30.)
    args = parser.parse_args()
    run = args.run.resolve()
    tactile = load_npz(run / "tactile.npz")
    trajectory = load_npz(run / "trajectory.npz")
    evaluation = json.loads((run / "evaluation.json").read_text()) if (run / "evaluation.json").exists() else {}
    videos = args.video or [run / name for name in ("overview.mp4", "task.mp4", "closeup.mp4") if (run / name).exists()][:2]
    if not 1 <= len(videos) <= 2:
        raise ValueError("One or two simulation camera videos are required")
    readers = [imageio.get_reader(path) for path in videos]
    counts = [reader.count_frames() for reader in readers]
    frames = len(tactile["frame_time_s"])
    if any(count != frames for count in counts):
        raise ValueError(f"Camera/tactile frame mismatch: video={counts}, tactile={frames}")
    steps = np.asarray(trajectory.get("render_control_steps", np.arange(frames)), dtype=np.int64)
    if len(steps) != frames:
        raise ValueError("trajectory.render_control_steps and tactile frame counts differ")
    if "dt" in trajectory:
        camera_times = (steps + 1) * float(trajectory["dt"])
        if not np.allclose(camera_times, tactile["frame_time_s"], atol=1e-5):
            raise ValueError("Camera and tactile simulation timestamps differ")
    if args.max_mb <= 0 or args.fps <= 0:
        raise ValueError("max-mb and fps must be positive")
    if any(index < 0 or index >= frames for index in args.extra_keyframes):
        raise ValueError("Extra keyframe index outside the recorded video")
    if frames > 1:
        intervals = np.diff(tactile["frame_time_s"])
        # The runner records a final terminal control step even if it falls
        # halfway through the usual two-control-step video interval.
        if (not np.allclose(intervals[:-1], 1 / args.fps, atol=1e-5)
                or not 0 < intervals[-1] <= 1 / args.fps + 1e-5):
            raise ValueError("Frame spacing does not match requested playback fps")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = frames / args.fps
    bitrate = max(64_000, int(args.max_mb * 1e6 * 8 * .91 / max(duration, 1.)))
    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", bitrate=str(bitrate),
                                macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"])
    composer = Composer(tactile, trajectory, evaluation, args.title, args.contact_note)
    # Pick evidence-driven frames: early grasp, maximum lift and final result.
    key_indexes = {0, frames - 1, int(np.argmax(tactile["normal_load_N"].sum(-1)))}
    key_indexes.update(args.extra_keyframes)
    if "lift" in trajectory:
        lift = np.asarray(trajectory["lift"])[steps].reshape(frames, -1)[:, 0]
        peak_lift = int(np.argmax(lift))
        key_indexes.add(peak_lift)
    else:
        peak_lift = frames // 2
    keyframes = []
    try:
        for index, control_step in enumerate(steps):
            canvas = composer.frame([reader.get_data(index) for reader in readers], index, int(control_step))
            writer.append_data(np.asarray(canvas))
            if index in key_indexes:
                path = args.output.with_name(f"{args.output.stem}_frame_{index:05d}.png")
                canvas.save(path)
                keyframes.append(str(path))
            if index == peak_lift:
                canvas.save(args.output.with_name(args.output.stem + "_cover.png"))
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    check = imageio.get_reader(args.output)
    decoded = check.count_frames()
    check.close()
    if decoded != frames:
        raise RuntimeError("Output frame count verification failed")
    if args.output.stat().st_size > args.max_mb * 1e6:
        raise RuntimeError("Encoded output exceeds requested delivery size; lower --max-mb and recompose")
    report = {"source_run": str(run), "source_videos": list(map(str, videos)),
              "source_tactile_sha256": hashlib.sha256((run / "tactile.npz").read_bytes()).hexdigest(),
              "frames": frames, "decoded_frames": decoded, "fps": args.fps, "playback_speed": 1.,
              "tactile_sampling": "Every physics substep, dt-weighted mean per video interval",
              "tactile_visualization": "Nearest-surface force bins; linear shared colour scales; no spatial smoothing. Contact location glyphs enlarged for legibility; their area is not physical contact area.",
              "contact_note": args.contact_note,
              "extra_keyframe_indices": args.extra_keyframes,
              "map_scale_N_per_bin": composer.limits, "keyframes": keyframes,
              "bytes": args.output.stat().st_size,
              "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
