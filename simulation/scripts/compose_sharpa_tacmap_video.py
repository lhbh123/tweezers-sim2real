#!/usr/bin/env python3
"""Compose native W01 views, official TacMap uint8 output and separate PhysX forces.

Inputs in RUN: tacmap.npz, tactile.npz, trajectory.npz, overview.mp4, task.mp4.
TacMap schema: frame_time_s [F], deformation_uint8 uint8 [F,5,H,W],
finger_names [5], surface_valid bool [5,H,W], metadata_json scalar JSON.
The TacMap producer must use metadata.sampling='render_endpoint'. Its already
quantized and blurred output is displayed on a fixed 0--255 grayscale. No depth
inversion, image normalization, extra blur, inferred deformation or force-derived
image is performed. The independently sampled PhysX force labels are interval
averages from tactile.npz, not measurements derived from TacMap intensities.
The force reader addresses merged distal-phalanx bodies. Their skin and shell
colliders are not separated by the tensor API; the labels are whole-body contact
loads, not isolated skin forces or pressures.
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
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGERS_ZH = ("拇指", "食指", "中指", "无名指", "小指")
PHASES = ("初始持镊", "接近方块", "下降对准", "挤压夹紧", "提起方块",
          "搬运至目标", "下降放置", "松开镊尖", "撤离镊尖", "稳定留置")
BG, PANEL = (10, 17, 28), (18, 29, 44)
WHITE, MUTED = (232, 240, 250), (151, 172, 196)
CYAN, AMBER = (66, 212, 201), (255, 180, 78)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in (keys or archive.files)}


def finger_order(names, archive_name):
    if np.asarray(names).dtype.kind not in "US":
        raise ValueError(f"{archive_name}.finger_names must be Unicode/string names")
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in names]
    if len(names) != 5 or set(names) != set(FINGERS):
        raise ValueError(f"{archive_name} must contain each W01 finger once")
    return [names.index(finger) for finger in FINGERS]


def validate_inputs(tacmap, tactile, trajectory, fps):
    """Reject schema/timing mismatches rather than silently interpolate samples."""
    maps = tacmap["deformation_uint8"]
    if maps.dtype != np.uint8 or maps.ndim != 4 or maps.shape[1] != 5 or min(maps.shape) < 1:
        raise ValueError("deformation_uint8 must be uint8 [F,5,H,W]")
    count = maps.shape[0]
    valid = tacmap["surface_valid"]
    if valid.dtype != np.bool_ or valid.shape != maps.shape[1:]:
        raise ValueError("surface_valid must be bool [5,H,W] matching TacMap")
    if not np.all(valid.reshape(5, -1).any(axis=1)):
        raise ValueError("Each finger requires a nonempty surface_valid mask")
    order = finger_order(tacmap["finger_names"], "tacmap")
    force_order = finger_order(tactile["finger_names"], "tactile")
    times = np.asarray(tacmap["frame_time_s"])
    if times.shape != (count,) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("TacMap timestamps must be finite, nonnegative and shape [F]")
    if count > 1:
        intervals = np.diff(times)
        if not (intervals > 0).all():
            raise ValueError("TacMap timestamps must increase strictly")
        if not np.allclose(intervals[:-1], 1 / fps, atol=1e-5, rtol=0) or intervals[-1] > 1 / fps + 1e-5:
            raise ValueError("TacMap timestamps do not match native video playback rate")
    force_times = np.asarray(tactile["frame_time_s"])
    if force_times.shape != times.shape or not np.allclose(force_times, times, atol=1e-5, rtol=0):
        raise ValueError("TacMap and PhysX force timestamps differ")
    for key in ("normal_load_N", "friction_load_N"):
        values = tactile[key]
        if values.shape != (count, 5) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{key} must be finite, nonnegative [F,5]")
    steps = trajectory["render_control_steps"]
    if steps.shape != (count,) or not np.issubdtype(steps.dtype, np.integer) or (steps < 0).any():
        raise ValueError("render_control_steps must contain one integer index per frame")
    if not np.isfinite(float(trajectory["dt"])) or float(trajectory["dt"]) <= 0:
        raise ValueError("Control dt must be positive")
    if not np.allclose((steps + 1) * float(trajectory["dt"]), times, atol=1e-5, rtol=0):
        raise ValueError("TacMap and trajectory endpoint timestamps differ")
    for key in ("phase", "lift", "goal_error", "both", "success"):
        if key in trajectory and (len(trajectory[key]) <= int(steps.max()) or not np.isfinite(trajectory[key]).all()):
            raise ValueError(f"Trajectory field {key} is missing frames or contains nonfinite values")
    raw_metadata = tacmap["metadata_json"]
    if raw_metadata.shape != () or raw_metadata.dtype.kind not in "US":
        raise ValueError("metadata_json must be a scalar Unicode/string JSON value")
    metadata_text = raw_metadata.item()
    metadata = json.loads(metadata_text.decode() if isinstance(metadata_text, bytes) else metadata_text)
    if not isinstance(metadata, dict) or metadata.get("sampling") != "render_endpoint":
        raise ValueError("TacMap metadata must declare sampling='render_endpoint'")
    tacmap = dict(tacmap, deformation_uint8=maps[:, order], surface_valid=valid[order], frame_time_s=times)
    tactile = dict(tactile, normal_load_N=tactile["normal_load_N"][:, force_order],
                   friction_load_N=tactile["friction_load_N"][:, force_order])
    return tacmap, tactile, steps.astype(np.int64), metadata


class Composer:
    def __init__(self, tacmap, tactile, trajectory, title, contact_note="", tacmap_note=""):
        self.tacmap, self.tactile, self.trajectory = tacmap, tactile, trajectory
        self.title, self.contact_note, self.fonts = title, contact_note, {}
        self.tacmap_note = tacmap_note

    def label(self, draw, xy, text, size=23, colour=WHITE):
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(str(ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"), size)
        draw.text(xy, str(text), font=self.fonts[size], fill=colour)

    def fitted_label(self, draw, xy, text, max_width, size=23, colour=WHITE, min_size=14):
        """Keep full labels inside their assigned header/panel region."""
        for fitted_size in range(size, min_size - 1, -1):
            if fitted_size not in self.fonts:
                self.fonts[fitted_size] = ImageFont.truetype(
                    str(ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"), fitted_size)
            if draw.textlength(str(text), font=self.fonts[fitted_size]) <= max_width:
                self.label(draw, xy, text, fitted_size, colour)
                return
        raise ValueError(f"Display label exceeds its {max_width}-pixel region: {text}")

    @staticmethod
    def camera(canvas, pixels, box):
        picture = Image.fromarray(np.asarray(pixels, np.uint8)[:, :, :3])
        x, y, width, height = box
        scale = min(width / picture.width, height / picture.height)
        picture = picture.resize((round(picture.width * scale), round(picture.height * scale)), Image.Resampling.LANCZOS)
        canvas.paste(picture, (x + (width - picture.width) // 2, y + (height - picture.height) // 2))

    def frame(self, camera_frames, index, control_step):
        canvas = Image.new("RGB", (1920, 1280), BG)
        draw = ImageDraw.Draw(canvas)
        time = float(self.tacmap["frame_time_s"][index])
        self.fitted_label(draw, (30, 15), self.title, 1210, 38, min_size=22)
        self.fitted_label(draw, (32, 68), "W01 · 官方 SharpaTacmap 输出  |  PhysX 合并DP接触载荷独立记录", 985, 23, CYAN)
        if self.contact_note:
            self.fitted_label(draw, (1040, 76), self.contact_note, 850, 20, AMBER)
        phase = int(self.trajectory["phase"][control_step].reshape(-1)[0]) if "phase" in self.trajectory else -1
        phase_label = PHASES[phase] if 0 <= phase < len(PHASES) else "未记录阶段"
        self.fitted_label(draw, (1270, 25), f"{time:6.2f} s  |  {phase_label}", 620, 29)
        for number, (name, pixels) in enumerate(zip(("环境视图", "操作近景"), camera_frames)):
            x = 30 + 950 * number
            self.label(draw, (x, 112), name, 20, MUTED)
            box = (x, 145, 920, 510)
            draw.rectangle((x, 145, x + 920, 655), fill=(0, 0, 0))
            self.camera(canvas, pixels, box)
        metrics = []
        for key, label in (("lift", "方块提升"), ("goal_error", "目标误差")):
            if key in self.trajectory:
                value = float(self.trajectory[key][control_step].reshape(-1)[0]) * 1000
                metrics.append(f"{label} {value:.1f} mm")
        if "both" in self.trajectory:
            both = bool(self.trajectory["both"][control_step].reshape(-1)[0])
            metrics.append("镊尖双侧接触：" + ("有" if both else "无"))
        success = bool(self.trajectory["success"][control_step].reshape(-1)[0]) if "success" in self.trajectory else False
        metrics.append("任务验收：已通过" if success else "任务验收：尚未通过")
        self.label(draw, (32, 673), "    |    ".join(metrics), 26, CYAN if success else WHITE)
        self.fitted_label(draw, (32, 717), self.tacmap_note or "五指 TacMap：官方量化形变等级 0–255 · 原始灰度 · 不等同力或压力", 1856, 23, AMBER if self.tacmap_note else MUTED)
        gradient = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
        for finger, name in enumerate(FINGERS_ZH):
            x, y = 30 + 378 * finger, 760
            draw.rounded_rectangle((x, y, x + 360, 1205), 12, fill=PANEL)
            values = self.tacmap["deformation_uint8"][index, finger]
            self.label(draw, (x + 16, y + 9), name, 28)
            self.label(draw, (x + 118, y + 16), f"峰值 {int(values.max())} / 255", 20, MUTED)
            # Keep all pixels, including any upstream blur across the valid-mask
            # boundary. surface_valid never fills, suppresses or changes values.
            picture = Image.fromarray(values, mode="L").convert("RGB")
            scale = min(256 / picture.width, 256 / picture.height)
            picture = picture.resize((round(picture.width * scale), round(picture.height * scale)), Image.Resampling.NEAREST)
            canvas.paste(picture, (x + 52 + (256 - picture.width) // 2, y + 54 + (256 - picture.height) // 2))
            canvas.paste(Image.fromarray(gradient, mode="L").convert("RGB"), (x + 52, y + 316))
            self.label(draw, (x + 50, y + 328), "0", 15, MUTED)
            self.label(draw, (x + 164, y + 328), "128", 15, MUTED)
            self.label(draw, (x + 278, y + 328), "255", 15, MUTED)
            active = int(np.count_nonzero(values))
            self.label(draw, (x + 16, y + 352), f"非零像素 {active} / {values.size}", 18, MUTED)
            normal, friction = (float(self.tactile[key][index, finger]) for key in ("normal_load_N", "friction_load_N"))
            self.fitted_label(draw, (x + 16, y + 380), f"合并DP 法向载荷Σ|Fn| {normal:.3f} N", 328, 18, CYAN)
            self.fitted_label(draw, (x + 16, y + 411), f"合并DP 摩擦载荷Σ|Ft| {friction:.3f} N", 328, 18, AMBER)
        self.fitted_label(draw, (32, 1218), "PhysX：合并DP法向载荷Σ|Fn|、摩擦载荷Σ|Ft|；均为对应帧区间均值。DP含皮肤/壳体，接触归属待查。", 1856, 20, MUTED)
        self.fitted_label(draw, (32, 1247), "TacMap：原始0–255灰度；以上载荷不是皮肤独立受力或压力，灰度不是压力或硬件读数。", 1856, 20, MUTED)
        return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tacmap", type=Path, help="Override tacmap.npz input")
    parser.add_argument("--video", type=Path, action="append", help="Two native videos, in overview/task order")
    parser.add_argument("--title", default="SharpaWave W01：官方 TacMap 与镊子夹取放置")
    parser.add_argument("--contact-note", default="")
    parser.add_argument("--tacmap-note", default="", help="Visible observation caveat above the five unchanged raw TacMap panels")
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--max-mb", type=float, default=18.)
    parser.add_argument("--extra-keyframes", type=int, nargs="*", default=[])
    parser.add_argument("--require-validated", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace this output path")
    args = parser.parse_args()
    if args.fps <= 0 or args.max_mb <= 0:
        raise ValueError("fps and max-mb must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace existing media: {args.output}")
    run = args.run.resolve()
    tacmap_path = args.tacmap.resolve() if args.tacmap else run / "tacmap.npz"
    force_path, trajectory_path = run / "tactile.npz", run / "trajectory.npz"
    validation_path = run / "tacmap_validation.json"
    if args.require_validated:
        validation = json.loads(validation_path.read_text())
        if validation.get("valid") is not True:
            raise ValueError("Independent TacMap validation has not passed")
        if validation.get("tacmap_npz_sha256") != sha256(tacmap_path):
            raise ValueError("Independent TacMap validation does not match this exact archive")
    tacmap = load_npz(tacmap_path, ("frame_time_s", "deformation_uint8", "finger_names", "surface_valid", "metadata_json"))
    tactile = load_npz(force_path, ("frame_time_s", "finger_names", "normal_load_N", "friction_load_N"))
    trajectory = load_npz(trajectory_path)
    tacmap, tactile, steps, metadata = validate_inputs(tacmap, tactile, trajectory, args.fps)
    count = len(steps)
    if any(index < 0 or index >= count for index in args.extra_keyframes):
        raise ValueError("Extra keyframe outside native video range")
    videos = args.video or [run / "overview.mp4", run / "task.mp4"]
    if len(videos) != 2:
        raise ValueError("Exactly two native camera videos are required")
    readers = []
    writer = None
    keyframes = []
    try:
        readers = [imageio.get_reader(path) for path in videos]
        counts = [reader.count_frames() for reader in readers]
        if counts != [count, count]:
            raise ValueError(f"Native video/TacMap frame mismatch: {counts} vs {count}")
        if any(abs(float(reader.get_meta_data()["fps"]) - args.fps) > 1e-5 for reader in readers):
            raise ValueError("Native video fps differs from requested playback")
        peak_lift = int(np.argmax(trajectory["lift"][steps].reshape(count, -1)[:, 0])) if "lift" in trajectory else 0
        peak_deformation = int(np.argmax(tacmap["deformation_uint8"].reshape(count, -1).max(axis=1)))
        keys = {0, count - 1, peak_lift, peak_deformation, *args.extra_keyframes}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bitrate = max(64_000, int(args.max_mb * 1e6 * 8 * .91 / max(count / args.fps, 1.)))
        writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", bitrate=str(bitrate),
                                    macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"])
        composer = Composer(tacmap, tactile, trajectory, args.title, args.contact_note, args.tacmap_note)
        for index, step in enumerate(steps):
            frame = composer.frame([reader.get_data(index) for reader in readers], index, int(step))
            writer.append_data(np.asarray(frame))
            if index in keys:
                path = args.output.with_name(f"{args.output.stem}_frame_{index:05d}.png")
                frame.save(path)
                keyframes.append(str(path))
            if index == peak_lift:
                frame.save(args.output.with_name(args.output.stem + "_cover.png"))
    finally:
        if writer is not None:
            writer.close()
        for reader in readers:
            reader.close()
    with imageio.get_reader(args.output) as reader:
        decoded = reader.count_frames()
        dimensions = list(reader.get_meta_data()["size"])
    if decoded != count or dimensions != [1920, 1280]:
        raise RuntimeError("Encoded frame count or dimensions differ from the source layout")
    if args.output.stat().st_size > args.max_mb * 1e6:
        raise RuntimeError("Encoded file exceeds requested delivery size")
    maps, valid = tacmap["deformation_uint8"], tacmap["surface_valid"]
    stats = {}
    for finger, name in enumerate(FINGERS):
        values = maps[:, finger]
        stats[name] = {"maximum_uint8": int(values.max()), "nonzero_pixel_samples": int(np.count_nonzero(values)),
                       "frames_with_nonzero_output": int(np.any(values > 0, axis=(1, 2)).sum()),
                       "nonzero_samples_outside_surface_valid": int(np.count_nonzero(values[:, ~valid[finger]]))}
    report = {"source_run": str(run), "source_tacmap": str(tacmap_path), "source_tacmap_sha256": sha256(tacmap_path),
              "source_tactile_sha256": sha256(force_path), "source_trajectory_sha256": sha256(trajectory_path),
              "source_videos": [{"path": str(Path(path).resolve()), "sha256": sha256(path)} for path in videos],
              "tacmap_metadata": metadata, "finger_display_order": list(FINGERS), "tacmap_stats": stats,
              "tacmap_display": "Original uint8 values, fixed 0--255 grayscale, nearest-neighbour resize. No added smoothing, threshold, fill, normalization, mask suppression or inferred millimetres.",
              "physx_labels": "Merged distal-phalanx (DP) body normal-contact load sum(|Fn|) and friction-anchor load sum(|Ft|), in N; dt-weighted mean over each video interval. Skin and shell contact contributions are not separately attributed. These are not isolated skin forces or pressures and are not inferred from TacMap.",
              "tacmap_time_semantics": "Render endpoint; distinct from PhysX frame-interval force averaging",
              "validation_required": args.require_validated,
              "validation_sha256": sha256(validation_path) if args.require_validated else None,
              "title": args.title, "contact_note": args.contact_note, "tacmap_note": args.tacmap_note, "frames": count, "decoded_frames": decoded,
              "fps": args.fps, "playback_speed": 1., "dimensions": dimensions, "keyframes": keyframes,
              "bytes": args.output.stat().st_size, "output_sha256": sha256(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "frames": count, "bytes": report["bytes"],
                      "sha256": report["output_sha256"], "tacmap_stats": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
