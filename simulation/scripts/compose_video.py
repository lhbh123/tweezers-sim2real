#!/usr/bin/env python3
"""Compose original simulation camera videos and synchronized telemetry HUD.

Example:
    /cpfs/user/lvzhaoyang/apps/bin/isaac-python scripts/compose_video.py \
        runs/my_run --output-dir runs/my_run/delivery

Inputs: overview.mp4, closeup.mp4, telemetry.json, metadata.json and optionally
result.json. The camera images are resized with aspect ratio preserved; no scene
content is inserted, removed, interpolated, relit or altered. The canvas and all
labels are outside the camera rectangles. No synthetic success label is used.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
BG = (10, 16, 24)
PANEL = (19, 28, 39)
BORDER = (43, 59, 75)
WHITE = (235, 242, 248)
MUTED = (140, 160, 180)
CYAN = (74, 216, 208)
AMBER = (255, 194, 102)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="Directory containing both camera videos and telemetry")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--output-name")
    p.add_argument("--closeup-only", action="store_true", help="Single enlarged 4:3 closeup with outside-image HUD")
    p.add_argument("--playback-speed", type=float, default=1.0, help="0.5 plays every original frame at half speed; HUD retains simulation time")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--max-mb", type=float, default=24.0, help="Decimal MB; default safely below 25 MB")
    p.add_argument("--cover-time", type=float, help="Simulation timestamp for cover; default peak recorded lift")
    p.add_argument("--time-offset", type=float, default=0.0, help="Simulation time at input frame 0")
    p.add_argument("--table-height", type=float, help="Override metadata table_height, in metres")
    p.add_argument("--block-size", type=float, default=0.012, help="Block side length, in metres")
    p.add_argument("--title", default="UR5e + XHand | Tweezer manipulation")
    a = p.parse_args()
    if a.width < 960 or a.width % 32:
        p.error("width must be at least 960 and divisible by 32")
    if a.max_mb <= 0 or a.block_size <= 0 or not 0 < a.playback_speed <= 8:
        p.error("max-mb/block-size must be positive and playback-speed must be in (0, 8]")
    a.output_name = a.output_name or ("closeup_demo.mp4" if a.closeup_only else "demo.mp4")
    if Path(a.output_name).name != a.output_name or not a.output_name.endswith(".mp4"):
        p.error("output-name must be an .mp4 basename")
    a.run = a.run.resolve()
    a.output_dir = (a.output_dir or a.run / "delivery").resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    return a


def read_json(path, optional=False):
    if optional and not path.exists():
        return {}
    return json.loads(path.read_text())


def finite_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def contact_forces(record):
    """Magnitudes of filtered block contact-resultant vectors, left then right."""
    value = record.get("force")
    if value is None:
        return (None, None)
    force = np.asarray(value, dtype=np.float64)
    if force.size != 6 or force.shape[-1] != 3:
        raise ValueError(f"Expected two 3D contact vectors, got shape {force.shape}")
    values = np.linalg.norm(force.reshape(2, 3), axis=1)
    return tuple(finite_float(v) for v in values)


class Compositor:
    """Draw in an explicit 1920x1080 reference layout at the requested scale."""

    def __init__(self, width, title, fixture, model, duration, baseline,
                 closeup_only=False, playback_speed=1.0, reset_release=None):
        self.scale = width / 1920
        self.closeup_only = closeup_only
        self.playback_speed = playback_speed
        self.reset_release = reset_release
        self.size = (width, 2 * round((1776 if closeup_only else 1080) * self.scale / 2))
        self.title = title
        self.fixture = fixture
        self.model = model
        self.duration = duration
        self.baseline = baseline
        self.fonts = {}

    def xy(self, x, y):
        return (round(x * self.scale), round(y * self.scale))

    def font(self, size, bold=False, mono=False):
        key = (size, bold, mono)
        if key not in self.fonts:
            stem = "DejaVuSansMono" if mono else "DejaVuSans"
            path = FONT_ROOT / (stem + ("-Bold" if bold else "") + ".ttf")
            self.fonts[key] = ImageFont.truetype(str(path), max(9, round(size * self.scale)))
        return self.fonts[key]

    def label(self, draw, text, x, y, size=24, color=WHITE, bold=False, mono=False):
        draw.text(self.xy(x, y), str(text), font=self.font(size, bold, mono), fill=color)

    def fit_label(self, draw, text, x, y, width, size=24, color=WHITE, bold=False):
        while size > 14 and draw.textlength(text, font=self.font(size, bold)) > width * self.scale:
            size -= 1
        self.label(draw, text, x, y, size, color, bold)

    def rect(self, draw, box, fill, outline=None, radius=14):
        draw.rounded_rectangle((*self.xy(*box[:2]), *self.xy(*box[2:])),
                               radius=round(radius * self.scale), fill=fill,
                               outline=outline, width=max(1, round(self.scale)))

    def camera(self, canvas, pixels, box):
        image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)[:, :, :3], "RGB")
        x, y, width, height = box
        bounds = self.xy(width, height)
        fit = min(bounds[0] / image.width, bounds[1] / image.height)
        image = image.resize((round(image.width * fit), round(image.height * fit)), Image.Resampling.LANCZOS)
        origin = self.xy(x, y)
        canvas.paste(image, (origin[0] + (bounds[0] - image.width) // 2,
                            origin[1] + (bounds[1] - image.height) // 2))

    def frame(self, overview, closeup, record, video_time):
        canvas = Image.new("RGB", self.size, BG)
        draw = ImageDraw.Draw(canvas)
        self.label(draw, "DEXTEROUS TOOL USE", 40, 22, 21, CYAN, True)
        self.fit_label(draw, self.title, 40, 55, 1360, 42, WHITE, True)
        self.rect(draw, (1505, 32, 1880, 84), PANEL, BORDER)
        self.label(draw, "PHYSX SIMULATION", 1531, 46, 22, MUTED, True)

        active = record.get("fixture_active")
        if active is not None and not isinstance(active, bool):
            raise ValueError("telemetry.fixture_active must be a boolean")
        if active is None and self.fixture is False and self.reset_release is not None:
            active = float(record["t"]) < self.reset_release
        # A permanent fixture declaration is never hidden by a transient flag.
        if self.fixture is True:
            model_text = "PRE-HELD HANDLE FIXTURE  |  PASSIVE SPRING TWEEZERS"
            model_color = AMBER
        elif active is True:
            model_text = "RESET FIXTURE ACTIVE  |  PASSIVE SPRING JAWS"
            model_color = AMBER
        elif self.fixture is False:
            model_text = "FRICTION-HELD TOOL  |  PASSIVE SPRING JAWS  |  NO HANDLE FIXTURE"
            model_color = CYAN
        else:
            model_text = "HANDLE CONSTRAINT: UNSPECIFIED  |  " + self.model.upper()
            model_color = AMBER
        self.fit_label(draw, model_text, 40, 116, 1420, 23, model_color, True)
        self.label(draw, f"PLAYBACK {self.playback_speed:g}x", 1640, 116, 20, MUTED, True)

        if self.closeup_only:
            self.label(draw, "HAND + TWEEZERS / CLOSEUP", 40, 160, 21, MUTED, True)
            views = [(closeup, (40, 200, 1840, 1380))]
        else:
            self.label(draw, "01 / ARM + HAND", 40, 160, 21, MUTED, True)
            self.label(draw, "02 / TOOL + BLOCK", 968, 160, 21, MUTED, True)
            views = [(overview, (40, 200, 912, 684)), (closeup, (968, 200, 912, 684))]
        # Both rectangles have the original 4:3 aspect ratio. The camera method
        # still letterboxes any differently sized source instead of cropping.
        for pixels, (x, y, width, height) in views:
            draw.rectangle((*self.xy(x - 1, y - 1), *self.xy(x + width + 1, y + height + 1)), fill=BORDER)
            draw.rectangle((*self.xy(x, y), *self.xy(x + width, y + height)), fill=(0, 0, 0))
            self.camera(canvas, pixels, (x, y, width, height))

        # The lower HUD is outside both source images.
        shift = 696 if self.closeup_only else 0
        cards = [(40, 430), (450, 735), (755, 1090), (1110, 1880)]
        for x0, x1 in cards:
            self.rect(draw, (x0, 907 + shift, x1, 1016 + shift), PANEL, BORDER)
        phase = str(record.get("phase", "unreported")).replace("_", " ").upper()
        self.label(draw, "CONTROL PHASE", 60, 923 + shift, 16, MUTED, True)
        self.fit_label(draw, phase, 60, 953 + shift, 350, 31, WHITE, True)
        self.label(draw, "SIMULATION TIME", 470, 923 + shift, 16, MUTED, True)
        sample_t = finite_float(record.get("t"))
        self.label(draw, "--" if sample_t is None else f"{sample_t:05.2f} s", 470, 951 + shift, 34, WHITE, True, True)
        block = record.get("block")
        height = finite_float(block[2]) if block is not None and len(block) >= 3 else None
        lift_mm = (height - self.baseline) * 1000 if height is not None else None
        self.label(draw, "BLOCK LIFT", 775, 923 + shift, 16, MUTED, True)
        self.label(draw, "--" if lift_mm is None else f"{lift_mm:6.1f} mm", 775, 951 + shift, 34, CYAN, True, True)
        left, right = contact_forces(record)
        self.label(draw, "JAW NORMAL CONTACT FORCE ON BLOCK", 1130, 923 + shift, 16, MUTED, True)
        left_text = "--" if left is None else f"{left:.3f} N"
        right_text = "--" if right is None else f"{right:.3f} N"
        self.fit_label(draw, f"LEFT {left_text}     RIGHT {right_text}", 1130, 956 + shift, 720, 29, WHITE, True)

        self.label(draw, "RAW SIMULATION CAMERAS + RECORDED TELEMETRY", 40, 1034 + shift, 15, MUTED)
        self.label(draw, "FORCES: NORMAL RESULTANTS", 1485, 1034 + shift, 15, MUTED)
        progress = np.clip(video_time / max(self.duration, 1e-6), 0, 1)
        draw.rectangle((*self.xy(40, 1065 + shift), *self.xy(1880, 1070 + shift)), fill=BORDER)
        draw.rectangle((*self.xy(40, 1065 + shift), *self.xy(40 + 1840 * progress, 1070 + shift)), fill=CYAN)
        return np.asarray(canvas)


def main():
    args = arguments()
    records = read_json(args.run / "telemetry.json")
    if isinstance(records, dict):
        records = records.get("records", records.get("telemetry"))
    if not isinstance(records, list) or not records:
        raise ValueError("telemetry.json must contain a non-empty record list")
    meta = read_json(args.run / "metadata.json")
    result = read_json(args.run / "result.json", optional=True)
    times = [float(r["t"]) for r in records]
    if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Telemetry timestamps must be finite and strictly increasing")
    table = args.table_height if args.table_height is not None else meta.get("table_height")
    if table is None:
        raise ValueError("table_height is required in metadata.json or --table-height")
    baseline = float(table) + args.block_size / 2
    fixture = meta.get("handle_fixed_to_hand", result.get("handle_fixed"))
    if fixture is not None:
        if not isinstance(fixture, bool):
            raise ValueError("Handle fixture metadata must be a boolean")
        if result.get("handle_fixed", fixture) != fixture:
            raise ValueError("metadata.json and result.json disagree about the handle fixture")
    names = ("closeup.mp4",) if args.closeup_only else ("overview.mp4", "closeup.mp4")
    paths = [args.run / name for name in names]
    readers = [imageio.get_reader(str(path), format="ffmpeg") for path in paths]
    try:
        video_meta = [reader.get_meta_data() for reader in readers]
        rates = [float(v["fps"]) for v in video_meta]
        if max(rates) - min(rates) > 1e-3:
            raise ValueError(f"Camera frame rates do not match: {rates}")
        counts = [int(reader.count_frames()) for reader in readers]
        if len(set(counts)) > 1 or counts[0] < 1:
            raise ValueError(f"Camera frame counts do not match or are empty: {counts}")
        fps, count = rates[0], counts[0]
        output_fps = fps * args.playback_speed
        duration = count / output_fps
        final_sim_time = args.time_offset + (count - 1) / fps
        if args.time_offset < times[0] - 1 / fps or final_sim_time > times[-1] + 1 / fps:
            raise ValueError("Video timestamps extend outside recorded telemetry; check --time-offset")
        if args.cover_time is None:
            valid = [r for r in records if r.get("block") is not None and len(r["block"]) >= 3]
            cover_time = max(valid, key=lambda r: float(r["block"][2]))["t"] if valid else times[-1]
        else:
            cover_time = args.cover_time
        cover_index = int(np.clip(round((cover_time - args.time_offset) * fps), 0, count - 1))
        renderer = Compositor(args.width, args.title, fixture,
                              str(meta.get("tool_model", "unspecified")), duration, baseline,
                              args.closeup_only, args.playback_speed, meta.get("reset_fixture_release_s"))
        limit_bytes = int(args.max_mb * 1_000_000)
        bitrate = min(8_000_000, int(limit_bytes * 8 * 0.82 / duration))
        if bitrate < 300_000:
            raise ValueError("Requested size cap is too small for this duration; increase --max-mb")
        output = args.output_dir / args.output_name
        cover_path = args.output_dir / ("closeup_cover.png" if args.closeup_only else "cover.png")
        writer = imageio.get_writer(
            str(output), format="ffmpeg", fps=output_fps, codec="libx264", quality=None,
            bitrate=str(bitrate), pixelformat="yuv420p", macro_block_size=1,
            ffmpeg_params=["-preset", "medium", "-maxrate", str(bitrate),
                           "-bufsize", str(bitrate), "-movflags", "+faststart", "-threads", "4"],
        )
        try:
            for index in range(count):
                sim_time = args.time_offset + index / fps
                # Last-observation-held avoids showing any future physics result.
                sample_index = max(0, bisect_right(times, sim_time + 1e-9) - 1)
                frames = [reader.get_data(index) for reader in readers]
                frame = renderer.frame(None if args.closeup_only else frames[0], frames[-1],
                                       records[sample_index], index / output_fps)
                writer.append_data(frame)
                if index == cover_index:
                    Image.fromarray(frame).save(cover_path)
                if index % max(1, int(fps * 2)) == 0:
                    print(f"Composed {index + 1}/{count} frames", flush=True)
        finally:
            writer.close()
        size = output.stat().st_size
        if size >= limit_bytes:
            raise RuntimeError(f"Encoded video exceeds size cap ({size} bytes); rerun with smaller --width or --max-mb")
        manifest = {
            "video": str(output), "cover": str(cover_path),
            "source_videos": [str(path) for path in paths],
            "source_telemetry": str(args.run / "telemetry.json"),
            "frames": count, "fps": output_fps, "source_fps": fps,
            "playback_speed": args.playback_speed, "duration_seconds": duration,
            "view_mode": "closeup_only" if args.closeup_only else "side_by_side",
            "resolution": list(renderer.size), "bytes": size, "size_cap_bytes": limit_bytes,
            "simulation_time_offset": args.time_offset, "cover_simulation_time": args.time_offset + cover_index / fps,
            "telemetry_alignment": "last observation at or before each video-frame simulation timestamp",
            "lift_formula": f"1000 * (block_z - {baseline:.9f}) mm",
            "contact_force_formula": "Euclidean norm of each of two recorded 3D contact-resultant vectors",
            "handle_fixed": fixture, "model_description": meta.get("tool_model"),
            "reset_fixture_release_s": meta.get("reset_fixture_release_s"),
            "fixture_label_source": "telemetry.fixture_active, with permanent-fixture metadata always taking precedence",
            "reported_result": result,
            "image_operations": ["aspect-ratio-preserving resize", "letterbox if needed", "outside-image HUD"],
        }
        manifest_name = "closeup_composition_manifest.json" if args.closeup_only else "composition_manifest.json"
        (args.output_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"video": str(output), "cover": str(cover_path),
                          "size_mb": round(size / 1_000_000, 3)}, indent=2))
    finally:
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"compose_video: {exc}", file=sys.stderr)
        raise SystemExit(2)
