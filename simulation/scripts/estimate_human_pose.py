#!/usr/bin/env python3
"""Extract reproducible human hand priors from an actual downloaded video.

Run inside .venv-human. The MediaPipe score is a handedness classification score,
not a calibrated landmark-localization or occlusion confidence. Monocular world
landmarks are a learned estimate, never motion-capture ground truth. Failed
detections remain in the output instead of being fabricated or interpolated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata

import imageio.v2 as imageio
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode


CHAINS = {"thumb": [0, 1, 2, 3, 4], "index": [0, 5, 6, 7, 8],
          "middle": [0, 9, 10, 11, 12], "ring": [0, 13, 14, 15, 16],
          "pinky": [0, 17, 18, 19, 20]}
COLORS = [(245, 95, 75), (255, 210, 70), (80, 220, 130), (65, 170, 250), (210, 110, 245)]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bend_angles(points):
    result = {}
    for name, chain in CHAINS.items():
        values = []
        for a, b, c in zip(chain, chain[1:], chain[2:]):
            u = points[b] - points[a]
            v = points[c] - points[b]
            denom = float(np.linalg.norm(u) * np.linalg.norm(v))
            values.append(float(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1))) if denom > 1e-10 else None)
        result[name] = values
    return result


def draw_hand(pixels, landmarks, label):
    im = Image.fromarray(pixels[:, :, :3])
    draw = ImageDraw.Draw(im)
    points = [(round(p[0] * im.width), round(p[1] * im.height)) for p in landmarks]
    for chain, color in zip(CHAINS.values(), COLORS):
        draw.line([points[i] for i in chain], fill=color, width=max(2, im.width // 300))
        for i in chain:
            x, y = points[i]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(10, 10, 10))
            draw.text((x + 4, y + 2), str(i), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    draw.rectangle((0, 0, im.width, 28), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255))
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--source-url", required=True)
    p.add_argument("--video-id", required=True)
    p.add_argument("--model", type=Path, default=Path("references/human/models/hand_landmarker.task"))
    p.add_argument("--output-dir", type=Path, default=Path("references/human"))
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float)
    p.add_argument("--sample-fps", type=float, default=2)
    p.add_argument("--times", help="Comma-separated exact sample times, overrides start/end/sample-fps")
    p.add_argument("--num-hands", type=int, default=2)
    p.add_argument("--review", type=Path, help="Optional explicit visual/context review file; defaults to output-dir/pose_review.json if present")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = a.output_dir / "keyframes"
    frames_dir.mkdir(exist_ok=True)
    reader = imageio.get_reader(str(a.video))
    info = reader.get_meta_data()
    # OpenCV retains the rational 24000/1001 rate that imageio rounds to 23.98.
    probe = cv2.VideoCapture(str(a.video))
    fps = float(probe.get(cv2.CAP_PROP_FPS)) or float(info["fps"])
    frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else float(info["duration"])
    probe.release()
    if a.times:
        times = sorted(set(float(t) for t in a.times.split(",")))
    else:
        times = np.arange(a.start, min(a.end or duration, duration), 1 / a.sample_fps).tolist()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(a.model)), running_mode=RunningMode.IMAGE,
        num_hands=a.num_hands, min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35, min_tracking_confidence=0.5,
    )
    records = []
    with HandLandmarker.create_from_options(options) as detector:
        for time in times:
            frame_index = round(time * fps)
            pixels = reader.get_data(frame_index)[:, :, :3]
            exact_time = frame_index / fps
            stem = f"{a.video_id}_{exact_time:07.3f}s"
            original = frames_dir / (stem + ".jpg")
            Image.fromarray(pixels).save(original, quality=95)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels)))
            if not result.hand_landmarks:
                records.append({"time_s": exact_time, "frame_index": frame_index, "detected": False,
                                "frame_path": str(original), "failure": "No hand returned at configured thresholds",
                                "method": "MediaPipe HandLandmarker IMAGE mode", "confidence": None,
                                "landmarks_world_m": None, "landmarks_normalized": None})
            for hand_index, (norm, world, handedness) in enumerate(zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness
            )):
                norm = [[float(v.x), float(v.y), float(v.z)] for v in norm]
                world = [[float(v.x), float(v.y), float(v.z)] for v in world]
                category = handedness[0]
                pixels_xy = np.asarray(norm)[:, :2] * [pixels.shape[1], pixels.shape[0]]
                annotation = frames_dir / (stem + f"_hand{hand_index}_landmarks.jpg")
                draw_hand(pixels, norm, f"{a.video_id} / {exact_time:.3f}s / {category.category_name} / model estimate").save(annotation, quality=95)
                records.append({
                    "time_s": exact_time, "frame_index": frame_index, "detected": True,
                    "hand_index": hand_index, "handedness": category.category_name,
                    "confidence": float(category.score), "confidence_kind": "handedness classification only",
                    "landmark_localization_confidence": None, "occlusion_confidence": None,
                    "landmarks_world_m": world, "landmarks_normalized": norm,
                    "landmarks_pixel_xy": pixels_xy.tolist(),
                    "joint_bend_proxy_rad": bend_angles(np.asarray(world)),
                    "joint_bend_proxy_2d_rad": bend_angles(pixels_xy),
                    "method": "MediaPipe HandLandmarker IMAGE mode, CPU, monocular learned 3D estimate",
                    "frame_path": str(original), "annotated_frame_path": str(annotation),
                    "occlusion_notes": "Automatic output; finger/tool occlusion and handedness require visual review. Per-point occlusion confidence is not supplied by this API.",
                    "visual_review": "pending",
                })
            print(f"time={exact_time:.3f}s hands={len(result.hand_landmarks)}", flush=True)
    reader.close()
    review_path = a.review or a.output_dir / "pose_review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        reviews = {(r["frame_index"], r.get("hand_index", 0)): r for r in review["frames"]}
        for record in records:
            record.update(reviews.get((record["frame_index"], record.get("hand_index", 0)), {}))
    output = {
        "source_url": a.source_url, "video_id": a.video_id, "video_path": str(a.video),
        "video_sha256": sha256(a.video), "model_sha256": sha256(a.model),
        "model_url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "mediapipe_version": importlib.metadata.version("mediapipe"),
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_fps": fps,
        "duration_s": duration, "resolution": list(info["size"]),
        "landmark_order": "MediaPipe 21: wrist 0; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20",
        "joint_angle_warning": "Angles are segment bend proxies. MCP proxy uses wrist-to-MCP; it is not an anatomical joint-coordinate measurement.",
        "world_coordinate_warning": "Monocular inferred depth and metric scale are uncertain. Source camera calibration is unavailable.",
        "frames": records,
        "review_path": str(review_path) if review_path.exists() else None,
    }
    (a.output_dir / "landmarks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {len(records)} hand/frame records to {a.output_dir / 'landmarks.json'}")


if __name__ == "__main__":
    main()
