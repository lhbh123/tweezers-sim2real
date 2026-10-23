#!/usr/bin/env python3
"""Render verified, real official TacMap closed-box controls on one fixed scale.

Only reads controls.npz after controls_validation.json has valid:true and an
archive hash matching the actual file. No simulation, generated control values,
image normalization, extra blur or depth inversion is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAMES = [model + suffix for model in ("TH", "4F")
                  for suffix in ("_clear", "_indent_0.1mm", "_indent_0.5mm", "_indent_1mm")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace existing image: {args.output}")
    run = args.run.resolve()
    archive, validation_path = run / "controls.npz", run / "controls_validation.json"
    validation = json.loads(validation_path.read_text())
    archive_hash = sha(archive)
    if validation.get("valid") is not True or validation.get("geometry_controls", {}).get("sha256") != archive_hash:
        raise ValueError("Requires passed independent controls validation tied to this exact archive")
    with np.load(archive, allow_pickle=False) as data:
        values = data["deformation_uint8"]
        names = data["case_names"].tolist()
        metadata = json.loads(str(data["metadata_json"].item()))
        counters = data["sensor_frame_counter"]
    step = int(metadata["resolution_step"])
    resolution = 240 // step
    if step not in (1, 3) or values.dtype != np.uint8 or values.shape != (8, 2, resolution, resolution):
        raise ValueError("Unexpected official control image dtype, shape or resolution")
    if names != EXPECTED_NAMES or [row["name"] for row in metadata["cases"]] != names:
        raise ValueError("Expected TH and 4F clear/0.1/0.5/1mm cases in verified order")
    if not np.array_equal(values[:, 0], values[:, 1]) or counters.shape != (8, 2) or not (np.diff(counters, axis=1) == 1).all():
        raise ValueError("Two actual fixed-fixture updates must agree and advance the sensor counter")
    if metadata.get("geometry_control_only") is not True or metadata.get("task_success_claimed") is not False:
        raise ValueError("Archive does not identify itself as separate geometry controls")
    surface_provenance = metadata.get("surface_provenance", {})
    derived = surface_provenance.get("kind") == "current_W01_CAD_derived_surface"
    canvas = Image.new("RGB", (1920, 1280), (10, 17, 28))
    draw = ImageDraw.Draw(canvas)
    fonts = {}
    white, muted, cyan, amber = (232, 240, 250), (151, 172, 196), (66, 212, 201), (255, 180, 78)

    def label(xy, text, size=26, colour=white):
        if size not in fonts:
            fonts[size] = ImageFont.truetype(str(ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf"), size)
        draw.text(xy, text, font=fonts[size], fill=colour)

    label((40, 18), "Sharpa 官方 TacMap：独立几何压入校验", 48)
    subtitle = (f"当前 W01 CAD 派生表面（非原版 npy）· 官方算法未改 · {resolution}×{resolution} 真实输出"
                if derived else f"真实官方输出 {resolution}×{resolution} · 独立解析核对通过 · 同一静态夹具两次实际更新一致")
    label((42, 91), subtitle, 27, cyan)
    headings = ("分离 1.0 mm · 负控", "名义压入 0.1 mm", "名义压入 0.5 mm", "名义压入 1.0 mm")
    for column, heading in enumerate(headings):
        label((250 + column * 400, 163), heading, 29)
    provenance = []
    control_rows = validation["geometry_controls"]["cases"]
    for row, model in enumerate(("TH", "4F")):
        y = 225 + row * 452
        label((42, y + 100), "拇指" if row == 0 else "四指共用", 31)
        label((42, y + 151), model + " 表面", 27, muted)
        for column in range(4):
            index, x = row * 4 + column, 250 + column * 400
            raw = values[index, 0]
            picture = Image.fromarray(raw).convert("RGB").resize((340, 340), Image.Resampling.NEAREST)
            canvas.paste(picture, (x, y))
            draw.rectangle((x - 1, y - 1, x + 340, y + 340), outline=(65, 83, 104), width=1)
            peak, active = int(raw.max()), int(np.count_nonzero(raw))
            verified = control_rows[index]
            if verified["case"] != names[index] or verified["actual_peak_code"] != peak or verified["nonzero_pixels"] != active:
                raise ValueError("Validation case statistics do not match the selected actual update")
            label((x, y + 355), f"峰值 {peak} / 255 · 非零 {active} 像素", 23, muted)
            label((x, y + 389), f"独立重算最大差 {verified['max_code_difference']} 个等级", 20, muted)
            provenance.append({"case": names[index], "case_index": index, "repeat_index": 0,
                               "native_size": [resolution, resolution], "display_size": [340, 340],
                               "raw_uint8_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                               "sensor_frame_counter": int(counters[index, 0]),
                               "peak_uint8": peak, "nonzero_pixels": active,
                               "nominal_surface_plane_indent_m": metadata["cases"][index]["nominal_surface_plane_indent_m"]})
    label((42, 1130), "独立静态几何校验；不代表夹放任务接触、FEM 形变或压力。", 31, amber)
    label((42, 1184), "名义深度是闭盒相对源表面支撑平面的位移；曲面各像素的法向穿入深度不同。", 24, muted)
    label((42, 1225), "图像保留官方量化与模糊后的 uint8 等级；统一灰度量程，不追加模糊、不归一化。", 24, muted)
    bar = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
    canvas.paste(Image.fromarray(bar).convert("RGB"), (1570, 1176))
    label((1568, 1200), "0", 20, muted)
    label((1680, 1200), "128", 20, muted)
    label((1790, 1200), "255", 20, muted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    report = {"image": str(args.output.resolve()), "size": list(canvas.size),
              "image_sha256": sha(args.output), "archive": str(archive), "archive_sha256": archive_hash,
              "validation": str(validation_path), "validation_sha256": sha(validation_path),
              "official_commit": metadata.get("official_commit"), "official_sources": metadata.get("official_sources"),
              "surface_provenance": surface_provenance,
              "compatibility": metadata.get("compatibility"), "source_cases": provenance,
              "scope": "Independent closed-box geometric sensing controls; not physical task contact, FEM, force or pressure.",
              "operations": ["fixed 0--255 grayscale", "nearest-neighbour display scaling", "layout", "captions outside images"],
              "selected_actual_update": 0, "repeat_equality_verified": True}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"image": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": report["image_sha256"], "source_archive_sha256": archive_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
