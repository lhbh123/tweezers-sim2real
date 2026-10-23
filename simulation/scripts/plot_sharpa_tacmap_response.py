#!/usr/bin/env python3
"""Plot validated geometric TacMap codes and independently sampled PhysX loads.

Both inputs must be completed, successful real runs with an independent TacMap
validation tied to the exact archive. No interpolation, smoothing, code-to-force
conversion, or image brightness adjustment is performed. The matched surface
must identify itself as derived from current W01 CAD; baseline assets must match
the archived original official files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np

from compose_sharpa_tacmap_video import ROOT, FINGERS, load_npz, sha256, validate_inputs


def read_run(run, derived):
    run = run.resolve()
    validation_file = run / "tacmap_validation.json"
    validation = json.loads(validation_file.read_text())
    archive_hash = sha256(run / "tacmap.npz")
    if validation.get("valid") is not True or validation.get("tacmap_npz_sha256") != archive_hash:
        raise ValueError(f"Independent TacMap validation missing or mismatched: {run}")
    force_hash = sha256(run / "tactile.npz")
    force_validation_file = run / "tactile_validation.json"
    force_validation = json.loads(force_validation_file.read_text())
    if force_validation.get("valid") is not True or force_validation.get("tactile_npz_sha256") != force_hash:
        raise ValueError(f"Independent force conservation validation missing or mismatched: {run}")
    evaluation = json.loads((run / "evaluation.json").read_text())
    if evaluation.get("episodes") != 1 or evaluation.get("successes") != 1:
        raise ValueError(f"Expected one completed successful task: {run}")
    tacmap = load_npz(run / "tacmap.npz", ("frame_time_s", "deformation_uint8", "finger_names", "surface_valid", "metadata_json"))
    force = load_npz(run / "tactile.npz", ("frame_time_s", "finger_names", "normal_load_N", "friction_load_N"))
    trajectory = load_npz(run / "trajectory.npz", ("render_control_steps", "dt", "phase", "lift", "goal_error", "both", "success"))
    tacmap, force, steps, metadata = validate_inputs(tacmap, force, trajectory, fps=30.)
    if metadata.get("backend") != "official SharpaTacmap" or metadata.get("output_semantics") != "quantized_blurred_normal_penetration":
        raise ValueError("Expected real official geometric TacMap output")
    provenance = metadata.get("surface_provenance", {})
    if derived:
        if provenance.get("kind") != "current_W01_CAD_derived_surface":
            raise ValueError("Matched input does not declare current W01 CAD-derived maps")
    else:
        if provenance.get("kind") not in (None, "unmodified_official_TacMap_surface"):
            raise ValueError("Baseline input is not original official maps")
        # The first baseline was recorded before surface_provenance was added.
        # Verify all actual archived map hashes rather than infer from the name.
        for sensor in metadata["sensors"]:
            for item in sensor["surface_assets"]:
                original = ROOT / "assets/sharpa_tactile/assets/tactilesensor_map" / Path(item["path"]).name
                if sha256(original) != item["sha256"]:
                    raise ValueError("Baseline surface differs from pinned original arrays")
    codes = tacmap["deformation_uint8"].max(axis=(2, 3))
    data = {"time_s": tacmap["frame_time_s"], "peak_code": codes,
            "normal_load_N": force["normal_load_N"],
            "friction_load_N": force["friction_load_N"],
            "phase": trajectory["phase"][steps].reshape(len(steps), -1)[:, 0]}
    provenance = {"run": str(run), "frame_count": len(steps), "tacmap_sha256": archive_hash,
                  "tactile_sha256": force_hash,
                  "trajectory_sha256": sha256(run / "trajectory.npz"),
                  "validation_sha256": sha256(validation_file),
                  "force_validation_sha256": sha256(force_validation_file),
                  "evaluation_sha256": sha256(run / "evaluation.json"),
                  "surface_provenance": provenance, "official_sources": metadata["official_sources"],
                  "contact_physics": metadata.get("contact_physics"),
                  "per_finger": {name: {"peak_code": int(codes[:, i].max()),
                                        "frames_with_nonzero_code": int((codes[:, i] > 0).sum()),
                                        "maximum_frame_mean_normal_load_N": float(force["normal_load_N"][:, i].max())}
                                 for i, name in enumerate(FINGERS)}}
    return data, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matched", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Refusing to replace existing response evidence")
    matched, matched_source = read_run(args.matched, derived=True)
    baseline, baseline_source = read_run(args.baseline, derived=False)
    if matched_source["official_sources"] != baseline_source["official_sources"]:
        raise ValueError("Comparison uses different official implementation hashes")
    if matched_source["contact_physics"] != baseline_source["contact_physics"]:
        raise ValueError("Comparison metadata declares different contact physics")
    alignment_file = args.matched / "tacmap_force_alignment.json"
    alignment = json.loads(alignment_file.read_text()) if alignment_file.exists() else None
    if alignment:
        expected = {"tacmap.npz": matched_source["tacmap_sha256"], "tactile.npz": matched_source["tactile_sha256"]}
        if alignment.get("source_sha256") != expected:
            raise ValueError("Force/code alignment report refers to different archives")

    font_manager.fontManager.addfont(ROOT / "assets/fonts/NotoSansCJKsc-Regular.otf")
    plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False,
                         "font.size": 15, "axes.labelsize": 15, "axes.titlesize": 18,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.subplots_adjust(left=.10, right=.89, top=.80, bottom=.17, hspace=.36)
    fig.suptitle("W01 实际任务：TacMap 量化响应与独立 PhysX 法向力", x=.5, y=.97, fontsize=24)
    fig.text(.5, .92, "当前 W01 CAD 派生表面（非原版 npy）· 官方算法未改 · 原接触参数", ha="center", fontsize=16)
    fig.text(.5, .883, "蓝色读取左轴：法向穿入的量化 / 模糊等级；橙色读取右轴：独立接触力 N", ha="center", fontsize=15)
    code_color, force_color, baseline_color = "#1464a0", "#c76413", "#69727a"
    names = ("拇指", "食指", "中指")
    longest = max(float(matched["time_s"][-1]), float(baseline["time_s"][-1]))
    code_upper = max(1, int(max(matched["peak_code"][:, :3].max(), baseline["peak_code"][:, :3].max())))
    shared_left_limits = (-max(.2, code_upper * .045), code_upper * 1.13 + .2)
    legend_handles = []
    axis_ranges = []
    for index, (name, ax) in enumerate(zip(names, axes)):
        other = ax.twinx()
        force = other.plot(matched["time_s"], matched["normal_load_N"][:, index],
                           color=force_color, linewidth=1.3, alpha=.75, label="匹配表面运行：PhysX 法向力（右轴 N）")[0]
        base = ax.step(baseline["time_s"], baseline["peak_code"][:, index], where="post",
                       color=baseline_color, linestyle="--", linewidth=1.6, label="原版表面运行：TacMap 峰值码（左轴）")[0]
        code = ax.step(matched["time_s"], matched["peak_code"][:, index], where="post",
                       color=code_color, linewidth=1.7, label="匹配表面运行：TacMap 峰值码（左轴）")[0]
        ax.set_zorder(other.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_ylim(*shared_left_limits)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, min_n_ticks=2))
        other.set_ylim(0, max(.01, float(matched["normal_load_N"][:, index].max()) * 1.12))
        ax.set_ylabel("TacMap 峰值码", color=code_color)
        other.set_ylabel("PhysX 法向力（N）", color=force_color)
        ax.tick_params(axis="y", colors=code_color)
        other.tick_params(axis="y", colors=force_color)
        ax.set_xlim(0, longest)
        ax.grid(axis="both", alpha=.17)
        ax.set_title(name, loc="left", pad=7)
        peak = int(matched["peak_code"][:, index].max())
        base_peak = int(baseline["peak_code"][:, index].max())
        ax.set_title(f"匹配表面峰值 {peak} / 255；原版峰值 {base_peak} / 255", loc="right", fontsize=13, pad=7)
        if alignment:
            entry = next(row for row in alignment["finger_statistics"] if row["finger"] == FINGERS[index])
            no_force = entry["all_frame_substeps_force_zero"]
            zero_indexes = np.asarray(no_force["frame_indexes"], dtype=np.int64)
            if len(zero_indexes):
                if (matched["peak_code"][zero_indexes, index] == 0).any() or (matched["normal_load_N"][zero_indexes, index] + matched["friction_load_N"][zero_indexes, index] > alignment["force_zero_tolerance_N"]).any():
                    raise ValueError("Force/code alignment summary disagrees with displayed arrays")
                for span in no_force["time_spans"]:
                    ax.axvspan(span["first_endpoint_time_s"], span["last_endpoint_time_s"], color="#914ab9", alpha=.15)
                target = zero_indexes[len(zero_indexes) // 2]
                ax.annotate(f"紫带 {len(zero_indexes)} 帧：几何图非零\n全部物理子步无接触力", xy=(matched["time_s"][target], matched["peak_code"][target, index]),
                            xytext=(.42, .53), textcoords="axes fraction", fontsize=12, color="#6a3488",
                            arrowprops={"arrowstyle": "->", "color": "#6a3488", "lw": 1.1})
        axis_ranges.append({"finger": FINGERS[index], "code_axis": list(ax.get_ylim()),
                            "force_N_axis": list(other.get_ylim())})
        if index == 0:
            legend_handles = [code, base, force]
    axes[-1].set_xlabel("各次真实运行的模拟时间（s）")
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.5, .855), ncol=3,
               frameon=False, fontsize=11, handlelength=2.7, columnspacing=1.3)
    fig.text(.5, .094, "曲线直接绘制每个采样点：TacMap 为相机时刻端点；PhysX 力为对应视频帧区间的时间均值。", ha="center", fontsize=14)
    fig.text(.5, .060, "两轴单位与量程独立；码值不是 N、压力或 FEM 形变。数值曲线放大可读范围，视频灰度仍固定 0–255。", ha="center", fontsize=14)
    fig.text(.5, .027, "原版与匹配表面为两次独立真实任务，各按自身时间轴绘制；未进行平滑、插值、力到灰度转换或补亮。", ha="center", fontsize=12, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    plt.close(fig)
    report = {"matched": matched_source, "baseline": baseline_source,
              "force_alignment_sha256": sha256(alignment_file) if alignment else None,
              "fingers": list(FINGERS[:3]), "axis_ranges": axis_ranges,
              "semantics": {"TacMap": "Per-frame maximum of actual official uint8 normal-penetration quantization followed by Gaussian blur; endpoint sampling; not force, pressure or FEM deformation.",
                            "PhysX": "Sum of magnitudes of solver normal-contact vectors, independently sampled every physical step and time-averaged over each frame interval; N.",
                            "comparison": "Independent real runs, each with its own recorded time axis; no interpolation or smoothing. Curve axis scaling is explicitly numeric and does not alter image gray levels."},
              "output_sha256": sha256(args.output), "output_bytes": args.output.stat().st_size}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "bytes": report["output_bytes"], "sha256": report["output_sha256"],
                      "matched_statistics": matched_source["per_finger"], "baseline_statistics": baseline_source["per_finger"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
