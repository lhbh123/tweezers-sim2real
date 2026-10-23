#!/usr/bin/env python3
"""Independently audit a recorded W01 tactile archive and optional camera run.

This checks saved physical evidence, not task success by inference from pictures.
Run with .venv-human/bin/python; no Isaac or scipy dependency is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


class Audit:
    def __init__(self):
        self.checks = {}
        self.warnings = []

    def check(self, name, passed, **details):
        self.checks[name] = {"passed": bool(passed), **details}

    def close(self, name, actual, expected, *, atol=2e-6, rtol=2e-5):
        actual, expected = np.asarray(actual), np.asarray(expected)
        same_shape = actual.shape == expected.shape
        self.check(name, same_shape and np.allclose(actual, expected, atol=atol, rtol=rtol),
                   max_absolute_error=float(np.max(np.abs(actual - expected))) if same_shape and actual.size else None,
                   actual_shape=list(actual.shape), expected_shape=list(expected.shape), atol=atol, rtol=rtol)


def cumulative_interval_means(values, times, dts, frame_times):
    """Integral interpolation: independent of the recorder's overlap binning."""
    shape = values.shape[1:]
    flat = values.reshape(len(times), -1)
    edges = np.r_[times[0] - dts[0], times]
    integral = np.vstack((np.zeros((1, flat.shape[1])), np.cumsum(flat * dts[:, None], axis=0)))
    bounds = np.r_[edges[0], frame_times]
    at_bounds = np.column_stack([np.interp(bounds, edges, integral[:, col]) for col in range(flat.shape[1])])
    return (np.diff(at_bounds, axis=0) / np.diff(bounds)[:, None]).reshape((len(frame_times),) + shape)


def audit_run(run, require_video=False, require_task_success=False):
    audit = Audit()
    with np.load(run / "tactile.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(data["metadata_json"]))
    times, dts, frame_times = (np.asarray(data[name], np.float64)
                              for name in ("substep_time_s", "substep_dt_s", "frame_time_s"))
    n, frames = len(times), len(frame_times)
    if n == 0 or frames == 0:
        raise ValueError("Empty physical sample or video-interval series")
    audit.check("five_fingers_present", data["finger_names"].tolist() == FINGERS)
    audit.check("finite_positive_sampling", np.isfinite(times).all() and np.isfinite(dts).all()
                and np.isfinite(frame_times).all() and (dts > 0).all()
                and (np.diff(times) > 0).all() and (np.diff(frame_times) > 0).all())
    audit.close("every_physics_substep_present", np.diff(times), dts[1:], atol=1e-7, rtol=1e-5)
    start = times[0] - dts[0]
    audit.check("frames_inside_physics_interval", frame_times[0] > start
                and frame_times[-1] <= times[-1] + 1e-6)
    audit.close("saved_frame_intervals", data["frame_interval_s"], np.diff(np.r_[start, frame_times]), atol=1e-9)
    poses = data["substep_sensor_pose_w"]
    audit.check("sensor_pose_shape_and_finiteness", poses.shape == (n, 5, 7) and np.isfinite(poses).all())
    audit.close("sensor_quaternion_unit_length", np.linalg.norm(poses[:, :, 3:], axis=-1), np.ones((n, 5)), atol=1e-6)
    threshold = float(metadata["max_mapping_distance_m"])
    stats = {}
    for channel in ("normal", "friction"):
        raw = data[channel + "_records"]
        if raw.ndim != 2 or raw.shape[1] != 15:
            raise ValueError(f"Invalid {channel} record shape: {raw.shape}")
        allowed_nan = [12] if channel == "friction" else []
        audit.check(channel + "_records_finite", np.isfinite(np.delete(raw, allowed_nan, axis=1)).all())
        sample = raw[:, 0].astype(np.int64)
        finger = raw[:, 1].astype(np.int64)
        taxel = raw[:, 13].astype(np.int64)
        indexes_ok = ((sample >= 0) & (sample < n) & (finger >= 0) & (finger < 5)
                      & (raw[:, 2] >= 0) & (raw[:, 2] < len(metadata["filter_paths"]))
                      & (taxel >= -1) & (taxel < data["surface_valid"].shape[-1]**2)).all()
        audit.check(channel + "_record_indexes", indexes_ok)
        if not indexes_ok:
            raise ValueError(f"Invalid {channel} record indexes")
        audit.check(channel + "_integer_indexes", (raw[:, [0, 1, 2, 13]] == raw[:, [0, 1, 2, 13]].astype(np.int64)).all())
        mapped = taxel >= 0
        audit.check(channel + "_mapping_threshold", np.array_equal(mapped, raw[:, 14] <= threshold),
                    threshold_m=threshold)
        flat_pair = sample * 5 + finger
        magnitude = np.linalg.norm(raw[:, 9:12], axis=-1)
        load = np.bincount(flat_pair, weights=magnitude, minlength=n * 5).reshape(n, 5)
        outside = np.bincount(flat_pair[~mapped], weights=magnitude[~mapped], minlength=n * 5).reshape(n, 5)
        vectors = np.stack([np.bincount(flat_pair, weights=raw[:, col], minlength=n * 5)
                            for col in (9, 10, 11)], axis=-1).reshape(n, 5, 3)
        counts = np.bincount(flat_pair, minlength=n * 5).reshape(n, 5)
        for suffix, expected in (("load_N", load), ("force_w_N", vectors),
                                 ("unmapped_load_N", outside), ("contact_count", counts)):
            audit.close(channel + "_raw_preserves_" + suffix, data["substep_" + channel + "_" + suffix], expected)
        for suffix in ("load_N", "force_w_N", "unmapped_load_N"):
            expected = cumulative_interval_means(data["substep_" + channel + "_" + suffix], times, dts, frame_times)
            audit.close(channel + "_frame_impulse_conservation_" + suffix, data[channel + "_" + suffix], expected)
        heatmap = data[channel + "_map_N"]
        audit.check(channel + "_map_shape_finite_nonnegative", heatmap.shape[:2] == (frames, 5)
                    and np.isfinite(heatmap).all() and (heatmap >= 0).all())
        audit.close(channel + "_mapped_plus_unmapped_conservation",
                    heatmap.sum((-1, -2)) + data[channel + "_unmapped_load_N"], data[channel + "_load_N"])
        audit.check(channel + "_peak_at_least_mean", (data[channel + "_peak_load_N"] + 1e-6 >= data[channel + "_load_N"]).all())
        valid = np.broadcast_to(data["surface_valid"], heatmap.shape)
        audit.check(channel + "_invalid_surface_bins_empty", not np.any(heatmap[~valid]))
        if len(raw):
            q = poses[sample, finger, 3:]
            local = raw[:, 6:9]
            twice_cross = 2 * np.cross(q[:, 1:], local)
            world = local + q[:, :1] * twice_cross + np.cross(q[:, 1:], twice_cross) + poses[sample, finger, :3]
            audit.close(channel + "_world_local_frame_consistency", world, raw[:, 3:6], atol=2e-7)
        if mapped.any():
            assigned = data["surface_points_m"].reshape(5, -1, 3)[finger[mapped], taxel[mapped]]
            distance = np.linalg.norm(assigned - raw[mapped, 6:9], axis=-1)
            audit.close(channel + "_assigned_surface_distance", distance, raw[mapped, 14], atol=1e-9)
        integrated = np.sum(load * dts[:, None], axis=0)
        integrated_outside = np.sum(outside * dts[:, None], axis=0)
        stats[channel] = {
            "raw_records": len(raw), "max_load_N_by_finger": load.max(0).tolist(),
            "active_substeps_by_finger": (load > 1e-8).sum(0).tolist(),
            "mapped_impulse_fraction_by_finger": [float(1 - b / a) if a > 1e-12 else None
                                                   for a, b in zip(integrated, integrated_outside)],
        }
    trajectory = {}
    if (run / "trajectory.npz").exists():
        with np.load(run / "trajectory.npz", allow_pickle=False) as archive:
            trajectory = {key: archive[key] for key in ("render_control_steps", "dt", "phase", "success") if key in archive}
        controls = len(trajectory["phase"])
        control_dt = float(trajectory["dt"])
        audit.close("all_control_steps_have_physics", times[-1] - start, controls * control_dt, atol=1e-6)
        steps = trajectory.get("render_control_steps", np.array([], np.int64))
        if len(steps):
            audit.close("video_timestamps_match_control_steps", frame_times, (steps + 1) * control_dt, atol=1e-6)
            audit.check("video_control_step_range", (steps >= 0).all() and (steps < controls).all()
                        and (np.diff(steps) > 0).all())
        elif require_video:
            audit.check("video_render_steps_present", False)
    tail = max(0., float(times[-1] - frame_times[-1]))
    if tail > 1e-6:
        audit.warnings.append(f"Video ends {tail:.9g} s before the last physics sample; raw tactile data retains the tail.")
    camera_paths = [run / name for name in ("overview.mp4", "task.mp4", "closeup.mp4") if (run / name).exists()]
    if (run / "camera_manifest.json").exists():
        manifest = json.loads((run / "camera_manifest.json").read_text())
        camera_paths = [run / (view["name"] + ".mp4") for view in manifest]
    if require_video:
        audit.check("camera_videos_present", len(camera_paths) > 0 and all(path.exists() for path in camera_paths))
    camera_stats = []
    if camera_paths:
        import imageio.v2 as imageio
        for path in camera_paths:
            with imageio.get_reader(path) as reader:
                count = reader.count_frames()
                fps = float(reader.get_meta_data()["fps"])
            audit.check("camera_" + path.stem + "_frame_count", count == frames, frames=count, tactile_frames=frames)
            spacing = np.diff(frame_times)
            valid_spacing = (not len(spacing) or (np.allclose(spacing[:-1], 1 / fps, atol=1e-5)
                              and 0 < spacing[-1] <= 1 / fps + 1e-5))
            audit.check("camera_" + path.stem + "_time_spacing", valid_spacing, fps=fps)
            camera_stats.append({"file": str(path), "frames": count, "fps": fps})
    task_success = None
    if (run / "evaluation.json").exists():
        evaluation = json.loads((run / "evaluation.json").read_text())
        task_success = bool(evaluation.get("success_per_env", [False])[0])
        if "success" in trajectory:
            audit.check("task_success_record_consistency", task_success == bool(trajectory["success"][:, 0].any()))
    if require_task_success:
        audit.check("requested_task_success", task_success is True)
    alignment = metadata.get("source_mesh_alignment_example")
    audit.warnings.append("Surface location is approximate. The distance threshold is a visualization correspondence tolerance, not physical sensor accuracy.")
    audit.warnings.append("Friction-anchor vectors are retained; complete contact torsional moments and six-axis wrenches are not measured.")
    return {
        "valid": all(check["passed"] for check in audit.checks.values()), "run": str(run),
        "task_success": task_success, "physics_substeps": n, "tactile_frames": frames,
        "physics_dt_range_s": [float(dts.min()), float(dts.max())],
        "recorded_duration_s": float(times[-1] - start), "video_uncovered_tail_s": tail,
        "force_channels": stats, "cameras": camera_stats,
        "surface_mapping_threshold_m": threshold, "source_mesh_alignment_example": alignment,
        "physics_contact_model": metadata.get("physics_contact_model", {}),
        "checks": audit.checks, "warnings": audit.warnings,
        "tactile_npz_sha256": hashlib.sha256((run / "tactile.npz").read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--require-task-success", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        report = audit_run(run, args.require_video, args.require_task_success)
    except (ValueError, KeyError, OSError, IndexError) as exc:
        report = {"valid": False, "run": str(run), "error": str(exc)}
    output = args.output or run / "tactile_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"valid": report["valid"], "task_success": report.get("task_success"),
                      "physics_substeps": report.get("physics_substeps"), "tactile_frames": report.get("tactile_frames"),
                      "failed_checks": [name for name, item in report.get("checks", {}).items() if not item["passed"]],
                      "error": report.get("error"), "report": str(output)}, ensure_ascii=False))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
