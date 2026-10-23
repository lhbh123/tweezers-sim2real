#!/usr/bin/env python3
"""Audit real W01 contacts against source shell/elastomer surfaces and tool forces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kinematics import Kinematics, transform, xyz


def quantiles_mm(values):
    return (np.quantile(values, [0, .5, .95, 1]) * 1000).tolist() if len(values) else []


def distances(mesh, points):
    return np.concatenate([trimesh.proximity.closest_point(mesh, points[i:i + 1000])[1]
                           for i in range(0, len(points), 1000)]) if len(points) else np.zeros(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--finger", choices=["thumb", "index", "middle", "ring", "pinky"], default="middle")
    args = parser.parse_args()
    run = args.run.resolve()
    launch = json.loads((run / "launch.json").read_text())
    pose_path = Path(launch["arguments"]["pose"])
    if not pose_path.is_absolute():
        pose_path = ROOT / pose_path
    archived_pose = run / "source" / pose_path.name
    cfg = json.loads((archived_pose if archived_pose.exists() else pose_path).read_text())
    data = np.load(run / "tactile.npz", allow_pickle=False)
    finger_id = data["finger_names"].tolist().index(args.finger)
    dp, skin = f"right_{args.finger}_DP", f"right_{args.finger}_elastomer"
    kin = Kinematics(Path(cfg["robot_asset_dir"]) / "robot.urdf", palm_link=cfg["palm_link"], hand_root_link=cfg["hand_root_link"])
    fk = kin.fk(root=dp)
    sensor_T_dp = np.linalg.inv(fk[skin])
    meshes, hashes = {}, {}
    for name in (dp, skin):
        pieces = []
        for col in kin.root.find(f"link[@name='{name}']").findall("collision"):
            element = col.find("geometry/mesh")
            path = kin.resolve_mesh(element.get("filename")).resolve()
            mesh = trimesh.load_mesh(path, process=False)
            mesh.apply_scale(xyz(element.get("scale", "1 1 1")))
            origin = col.find("origin")
            local = transform(xyz(origin.get("xyz")) if origin is not None else None,
                              Rotation.from_euler("xyz", xyz(origin.get("rpy"))).as_matrix() if origin is not None else None)
            mesh.apply_transform(sensor_T_dp @ fk[name] @ local)
            pieces.append(mesh)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        meshes[name] = trimesh.util.concatenate(pieces)
    report = {
        "run": str(run), "finger": args.finger,
        "source_tactile_sha256": hashlib.sha256((run / "tactile.npz").read_bytes()).hexdigest(),
        "task_success_per_env": json.loads((run / "evaluation.json").read_text()).get("success_per_env") if (run / "evaluation.json").exists() else None,
        "method": "Actual finger-frame raw PhysX contacts compared with source shell and elastomer collision triangles. Nearest-source attribution is geometric evidence, not collider-id telemetry. Raw data and projected maps are unchanged.",
        "mesh_sha256": hashes, "channels": {},
        "interpretation": "A projected taxel does not establish physical skin contact. The 3 mm display correspondence threshold can include shell/convex-hull contacts near the elastomer.",
    }
    phase_ids, phase_names = None, ["settle", "approach", "descend", "squeeze", "lift", "transfer", "lower", "release", "withdraw", "settle_at_goal"]
    if (run / "trajectory.npz").exists():
        trajectory = np.load(run / "trajectory.npz", allow_pickle=False)
        control_phases = trajectory["phase"].reshape(len(trajectory["phase"]), -1)[:, 0].astype(int)
        decimation = int(round(float(trajectory["dt"]) / float(np.median(data["substep_dt_s"]))))
        if decimation < 1 or len(control_phases) * decimation != len(data["substep_time_s"]):
            raise ValueError("Physical/control samples do not align for phase attribution")
        phase_ids = np.repeat(control_phases, decimation)
        report["phase_alignment"] = {
            "decimation": decimation, "physics_substeps": len(phase_ids), "control_steps": len(control_phases),
            "method": "trajectory.phase at the corresponding completed control interval; physical sample index // decimation",
            "boundary_caveat": "The phase label reflects the end of its control interval; a transition interval can include substeps executed under the preceding phase.",
        }
    for channel in ("normal", "friction"):
        raw = data[channel + "_records"]
        raw = raw[raw[:, 1] == finger_id]
        skin_distance = distances(meshes[skin], raw[:, 6:9])
        shell_distance = distances(meshes[dp], raw[:, 6:9])
        weights = np.linalg.norm(raw[:, 9:12], axis=1) * data["substep_dt_s"][raw[:, 0].astype(int)]
        def summarize(selected):
            points, skin_d, shell_d, impulse = raw[selected], skin_distance[selected], shell_distance[selected], weights[selected]
            def fraction(mask):
                return float(impulse[mask].sum() / impulse.sum()) if impulse.sum() > 0 else None
            return {
                "raw_records": len(points), "load_impulse_N_s": float(impulse.sum()),
                "records_below_tool_raw_reader_threshold_1e_minus_8_N": int((np.linalg.norm(points[:, 9:12], axis=1) < 1e-8).sum()),
                "surface_distance_mm_quantiles_0_50_95_100": quantiles_mm(skin_d),
                "shell_distance_mm_quantiles_0_50_95_100": quantiles_mm(shell_d),
                "elastomer_closer_impulse_fraction": fraction(skin_d < shell_d),
                "skin_closer_by_0p1mm_impulse_fraction": fraction(skin_d + .0001 < shell_d),
                "taxel_mapping_distance_mm_quantiles_0_50_95_100": quantiles_mm(points[:, 14]),
                "mapped_impulse_fraction": fraction(points[:, 13] >= 0),
            }
        report["channels"][channel] = summarize(np.ones(len(raw), dtype=bool))
        if phase_ids is not None:
            point_phases = phase_ids[raw[:, 0].astype(int)]
            groups = {}
            for phase in np.unique(phase_ids):
                group = summarize(point_phases == phase)
                group["phase_name"] = phase_names[phase] if 0 <= phase < len(phase_names) else str(phase)
                group["physics_substeps_in_phase"] = int((phase_ids == phase).sum())
                groups[str(int(phase))] = group
            report["channels"][channel]["by_control_phase"] = groups
    times = data["substep_time_s"]
    upward = -(data["substep_normal_force_w_N"][:, finger_id, 2] + data["substep_friction_force_w_N"][:, finger_id, 2])
    last = times >= times[-1] - 1. - 1e-8
    report["support"] = {
        "last_1s_net_up_on_tool_mean_N": float(upward[last].mean()),
        "last_1s_net_up_on_tool_min_max_N": [float(upward[last].min()), float(upward[last].max())],
        "all_substeps_contact": bool((data["substep_normal_contact_count"][:, finger_id] > 0).all()),
        "active_substeps": int((data["substep_normal_contact_count"][:, finger_id] > 0).sum()),
        "upward_source": "Sum of normal and friction vectors on the tool; opposite of measured force on the finger.",
    }
    errors = {"normal": [], "friction": []}
    def compare(index, points):
        for channel, kind in (("normal", "normal_contacts"), ("friction", "friction_anchors")):
            vectors = [p["force_on_tool_N"] for p in points[kind] if p["robot"] == dp]
            force = np.sum(vectors, axis=0) if vectors else np.zeros(3)
            errors[channel].append(float(np.max(abs(force + data["substep_" + channel + "_force_w_N"][index, finger_id]))))
    if (run / "raw_contact_points.jsonl").exists():
        for line in (run / "raw_contact_points.jsonl").open():
            row = json.loads(line)
            if row.get("env", 0) == 0:
                compare(int(np.argmin(abs(times - row["time_s"]))), row["points"])
    elif (run / "grasp_substeps.jsonl").exists():
        index = 0
        for line in (run / "grasp_substeps.jsonl").open():
            for row in json.loads(line)["substeps"]:
                compare(index, row["grasp_contact_points"])
                index += 1
        if index != len(times):
            raise ValueError("Raw tool-contact step count does not match tactile samples")
    report["raw_point_reader_opposite_force_max_abs_error_N"] = {k: max(v) if v else None for k, v in errors.items()}
    report["raw_point_reader_compared_samples"] = {k: len(v) for k, v in errors.items()}
    output = run / f"{args.finger}_contact_source_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output), "support": report["support"],
                      "channels": {k: {key: value for key, value in v.items() if key != "by_control_phase"} for k, v in report["channels"].items()},
                      "opposite_force_error_N": report["raw_point_reader_opposite_force_max_abs_error_N"]}))


if __name__ == "__main__":
    main()
