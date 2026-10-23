"""Summarize recorded task dynamics without starting a simulation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+")
args = parser.parse_args()

for run in args.runs:
    path = Path(run)
    telemetry = json.loads((path / "telemetry.json").read_text())
    trajectory = json.loads((path / "trajectory.json").read_text())
    meta = json.loads((path / "metadata.json").read_text())
    dt = meta["args"]["dt"]
    handle = np.array([row["handle"] for row in telemetry])
    grip = np.array([row.get("grip_force", np.zeros((2, 2, 3))) for row in telemetry])
    jaw_force = np.array([row["force"] for row in telemetry]).reshape(len(telemetry), -1, 3)
    grip_magnitude = np.linalg.norm(grip, axis=-1)
    jaw_magnitude = np.linalg.norm(jaw_force, axis=-1)
    grip_each_blade = grip_magnitude.sum(axis=-1)
    bilateral = (grip_each_blade > 0.01).all(axis=-1)
    tilt = []
    for row in trajectory:
        quaternion = np.asarray(row["handle_pose"][1])
        axis = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_matrix()[:, 2]
        tilt.append(float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))))
    last_two = min(round(2 / dt), len(telemetry))
    report = {
        "run": str(path),
        "first_handle_m": handle[0].tolist(),
        "first_grip_force_magnitudes_n": grip_magnitude[0].tolist(),
        "max_grip_force_n": float(grip_magnitude.max()),
        "max_handle_speed_m_s": float(np.linalg.norm(np.diff(handle, axis=0) / dt, axis=-1).max()),
        "max_handle_tilt_deg": max(tilt),
        "final_handle_tilt_deg": tilt[-1],
        "bilateral_finger_contact_fraction": float(bilateral.mean()),
        "final_two_s_bilateral_jaw_contact_fraction": float((jaw_magnitude[-last_two:] > 0.001).all(axis=-1).mean()),
        "final_two_s_block_speed_max_m_s": float(max(np.linalg.norm(row.get("block_velocity", [0, 0, 0])) for row in telemetry[-last_two:])),
        "samples": [
            {"t": row["t"], "handle": row["handle_pose"][0], "tilt_deg": angle}
            for row, angle in zip(trajectory, tilt)
            if abs(row["t"] * 10 - round(row["t"] * 10)) < 1e-6
        ],
    }
    (path / "dynamics_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
