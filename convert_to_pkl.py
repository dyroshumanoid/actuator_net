#!/usr/bin/env python3
"""Convert actuator experiment txt logs to pkl format for actuator network training.

File format:
  actuatornet/: has timestamp in col 0 → 14 cols (pos/vel/des), 27 cols (torque)
  pace/:        no timestamp           → 13 cols (pos/vel/des), 26 cols (torque)

Torque files have two halves: first N cols = joint torques, second N cols = motor torques.
We use only the joint torques (first half).

Output pkl: list of dicts, one per timestep, with keys:
  joint_names, joint_positions, joint_velocities, joint_efforts,
  joint_position_command, time_sec, time_nsec

joint_positions[0] is a dummy platform joint (0.0) that utils.py skips via [1:].
joint_positions[1:13] are the 12 actual joints in user-specified order.
"""

import pickle
import numpy as np
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_ROOT / "pkl"
OUTPUT_DIR.mkdir(exist_ok=True)

JOINT_NAMES = [
    "platform",
    "left_hip_roll",
    "left_hip_pitch",
    "left_hip_yaw",
    "left_knee_pitch",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_roll",
    "right_hip_pitch",
    "right_hip_yaw",
    "right_knee_pitch",
    "right_ankle_pitch",
    "right_ankle_roll",
]


def convert_experiment(exp_dir: Path, has_timestamp: bool, dt: float = 0.001):
    """Load txt logs from one experiment and return a list of timestep dicts."""
    pos = np.loadtxt(exp_dir / "joint_position_log.txt")
    vel = np.loadtxt(exp_dir / "joint_velocity_log.txt")
    des = np.loadtxt(exp_dir / "joint_desired_log.txt")
    trq = np.loadtxt(exp_dir / "torque_joint_log.txt")

    if has_timestamp:
        # col 0 = timestamp, cols 1-12 = 12 joints, col 13 = platform
        timestamps = pos[:, 0]
        pos_data   = pos[:, 1:13]
        vel_data   = vel[:, 1:13]
        des_data   = des[:, 1:13]
        platform_pos = pos[:, 13]
        platform_vel = vel[:, 13]
        platform_des = des[:, 13]
        # torque: col 0 = timestamp, cols 1-12 = joint torques, col 13 = platform torque
        trq_data        = trq[:, 1:13]
        platform_trq    = trq[:, 13]
    else:
        # no timestamp column; cols 0-11 = 12 joints, col 12 = platform
        n = pos.shape[0]
        timestamps = np.arange(n, dtype=float) * dt
        pos_data   = pos[:, 0:12]
        vel_data   = vel[:, 0:12]
        des_data   = des[:, 0:12]
        platform_pos = pos[:, 12]
        platform_vel = vel[:, 12]
        platform_des = des[:, 12]
        # torque: cols 0-11 = joint torques, col 12 = platform torque, cols 13-25 = motor torques
        trq_data        = trq[:, 0:12]
        platform_trq    = trq[:, 12]

    records = []
    for i, t in enumerate(timestamps):
        t_sec = int(t)
        t_nsec = int(round((t - t_sec) * 1e9))
        records.append({
            "joint_names":            JOINT_NAMES,
            "joint_positions":        [platform_pos[i]] + pos_data[i].tolist(),
            "joint_velocities":       [platform_vel[i]] + vel_data[i].tolist(),
            "joint_efforts":          [platform_trq[i]] + trq_data[i].tolist(),
            "joint_position_command": [platform_des[i]] + des_data[i].tolist(),
            "time_sec":               t_sec,
            "time_nsec":              t_nsec,
        })
    return records


def convert_directory(data_dir: Path, has_timestamp: bool, dt: float = 0.001):
    for exp_dir in sorted(data_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        print(f"  {exp_dir.name} ...", end=" ", flush=True)
        records = convert_experiment(exp_dir, has_timestamp=has_timestamp, dt=dt)
        out_path = OUTPUT_DIR / f"{exp_dir.name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(records, f)
        print(f"{len(records)} steps → {out_path.name}")


print("Converting actuatornet experiments (with timestamp)...")
convert_directory(DATA_ROOT / "actuatornet", has_timestamp=True)

print("Converting pace experiments (no timestamp, dt=0.001 s)...")
convert_directory(DATA_ROOT / "pace", has_timestamp=False, dt=0.001)

print(f"\nDone. PKL files saved to: {OUTPUT_DIR}")
