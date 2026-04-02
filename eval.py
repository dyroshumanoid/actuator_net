"""Evaluate trained actuator networks on the held-out eval set.

Loads data_period2.0_radius_0.03.pkl, runs inference for each joint group,
prints loss/MAE, and saves a prediction plot per group.
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    JOINT_GROUPS, EVAL_PKL_NAME,
    load_single_experiment, prepare_data_for_joint_group,
)

EXPERIMENT_DIR    = '/home/dyros/scraps/actuator_net/data/pkl'
ACTUATOR_NET_PATH = 'hashi.pt'
BEST_PARAMS_PATH  = 'best_params.json'
DEVICE            = 'cpu'

eval_pkl_path = os.path.join(EXPERIMENT_DIR, EVAL_PKL_NAME)

if not os.path.exists(eval_pkl_path):
    raise FileNotFoundError(f"Eval pkl not found: {eval_pkl_path}")

if not os.path.exists(BEST_PARAMS_PATH):
    raise FileNotFoundError(f"best_params.json not found — run train.py first")

with open(BEST_PARAMS_PATH) as f:
    best_params = json.load(f)

jpe, jv, te = load_single_experiment(eval_pkl_path, torque_scaling=0.01)

print(f"Eval set: {EVAL_PKL_NAME}  ({jpe.shape[0]} timesteps)\n")

results = {}
for joint_indices, group_name in JOINT_GROUPS:
    group_net_path = ACTUATOR_NET_PATH.replace(".pt", f"_{group_name}.pt")

    if not os.path.exists(group_net_path):
        print(f"[{group_name}] No model found at {group_net_path}, skipping.")
        continue

    params = best_params.get(group_name)
    if params is None:
        print(f"[{group_name}] No saved hyperparams, skipping.")
        continue

    model = torch.jit.load(group_net_path, map_location=DEVICE)
    model.eval()

    xs, ys = prepare_data_for_joint_group(jpe, jv, te, joint_indices, params['num_samples_in_history'])

    with torch.no_grad():
        y_pred = model(xs)

    loss = ((y_pred - ys) ** 2).mean().item()
    mae  = (y_pred - ys).abs().mean().item()
    print(f"[{group_name}]  MSE={loss:.6f}  MAE={mae:.6f}")
    results[group_name] = {"mse": loss, "mae": mae}

    # Plot per output joint
    num_joints = len(joint_indices)
    fig, axes = plt.subplots(num_joints, 1, figsize=(14, 4 * num_joints), squeeze=False)
    fig.suptitle(f"{group_name} — eval on {EVAL_PKL_NAME}", fontsize=10)
    for j in range(num_joints):
        axes[j][0].plot(ys[:, j].numpy(),      label="Measured",  color="green", linewidth=0.5)
        axes[j][0].plot(y_pred[:, j].numpy(),  label="Predicted", color="red",   linewidth=0.5)
        axes[j][0].set_ylabel("Torque (scaled)")
        axes[j][0].set_title(f"joint index {joint_indices[j]}")
        axes[j][0].legend(fontsize=7)
    axes[-1][0].set_xlabel("Timestep")
    plt.tight_layout()
    out_path = f"eval_{group_name}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  → saved {out_path}")

print("\nSummary:")
for name, r in results.items():
    print(f"  {name:20s}  MSE={r['mse']:.6f}  MAE={r['mae']:.6f}")
