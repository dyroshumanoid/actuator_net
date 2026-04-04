"""Evaluate trained actuator networks on the held-out eval set.

Loads EVAL_PKL_NAME, runs inference for all 12 joints, and saves a single
PNG with all joints in one figure (6 rows x 2 cols).
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

from utils import (
    JOINT_GROUPS, EVAL_PKL_NAME,
    load_single_experiment, prepare_data_for_joint_group,
)

EXPERIMENT_DIR    = '/home/dyros/scraps/actuator_net/data/pkl'
ACTUATOR_NET_PATH = '/home/dyros/scraps/p73.pt'
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

n_joints = len(JOINT_GROUPS)
n_cols = 2
n_rows = (n_joints + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
axes = axes.flatten()

results = {}
for idx, (joint_indices, group_name) in enumerate(JOINT_GROUPS):
    group_net_path = ACTUATOR_NET_PATH.replace(".pt", f"_{group_name}.pt")
    ax = axes[idx]

    if not os.path.exists(group_net_path):
        ax.set_title(f"{group_name}\n(no model)", fontsize=8)
        ax.axis('off')
        print(f"[{group_name}] No model found at {group_net_path}, skipping.")
        continue

    params = best_params.get(group_name)
    if params is None:
        ax.set_title(f"{group_name}\n(no params)", fontsize=8)
        ax.axis('off')
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

    ax.plot(ys[:, 0].numpy(),     label="Measured",  color="green", linewidth=1.5)
    ax.plot(y_pred[:, 0].numpy(), label="Predicted", color="red",   linewidth=0.6)
    ax.set_title(f"{group_name}  MSE={loss:.4f}  MAE={mae:.4f}", fontsize=8)
    ax.set_ylabel("Torque (scaled)", fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

# hide any unused subplots
for i in range(n_joints, len(axes)):
    axes[i].axis('off')

axes[-2].set_xlabel("Timestep", fontsize=8)
axes[-1].set_xlabel("Timestep", fontsize=8)

fig.suptitle(f"Actuator net eval — {EVAL_PKL_NAME}", fontsize=11)
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"eval_all_joints_{timestamp}.png"
plt.savefig(out_path, dpi=120)
plt.close()
print(f"\nSaved: {out_path}")

print("\nSummary:")
for name, r in results.items():
    print(f"  {name:20s}  MSE={r['mse']:.6f}  MAE={r['mae']:.6f}")
