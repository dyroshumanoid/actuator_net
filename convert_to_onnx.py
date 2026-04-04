import torch
import os

path = "/home/dyros/ros2_ws/src/p73_walker_controller/p73_lib/src/actuatornet_models/"

input_sizes = {
    "p73_left_hip_roll":      6,
    "p73_left_hip_pitch":     6,
    "p73_left_hip_yaw":       6,
    "p73_left_knee_pitch":    6,
    "p73_left_ankle_pitch":   6,
    "p73_left_ankle_roll":    6,
    "p73_right_hip_roll":     6,
    "p73_right_hip_pitch":    6,
    "p73_right_hip_yaw":      6,
    "p73_right_knee_pitch":   6,
    "p73_right_ankle_pitch":  6,
    "p73_right_ankle_roll":   6,
}

for name, in_size in input_sizes.items():
    pt_path   = os.path.join(path, f"{name}.pt")
    onnx_path = os.path.join(path, f"{name}.onnx")

    if not os.path.exists(pt_path):
        print(f"Skipped (not found): {pt_path}")
        continue

    m = torch.jit.load(pt_path, map_location="cpu")
    m.eval()
    torch.onnx.export(
        m,
        torch.zeros(1, in_size),
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Converted: {name}.onnx")
