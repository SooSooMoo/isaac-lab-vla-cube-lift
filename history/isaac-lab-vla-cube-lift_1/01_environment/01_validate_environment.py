"""Validate Cube Lift IK-Rel visuomotor observations."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor"

env_cfg = parse_env_cfg(
    TASK,
    device=args_cli.device,
    num_envs=1,
    use_fabric=True,
)

env = gym.make(TASK, cfg=env_cfg)
obs, _ = env.reset()

actions = torch.zeros(
    env.unwrapped.action_space.shape,
    device=env.unwrapped.device,
)

# Render several frames so RTX cameras are fully initialized.
for _ in range(10):
    obs = env.step(actions)[0]

policy = obs["policy"]

print("=== policy observation keys ===")
print(sorted(policy.keys()))

required = (
    "eef_pos",
    "eef_quat",
    "gripper_pos",
    "table_cam",
    "wrist_cam",
)

for name in required:
    assert name in policy, f"Missing observation: {name}"
    value = policy[name]
    if hasattr(value, "torch"):
        value = value.torch

    tensor = value.detach()
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} "
        f"min={float(tensor.min()):.4f} "
        f"max={float(tensor.max()):.4f} "
        f"mean={float(tensor.float().mean()):.4f} "
        f"std={float(tensor.float().std()):.4f}"
    )

    if name in ("table_cam", "wrist_cam"):
        assert tensor.ndim == 4
        assert tensor.shape[-1] in (3, 4)
        assert float(tensor.float().std()) > 1.0, f"{name} appears blank"

print("[RESULT] Lift visuomotor observation validation OK")

env.close()
simulation_app.close()
