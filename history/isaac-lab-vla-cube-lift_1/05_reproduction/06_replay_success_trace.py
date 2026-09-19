"""Replay an exact successful IK-Rel action trace."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--trace", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


trace_path = Path(args_cli.trace)
trace = np.load(trace_path)
actions = trace["actions"].astype(np.float32)
expected_position = trace["initial_object_position"].astype(np.float32)

torch.manual_seed(args_cli.seed)
torch.cuda.manual_seed_all(args_cli.seed)
np.random.seed(args_cli.seed)

env_cfg = parse_env_cfg(
    args_cli.task,
    device=args_cli.device,
    num_envs=1,
    use_fabric=True,
)
env_cfg.seed = args_cli.seed
env_cfg.observations.policy.concatenate_terms = False
env_cfg.terminations.time_out = None
env_cfg.recorders = None

success_term = env_cfg.terminations.success
env_cfg.terminations.success = None

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
env.seed(args_cli.seed)
obs, _ = env.reset()

actual_position = (
    env.scene["object"].data.root_pos_w.torch[0]
    - env.scene.env_origins[0]
).detach().cpu().numpy()

print("trace:", trace_path)
print("actions:", actions.shape)
print("expected initial object:", expected_position.tolist())
print("actual initial object:  ", actual_position.tolist())
print(
    "initial position error:",
    float(np.linalg.norm(actual_position - expected_position)),
)

frames = [
    obs["policy"]["table_cam"][0, ..., :3]
    .detach().cpu().numpy().astype(np.uint8)
]

success = False
success_step = None

with torch.inference_mode():
    for step, action in enumerate(actions):
        action_tensor = torch.from_numpy(action).to(
            device=env.device,
            dtype=torch.float32,
        ).view(1, 7)

        obs, _, terminated, truncated, _ = env.step(action_tensor)

        frames.append(
            obs["policy"]["table_cam"][0, ..., :3]
            .detach().cpu().numpy().astype(np.uint8)
        )

        if bool(success_term.func(env, **success_term.params)[0]):
            success = True
            success_step = step + 1
            break

        if bool(terminated[0]) or bool(truncated[0]):
            break

status = "success" if success else "failed"
video_path = trace_path.with_name(
    trace_path.stem + f"_exact_replay_{status}.mp4"
)

clip = ImageSequenceClip(frames, fps=50)
clip.write_videofile(
    str(video_path),
    codec="libx264",
    audio=False,
    logger=None,
)
clip.close()

print("replay success:", success)
print("success step:", success_step)
print("video:", video_path)
print(
    "[RESULT]",
    "EXACT_ACTION_REPLAY_SUCCESS"
    if success
    else "EXACT_ACTION_REPLAY_FAILED",
)

env.close()
simulation_app.close()
