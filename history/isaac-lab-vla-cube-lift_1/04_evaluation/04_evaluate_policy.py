# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)

parser.add_argument(
    "--stop_on_success",
    action="store_true",
    help="Stop after recording the first successful rollout.",
)

parser.add_argument(
    "--fixed_trial_seed",
    action="store_true",
    help="Use exactly --seed for every rollout instead of seed + trial.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from moviepy.editor import ImageSequenceClip
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    initial_object_position = (
        env.scene["object"].data.root_pos_w.torch[0]
        - env.scene.env_origins[0]
    ).detach().cpu().numpy().astype(np.float32)

    traj = dict(actions=[], obs=[], next_obs=[])
    frames = [
        obs_dict["policy"]["table_cam"][0, ..., :3]
        .detach().cpu().numpy().astype(np.uint8)
    ]

    for i in range(horizon):
        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # Check if environment image observations
        if hasattr(env.cfg, "image_obs_list"):
            # Process image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from chw uint8 to hwc normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        traj["obs"].append(obs)

        # Compute actions
        actions = policy(obs)

        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)

        frame = (
            obs_dict["policy"]["table_cam"][0, ..., :3]
            .detach().cpu().numpy().astype(np.uint8)
        )
        frames.append(frame)
        obs = obs_dict["policy"]

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj, frames, initial_object_position
        elif terminated or truncated:
            return False, traj, frames, initial_object_position

    return False, traj, frames, initial_object_position


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    # Set the seed before environment construction for reproducible resets.
    env_cfg.seed = args_cli.seed

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    video_dir = Path(
        "/workspace/step2/evaluation/"
        f"robomimic_epoch_200_recorded_seed_{args_cli.seed}_horizon_{args_cli.horizon}/videos"
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    records = []

    with torch.inference_mode():
        # Run policy
        results = []
        for trial in range(args_cli.num_rollouts):
            trial_seed = (
                args_cli.seed
                if args_cli.fixed_trial_seed
                else args_cli.seed + trial
            )
            print(f"[INFO] Starting trial {trial} with seed {trial_seed}")

            torch.manual_seed(trial_seed)
            np.random.seed(trial_seed)
            random.seed(trial_seed)
            env.seed(trial_seed)

            policy, _ = FileUtils.policy_from_checkpoint(
                ckpt_path=args_cli.checkpoint,
                device=device,
            )
            terminated, traj, frames, initial_object_position = rollout(
                policy,
                env,
                success_term,
                args_cli.horizon,
                device,
            )
            results.append(terminated)
            print(f"[INFO] Trial {trial}: {terminated}\n")

            status = "success" if terminated else "failed"
            video_path = video_dir / (
                f"trial_{trial:02d}_seed_{trial_seed}_{status}.mp4"
            )

            clip = ImageSequenceClip(frames, fps=50)
            clip.write_videofile(
                str(video_path),
                codec="libx264",
                audio=False,
                logger=None,
            )
            clip.close()

            action_trace_path = video_path.with_suffix(".npz")
            np.savez_compressed(
                action_trace_path,
                actions=np.asarray(
                    traj["actions"],
                    dtype=np.float32,
                ).reshape(-1, 7),
                seed=np.asarray(trial_seed, dtype=np.int64),
                initial_object_position=np.asarray(
                    initial_object_position,
                    dtype=np.float32,
                ),
                success=np.asarray(bool(terminated)),
            )
            print(f"[TRACE] {action_trace_path}")

            records.append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "initial_x": float(initial_object_position[0]),
                    "initial_y": float(initial_object_position[1]),
                    "initial_z": float(initial_object_position[2]),
                    "success": int(terminated),
                    "steps": len(traj["actions"]),
                    "video": str(video_path),
                }
            )
            print(f"[VIDEO] {video_path}")
            print(
                f"[TRIAL] seed={trial_seed} "
                f"initial_object={initial_object_position.tolist()} "
                f"success={terminated}"
            )

            if terminated and args_cli.stop_on_success:
                print(f"[RESULT] reproducible_success_seed={trial_seed}")
                break

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    csv_path = video_dir.parent / "results.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trial",
                "seed",
                "initial_x",
                "initial_y",
                "initial_z",
                "success",
                "steps",
                "video",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"[RESULT] csv={csv_path}")
    print(f"[RESULT] videos={video_dir}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
