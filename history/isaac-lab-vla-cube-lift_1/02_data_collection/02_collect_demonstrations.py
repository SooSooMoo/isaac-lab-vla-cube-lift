# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    uv run python scripts/environments/state_machine/lift_cube_sm.py --num_envs 32 --viz kit

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_demos", type=int, default=1, help="Number of successful demonstrations to collect.")
parser.add_argument("--max_attempts", type=int, default=10, help="Maximum attempted episodes.")
parser.add_argument("--output_dir", type=str, default="/workspace/step2/datasets/vla_lift", help="Dataset output directory.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything else."""

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from pathlib import Path
from PIL import Image

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # convert to torch
        return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)



def quat_conjugate(q):
    result = q.clone()
    result[..., 1:] *= -1.0
    return result


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ), dim=-1)


def quat_to_rotvec(q):
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    xyz_norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, q[..., :1].clamp_min(1.0e-8))
    axis = xyz / xyz_norm.clamp_min(1.0e-8)
    rotvec = axis * angle
    return torch.where(xyz_norm < 1.0e-6, 2.0 * xyz, rotvec)


def absolute_to_relative(target, current_pos, current_quat, scale=0.5):
    delta_pos = (target[:, :3] - current_pos) / scale
    q_error = quat_multiply(target[:, 3:7], quat_conjugate(current_quat))
    delta_rot = quat_to_rotvec(q_error) / scale
    arm = torch.cat((delta_pos, delta_rot), dim=-1).clamp(-1.0, 1.0)
    return torch.cat((arm, target[:, 7:8]), dim=-1)



def main():
    """Collect successful Cube Lift demonstrations in Isaac Lab HDF5 format."""

    from isaaclab.envs.mdp.recorders.recorders_cfg import (
        ActionStateRecorderManagerCfg,
    )
    from isaaclab.managers import DatasetExportMode

    task = "IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor"

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_stem = "lift_robomimic"

    env_cfg: LiftEnvCfg = parse_env_cfg(
        task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    # We finalize episodes ourselves after 10 continuous successful steps.
    # This avoids an automatic reset exporting an episode before success is marked.
    env_cfg.terminations.time_out = None
    env_cfg.terminations.object_dropping = None
    env_cfg.terminations.success = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.env_name = task

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = str(output_dir)
    env_cfg.recorders.dataset_filename = dataset_stem
    env_cfg.recorders.dataset_export_mode = (
        DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    )

    env = gym.make(task, cfg=env_cfg).unwrapped
    env.reset()

    actions = torch.zeros(
        env.action_space.shape,
        device=env.device,
    )
    actions[:, -1] = 1.0

    desired_orientation = torch.zeros(
        (env.num_envs, 4),
        device=env.device,
    )
    desired_orientation[:, 1] = 1.0

    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation,
        env.num_envs,
        env.device,
        position_threshold=0.01,
    )

    successful_demos = 0
    attempts = 0
    episode_steps = 0
    success_steps = 0
    max_episode_steps = 250
    required_success_steps = 10

    print(f"[DATA] task={task}")
    print(f"[DATA] output={output_dir / (dataset_stem + '.hdf5')}")
    print(f"[DATA] target_demos={args_cli.num_demos}")

    while (
        simulation_app.is_running()
        and successful_demos < args_cli.num_demos
        and attempts < args_cli.max_attempts
    ):
        with torch.inference_mode():
            # Isaac Lab Recorder records the current observation and this action
            # through the normal env.step() hooks.
            env.step(actions)
            episode_steps += 1

            ee_frame_sensor = env.scene["ee_frame"]
            tcp_rest_position = (
                ee_frame_sensor.data.target_pos_w.torch[..., 0, :].clone()
                - env.scene.env_origins
            )
            tcp_rest_orientation = (
                ee_frame_sensor.data.target_quat_w.torch[..., 0, :].clone()
            )

            object_data: RigidObjectData = env.scene["object"].data
            object_position = (
                object_data.root_pos_w.torch
                - env.scene.env_origins
            )

            desired_position = env.command_manager.get_command(
                "object_pose"
            )[..., :3]

            absolute_target = pick_sm.compute(
                torch.cat(
                    [tcp_rest_position, tcp_rest_orientation],
                    dim=-1,
                ),
                torch.cat(
                    [object_position, desired_orientation],
                    dim=-1,
                ),
                torch.cat(
                    [desired_position, desired_orientation],
                    dim=-1,
                ),
            )

            actions = absolute_to_relative(
                absolute_target,
                tcp_rest_position,
                tcp_rest_orientation,
                scale=0.5,
            )

            goal_distance = torch.linalg.norm(
                object_position - desired_position,
                dim=-1,
            )

            if bool((goal_distance < 0.08)[0]):
                success_steps += 1
            else:
                success_steps = 0

            episode_succeeded = success_steps >= required_success_steps
            episode_timed_out = episode_steps >= max_episode_steps

            if episode_succeeded:
                attempts += 1

                # Follow the exact export order used by scripts/tools/record_demos.py.
                env.recorder_manager.record_pre_reset(
                    [0],
                    force_export_or_skip=False,
                )
                env.recorder_manager.set_success_to_episodes(
                    [0],
                    torch.tensor(
                        [[True]],
                        dtype=torch.bool,
                        device=env.device,
                    ),
                )
                env.recorder_manager.export_episodes([0])

                successful_demos += 1
                print(
                    f"[DATA] saved successful demo "
                    f"{successful_demos}/{args_cli.num_demos} "
                    f"(attempts={attempts}, steps={episode_steps}, "
                    f"goal_distance={float(goal_distance[0]):.4f})"
                )

            elif episode_timed_out:
                attempts += 1
                print(
                    f"[DATA] rejected failed attempt {attempts} "
                    f"(steps={episode_steps}, "
                    f"best condition not maintained)"
                )

            if episode_succeeded or episode_timed_out:
                # Clear the completed/discarded in-memory episode and reset.
                env.recorder_manager.reset()
                env.reset()
                pick_sm.reset_idx(
                    torch.tensor([0], dtype=torch.long, device=env.device)
                )

                actions.zero_()
                actions[:, -1] = 1.0
                episode_steps = 0
                success_steps = 0

    print(
        f"[RESULT] successful_demos={successful_demos} "
        f"attempts={attempts}"
    )
    print(
        f"[RESULT] dataset="
        f"{output_dir / (dataset_stem + '.hdf5')}"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
