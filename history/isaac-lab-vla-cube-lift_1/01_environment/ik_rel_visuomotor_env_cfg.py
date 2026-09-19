# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg as BaseFrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.contrib.stack import mdp as stack_mdp
from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings



def object_reached_lift_goal(
    env,
    threshold: float = 0.08,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Return true when the cube is within threshold of the commanded lift goal."""
    object_position = (
        env.scene[object_cfg.name].data.root_pos_w.torch
        - env.scene.env_origins
    )
    desired_position = env.command_manager.get_command(command_name)[..., :3]
    distance = torch.linalg.norm(
        object_position - desired_position,
        dim=-1,
    )
    return distance < threshold


@configclass
class VisuomotorObservationsCfg:

    """Policy observations matching the official Robomimic visuomotor example."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Robot state available at both training and inference time.
        eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=stack_mdp.gripper_pos)

        # RGB observations. Privileged cube coordinates are intentionally excluded.
        table_cam = ObsTerm(
            func=stack_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        wrist_cam = ObsTerm(
            func=stack_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCubeLiftVisuomotorEnvCfg(BaseFrankaCubeLiftEnvCfg):
    """Cube Lift IK-Rel environment with official-style visuomotor observations."""

    observations: VisuomotorObservationsCfg = VisuomotorObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Required by the official stack_mdp.gripper_pos observation.
        self.gripper_joint_names = ["panda_finger_joint.*"]

        # Used by the official Robomimic play.py success-rate calculation.
        self.terminations.success = DoneTerm(
            func=object_reached_lift_goal,
            params={
                "threshold": 0.08,
                "command_name": "object_pose",
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(0.03701, 0.03701, -0.70614, -0.70614),
                convention="ros",
            ),
        )

        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4),
                rot=(-0.61237, -0.61237, 0.35355, 0.35355),
                convention="ros",
            ),
        )

        self.num_rerenders_on_reset = 3

        for camera_cfg in (self.scene.table_cam, self.scene.wrist_cam):
            set_isaac_rtx_global_settings(
                camera_cfg.renderer_cfg,
                antialiasing_mode="DLAA",
            )

        self.image_obs_list = ["table_cam", "wrist_cam"]
