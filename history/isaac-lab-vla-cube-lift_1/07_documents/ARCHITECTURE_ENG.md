# System Architecture

## Overview

This project trains a visuomotor imitation-learning policy in Isaac Lab.

```text
Isaac Sim + Isaac Lab
  -> Table and wrist RGB cameras + robot state
  -> Robomimic BC-RNN-GMM
  -> 6D IK-Rel command + gripper command
  -> Franka Panda
```

## Policy Inputs

- `table_cam`: 200 x 200 RGB image
- `wrist_cam`: 200 x 200 RGB image
- `eef_pos`: end-effector position
- `eef_quat`: end-effector orientation
- `gripper_pos`: gripper joint state

The fixed text `pick up the cube` is HDF5 metadata and is not a model input.

## Policy Output

```text
[x, y, z, rx, ry, rz, gripper]
```

The first six values control the relative end-effector pose. The final value controls the gripper.

## Workflow

1. Collect successful state-machine demonstrations.
2. Store images, robot state, and actions in Robomimic HDF5.
3. Train BC-RNN-GMM for 200 epochs.
4. Run closed-loop evaluation in Isaac Lab.
5. Save metrics, success videos, and failure videos.

## Platform Roles

- Isaac Sim: robot, physics, sensors, and rendering.
- Isaac Lab: task, control, observations, data collection, and evaluation.

## Reproduction

The closed-loop success was not deterministic from seed alone. The successful action sequence is saved and can be replayed from its recorded initial cube position.