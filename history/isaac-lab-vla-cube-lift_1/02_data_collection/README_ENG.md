# 02. Data Collection

This folder contains the code and dataset information used to collect successful Franka Cube Lift demonstrations with an Isaac Lab state machine as the expert.

## Files

| Path | Description |
|---|---|
| `02_collect_demonstrations.py` | Collects successful demonstrations using the state machine |
| `data/metadata/` | Stores metadata such as dataset size, shapes, and collection conditions |
| `data/samples/` | Stores a small public sample for inspecting the dataset structure on GitHub |

## Collection Procedure

The Franka Panda Pick and Lift state machine is used as the expert policy.

For each attempt, the initial cube position is randomized and the following sequence is executed:

1. Move above the cube
2. Approach the cube
3. Close the gripper
4. Grasp the cube
5. Lift the cube toward the target position

Only attempts that satisfy the success condition are saved to the HDF5 dataset.

## Dataset

- Format: Robomimic HDF5
- Demonstrations: 50 successful episodes
- Expert: Isaac Lab state machine
- Robot: Franka Panda
- Controller: IK-Rel
- Task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`

## Observations

| Key | Description |
|---|---|
| `table_cam` | Table RGB camera image, 200 x 200 x 3 |
| `wrist_cam` | Wrist RGB camera image, 200 x 200 x 3 |
| `eef_pos` | End-effector position |
| `eef_quat` | End-effector orientation |
| `gripper_pos` | Gripper joint state |

## Actions

The expert action has seven dimensions.

```text
[x, y, z, rx, ry, rz, gripper]
```

The first six dimensions are relative IK commands, and the final dimension is the gripper command.

## Language Data

The Robomimic HDF5 dataset stores the following fixed instruction as dataset-level metadata:

```text
language_conditioning: fixed_instruction
language_instruction: pick up the cube
```

However, this text is not stored in the per-step `obs` group and is not used as an input to the current Robomimic BC-RNN policy.

```text
Fixed instruction stored as HDF5 metadata: yes
Language instruction provided to the model: no
Language-conditioned action selection: no
```

Because every demonstration performs the same Cube Lift task, the intention to lift the cube is implicitly fixed across the dataset.

A language-conditioned VLA extension will require multiple instructions and data containing different targets or behaviors selected by those instructions.

## Publication Plan

The complete HDF5 dataset will be published through GitHub Releases or external storage after its file size is confirmed.

The `data/samples/` directory will contain only a small sample for inspecting the dataset structure. The public sample and the complete training dataset must be clearly distinguished.
