# 01. Environment

This folder contains the Isaac Lab environment definition, environment validation script, and RunPod dependency information used by this project.

## Files

| File | Description |
|---|---|
| `01_validate_environment.py` | Validates environment creation, observations, and image data |
| `ik_rel_visuomotor_env_cfg.py` | Defines the Franka Cube Lift IK-Rel Visuomotor environment |
| `requirements/requirements-runpod-snapshot.txt` | Snapshot of the Python packages installed on RunPod |

## Environment

- GPU: NVIDIA GeForce RTX 4090
- Isaac Sim: 6.0.1 RC
- Isaac Lab: develop branch
- Robot: Franka Panda
- Controller: Differential IK, relative pose
- Control frequency: 50 Hz
- Task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`

## Connecting Isaac Sim and Isaac Lab

Isaac Sim provides physics simulation, rendering, robots, and sensors.

Isaac Lab runs on top of Isaac Sim and provides tasks, environments, controllers, observations, data collection, and learning workflows.

In this environment, Isaac Sim is connected to Isaac Lab through the following symbolic link:

```text
/workspace/IsaacLab-develop/_isaac_sim -> /isaac-sim
```

## Environment Validation

Run the validation script from the Isaac Lab repository:

```bash
cd /workspace/IsaacLab-develop

./isaaclab.sh -p \
  /workspace/step2/isaac-lab-vla-cube-lift_1/01_environment/01_validate_environment.py \
  --device cuda:0 \
  --viz none \
  --kit_args "--enable omni.replicator.core"
```

## Validation Targets

The expected observation keys are:

```text
eef_pos
eef_quat
gripper_pos
table_cam
wrist_cam
```

The expected RGB image shape is:

```text
(num_envs, 200, 200, 3)
```

A successful validation ends with:

```text
[RESULT] Lift visuomotor observation validation OK
```

## Relationship to VLA

This environment provides the Vision and Action foundations of the planned VLA system.

- Vision: table camera and wrist camera
- Action: 6DoF IK-Rel and gripper control
- Language: not implemented

Language instructions are not currently included in the observation space.
