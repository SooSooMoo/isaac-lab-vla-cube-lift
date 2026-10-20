# Grasp and lift a cube at an arbitrary location using verbal instructions and images.

![Cube lift and hold in four quadrants](result/evidence/images/four_quadrants.png)

Frames from the reviewed recordings during lift and hold in four quadrants. Each panel shows the table camera on the left and wrist camera on the right.


[日本語](README_JPN.md)

**Given “pick up the cube” and camera images, Franka grasps a cube, lifts it, and holds it for three seconds.**

This simulated VLA policy takes table-camera and wrist-camera images, robot state, and a language instruction. A learned model outputs arm and gripper actions.

Model weights are not published. This repository provides implementation, evaluation evidence, and videos. Inference requires a separate checkpoint and cannot run from the public files alone.

## Videos

Four quadrant demonstrations of the same model. Each video shows the table camera on the left and wrist camera on the right at real-time 50 fps. All four recordings passed pick and three-second hold checks and were visually reviewed. Recording trials are not added to the evaluation counts below.

| X−24, Y＋9 mm | X＋24, Y＋9 mm |
|---|---|
| [▶ X−24 / Y＋9](result/evidence/videos/x-24_y9.mp4) | [▶ X＋24 / Y＋9](result/evidence/videos/x24_y9.mp4) |
| **X−24, Y−9 mm** | **X＋24, Y−9 mm** |
| [▶ X−24 / Y−9](result/evidence/videos/x-24_y-9.mp4) | [▶ X＋24 / Y−9](result/evidence/videos/x24_y-9.mp4) |


## Results

| Evaluation | Result |
|---|---:|
| 29 positions used during development | 29/29 successful |
| 8 additional positions first evaluated with the frozen model | 7/8 successful |

Positions lie within ±30 mm on each XY axis relative to the reference placement. The additional set contains two positions per quadrant. These are individual trials, not a guarantee of success throughout the square or an estimate of general reliability.

At `(11, −23) mm`, approach misalignment led to a failed lift. The failure is included in the evidence. This portfolio freezes the model from that evaluation, separately from subsequent repair experiments.

## Language, vision, and action

- **Language:** Data use `pick up the cube`; inference tokenizes that instruction and executes the language encoder.
- **Vision:** RGB images from the table and wrist cameras.
- **Action:** Fused visual, language, and robot-state features feed a causal GRU that outputs seven action dimensions.

No teacher overrides actions during these evaluations. Ground-truth cube coordinates are used for scoring and teacher collection, not as policy inputs.

Training and evaluation use a single instruction. Behavioral changes in response to different instruction meanings have not been verified.

## Documentation

- [Architecture and criteria](document/README_ENG.md)
- [All 37 evaluated positions](result/evidence/README_ENG.md)
- [Model](result/models/README_ENG.md)
- [Datasets](datasets/README_ENG.md)
- [Code and reproduction readiness](code/README_ENG.md)

## Limitations

Results use fixed initial states, cameras, and a simulation environment. Real hardware, changed lighting or object shapes, and general success probabilities are untested. Grasping is assessed through lift and hold behavior, not contact-force validation.

The earlier visual-position-estimator plus staged-controller result at 35 positions is not a result of this VLA model.

## Future work

The goal is complete success: one model achieving pick and three-second hold at every designated evaluation position. The challenge is to correct failures while preserving performance at positions that already succeed.

Repeated trials and evaluations at new positions will assess reliability and help expand the range of stable operation. Success at every tested position will not, by itself, be treated as a guarantee at every position in the region.
