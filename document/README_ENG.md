# Methods and reproduction

[日本語](README_JPN.md)


Frozen ResNet18 features through layer2 are pooled to 5×5 per camera, combined with robot state (6409 dimensions), and passed through a 256→64→3 GELU head to estimate Cube minus EEF position. An explicit staged controller uses the estimate, smoothed in environment coordinates with EMA alpha 0.2.

Approach/descend transitions require position error within 10 mm and rotation error within 0.1 rad for five consecutive steps. Closing lasts 20 steps before lifting. Pick checks require Cube height at least 100 mm above the settled height, a closed command, and EEF–Cube distance at most 80 mm for ten steps. A further 150 steps at dt=0.02 s test retention. Horizontal displacement is limited to 50 mm from the settled position. Ground truth is used for scoring. Contact forces are not verified.

Offsets are relative to initial Cube center approximately (0.573153, 0.039928, 0.055) m. Reported height is an environment-coordinate height, not lift distance. Initial pose, lighting and Cube color are fixed. Above was not re-evaluated here.

`code/snapshots/` preserves experimental scripts with absolute RunPod paths. They require the original initial-state HDF5, smoke_utils, Isaac Lab source and local assets. This repository alone is not a standalone reproduction environment. Training images and NVIDIA assets are not included. The model and videos were published after SHA-256 verification. Fresh-environment reproduction remains unverified.

Recorded environment: RTX 4090; torch 2.11.0+cu128; torchvision 0.26.0+cu128; numpy 2.5.3; gymnasium 1.2.1. The training protocol and evaluation protocols identify inputs and settings.

## Documentation map
[Code](../code/README_ENG.md) · [Data](../datasets/README_ENG.md) · [Model](../result/models/README_ENG.md) · [Evidence](../result/evidence/README_ENG.md) · [Videos](../result/evidence/videos/README_ENG.md)
