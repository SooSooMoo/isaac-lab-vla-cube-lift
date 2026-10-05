# Code reference and execution order

[日本語](README_JPN.md)

## Purpose
The learned component estimates Cube position from images. Explicit approach, descend, close, lift and hold stages control the robot. This differs from the previous language/progress-conditioned action model.

## Scripts
| File | Purpose and timing |
|---|---|
| [check_visual_validation26.py](snapshots/check_visual_validation26.py) | Evaluate the selected candidate at 26 existing positions |
| [check_visual_random30.py](snapshots/check_visual_random30.py) | Evaluate five new random locations with frozen weights |
| [check_visual_positive_x30.py](snapshots/check_visual_positive_x30.py) | Supplement coverage with four positive-X strata |
| [record_portfolio_video3.py](snapshots/record_portfolio_video3.py) | Record and score three runs; verify decoded frame counts |
| [train_spatial_cube_oracle30_refit.py](snapshots/train_spatial_cube_oracle30_refit.py) | Train the final candidate; unnecessary for evaluation-only use |
| [artifact_import.json](configs/artifact_import.json) | Source paths and SHA-256 identities of published artifacts |

Launchers store their worker in the SOURCE string, write it to a temporary directory, and launch Isaac Lab. Loading, control and scoring remain in that worker; implementation was not relocated to the explanatory folders.

## Reading order
[Model](model/README_ENG.md) → [Inference](inference/README_ENG.md) → [Collection](data_collection/README_ENG.md) → [Training](training/README_ENG.md) → [Evidence](../result/evidence/README_ENG.md).

## Running
On the original prepared Pod, use `/isaac-sim/python.sh -u code/snapshots/check_visual_validation26.py` and the corresponding other launchers. Absolute-path data, weights, initial state, smoke_utils and custom Isaac Lab are required. These are not standalone checkout-and-run CLIs. Documentation updates do not require retraining or simulation.
