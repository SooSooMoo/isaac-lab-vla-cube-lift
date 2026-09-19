# Experiment Summary

## Training

- Environment: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`
- Robot: Franka Panda
- Control: Differential IK relative pose control
- Algorithm: Robomimic BC-RNN-GMM
- Demonstrations: 50 successful episodes
- Total samples: 7,720
- Epochs: 200
- Checkpoint: `model_epoch_200.pth`

## Policy

Inputs include table and wrist RGB images, end-effector position and orientation, and gripper joint state. The output is a seven-dimensional IK-Rel and gripper action.

The fixed instruction `pick up the cube` is stored as HDF5 metadata but is not used as a language input by the current model.

## Evaluation

| Metric | Result |
| --- | --- |
| Initial evaluation | 1/10 successful rollouts |
| Recorded seed search | 1/48 successful rollouts |
| Successful seed | 2058 |
| Initial cube position | `(0.42618394, 0.17994991, 0.055)` |
| Exact action replay | 1/1 |
| Exact replay success step | 101 |
| Offline overall MAE | 0.1127 |
| Gripper accuracy | 97.99% |

## Interpretation

The policy produced one genuine closed-loop cube-lift success, but its overall success rate remained low. Typical failures included diagonal contact, pushing the cube, missing the grasp, and dropping it.

The success was not reliably reproduced using only the same seed and initial position. Replaying the saved successful action sequence from the recorded initial position succeeded.

This project is an Isaac Lab visuomotor imitation-learning baseline toward a future language-conditioned VLA system. The current policy is not language-conditioned.