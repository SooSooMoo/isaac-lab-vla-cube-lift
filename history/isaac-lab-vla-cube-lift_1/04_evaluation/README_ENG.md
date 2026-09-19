# 04. Evaluation

This folder contains the code and result summary for closed-loop evaluation, video recording, and offline teacher-forced diagnostics of the trained Robomimic policy.

## Files

| File | Description |
|---|---|
| `04_evaluate_policy.py` | Evaluates the trained policy in Isaac Lab and saves videos, action traces, and CSV results |
| `05_offline_diagnostic.py` | Compares model predictions against recorded expert actions |
| `evaluation_summary.csv` | Stores the main evaluation results in machine-readable format |

## Closed-Loop Evaluation

At every step, the current camera observations and robot state are provided to the policy. The predicted action is applied to the Isaac Lab environment.

```text
Current observation
        ↓
Robomimic BC-RNN-GMM
        ↓
Predicted action
        ↓
Isaac Lab environment
        ↓
Next observation
```

### Example

```bash
cd /workspace/IsaacLab-develop

./isaaclab.sh -p   /workspace/step2/isaac-lab-vla-cube-lift_1/04_evaluation/04_evaluate_policy.py   --task IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor   --checkpoint /workspace/IsaacLab-develop/logs/robomimic/IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor/bc_rnn_image_franka_lift/20260829133122/models/model_epoch_200.pth   --horizon 250   --num_rollouts 10   --seed 2011   --device cuda:0   --viz kit   --kit_args "--enable omni.replicator.core"
```

Main arguments:

| Argument | Description |
|---|---|
| `--checkpoint` | Robomimic checkpoint to evaluate |
| `--horizon` | Maximum number of steps per rollout |
| `--num_rollouts` | Number of rollouts |
| `--seed` | Initial random seed |
| `--fixed_trial_seed` | Uses the same seed for every trial |
| `--stop_on_success` | Stops after the first successful rollout |

## Offline Teacher-Forced Diagnostic

The offline diagostic sequentially feeds observations recorded in the HDF5 dataset to the model and compares predictions with expert actions. Predicted actions are not applied to the Isaac Lab environment.

```bash
cd /workspace/IsaacLab-develop

./isaaclab.sh -p   /workspace/step2/isaac-lab-vla-cube-lift_1/04_evaluation/05_offline_diagnostic.py
```

The diagnostic uses the first 10 of the 50 demonstrations.

## Results

| Evaluation | Successes | Trials | Success rate |
|---|---:|---:|---:|
| Initial rollouts | 1 | 10 | 10.0% |
| Seed search | 1 | 48 | 2.08% |

Successful condition:

- Seed: 2058
- Initial cube position: `(0.426184, 0.179950, 0.055000)`
- Success step during seed search: 156

Offline diagnostic:

- Overall MAE: 0.1127
- Gripper accuracy: 97.99%

## Interpretation

The offline teacher-forced diagnostic showed correlation with expert actions, while closed-loop success remained low.

In behavior cloning, small prediction errors can move the robot into states not represented in the training dataset, causing errors to accumulate.

Seed 2058 alone did not reproduce success on every run. Reproduction of the successful action sequence is documented in `05_reproduction`.

## Language Input

The fixed instruction `pick up the cube` is stored as HDF5 metadata, but the evaluated policy does not use language as an input.
