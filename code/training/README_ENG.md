# Training

[日本語](README_JPN.md)

## Implementation
[train_spatial_cube_oracle30_refit.py](../snapshots/train_spatial_cube_oracle30_refit.py) created the selected candidate. Frozen visual features train a position head. It rebuilds the head from recorded source weights and data rather than continuing from the final inference checkpoint.

## Settings and selection
Seed 2083; AdamW lr 0.0003, weight decay 0.0001; 80 epochs; batch 128. Sampling with replacement balances positions. Targets are scaled by 0.1 m with SmoothL1 beta 0.05. Epoch 62 was selected by lowest validation RMSE.

The 23 episodes span training, validation and diagnostics. Validation IDs are 4 and 13. The held-out position is (+20,-20 mm); older (-20,+20 mm) trajectories form recovery_check. These diagnostics were inspected during development and are not a fresh blind test.

See [protocol and input hashes](../../result/training/protocol.json) and [offline results](../../result/training/results.json). Offline fit and rollout success are separate checks. Evaluation of existing weights does not require retraining.
