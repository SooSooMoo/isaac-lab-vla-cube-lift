# Position estimator

[日本語](README_JPN.md)

## Inputs and output
Inputs are two 200×200 RGB views, EEF position (3), xyzw orientation (4, sign flipped if w is negative), and finger position (2). The output is Cube minus EEF in environment axes, in metres. Language and progress are not inputs.

## Architecture
Frozen ResNet18 through layer2 produces 128×5×5 pooled features per view. The 6400 image features and 9 state values feed a 6409→256 GELU→64 GELU→3 regression head. Images use ImageNet normalization; feature and target normalization parameters are stored in the checkpoint.

See the [training script](../snapshots/train_spatial_cube_oracle30_refit.py) and `load_cube_estimator` inside the [evaluation launcher](../snapshots/check_visual_validation26.py).

## Weights
[One checkpoint](../../result/models/README_ENG.md) contains all learned inference weights. Previous policy corrections are unnecessary for inference; training still references an older checkpoint to initialize the feature extractor.
