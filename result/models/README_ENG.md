# Current model

[日本語](README_JPN.md)

## File and size
[`cube_position_estimator.pt`](cube_position_estimator.pt) is 9,455,338 bytes (~9.46 MB, 9.02 MiB). This single inference checkpoint includes the image encoder, position head and normalization values. Staged control resides in Python.

SHA-256: `9a5004fe9ad27b4266b08002842cd5d48eac57235d3d57f8405bbc38feec719f`

See [source identity](manifest.json). Selected epoch: 62; format: spatial_cube_estimator_v1. Unlike the prior four-artifact stack, only one learned inference checkpoint is needed. The initial state and custom simulator environment remain separate dependencies.
