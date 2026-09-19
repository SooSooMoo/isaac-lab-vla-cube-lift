# Limitations and VLA Roadmap

## Current Achievement

This project implemented and verified:

- Collection of 50 successful demonstrations from randomized cube positions
- Storage of table and wrist images, robot state, and actions in HDF5
- Training of a Robomimic BC-RNN-GMM policy for 200 epochs
- Closed-loop policy evaluation with video recording
- One genuine closed-loop cube-lift success
- Exact replay of the saved successful action sequence

## Current Limitations

- Closed-loop success remains low
- Initial evaluation: 1/10; recorded seed search: 1/48
- The gripper may approach diagonally and push the cube
- The cube may be dropped during grasping
- The same seed and initial position do not guarantee the same result
- `pick up the cube` is metadata, not a model input
- The current policy is a visuomotor imitation-learning baseline, not a language-conditioned VLA

## Path Toward 30% or Higher Success

1. Add demonstrations that approach the cube vertically.
2. Balance approach, grasp, and lift phases in the dataset.
3. Use DAgger to add teacher corrections at policy failure states.
4. Tune action speed and IK-Rel scale to reduce impacts.
5. Explicitly separate training and evaluation cube positions.
6. Evaluate multiple seeds and select checkpoints by rollout success.

## Next Stage: Language-Conditioned VLA

### Data

Use multiple colored cubes and instructions:

```text
pick up the red cube
pick up the blue cube
do not pick up any cube
```

The dataset must contain cases where the same image requires a different action depending on the instruction.

### Model

```text
image features + language features + robot state -> action
```

A language encoder can be fused with image and state encoders before a BC-RNN, Transformer, or Diffusion Policy action model.

### Evaluation

- Correct target-color lift rate
- Wrong-target selection rate
- Success on unseen cube positions
- Success on paraphrased instructions
- No-action accuracy for `do not pick up`

## Final Goal

Demonstrate that changing the language instruction changes the selected object or behavior, extending the current single-task visuomotor policy into a genuine language-conditioned VLA system.