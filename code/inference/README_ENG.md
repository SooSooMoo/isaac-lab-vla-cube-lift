# Inference and staged control

[日本語](README_JPN.md)

## Pipeline
1. Capture pre-action images and robot state.
2. Estimate relative position and add EEF position to obtain environment coordinates.
3. Apply EMA: 80% previous estimate and 20% current estimate, reset per trial.
4. Approach targets 100 mm above the estimate; descend and close target the estimate itself.
5. Five consecutive steps within 10 mm and 0.1 rad trigger approach→descend and descend→close transitions. Twenty close steps precede lift.
6. Lift targets 180 mm above the estimated anchor captured at closure transition. Hold retains the fixed goal.

True Cube position is used for scoring and recording, not visual-control input. The SOURCE worker in the [evaluation launcher](../snapshots/check_visual_validation26.py) uses relative-pose DLS IK and checks consistency of sensor and IK frames.

## Limits
Prediction errors change images and subsequent trajectories. Small offline error does not establish rollout success. EMA alone did not resolve the 12/13 result; the final model also incorporates additional training data.
