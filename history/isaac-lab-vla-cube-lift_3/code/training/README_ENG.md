# Training and final adjustment

`fit_gripper_reference.py` is the original final gripper-adjustment script. It depends on absolute paths, feature caches, and predecessor models from RunPod that are not bundled. It is not a standalone reproduction of the full training pipeline.

The final adjustment freezes the recurrent model and solves gripper output weights with linear programming. Runtime uses the resulting learned model output.
