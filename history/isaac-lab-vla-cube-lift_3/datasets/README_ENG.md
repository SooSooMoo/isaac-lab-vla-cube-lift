# Datasets

HDF5 episodes store pre-action camera images, robot state, and seven-dimensional actions. The `instruction` attribute is `pick up the cube`. Teacher corrections retain the causal policy prefix and identify supervised teacher steps with `teacher_mask`.

The materials ZIP does not include HDF5 files. The complete training dependency inventory and distribution plan remain pending. Earlier position-estimation datasets are not presented as the training data for this policy.


[Dependency manifest](dependency_manifest.json)

Evaluation reads `data/demo_0/initial_state` and the first EEF observation from `baseline_inputs/datasets/lift_robomimic_language.hdf5`. This is not the complete training dataset.
