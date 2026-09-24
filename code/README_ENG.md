# Code Reference and Execution Workflow

## Purpose

This directory contains the code for a shared policy that uses table-camera and wrist-camera images, robot state, and a language instruction to perform two tasks:

- `move above the cube`: move to and hold a target position above the cube.
- `pick up the cube`: grasp and lift the cube.

The final stack consists of four artifacts: a base model, a visual adapter, a gripper correction, and XY attenuation. Both tasks use the same stack. See the [model manifest](../result/model/manifest.json) for artifact identities and the [verification records](../result/evidence/README.md) for numerical results.

**These are experiment scripts tied to the verified Pod environment, not a sequence in which every file should be executed.** They also require files outside this repository, including training data, previous experiment outputs, and the custom Isaac Lab environment. Check the fixed paths and hashes inside each script.

## 1. File Reference

All paths below are relative to `code/`. The tables cover code and configuration files; explanatory README files are excluded.

### Model and Inference

| File | Summary | Details and when to use it |
|---|---|---|
| [model/language_vla/model.py](model/language_vla/model.py) | Base VLA model | Processes two camera images, robot state, and language to produce arm and gripper outputs. Start here to understand the architecture. Additional learned corrections are required for the final stack. |
| [model/language_vla/__init__.py](model/language_vla/__init__.py) | Python package marker | Makes the model directory a Python package. No standalone execution is needed. |
| [inference/shared_loader.py](inference/shared_loader.py) | Shared loader for the final stack | Checks weights and dependency hashes, installs visual residuals, gripper correction, and feature-based XY attenuation, and prepares images and state. Extracted from the verified evaluation script as a reference implementation; it is not a standalone CLI. |

Each evaluation script contains its own verified loader implementation. Editing `inference/shared_loader.py` alone does not update all evaluation scripts.

### Simulator Runtime

| File | Summary | Details and when to use it |
|---|---|---|
| [runtime/evaluate.py](runtime/evaluate.py) | Evaluation loop | Launches the environment, obtains observations, runs inference, executes actions, records trajectories, and checks termination conditions. Use the `evaluation/` scripts below to evaluate the final stack. |
| [runtime/evaluate_above_wrapper.py](runtime/evaluate_above_wrapper.py) | Preflight checks and evaluation launch | Checks initial inputs and PyTorch settings before launching evaluation. Experiment scripts read and adapt the corresponding wrapper on the original Pod. |
| [runtime/smoke_utils.py](runtime/smoke_utils.py) | Shared utilities | Provides language tokenization, observation extraction, episode export, initial-state comparison, and policy loading. Final evaluation replaces the loading routine with the shared stack loader. |
| [runtime/collect.py](runtime/collect.py) | Teacher episode collection | Runs teacher control for a specified instruction and initial state, saving observations and actions. Used during early data preparation; teacher-controlled outcomes are distinct from autonomous policy success. |
| [runtime/pack.py](runtime/pack.py) | Two-instruction dataset packaging | Checks the pick and above HDF5 episodes for their instructions, initial states, and pre-action observation timing, then creates a combined dataset. Used when preparing teacher data. |
| [runtime/language_vla/model.py](runtime/language_vla/model.py) | Runtime copy of the model | Keeps the same source as `model/language_vla/model.py` in the runtime package layout. |
| [runtime/language_vla/dataset.py](runtime/language_vla/dataset.py) | Training dataset reader | Constructs image, state, language, and action samples from HDF5 and provides batch collation. Refer to it when inspecting or extending the data format. |
| [runtime/language_vla/__init__.py](runtime/language_vla/__init__.py) | Runtime package marker | Defines the runtime `language_vla` package. No standalone execution is needed. |

### Additional Data Collection

| File | Summary | Details and when to use it |
|---|---|---|
| [data_collection/collect_current_arm_hold623.sh](data_collection/collect_current_arm_hold623.sh) | Teacher data for settling and closure | After a 623-step approach, collects 20 settling steps followed by 40 closed-gripper holding steps. This is a teacher-intervention experiment used while developing the arm-hold adapter. |
| [data_collection/collect_visual_inputs13.sh](data_collection/collect_visual_inputs13.sh) | Observations of visual variation | Saves 13 observations from each of a reference run and a run with a correction. Used for visual consistency training; it does not supply new expert action labels. |
| [data_collection/collect_retention_inputs700.sh](data_collection/collect_retention_inputs700.sh) | Gripper-retention inputs | Saves features and actions over 700 steps. The 86 observations from step 614 onward use diagnostic forced-closure targets. Run before training the gripper correction. |
| [data_collection/collect_xy_correction879.sh](data_collection/collect_xy_correction879.sh) | XY correction inputs | Saves preservation targets for the first 800 observations and XY-suppression targets for the following 79. Used to train XY attenuation; intervention success is not counted as autonomous success. |

### Training

| File | Summary | Details and when to use it |
|---|---|---|
| [training/train_current_hold_adapter.sh](training/train_current_hold_adapter.sh) | Arm-hold adapter training | Uses teacher settling/closure data and observations that constrain changes to existing actions. Produces the adapter used in the subsequent visual-correction development stage. |
| [training/train_visual_consistency13.sh](training/train_visual_consistency13.sh) | Visual consistency training | Reuses 13 observation pairs to reduce output differences caused by wrist-image variation. Builds on an existing adapter and produces the final stack's `visual_adapter.pt`. |
| [training/train_retention_gripper.sh](training/train_retention_gripper.sh) | Gripper correction training | Learns a linear correction from fused features. Preserves gripper decisions on the first 614 observations and above observations while fitting 86 closure targets. Arm outputs are unchanged for identical inputs. |
| [training/train_xy_attenuation.sh](training/train_xy_attenuation.sh) | XY attenuation training | Learns a feature-based attenuation value clipped to 0–1 and applies it to XY commands. On cached observations, it leaves approach and above outputs unchanged while fitting the 79 XY-suppression targets. |

Passing a training check means that constraints on saved inputs were met. New weights must still be evaluated in the simulator. Retraining is unnecessary when reproducing evaluations or videos with the existing final model.

### Diagnostics

| File | Summary | Details and when to use it |
|---|---|---|
| [diagnostics/diagnose_xy_suppression800.sh](diagnostics/diagnose_xy_suppression800.sh) | Horizontal-command intervention | Zeros XY commands from step 800 onward to investigate horizontal motion. The adopted policy does not use this fixed-step override. |
| [diagnostics/probe_xy_feature_separation.sh](diagnostics/probe_xy_feature_separation.sh) | Feature separability probe | Tests whether cached features linearly distinguish correction regions from preservation regions. Uses blocks from the same trajectories; this is not an independent generalization evaluation. |

### Final Evaluation and Repeatability

| File | Summary | Details and when to use it |
|---|---|---|
| [evaluation/verify_xy_attenuation_above.sh](evaluation/verify_xy_attenuation_above.sh) | Final above evaluation | Evaluates the complete shared stack against the requirement to remain within 30 mm of the target for 50 steps. Run after selecting the final weights. |
| [evaluation/verify_xy_attenuation1300.sh](evaluation/verify_xy_attenuation1300.sh) | Final pick evaluation | Runs for up to 1,300 steps, checking cube height of approximately 121 mm for 10 consecutive steps and horizontal displacement within 50 mm. Terminates early on success. |
| [evaluation/repeat_both_xy_attenuation.sh](evaluation/repeat_both_xy_attenuation.sh) | Repeat both tasks | Repeats the successful configuration and compares seven arrays covering actions, end-effector positions/orientations, cube positions, and finger positions with the reference trajectories. Run after both numerical evaluations pass. |

### Video

| File | Summary | Details and when to use it |
|---|---|---|
| [video/record_both_policy_videos.sh](video/record_both_policy_videos.sh) | Record original videos | Reruns the same policy and records table and wrist views side by side at 50 fps. Checks trajectory agreement and video decoding. |
| [video/record_both_continued3s_v2.sh](video/record_both_continued3s_v2.sh) | Record three seconds after success | Continues inference and control with the same model for 150 additional steps. Does not freeze the robot or duplicate still frames. Reveals post-success behavior, including any loss of stability. |
| [video/trim_final_videos.sh](video/trim_final_videos.sh) | Trim presentation videos | Keeps 495 frames for above and 930 for pick from the extended recordings, saving separate files and preserving originals. Stops without overwriting if edited files already exist. |

### Configuration and Package Checks

| File | Summary | Details and when to use it |
|---|---|---|
| [configs/success.json](configs/success.json) | Success criteria and identity | Records the base-model SHA, progress denominator, timestep, and numerical criteria. It is a record of settings, not a configuration loader that automatically changes evaluation scripts. |
| [configs/artifact_import.json](configs/artifact_import.json) | Artifact import definitions | Lists source paths, destinations, required artifacts, hashes, and video frame counts for models, verification JSON, and videos. Used by the Pod update program. |
| [tools/check_package.py](tools/check_package.py) | Package integrity checks | Checks Python, JSON, Markdown links, embedded Python in shell scripts, and file hashes. Add `--require-imported` to check imported artifacts. Run after documentation updates and before publication. |

## 2. Project Workflow and Execution Timing

| Order | Stage | When to run it | What to confirm before proceeding |
|---:|---|---|---|
| 1 | Check environment and inputs | After moving to a new Pod or changing the environment | Custom task, dependencies, initial observations, and model hashes match the expected setup. |
| 2 | Collect teacher/correction data | After identifying a motion problem and defining correction targets | Observation/action alignment, pre-intervention trajectory agreement, and saved-data integrity. |
| 3 | Train and check cached inputs | Once the required data is available | Finite values, preservation of existing outputs, improvement on target outputs, and successful weight saving/reloading. |
| 4 | Evaluate both tasks with the shared stack | Before adopting a new model candidate | Both above position-hold and pick height/horizontal-displacement criteria pass. |
| 5 | Repeat and compare | After both numerical evaluations pass | Trajectory agreement under identical conditions, reported as fixed-condition repeatability. |
| 6 | Record and review | When preparing a demo of the verified model | Recording trajectory agreement, cube visibility, unwanted oscillation, and a suitable ending, including human review. |
| 7 | Trim the ending if needed | After selecting the presentation interval | Original footage is preserved and edits are documented. Editing does not improve the underlying controller. |
| 8 | Import artifacts and prepare publication | Once models, results, and videos are finalized | Backups, hashes, consistency between documentation and results, and working links. |

The adopted stack has completed stages 4–6, and the presentation intervals have been selected. During three seconds of continued inference, above drifted and pick dropped the cube, so presentation clips were trimmed. Sustained post-success stability has not been established.

### Training Dependencies During Development

```text
Base model + existing teacher data and preservation observations
    → Arm-hold adapter training
Existing adapter + 13 observation pairs
    → Visual consistency training
Visual adapter + collected gripper-retention data
    → Gripper correction training
Visual adapter + gripper correction + collected XY-intervention data
    → Feature separability check and XY attenuation training
Final shared stack → Both-task evaluation → Repeat runs → Recording
```

This describes dependencies rather than an automatic build pipeline. Paths are fixed to selected experiment outputs, whereas retraining creates new output locations. The diagram alone is not a from-scratch reproduction procedure.

## 3. Common Commands

Run these commands on the verified Pod. The original experiment files at the fixed paths are also required.

### Check Documentation and Packaged Files

```bash
cd /workspace/step3/isaac-lab-vla-cube-lift_2
/isaac-sim/python.sh code/tools/check_package.py --require-imported
```

This does not train a model or run the simulator. When a README changes, update its hash in the repository-root `source_inventory.json` as well.

### Re-evaluate Both Tasks with the Current Model

```bash
cd /workspace/step3/isaac-lab-vla-cube-lift_2
bash code/evaluation/verify_xy_attenuation_above.sh
bash code/evaluation/verify_xy_attenuation1300.sh
```

These commands run the simulator and create new result files. They do not retrain weights.

### Check Repeatability of the Same Stack

```bash
cd /workspace/step3/isaac-lab-vla-cube-lift_2
bash code/evaluation/repeat_both_xy_attenuation.sh
```

Reference paths are fixed to the adopted experiment records. A changed model requires evaluation and comparison procedures appropriate to that candidate.

## 4. Before Running

- See the [environment notes](../document/README_EMG.md#4-runtime-environment) and [reproduction guide](../document/README_EMG.md#6-reproduction-and-verification-order) for dependencies.
- Training, recording, and repeated simulation consume storage. Account for the Pod's disk quota and avoid unnecessary reruns.
- Collection and diagnostic scripts may use teacher interventions. Keep those interventions separate from final autonomous evaluation.
- Preserving arm or gripper outputs for identical inputs does not guarantee unchanged future actions once observations evolve. Training error alone is not a success criterion.
- The final stack includes progress as an input. Generalization to new layouts or paraphrased instructions, and contact-force-based grasp verification, are outside the demonstrated scope.
