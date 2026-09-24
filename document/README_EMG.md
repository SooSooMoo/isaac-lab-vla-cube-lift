# Technical Documentation

[日本語](README_JPN.md)

## 1. Purpose and evaluation scope

This Isaac Lab project executes `move above the cube` and `pick up the cube` using the same language-conditioned model stack. Evaluation covers a fixed initial state and these two exact instructions.

This document consolidates the method, environment, development history, reproduction steps, results, and publication status. Code is in `../code/`, training-related data in `../datasets/`, and models and evaluation records in `../result/`.

## 2. Method and model stack

1. Normalize 200×200 RGB images from the table and wrist cameras using ImageNet mean and standard deviation.
2. Provide end-effector position, xyzw orientation, finger positions, progress as `step/1200`, and a language embedding.
3. Apply a learned residual to the base policy's fused features and generate six arm components and a gripper output.
4. Add a learned linear correction to the gripper logit.
5. Compute a learned score from standardized fused features, clamp it to 0–1, and multiply XY commands by `1 - score`.

The final stack consists of four files: the base model, visual adapter, gripper correction, and XY attenuation. Both instructions use the same stack, without a fixed-step switch to another policy. Progress is nevertheless included in the input features.

Diagnostic trajectories with teacher interventions supplied additional training targets. Results obtained with interventions are distinguished from autonomous evaluation results. Preserving outputs on fixed inputs does not guarantee trajectory preservation when observations evolve during execution.

See [shared_loader.py](../code/inference/shared_loader.py) for the implementation and [manifest.json](../result/model/manifest.json) for model identities.

## 3. Verified results

| Item | above | pick |
|---|---|---|
| Executed steps | 424 | 879 |
| Position or height | Final target error 20.887 mm; limit 30 mm | Final cube height 124.960 mm; required height 120.9996 mm |
| Duration criterion | Held for 50 steps | Height criterion met for 10 consecutive steps |
| Horizontal cube displacement | — | Maximum 31.025 mm after first closure; limit 50 mm |
| Repeat execution | All seven recorded arrays identical | All seven recorded arrays identical |

The compared arrays contain actions, pre/post-action end-effector positions, pre/post-action orientations, post-action cube positions, and finger positions. Reported values are in [reported_results.json](../result/evidence/reported_results.json).

The user approved the original videos' motion and the use of versions with trimmed endings. Continuing inference and control for three seconds after success revealed position drift for above and a dropped cube for pick. Trimming changes the displayed interval; it does not improve sustained holding. Secure grasp has not been verified using contact forces.

## 4. Runtime environment

The reported environment used Python 3.12, PyTorch 2.11.0+cu128, and an NVIDIA GeForce RTX 4090. Python is launched through `/isaac-sim/python.sh`, and Isaac Lab through `/workspace/IsaacLab-develop/isaaclab.sh`. Video encoding uses the ffmpeg bundled with imageio-ffmpeg.

The required custom task is `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`. Evaluation checks the pose function's xyzw convention. An environment using wxyz or lacking the custom task is not directly interchangeable.

The recorded NGX libraries reside at `/workspace/step3/ngx_runtime_580.65.06_cgbe5u2a/lib` and are added to the evaluation process's `LD_LIBRARY_PATH`. Initial camera inputs were verified after the NGX repair. NVIDIA binaries require redistribution review and are not included in ordinary source distribution.

The guarded runtime checks that cuDNN benchmark is initially True, then changes it to False. Other expected values are deterministic algorithms=False, cuDNN deterministic=False, matmul TF32=False, cuDNN TF32=True, and an unset `CUBLAS_WORKSPACE_CONFIG`. These describe the experiment, not general recommended settings.

Consult preserved environment records for exact dependencies. Existing Isaac Sim/Kit, custom Isaac Lab code, Python packages, and external USD assets are required. A clean-environment rebuild has not been verified.

## 5. Development history and data

- The initial height-only criterion did not exclude large horizontal movement. Horizontal displacement checks and visual review were added.
- Small fitting errors did not prevent trajectories from diverging as camera observations changed. Validation therefore progressed from short executions to longer runs.
- Gripper correction improved premature release, and learned XY attenuation preserved the approach while reducing horizontal motion during lifting.
- The September 19, 2026 update backed up the existing folder, checked hashes, and organized models, evaluation records, and videos.
- Dataset organization imported 18 files totaling 110.45 MiB and verified copy hashes against the sources. This does not establish complete training reproducibility.
- A subsequent GitHub update added 13 previously missing NPZ files, approximately 14.62 MiB, verified all 13 hashes, and preserved existing files.

See `../datasets/dependency_manifest.json` for data dependencies. Training used additional observations, preservation data, and intervention-derived targets alongside the original HDF5 dataset.

## 6. Reproduction and verification order

The current repository path on the Pod is `/workspace/step3/isaac-lab-vla-cube-lift`. Run the following only on a Pod with the required environment and original experiment files available.

```bash
cd /workspace/step3/isaac-lab-vla-cube-lift
/isaac-sim/python.sh code/tools/check_package.py
/isaac-sim/python.sh code/tools/check_package.py --require-imported
```

After package checks, run evaluation when needed:

```bash
bash code/evaluation/verify_xy_attenuation_above.sh
bash code/evaluation/verify_xy_attenuation1300.sh
bash code/evaluation/repeat_both_xy_attenuation.sh
```

Evaluation scripts also depend on absolute paths and wrappers from the original Pod. Obtaining the repository alone does not establish runtime readiness. Source integrity, imported artifacts, and simulator evaluation are separate checks.

The historical `update_portfolio_on_pod.py` targets the former folder name `isaac-lab-vla-cube-lift_2` and its contemporary distribution ZIP. Check its targets and changes before rerunning it against the current folder.

## 7. Publication status and handoff

- Verified: numerical criteria for both tasks in the fixed initial state, identical repeat executions, recording trajectory agreement, and user review of the videos.
- Verified: artifact import into the Pod and publication of the additional 13 NPZ files on GitHub.
- Not verified: language-pair reevaluation of the final complete stack, full training reproduction from scratch, clean-environment restoration, sustained stable holding, contact-force validation, and evaluation on unseen arrangements.

Do not cite intermediate September 17 results as final results. Original, extended, and trimmed videos serve different purposes. Disk quota was repeatedly exhausted; check available capacity before saving additional artifacts and protect verified models and data.

The six former individual documents have been consolidated into this bilingual documentation. The English and Japanese versions cover the same scope. Earlier versions remain available in Git history.
