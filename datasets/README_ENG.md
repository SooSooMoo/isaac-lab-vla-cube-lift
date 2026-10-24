# Datasets

This project collected teacher episodes for grasping the cube specified by its color and used them for imitation learning.
The final training run used 50 episodes: 46 for training and four for validation.

- Observations: table/wrist camera images, end-effector position and orientation, and finger positions.
- Instructions: `pick up the red cube` / `pick up the blue cube`.
- Actions: three relative translation components, three relative rotation components, and a gripper command.
- Recovery data: observation history obtained by replaying a failed trajectory, followed by corrective teacher actions. Replayed actions are excluded from supervised action targets.

**The original training HDF5 files and image data are not included in this repository.**
This directory documents the data. The same training run cannot be reproduced using the repository alone.

See [result/evidence](../result/evidence/) for training settings and audit records, and [code](../code/) for collection and audit scripts.

## Scope of the HDF5 Description

The following description is based on the published [collection script](../code/collect_step5_color_replay_recovery_red1_v1.py), [audit script](../code/audit_step5_color_replay_recovery_v1.py), [training script](../code/train_step5_color_replay_recovery_v1.py), and [audit results](../result/evidence/recovery_audit.json). It is not a specification obtained by directly inspecting the unpublished original HDF5 files.

HDF5 stores groups (similar to directories), datasets (arrays), and attributes (metadata) in a single file. In this project, each condition-specific `episode.hdf5` generally contains one episode, read from `data/demo_0`. The complete set of 50 episodes is managed through a `training_manifest.json` that references the individual files.

Here, `T` is the number of steps with teacher action labels, `C` is the replay-context length in a recovery episode, and `L` is the number of instruction tokens. The final selected recovery episode has **T=565 and C=500**. T varies across standard episodes.

### 1. Data Types

| Type | Root attribute `schema` | Purpose |
|---|---|---|
| Standard teacher data | `color_teacher_v1` | Learn from teacher observations and action labels |
| Recovery data with replay context | `color_recovery_teacher_v1` | Warm up the GRU state with replay observations, then learn the subsequent teacher actions |
| Learned-policy evaluation/recording data | `color_policy_evaluation_v1` | Evaluate and record model behavior; not teacher training data |

The tree below focuses on recovery data. Standard teacher episodes do not require `recovery_context`. The existing 49 episodes are inherited through an audited manifest, so auxiliary groups and attributes from older collection versions are not necessarily identical.

### 2. Group Structure

```text
episode.hdf5
├── [root attributes] schema, step_dt, action_scale, ...
├── data/
│   └── demo_0/
│       ├── [attributes] language_instruction, layout, success, num_samples, ...
│       ├── obs/
│       │   ├── table_cam           (T, 200, 200, 3)
│       │   ├── wrist_cam           (T, 200, 200, 3)
│       │   ├── eef_pos             (T, 3)
│       │   ├── eef_quat            (T, 4)
│       │   └── gripper_pos         (T, 2)
│       ├── actions                 (T, 7)
│       ├── language_tokens         (L,)
│       └── reset_state_before_settle/
│           ├── articulation/robot/...
│           └── rigid_object/object, object_blue/...
├── diagnostics/
│   ├── target_post_pos             (T, 3)
│   ├── other_post_pos              (T, 3)
│   ├── joint_pos                   (T, 9)
│   └── phase                       (T,)
└── recovery_context/               [recovery data only]
    ├── obs/
    │   ├── table_cam               (C, 200, 200, 3)
    │   ├── wrist_cam               (C, 200, 200, 3)
    │   ├── eef_pos                 (C, 3)
    │   ├── eef_quat                (C, 4)
    │   └── gripper_pos             (C, 2)
    └── replayed_actions            (C, 7)
```

Items marked `[attributes]` are HDF5 attributes, not array datasets. The `reset_state_before_settle` subtree is schematic: the simulator state dictionary is stored recursively.

### 3. Observation, Action, and Language Datasets

Paths in this table are relative to `data/demo_0/`.

| Path | Shape | Type / units | Contents |
|---|---|---|---|
| `obs/table_cam` | `(T,200,200,3)` | `uint8`, 0–255 | Table-camera RGB images; axes are time, height, width, and color |
| `obs/wrist_cam` | `(T,200,200,3)` | `uint8`, 0–255 | Wrist-camera RGB images with the same axis order |
| `obs/eef_pos` | `(T,3)` | Floating point, m | End-effector position `(x,y,z)` |
| `obs/eef_quat` | `(T,4)` | Floating point, dimensionless | End-effector unit quaternion `(x,y,z,w)` |
| `obs/gripper_pos` | `(T,2)` | Floating point, m | Left and right finger joint positions, not gripper commands |
| `actions` | `(T,7)` | Floating point | Six relative-IK translation/rotation command components plus one gripper command |
| `language_tokens` | `(L,)` | `int64` | Instruction tokens for the entire episode; not repeated at every step |

**Stored floating-point precision is inherited from the source arrays.** The collection script does not uniformly cast every numerical array to float32 before saving. Check `dataset.dtype` in the actual file for its precise storage type. The training code converts states and target actions to float32.

#### Coordinates, Orientations, and Images

- `eef_pos` is the end-effector position observation produced by the environment. The audit compares it with cube coordinates after subtracting the environment origin. Do not assume its frame is interchangeable with the action-command frame; preserve the original observation and IK configurations.
- `eef_quat` uses `xyzw` order. The audit checks that its norm is close to one and its final component w is nonnegative, allowing numerical tolerance.
- Orientations inside `reset_state_before_settle` use the simulator's internal representation. Do not apply the observation-level `xyzw` convention indiscriminately to those internal arrays.
- Stored images use RGB and HWC layout. Training converts them to CHW, divides by 255, and normalizes with mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.

#### Meaning of the Seven Action Components

```text
actions[t] = [dx/scale, dy/scale, dz/scale,
              rx/scale, ry/scale, rz/scale, gripper]
scale = action_scale = 0.5
```

`dx,dy,dz` are relative position commands in meters. `rx,ry,rz` form an axis-angle rotation vector in radians, not Euler angles. These are commands passed to the IK controller, not measured displacements or joint angles. The collection teacher computes errors in the robot-relative IK frame to generate actions.

`gripper=+1` commands opening, and `gripper=−1` commands closing. Do not multiply the gripper component by `action_scale`. The teacher-data audit also checks that the per-step translation-command norm is approximately at most 0.01 m and the rotation-command norm is approximately at most 0.08 rad.

#### Language Tokens

The instruction is stored as a string in the `language_instruction` attribute and as an integer sequence in `language_tokens`. Both supported instructions contain five words, so `L=5`.

Collection and auditing tokenize by lowercasing, removing periods and commas, splitting on whitespace, and computing the following ID for each of up to 16 words:

```python
1 + int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), 'little') % 4095
```

This is a word-hashing scheme specific to this implementation. It is not a standard large-language-model tokenizer, nor a format from which words can be reconstructed using integer IDs alone.

### 4. Observation–Action Timing

The root attribute `observation_timing='pre_action'` indicates this recording order:

```text
capture obs[t] → record actions[t] → env.step(actions[t])
              → record post-action diagnostics[t] → obs[t+1]
```

The supervised pair is therefore `(obs[t], instruction, actions[t])`. Since `diagnostics[t]` is recorded after the action, it corresponds in time to the next `obs[t+1]`. Avoid shifting observations and action labels by one step. The observation after the final action is not necessarily saved as an additional `obs[T]` entry.

The final training code requires **step_dt=0.02 seconds (50 Hz)** and **action_scale=0.5**. `step_dt` is the control-step interval, distinct from the simulator's internal physics-substep interval.

### 5. Attributes (Metadata)

#### File Root Attributes

| Attribute | Value / meaning |
|---|---|
| `schema` | Data-format identifier listed above |
| `training_eligible` | Whether the episode is a training candidate; set according to success in recovery collection. This flag alone does not establish audit success |
| `teacher_only` | True for teacher data. It does not mean that replayed actions in a recovery episode are teacher labels |
| `training_performed` | False during collection/evaluation; the process generating this file did not perform training |
| `observation_timing` | `pre_action` |
| `step_dt` | Control-step interval; 0.02 seconds for final training |
| `action_scale` | Relative-IK scaling factor; 0.5 for final training |
| `quaternion_order` | Observation orientation uses `xyzw` |
| `uses_progress_input` | False; teacher phase/progress is not a model input |
| `action_space` | String describing seven-dimensional relative IK and gripper signs |
| `result_json` | JSON string containing results such as success, conditions, and sample counts |

Recovery data additionally stores `replay_source`, `replay_source_sha256`, `replay_steps=500`, `context_is_teacher=false`, and `context_usage`. Original absolute paths record provenance; they are not download locations within the repository.

#### Attributes of `data/demo_0`

| Attribute | Value / meaning |
|---|---|
| `language_instruction` | English instruction specifying the red or blue cube |
| `layout` | `red_yminus_blue_yplus` or `blue_yminus_red_yplus` |
| `target_color` | `red` or `blue` |
| `dx_mm`, `dy_mm` | XY offsets from the baseline layout in millimeters; both cubes move together |
| `success` | Episode pass/fail status |
| `num_samples` | T, the teacher-segment length; excludes context length C in recovery data |
| `settle_physics_steps` | 90 in the published recovery collection script |
| `camera_warmup_renders` | 12 in that script |

### 6. Diagnostics and Reset State

`diagnostics` is stored separately from observations supplied to the learned model.

| Path | Shape / type | Contents |
|---|---|---|
| `diagnostics/target_post_pos` | `(T,3)`, floating point, m | Target cube position after the action, with the environment origin subtracted from world position |
| `diagnostics/other_post_pos` | `(T,3)`, floating point, m | Non-target cube position after the action, in the same coordinate representation |
| `diagnostics/joint_pos` | `(T,9)`, floating point | Post-action positions of seven arm joints (rad) and two finger joints (m) |
| `diagnostics/phase` | `(T,)`, `int8` | Teacher phase: 0=approach, 1=descend, 2=close, 3=lift |

Cube coordinates and `phase` support teacher generation, success checks, and auditing. They are not inputs to the final VLA model.

`data/demo_0/reset_state_before_settle` stores the simulator state before settling. It recursively saves a state dictionary containing fields such as the robot's `joint_position` and the cubes' `root_pose`. In recovery data, this is the state **before the 500-step replay begins**, not the state at the teacher handoff. Internal keys, shapes, and orientation conventions depend on the environment implementation; treat this state as input to `reset_to` in the same environment.

### 7. The 500-Step Recovery Context

The published audit records the recovery episode as follows:

| Segment | Steps | Storage | Training treatment |
|---|---|---|---|
| Observation history from the failed-trajectory replay | C=500 | `recovery_context/obs` | GRU state warm-up only; no loss or gradients |
| Actions used for replay | 500 | `recovery_context/replayed_actions` | Replay verification; not supervised labels |
| Corrective teacher actions | T=565 | `data/demo_0/obs` and `actions` | Supervised action training |

Conceptually, observations span 500+565=1,065 sequential steps, but there are only 565 action labels. Do not concatenate the 500 replayed actions with `actions` and train on all 1,065 as teacher labels.

The [training protocol](../result/evidence/training_protocol.json) reports 46 training episodes with 40,301 action labels and four validation episodes with 3,534 action labels. The 500 replay-context steps are excluded from these supervised sample counts. Validation uses XY offset `(0,+10)` mm; the recovery episode belongs to the training split.

### 8. Storage and Differences from Evaluation HDF5

The published recovery collection script makes the leading time axis extensible and uses one-step chunks for time-series datasets. Arrays written through `append()` use lossless gzip compression at level 9. `language_tokens`, `replayed_actions`, and reset-state arrays use separate `create_dataset()` calls, so not every dataset has the same compression settings. These settings also should not be assumed to apply to all 49 existing episode files.

HDF5 files from the [evaluation script](../code/evaluate_step5_color_replay_recovery32_v1.py) and [recording script](../code/record_step5_color_portfolio8_v1.py) differ as follows:

- They use `schema='color_policy_evaluation_v1'`, `teacher_only=false`, and `training_eligible=false`.
- Stored `obs` contains end-effector position, orientation, and finger positions, without per-step RGB images. Inference itself still uses images at every step.
- Actions are learned-model outputs, not teacher labels.
- Teacher `phase` and `recovery_context` are absent.
- Time-series arrays use gzip level 1. Images are stored separately as PNG files and, during recording, in MP4 videos.

Evaluation HDF5 files therefore cannot simply replace teacher HDF5 files for training. The older failed rollout referenced by recovery collection also does not necessarily follow the image-storage policy of this final evaluation script.

### 9. Example: Inspect an Actual File

Run this in an environment where you separately have the HDF5 file. The example does not assume that the file is included in this repository.

```python
import sys
import h5py

# Usage: python inspect_hdf5.py /path/to/episode.hdf5
with h5py.File(sys.argv[1], 'r') as f:
    def show(name, obj):
        label = '/' + name if name else '/'
        if isinstance(obj, h5py.Dataset):
            print(label, 'shape=', obj.shape, 'dtype=', obj.dtype,
                  'chunks=', obj.chunks, 'compression=', obj.compression,
                  'compression_opts=', obj.compression_opts)
        else:
            print(label, '(group)')
        for key, value in obj.attrs.items():
            print('  attr:', key, '=', value)

    show('', f)
    f.visititems(show)

    demo = f['data/demo_0']
    print('action steps:', len(demo['actions']))
    print('instruction:', demo.attrs['language_instruction'])
    if 'recovery_context' in f:
        print('context steps:', len(f['recovery_context/replayed_actions']))
```

This example displays the structure in read-only mode. It does not audit the data or determine training eligibility.

### 10. References

- [Recovery-data collection script](../code/collect_step5_color_replay_recovery_red1_v1.py)
- [HDF5 and observation–action consistency audit](../code/audit_step5_color_replay_recovery_v1.py)
- [Final training script](../code/train_step5_color_replay_recovery_v1.py)
- [Recovery-data audit results](../result/evidence/recovery_audit.json)
- [Final training protocol](../result/evidence/training_protocol.json)

The added details are based on public code and records reviewed on October 24, 2026.
