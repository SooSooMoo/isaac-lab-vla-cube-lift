# Isaac Lab : VLA : language-conditioned cube manipulation

**One shared model stack performs two instructions: move above a cube and pick it up.**

<table>
  <tr>
    <td align="center">
      <img src="result/evidence/videos/above.png" width="100%">
      <br><b> above </b>
    </td>
    <td align="center">
      <img src="result/evidence/videos/pick.png" width="100%">
      <br><b> pick </b>
    </td>
  </tr>
</table>

[日本語](README_JPN.md) · [Method](document/README_EMG.md#2-method-and-model-stack) · [Reproduction](document/README_EMG.md#6-reproduction-and-verification-order)

| Instruction | Reported result | Criterion | Steps |
|---|---|---|---:|
| `move above the cube` | Final error 20.887 mm; 50-step hold | Within 30 mm for 50 steps | 424 |
| `pick up the cube` | Final cube height 124.960 mm; maximum horizontal displacement 31.025 mm | Height ≥120.9996 mm for 10 steps; horizontal displacement ≤50 mm | 879 |

Each task reproduced all seven recorded trajectory arrays exactly in a repeat run. Recording also matched the verified trajectory. Heights are world coordinates, not lift distances.

The stack combines a base policy, learned visual residuals, a learned gripper correction and learned continuous XY attenuation. Both tasks use the same stack. No teacher actions or explicit step-triggered overrides are used at evaluation. Inputs include progress, so this is not evidence of purely physical-state-based understanding.

## Videos

- [`above_trimmed.mp4`](result/evidence/videos/above_trimmed.mp4)：above Video
- [`pick_trimmed.mp4`](result/evidence/videos/pick_trimmed.mp4)：pick Video
Playback is real-time at 50 fps. Only the tail is removed; no frozen frames or substituted actions. The user approved the original movement and accepted the edited videos.

## Scope

Fixed initial state and two exact instructions. Geometric criteria do not establish contact-force verification or generalization. Earlier 8/8 language development pairs are not a final-stack recheck.

With three seconds of continued inference, above drifts and pick drops the cube. Presentation clips end before these events; sustained post-success stability is not claimed. Full recordings are retained.

The previous snapshot is retained in an external backup; the history directory contains the earlier Version 1 project. It is superseded by the September 18 numerical results and September 19 video edits. This task-specific policy is not a foundation-model VLA benchmark.

## Folders

- code/: Model, inference, training, evaluation, video processing
- datasets/: Observation data, preservation anchors, language targets, metadata, and dataset examples
- result/: Model identification details, validation records, videos
- document/: Methodology, environment, reproduction steps, update history
- history/: [Earlier Version 1 project](history/isaac-lab-vla-cube-lift_1/README_ENG.md)

## Notes

Moved and backed up the large dataset (two_instruction.hdf5) and trained model checkpoint (shared_full_candidate.pt) to the backup directory (isaac-lab-vla-cube-lift_bk_data_and_model).

<!-- portfolio-next-steps:start -->
## Next Steps

The following development milestones have not yet been achieved. They are separate from the verified results reported above and will be evaluated one stage at a time.

| Stage | Next action | What to verify |
|---|---|---|
| 1 | Pick up the cube from a fixed initial position and hold it lifted for 3 seconds | No drop or large horizontal movement after reaching the lift target |
| 2 | Pick up the cube with a randomized initial position | Approach and lift the cube at positions not used for training |
| 3 | Place two differently colored cubes together and pick up the color specified by the instruction | Select the intended object from an instruction such as "pick up the red cube" |

In Stage 3, swap the cubes' positions as well, to check that selection follows color and language rather than a fixed location.

Limit the conditions changed at each stage, and record trial counts, success rates, drops, and horizontal displacement. Define evaluation conditions before running the trials. In Stages 2 and 3, also check that the same model stack retains the existing `move above the cube` behavior.
<!-- portfolio-next-steps:end -->
