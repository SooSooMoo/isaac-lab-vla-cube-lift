# Isaac Lab: language-conditioned cube manipulation

**One shared model stack performs two instructions: move above a cube and pick it up.**

[日本語](README_JPN.md) · [Method](document/method.md) · [Reproduction](document/reproduction.md)

| Instruction | Reported result | Criterion | Steps |
|---|---|---|---:|
| `move above the cube` | Final error 20.887 mm; 50-step hold | Within 30 mm for 50 steps | 424 |
| `pick up the cube` | Final cube height 124.960 mm; maximum horizontal displacement 31.025 mm | Height ≥120.9996 mm for 10 steps; horizontal displacement ≤50 mm | 879 |

Each task reproduced all seven recorded trajectory arrays exactly in a repeat run. Recording also matched the verified trajectory. Heights are world coordinates, not lift distances.

The stack combines a base policy, learned visual residuals, a learned gripper correction and learned continuous XY attenuation. Both tasks use the same stack. No teacher actions or explicit step-triggered overrides are used at evaluation. Inputs include progress, so this is not evidence of purely physical-state-based understanding.

## Videos

The Pod updater imports `result/evidence/videos/above_trimmed.mp4` (9.90 s) and `pick_trimmed.mp4` (18.60 s), plus the original and full continuation versions. Upload presentation videos to a GitHub Release and add its URLs when publishing. Playback is real-time at 50 fps. Only the tail is removed; no frozen frames or substituted actions. The user approved the original movement and accepted the edited videos.
- [`above_trimmed.mp4`](result/evidence/videos/above_trimmed.mp4)：above Video
- [`pick_trimmed.mp4`](result/evidence/videos/pick_trimmed.mp4)：pick Video

## Scope

Fixed initial state and two exact instructions. Geometric criteria do not establish contact-force verification or generalization. Earlier 8/8 language development pairs are not a final-stack recheck.

With three seconds of continued inference, above drifts and pick drops the cube. Presentation clips end before these events; sustained post-success stability is not claimed. Full recordings are retained.

The previous snapshot is retained in an external backup; the history directory is reserved for future records. It is superseded by the September 18 numerical results and September 19 video edits. This task-specific policy is not a foundation-model VLA benchmark.

## Notes

Moved and backed up the large dataset (two_instruction.hdf5) and trained model checkpoint (shared_full_candidate.pt) to the backup directory (isaac-lab-vla-cube-lift_bk_data_and_model).
