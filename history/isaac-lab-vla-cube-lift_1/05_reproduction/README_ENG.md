# 05 Successful Trajectory Reproduction

This directory contains the saved action sequence from a successful closed-loop rollout and the script used to replay it in Isaac Lab.

## Files

| File | Description |
| --- | --- |
| `06_replay_success_trace.py` | Replays the recorded actions |
| `success_action_trace.npz` | Stores 101 actions and the initial cube position |
| `experiment_summary_JPN.md` | Japanese experiment summary |
| `experiment_summary_ENG.md` | English experiment summary |

## Recorded Conditions

- Task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`
- Seed: `2058`
- Initial cube position: `(0.42618394, 0.17994991, 0.055)`
- Action shape: `101 x 7`
- Replay success step: `101`
- Initial position error: `0.0 m`

## Reproduction Command

```bash
cd /workspace/IsaacLab-develop
./isaaclab.sh -p /workspace/step2/isaac-lab-vla-cube-lift_1/05_reproduction/06_replay_success_trace.py --task IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor --trace /workspace/step2/isaac-lab-vla-cube-lift_1/05_reproduction/success_action_trace.npz --seed 2058 --device cuda:0 --viz kit --kit_args "--enable omni.replicator.core"
```

## Interpretation

Exact replay success does not mean that the trained policy is consistently successful in closed-loop inference. It shows that the policy generated at least one successful trajectory and that the saved actions reproduce it from the recorded initial cube position.

Related videos:

- `../06_videos/policy_success.mp4`
- `../06_videos/exact_replay_success.mp4`