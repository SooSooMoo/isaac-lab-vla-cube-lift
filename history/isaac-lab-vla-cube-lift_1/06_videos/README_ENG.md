# 06 Videos

This directory contains videos and an image documenting the trained policy evaluation and exact replay of the successful trajectory.

## Files

| File | Description |
| --- | --- |
| `policy_success.mp4` | Genuine closed-loop cube grasp and lift by the trained BC-RNN-GMM policy |
| `policy_failure.mp4` | Representative closed-loop grasp failure |
| `exact_replay_success.mp4` | Exact replay of the saved 101-step successful action sequence |
| `franka_cube_lift_success.png` | Success frame displayed near the top of the main README files |

## Important Distinction

In `policy_success.mp4`, the trained policy observes camera images and robot state and produces actions during simulation. This is a closed-loop policy success.

In `exact_replay_success.mp4`, the action sequence saved during that success is replayed from the recorded initial cube position. The model does not generate a new action at every step in this video.

## Evaluation Results

- Initial evaluation: 1/10 successful rollouts
- Recorded seed search: 1/48 successful rollouts
- Successful seed: 2058
- Exact replay: 1/1 successful

The closed-loop success rate remains low. Both the success and representative failure videos are included to present the achieved result and its limitations transparently.

## GitHub Display

GitHub may not always embed MP4 playback directly in a README. The main README files link to each video and display the success frame as a PNG image.