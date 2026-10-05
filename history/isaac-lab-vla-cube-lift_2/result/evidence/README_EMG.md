# Verification Evidence

[日本語](README_JPN.md)

[Reported results](reported_results.json) transcribe the Pod output shared by the user. Original evaluation JSON files are imported into `verification/` and used to check the reported values.

| Item | Result | Criterion |
|---|---:|---:|
| Final above target error | 20.887 mm | At most 30 mm |
| above hold duration | 50 steps | 50 steps |
| Final pick cube height | 124.960 mm | At least 120.9996 mm |
| Consecutive steps meeting the pick height criterion | 10 steps | 10 steps |
| Maximum horizontal displacement for pick | 31.025 mm | At most 50 mm |

Repeat runs reproduced all seven recorded arrays exactly for both tasks. Pod reports also confirm that recording runs matched the verified trajectories. The user reviewed video quality and motion. Language-pair reevaluation of the final complete model stack and contact-force verification have not been performed.

`import_manifest.json` records hashes, sizes, and video frame counts of artifacts at import time.
