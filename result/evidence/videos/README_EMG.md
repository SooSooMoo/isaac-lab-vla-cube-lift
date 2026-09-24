# Demonstration Videos

[日本語](README_JPN.md)

| Version | above | pick |
|---|---:|---:|
| original (up to success) | 424 frames / 8.48 s | 879 frames / 17.58 s |
| continued3s (continued inference) | 574 frames / 11.48 s | 1029 frames / 20.58 s |
| trimmed (for presentation) | 495 frames / 9.90 s | 930 frames / 18.60 s |

File names include `above_trimmed.mp4`. Videos play in real time at 50 fps, with the table camera on the left and the wrist camera on the right. Each frame shows the pre-action observation.

Trimmed versions retain a continuous prefix of the extended recording: approximately 1.4 seconds after success for above and 1.0 second for pick. No freeze frames or different actions are inserted. Full extended recordings include position drift for above and a dropped cube for pick. Trimming does not improve sustained holding by the controller.

Videos are placed by the Pod import process. `.gitignore` excludes videos from ordinary additions, so available videos can differ between the Pod and GitHub. Upload presentation videos separately and add links to their published locations.
