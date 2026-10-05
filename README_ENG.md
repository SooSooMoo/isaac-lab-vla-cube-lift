# Franka Cube Pick with Learned Visual Localization and Staged Control

[日本語](README_JPN.md)

## Goal and results
One visual position estimator and one staged controller passed pick and three-second hold checks at 35 distinct Cube placements within XY offsets of ±30 mm from the baseline placement. Each location contributed one trial to this aggregate; this is not a guarantee for the entire region or a general 100% success rate.

## Architecture
Table and wrist camera images and robot state feed a learned Cube-relative position estimator. Estimated environment coordinates are smoothed with an EMA coefficient of 0.2. Explicit approach, descend, close, lift, and hold stages drive the robot. Ground-truth Cube position is used for scoring, not as a control input during visual evaluation.

This result concerns learned perception plus staged control, not an end-to-end learned action policy or newly verified language instruction switching.

## Evaluation
- Previously evaluated locations: 26/26
- New random locations within ±30 mm: 5/5
- Additional stratified positive-X locations: 4/4
- Lowest hold height across the 35 trials: 127.87 mm
- Largest horizontal displacement: 10.45 mm
- Checks include lifting at least 100 mm above settled height, maintaining the required height for three seconds after success, and horizontal displacement within 50 mm. Contact-force validation is not included.

## Lessons
Downward bias in estimated Cube height prevented closure. Aligned observations from a failed rollout helped repair the ±20 mm case. For a remaining ±30 mm case, adding only a failed trajectory degraded approach behavior. Ground-truth control succeeded there; adding its complete successful approach-to-hold trajectory produced the model that passed the 35-location evaluation.

## Demonstration videos

Original pre-action table (left) and wrist (right) inputs at 50 fps, real-time playback. No trimming, speed-up, or action replacement. All three recording trials passed pick and three-second hold checks and were visually approved by the user.

- [Center: 17.78 s](result/evidence/videos/position_01.mp4)
- [X−30, Y+30 mm: 18.20 s](result/evidence/videos/position_02.mp4)
- [X+30, Y−30 mm: 18.46 s](result/evidence/videos/position_03.mp4)

[35-location results](result/evidence/README.md) · [Methods and reproduction](document/README_ENG.md) · [Previous above/pick portfolio](history/isaac-lab-vla-cube-lift_2/README_ENG.md)


## Limitations and future work
Repeated-trial success rates, full-region coverage, and different initial poses or camera conditions remain unverified. Expansion to ±40 or ±50 mm is undecided. Color recognition is a separate project.
