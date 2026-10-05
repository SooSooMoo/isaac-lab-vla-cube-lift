# Verification evidence

[日本語](README_JPN.md)

One model passed 26 existing locations, five new random locations, and four supplementary locations: 35 distinct positions, one trial each. This is not a full-region guarantee or a statistical 100% success rate. Three recording runs are separate and do not add three new positions.

[26-location JSON](validation26/results.json) · [New five](random5/results.json) · [Supplementary four](positive4/results.json) · [CSV](positions35.csv)

| Group | X (mm) | Y (mm) | Pick / hold | Min height (mm) | Max XY (mm) |
|---|---:|---:|---|---:|---:|
| validation26 | 0 | 0 | PASS / PASS | 128.36 | 7.19 |
| validation26 | 10 | 0 | PASS / PASS | 128.46 | 6.74 |
| validation26 | -10 | 0 | PASS / PASS | 128.10 | 7.12 |
| validation26 | 0 | -10 | PASS / PASS | 128.46 | 6.86 |
| validation26 | 0 | 10 | PASS / PASS | 128.12 | 5.91 |
| validation26 | -20 | 0 | PASS / PASS | 128.12 | 7.86 |
| validation26 | 20 | 0 | PASS / PASS | 127.92 | 6.57 |
| validation26 | 0 | -20 | PASS / PASS | 128.55 | 7.38 |
| validation26 | 0 | 20 | PASS / PASS | 128.31 | 6.90 |
| validation26 | -20 | -20 | PASS / PASS | 128.36 | 9.28 |
| validation26 | -20 | 20 | PASS / PASS | 128.71 | 7.01 |
| validation26 | 20 | -20 | PASS / PASS | 128.11 | 7.56 |
| validation26 | 20 | 20 | PASS / PASS | 128.10 | 8.33 |
| validation26 | -8.927 | -8.147 | PASS / PASS | 128.27 | 6.91 |
| validation26 | -13.82 | 2.136 | PASS / PASS | 128.72 | 7.34 |
| validation26 | -19.123 | -15.53 | PASS / PASS | 128.83 | 8.96 |
| validation26 | -8.512 | -8.622 | PASS / PASS | 128.15 | 6.69 |
| validation26 | 15.344 | 5.632 | PASS / PASS | 127.94 | 6.95 |
| validation26 | -30 | 0 | PASS / PASS | 128.85 | 9.33 |
| validation26 | 30 | 0 | PASS / PASS | 128.16 | 7.35 |
| validation26 | 0 | -30 | PASS / PASS | 128.44 | 7.99 |
| validation26 | 0 | 30 | PASS / PASS | 128.32 | 7.44 |
| validation26 | -30 | -30 | PASS / PASS | 128.48 | 10.45 |
| validation26 | -30 | 30 | PASS / PASS | 128.82 | 9.82 |
| validation26 | 30 | -30 | PASS / PASS | 128.30 | 7.08 |
| validation26 | 30 | 30 | PASS / PASS | 128.14 | 10.25 |
| random5 | -2.12 | 15.26 | PASS / PASS | 128.41 | 6.46 |
| random5 | -21.914 | -6.215 | PASS / PASS | 128.60 | 7.78 |
| random5 | -24.981 | -19.628 | PASS / PASS | 128.44 | 9.75 |
| random5 | -4.286 | 24.037 | PASS / PASS | 128.62 | 6.95 |
| random5 | -18.254 | 4.408 | PASS / PASS | 128.55 | 8.07 |
| positive4 | 14.972 | -23.836 | PASS / PASS | 128.31 | 8.01 |
| positive4 | 12.324 | 0.347 | PASS / PASS | 128.30 | 6.61 |
| positive4 | 18.082 | -2.793 | PASS / PASS | 128.24 | 6.79 |
| positive4 | 15.143 | 19.982 | PASS / PASS | 127.87 | 7.65 |

Height is an environment coordinate; XY displacement is relative to the settled Cube. Pass includes pick, three-second hold, finite values and joint margins. Contact forces and exact replay across all 35 positions are not established.
