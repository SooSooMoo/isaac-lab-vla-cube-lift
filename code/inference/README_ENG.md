# Inference and evaluation

**Model weights are not published.** This repository provides implementation, evaluation evidence, and videos. Inference requires a separate checkpoint and cannot run from the public files alone.

Inference is based on the evaluated worker, with paths and descriptions adapted for publication. The three model modules are byte-identical to the originals. The publication launcher is syntax-checked; a simulation using the reorganized code has not been run.

An existing Isaac Lab installation must register the custom task `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`. The task configuration and initial-state HDF5 are not included. Additional setup is required on another machine.
