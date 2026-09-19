# Environment

Use the successful old Pod and `/workspace/IsaacLab-develop/isaaclab.sh`, with `/isaac-sim/python.sh` (Python 3.12 in reported traces). RTX4090; current NGX/driver library family 580.65.06. Exact package versions and Lab revision must be imported from the preservation environment JSON; do not infer them from the Docker image label.

Required custom task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`. The evaluator checks the installed pose function against xyzw. A different Pod using wxyz and lacking this task is not interchangeable.

NGX path: `/workspace/step3/ngx_runtime_580.65.06_cgbe5u2a/lib`. Evaluation prepends it to LD_LIBRARY_PATH for that process. After the NGX repair, both initial camera inputs matched saved observations exactly. Do not publish NVIDIA binary libraries to Git without reviewing redistribution terms; they are kept in the private preservation archive.

The guarded runtime expects cuDNN benchmark=True initially and switches it to False. Other expected defaults are deterministic algorithms=False, cuDNN deterministic=False, matmul TF32=False, cuDNN TF32=True, CUBLAS_WORKSPACE_CONFIG absent. Failures stop evaluation rather than silently changing the protocol.

This repository does not install drivers or replace the environment. Existing Isaac Sim/Kit, custom Lab runtime, Python packages and external USD assets are required. A clean-machine rebuild has not been verified.

## September 18–19 record

Reported training runtime: Python 3.12, PyTorch 2.11.0+cu128, RTX 4090. Video encoder: imageio-ffmpeg bundled ffmpeg. The newer Pod reproduced the final trajectories; exact environment dependencies remain required.
