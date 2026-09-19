#!/usr/bin/env bash
set -euo pipefail

ISAAC_LAB_ROOT="${ISAAC_LAB_ROOT:-/workspace/IsaacLab-develop}"
DATASET_PATH="${DATASET_PATH:-/workspace/step2/datasets/robomimic_lift_50/lift_robomimic.hdf5}"
TASK_NAME="${TASK_NAME:-IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bc_rnn_image_franka_lift}"
EPOCHS="${EPOCHS:-200}"
LOG_DIR="${LOG_DIR:-robomimic}"

if [[ ! -d "${ISAAC_LAB_ROOT}" ]]; then
    echo "[ERROR] Isaac Lab directory not found: ${ISAAC_LAB_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "[ERROR] Dataset not found: ${DATASET_PATH}" >&2
    exit 1
fi

cd "${ISAAC_LAB_ROOT}"

./isaaclab.sh -p     scripts/imitation_learning/robomimic/train.py     --task "${TASK_NAME}"     --dataset "${DATASET_PATH}"     --name "${EXPERIMENT_NAME}"     --epochs "${EPOCHS}"     --log_dir "${LOG_DIR}"
