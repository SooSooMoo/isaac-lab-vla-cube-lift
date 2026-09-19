"""Compare Robomimic predictions against recorded teacher actions."""

from pathlib import Path

import h5py
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


CHECKPOINT = Path(
    "/workspace/IsaacLab-develop/logs/robomimic/"
    "IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor/"
    "bc_rnn_image_franka_lift/20260829133122/models/"
    "model_epoch_200.pth"
)

DATASET = Path(
    "/workspace/step2/datasets/"
    "robomimic_lift_50/lift_robomimic.hdf5"
)

NUM_DEMOS = 10

device = TorchUtils.get_torch_device(try_to_use_cuda=True)
policy, _ = FileUtils.policy_from_checkpoint(
    ckpt_path=str(CHECKPOINT),
    device=device,
)

all_targets = []
all_predictions = []

with h5py.File(DATASET, "r") as f:
    data = f["data"]
    demo_names = sorted(
        (name for name in data if name.startswith("demo_")),
        key=lambda name: int(name.split("_")[-1]),
    )[:NUM_DEMOS]

    for demo_name in demo_names:
        demo = data[demo_name]
        recorded_obs = demo["obs"]
        targets = demo["actions"][:]

        policy.start_episode()
        predictions = []

        for step in range(len(targets)):
            obs = {}

            for name in ("eef_pos", "eef_quat", "gripper_pos"):
                value = recorded_obs[name][step]
                obs[name] = torch.from_numpy(value).float().to(device)

            for name in ("table_cam", "wrist_cam"):
                image = recorded_obs[name][step]
                image = torch.from_numpy(image).to(device)
                image = image[..., :3].permute(2, 0, 1).float() / 255.0
                obs[name] = image.clip(0.0, 1.0)

            prediction = policy(obs)
            predictions.append(np.asarray(prediction, dtype=np.float32))

        predictions = np.stack(predictions)

        all_targets.append(targets.astype(np.float32))
        all_predictions.append(predictions)

        demo_mae = float(np.abs(predictions - targets).mean())
        print(
            f"{demo_name}: samples={len(targets)} "
            f"MAE={demo_mae:.6f}"
        )

targets = np.concatenate(all_targets, axis=0)
predictions = np.concatenate(all_predictions, axis=0)

error = predictions - targets
mae_per_dim = np.abs(error).mean(axis=0)
rmse_per_dim = np.sqrt(np.square(error).mean(axis=0))

target_mean_abs = np.abs(targets).mean(axis=0)
prediction_mean_abs = np.abs(predictions).mean(axis=0)

gripper_accuracy = np.mean(
    np.sign(predictions[:, 6]) == np.sign(targets[:, 6])
)

correlations = []
for index in range(7):
    if targets[:, index].std() < 1.0e-8:
        correlations.append(float("nan"))
    else:
        correlations.append(
            float(np.corrcoef(
                targets[:, index],
                predictions[:, index],
            )[0, 1])
        )

np.set_printoptions(precision=4, suppress=True)

print("\n=== offline teacher-forced diagnostic ===")
print("samples:", len(targets))
print("target mean abs:    ", target_mean_abs)
print("prediction mean abs:", prediction_mean_abs)
print("MAE per dimension:  ", mae_per_dim)
print("RMSE per dimension: ", rmse_per_dim)
print("correlation:        ", np.asarray(correlations))
print("gripper accuracy:   ", float(gripper_accuracy))
print("overall MAE:        ", float(np.abs(error).mean()))

assert np.isfinite(predictions).all()
print("[RESULT] Offline Robomimic diagnostic completed")
