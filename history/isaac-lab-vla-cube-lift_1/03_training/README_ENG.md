# 03. Training

This folder contains the Robomimic BC-RNN-GMM configuration, reproducible training script, and model publication information.

## Files

| File | Description |
|---|---|
| `bc_rnn_image_lift.json` | BC-RNN-GMM training configuration |
| `03_train_robomimic.sh` | Reproduces the 200-epoch training run |
| `models/` | Stores checkpoint publication information |

## Training Configuration

- Algorithm: Behavior Cloning
- Policy: BC-RNN-GMM
- Demonstrations: 50 successful episodes
- Epochs: 200
- Dataset: Robomimic HDF5
- Task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`
- Checkpoint: `model_epoch_200.pth`

## Usage

```bash
cd /workspace/step2/isaac-lab-vla-cube-lift_1/03_training
./03_train_robomimic.sh
```

Paths and epoch count can be overridden with environment variables.

```bash
DATASET_PATH=/path/to/dataset.hdf5 EPOCHS=200 ./03_train_robomimic.sh
```

## Output

By default, logs and checkpoints are stored under:

```text
/workspace/IsaacLab-develop/logs/robomimic/
```

The evaluated checkpoint is `model_epoch_200.pth`, produced after 200 training epochs.

## Language Input

The fixed instruction `pick up the cube` is stored as HDF5 metadata, but language is not included in the current BC-RNN observations. This training run is therefore visuomotor behavior cloning from images and robot state.
