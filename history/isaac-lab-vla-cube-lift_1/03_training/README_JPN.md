# 03. 学習

このフォルダには、Robomimic BC-RNN-GMMの学習設定、再現用スクリプト、モデル公開情報を収録しています。

## ファイル

| ファイル | 内容 |
|---|---|
| `bc_rnn_image_lift.json` | BC-RNN-GMMの学習設定 |
| `03_train_robomimic.sh` | 200 epoch学習を再現するスクリプト |
| `models/` | checkpointの公開情報と配置先 |

## 学習条件

- Algorithm: Behavior Cloning
- Policy: BC-RNN-GMM
- Demonstrations: 50 successful episodes
- Epochs: 200
- Dataset: Robomimic HDF5
- Task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`
- Checkpoint: `model_epoch_200.pth`

## 実行方法

```bash
cd /workspace/step2/isaac-lab-vla-cube-lift_1/03_training
./03_train_robomimic.sh
```

環境変数でデータセットのパスやepoch数を変更できます。

```bash
DATASET_PATH=/path/to/dataset.hdf5 \
EPOCHS=200 \
./03_train_robomimic.sh
```

## 出力

既定では、Isaac Labリポジトリの次の場所へログとcheckpointが保存されます。

```text
/workspace/IsaacLab-develop/logs/robomimic/
```

今回評価したcheckpointは、200 epoch学習後の `model_epoch_200.pth` です。

checkpointの元の絶対パスは、次のファイルに記録しています。

```text
models/checkpoint_path.txt
```

checkpoint本体はポートフォリオへコピーせず、元の場所に保存します。

## 言語入力について

HDF5には固定指示 `pick up the cube` がメタデータとして保存されています。

ただし、現在のBC-RNNのObservationには言語が含まれていません。そのため、この学習は画像とロボット状態から行動を模倣するVisuomotor Behavior Cloning(視覚運動模倣学習)です。
