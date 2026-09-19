# 04. 評価

このフォルダには、学習済みRobomimic方策の閉ループ評価、動画記録、オフライン教師強制診断に使用したコードと結果概要を収録しています。

## ファイル

| ファイル | 内容 |
|---|---|
| `04_evaluate_policy.py` | Isaac Lab環境で学習済み方策を閉ループ評価し、動画、Action trace、CSVを保存します |
| `05_offline_diagnostic.py` | 記録済み教師Actionとモデル予測を比較します |
| `evaluation_summary.csv` | 主要な評価結果を機械可読形式で記録します |

## 閉ループ評価

閉ループ評価では、各stepで現在のカメラ画像とロボット状態をモデルへ入力し、予測されたActionをIsaac Lab環境へ適用します。

```text
現在のObservation
        ↓
Robomimic BC-RNN-GMM
        ↓
予測Action
        ↓
Isaac Lab環境
        ↓
次のObservation
```

### 実行例

```bash
cd /workspace/IsaacLab-develop

./isaaclab.sh -p   /workspace/step2/isaac-lab-vla-cube-lift_1/04_evaluation/04_evaluate_policy.py   --task IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor   --checkpoint /workspace/IsaacLab-develop/logs/robomimic/IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor/bc_rnn_image_franka_lift/20260829133122/models/model_epoch_200.pth   --horizon 250   --num_rollouts 10   --seed 2011   --device cuda:0   --viz kit   --kit_args "--enable omni.replicator.core"
```

主な引数：

| 引数 | 内容 |
|---|---|
| `--checkpoint` | 評価するRobomimic checkpoint |
| `--horizon` | 1 rolloutの最大step数 |
| `--num_rollouts` | rollout回数 |
| `--seed` | 最初の乱数seed |
| `--fixed_trial_seed` | 全trialで同じseedを使用 |
| `--stop_on_success` | 最初の成功後に評価を終了 |

## オフライン教師強制診断

オフライン診断では、HDF5に記録されたObservationを順番にモデルへ入力し、モデル予測と教師Actionを比較します。Isaac Lab環境へActionは適用しません。

```bash
cd /workspace/IsaacLab-develop

./isaaclab.sh -p   /workspace/step2/isaac-lab-vla-cube-lift_1/04_evaluation/05_offline_diagnostic.py
```

診断対象は50デモのうち先頭10デモです。

## 評価結果

| 評価 | 成功数 | 試行数 | 成功率 |
|---|---:|---:|---:|
| 初期rollout | 1 | 10 | 10.0% |
| seed探索 | 1 | 48 | 2.08% |

成功が確認された条件：

- Seed: 2058
- 初期キューブ位置: `(0.426184, 0.179950, 0.055000)`
- seed探索時の成功step: 156

オフライン診断：

- Overall MAE: 0.1127
- Gripper accuracy: 97.99%

## 結果の解釈

オフライン教師強制診断では教師Actionとの相関が確認されましたが、閉ループ成功率は低い結果でした。

Behavior Cloningでは、小さな予測誤差によって学習データにない状態へ移動すると、その後の誤差が蓄積することがあります。

seed 2058だけでは成功を毎回再現できませんでした。成功Action列の再現は `05_reproduction` に記録しています。

## 言語入力について

HDF5には固定指示 `pick up the cube` がメタデータとして保存されていますが、評価対象モデルは言語を入力として使用していません。
