# 01. 環境

このフォルダには、本プロジェクトで使用したIsaac Lab環境の定義、環境検証スクリプト、RunPodの依存関係情報を収録しています。

## ファイル

| ファイル | 内容 |
|---|---|
| `01_validate_environment.py` | 環境生成、Observation、画像データを検証します |
| `ik_rel_visuomotor_env_cfg.py` | Franka Cube Lift IK-Rel Visuomotor環境を定義します |
| `requirements/requirements-runpod-snapshot.txt` | RunPodに導入されたPythonパッケージのスナップショットです |

## 使用環境

- GPU: NVIDIA GeForce RTX 4090
- Isaac Sim: 6.0.1 RC
- Isaac Lab: develop branch
- Robot: Franka Panda
- Controller: Differential IK, relative pose
- Control frequency: 50 Hz
- Task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`

## Isaac SimとIsaac Labの接続

Isaac Simは、物理シミュレーション、レンダリング、ロボット、センサーを提供します。

Isaac LabはIsaac Sim上で動作し、タスク、環境、コントローラ、Observation、データ収集、学習処理を提供します。

本環境では、次のシンボリックリンクでIsaac SimをIsaac Labへ接続しています。

```text
/workspace/IsaacLab-develop/_isaac_sim -> /isaac-sim
```

## 環境検証

Isaac Labリポジトリから検証スクリプトを実行します。

```bash
cd /workspace/IsaacLab-develop

./isaaclab.sh -p \
  /workspace/step2/isaac-lab-vla-cube-lift_1/01_environment/01_validate_environment.py \
  --device cuda:0 \
  --viz none \
  --kit_args "--enable omni.replicator.core"
```

## 検証対象

期待するObservation keyは次の5つです。

```text
eef_pos
eef_quat
gripper_pos
table_cam
wrist_cam
```

RGB画像の期待shapeは次のとおりです。

```text
(num_envs, 200, 200, 3)
```

正常終了時には、次の結果が表示されます。

```text
[RESULT] Lift visuomotor observation validation OK
```

## VLAとの関係

この環境は、計画しているVLAシステムのVisionとActionの基盤です。

- Vision: 卓上カメラと手首カメラ
- Action: 6DoF IK-Relとグリッパー制御
- Language: 未実装

現在、言語指示はObservationに含まれていません。
