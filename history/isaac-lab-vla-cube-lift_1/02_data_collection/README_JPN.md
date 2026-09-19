# 02. データ収集

このフォルダには、Isaac LabのState Machineを教師として、Franka Cube Liftの成功デモを収集するためのコードとデータ情報を収録します。

## ファイル構成

| パス | 内容 |
|---|---|
| `02_collect_demonstrations.py` | State Machineを使用して成功デモを収雀します |
| `data/metadata/` | データセット件数、shape、収集条件などのメタデータを保存します |
| `data/samples/` | GitHubで構造を確認するための小さなサンプルデータを保存します |

## 収集方法

Franka PandaのPick and Lift State Machineを教師方策として使用します。

各試行ではキューブの初期位置をランダム化し、次の動作を実行します。

1. キューブ上方へ移動
2. キューブへ接近
3. グリッパーを閉じる
4. キューブを把持する
5. 目標位置まで持ち上げる

成功判定を満たした試行だけをHDF5データセットへ保存します。

## データセット

- Format: Robomimic HDF5
- Demonstrations: 50 successful episodes
- Expert: Isaac Lab State Machine
- Robot: Franka Panda
- Controller: IK-Rel
- Task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`

## Observation

| Key | 内容 |
|---|---|
| `table_cam` | 卓上RGBカメラ画像、200 × 200 × 3 |
| `wrist_cam` | 手首RGBカメラ画像、200 × 200 × 3 |
| `eef_pos` | エンドエフェクタ位置 |
| `eef_quat` | エンドエフェクタ姿勢 |
| `gripper_pos` | グリッパー関節状態 |

## Action

教師Actionは7次元です。

```text
[x, y, z, rx, ry, rz, gripper]
```

先頭6次元は相対IK指令、最後の1次元はグリッパー指令です。

## 言語データについて

Robomimic HDF5のデータセット属性には、次の固定言語指示がメタデータとして保存されています。

```text
language_conditioning: fixed_instruction
language_instruction: pick up the cube
```

ただし、この文章は各stepの`obs`には含まれておらず、現在のRobomimic BC-RNNの入力としても使用されていません。

```text
HDF5メタデータに固定指示を保存: yes
モデルへ言語指示を入力: no
言語による行動選択: no
```

すべてのデモが同じCube Liftタスクであるため、「キューブを持ち上げる」という目的はデータセット全体に暗黙的に固定されています。

言語条件付きVLAへ発展させるには、複数の指示と、指示によって異なる対象または行動を含むデータが必要です。

## 公開方針

完全版HDF5はファイルサイズを確認し、GitHub Releasesまたは外部ストレージで公開します。

`data/samples/`には、データ構造を確認するための小さなサンプルだけを配置します。完全版データセットと公開用サンプルを混同しないように管理します。
