# 05 成功軌道の再現性確認

このフォルダには、学習済み方策が実際に成功したときの行動列と、その完全リプレイ用プログラムを収録しています。

## 目的

閉ループ推論では、同じseedと初期位置を指定しても、GPU計算やシミュレーションの非決定性によって成功結果を完全には再現できませんでした。

そこで、成功時に方策が出力した行動列を保存し、同じ初期状態からその行動を順番に適用することで、成功軌道を再現します。

## ファイル

| ファイル | 内容 |
| --- | --- |
| `06_replay_success_trace.py` | 保存した行動列をIsaac Lab環境で再生するプログラム |
| `success_action_trace.npz` | 成功時の101ステップ・7次元行動列と初期キューブ位置 |
| `experiment_summary_JPN.md` | 実験結果の日本語要約 |
| `experiment_summary_ENG.md` | 実験結果の英語要約 |

## 保存されている成功条件

- Task: `IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor`
- Seed: `2058`
- 初期キューブ位置: `(0.42618394, 0.17994991, 0.055)`
- 行動列: `101 x 7`
- 成功step: `101`
- 初期位置誤差: `0.0 m`

## 実行方法

実行例は次のとおりです。各引数は1行で指定しても実行できます。

```bash
cd /workspace/IsaacLab-develop
./isaaclab.sh -p /workspace/step2/isaac-lab-vla-cube-lift_1/05_reproduction/06_replay_success_trace.py --task IsaacContrib-Lift-Cube-Franka-IK-Rel-Visuomotor --trace /workspace/step2/isaac-lab-vla-cube-lift_1/05_reproduction/success_action_trace.npz --seed 2058 --device cuda:0 --viz kit --kit_args "--enable omni.replicator.core"
```

## 結果の解釈

完全リプレイ成功は、学習済み方策が毎回安定して成功することを意味しません。

この結果が示すのは、学習済み方策が閉ループ推論中に少なくとも1本の成功行動列を生成し、その行動列を同一の初期位置から再現できたということです。

関連動画:

- `../06_videos/policy_success.mp4`
- `../06_videos/exact_replay_success.mp4`