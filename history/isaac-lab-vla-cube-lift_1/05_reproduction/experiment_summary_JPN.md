# 成功再現結果

## 閉ループ推論による成功

- Checkpoint: `model_epoch_200.pth`
- Policy: Robomimic BC-RNN-GMM
- Seed: 2058
- 初期キューブ位置: `(0.42618394, 0.17994991, 0.055)`
- seed探索時の成功step: 156
- 成功動画: `../06_videos/policy_success.mp4`

## 同一seedの再実行

seed 2058を指定しただけでは、成功を毎回再現できませんでした。

同一プロセス内で繰り返した結果、trial 6で成功軌道が得られました。これは、seed以外にも観測、RNN内部状態、レンダリング、シミュレーション初期化などの影響が残っている可能性を示しています。

## 成功行動列の完全リプレイ

- 保存行動shape: `(101, 7)`
- 初期位置誤差: `0.0`
- Replay success: `True`
- Replay success step: `101`
- 行動列: `success_action_trace.npz`
- 動画: `../06_videos/exact_replay_success.mp4`

同じ初期キューブ位置と完全に同じ行動列を使用すると、成功軌道を再現できました。

この結果は物理軌道の再現性を示しますが、seedだけによる学習済み方策の決定論的な成功を示すものではありません。

## 評価上の注意

- `policy_success.mp4`は、学習済み方策の閉ループ推論で実際に得られた成功です。
- `exact_replay_success.mp4`は、その成功行動列を保存して再生した結果です。
- 両者を同じ種類の推論成功として集計しません。
