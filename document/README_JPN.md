# 方法と再現条件

[English](README_ENG.md)


## 構成
凍結ResNet18のlayer2までの特徴を各カメラで5×5へ集約し、ロボット状態と合わせて6409次元を入力します。256→64→3のGELUヘッドでCube−EEFの相対位置を推定します。学習は位置推定のみです。

制御は接近、下降、閉鎖、持上げ、維持。目標誤差10mm以内かつ姿勢誤差0.1rad以内を5ステップ連続で満たすと次段階に進み、閉鎖20ステップ後に持ち上げます。成功後150ステップ、dt=0.02秒を継続します。姿勢や段階の条件は保存コードを正とします。

## 判定と範囲
Cube静定高さ＋100mm、閉鎖指令、手先とCube間80mm以内を10ステップ連続で満たした後、3秒の高さ・水平変位を確認します。水平変位は静定位置から50mm以内。関節余裕、有限値も確認します。真値Cubeは採点用です。動画は動作前画像なので最終動作後の状態は数値ログで確認します。

XYオフセットは初期Cube中心約(0.573153, 0.039928, 0.055)mからの移動量です。高さの数値は持上げ量ではなく環境座標です。初期姿勢・照明・Cubeの色などは固定です。aboveは今回再評価していません。

## 保存コードと再現
`code/snapshots/`は実験時の実行コードです。RunPod上の`/workspace/step4`にあるモデル、初期状態HDF5、smoke_utils、Isaac Labソース、ローカル資産などへの絶対パスが残っています。GitHubの取得だけで別環境へ完全再現できるパッケージではありません。

現環境での再評価は、保存コードに記載された依存ファイルを確認し、`/isaac-sim/python.sh -u /workspace/step4/check_visual_validation26.py`を実行します。続いてランダム5地点とX正側4地点のスクリプトを使用します。各候補・元データの識別情報は`result/training/protocol.json`、評価設定は各`protocol.json`に保存されています。

初期状態データや学習画像、NVIDIAの資産はこの公開パッケージには同梱していません。新環境への完全再現手順は未検証です。モデルと動画はSHA-256照合後に公開済みです。

## 実行環境
RTX4090、torch 2.11.0+cu128、torchvision 0.26.0+cu128、numpy 2.5.3、gymnasium 1.2.1。これらは成功時の記録で、汎用互換性の保証ではありません。

## 資料の入口
[コード](../code/README_JPN.md) · [データ](../datasets/README_JPN.md) · [モデル](../result/models/README_JPN.md) · [検証](../result/evidence/README_JPN.md) · [動画](../result/evidence/videos/README_JPN.md)
