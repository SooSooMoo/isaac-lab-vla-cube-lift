# 学習

[English](README_ENG.md)

## 実行コード
[train_spatial_cube_oracle30_refit.py](../snapshots/train_spatial_cube_oracle30_refit.py)が最終採用候補を作ったスクリプトです。凍結画像特徴を抽出し、位置推定ヘッドを学習します。推論モデルからの継続学習ではなく、記録された元チェックポイント・データからヘッドを構築する処理です。

## 条件と選択
seed=2083、AdamW（lr=0.0003、weight_decay=0.0001）、80 epoch、バッチ128。位置ごとに重み付けし、復元抽出で学習します。相対位置を0.1mでスケールしSmoothL1（beta=0.05）を使用。検証RMSE最小のepoch 62を採用しました。

23エピソードを学習・検証・診断に分けます。検証IDは4と13、保留位置は(+20,−20mm)、以前の(−20,+20mm)軌道はrecovery_checkです。これらは開発で既に観察しており、完全に未見の盲検試験ではありません。

[設定・入力ハッシュ](../../result/training/protocol.json)と[オフライン結果](../../result/training/results.json)を参照。保存入力上の精度と実動作の合否は別です。既存重みで評価する際に学習を再実行する必要はありません。
