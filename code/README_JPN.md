# コード一覧と実行順序

[English](README_ENG.md)

## 役割
今回の学習対象はCubeの画像位置推定です。接近・下降・把持・持上げ・維持は段階制御で実行します。言語・進行度を使う前回の行動モデルとは別構成です。

## 保存コード
| ファイル | 役割・実行タイミング |
|---|---|
| [check_visual_validation26.py](snapshots/check_visual_validation26.py) | 候補確定後に固定点と既存ランダム点の26地点を評価 |
| [check_visual_random30.py](snapshots/check_visual_random30.py) | モデルを固定して新規ランダム5地点を評価 |
| [check_visual_positive_x30.py](snapshots/check_visual_positive_x30.py) | X正側の4領域から各1地点を追加評価 |
| [record_portfolio_video3.py](snapshots/record_portfolio_video3.py) | 3地点で録画・数値判定・フレーム数検証 |
| [train_spatial_cube_oracle30_refit.py](snapshots/train_spatial_cube_oracle30_refit.py) | 最終候補の学習。評価だけなら再学習不要 |
| [artifact_import.json](configs/artifact_import.json) | 公開モデル・動画の元パスとSHA-256 |

評価ランチャーは文字列SOURCEに実行本体を保持し、一時ディレクトリへ書き出してIsaac Labで起動します。モデルのロード・段階制御・採点はこの本体内にあります。README用フォルダへ実装を移したわけではありません。

## 読む順序
[モデル](model/README_JPN.md) → [推論](inference/README_JPN.md) → [収集](data_collection/README_JPN.md) → [学習](training/README_JPN.md) → [検証記録](../result/evidence/README_JPN.md)。

## 再実行
成功時Podの依存が揃った状態で、`/isaac-sim/python.sh -u code/snapshots/check_visual_validation26.py`等を実行します。絶対パスのデータ・モデル・初期状態・smoke_utils・カスタムIsaac Labが必要です。GitHub取得だけで動くCLIではありません。資料の閲覧・更新に学習や評価の実行は不要です。
