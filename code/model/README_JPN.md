# 位置推定モデル

[English](README_ENG.md)

## 入出力
卓上・手首の200×200 RGB画像、EEF位置3成分、xyzw姿勢4成分、指位置2成分を入力します。姿勢はwが負なら符号を反転します。出力は環境軸に沿うCube−EEFの相対位置3成分（m）です。言語と進行度は入力しません。

## 構造
凍結ResNet18のlayer2までを使用し、各カメラを128×5×5特徴へ集約します。2画像6400成分と状態9成分を結合し、6409→256 GELU→64 GELU→3のヘッドで回帰します。画像はImageNet平均・標準偏差で正規化。特徴平均・スケール、目標平均・スケールもチェックポイントに保存します。

構築コードは[学習スクリプト](../snapshots/train_spatial_cube_oracle30_refit.py)、推論の`load_cube_estimator`は[評価スクリプト](../snapshots/check_visual_validation26.py)内です。

## 重み
今回の推論用重みは[1ファイル](../../result/models/README_JPN.md)に格納。前回の基礎モデルや複数補正は推論時には不要ですが、学習スクリプトは特徴抽出器の初期化に古いチェックポイントを参照します。
