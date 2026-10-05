# 現行モデル

[English](README_ENG.md)

## 配置・容量
[`cube_position_estimator.pt`](cube_position_estimator.pt)は9,455,338 bytes（約9.46MB、9.02MiB）。今回の学習済み推論重みはこの1ファイルで、画像エンコーダー・位置ヘッド・正規化値を含みます。段階制御はPythonコードです。

SHA-256：`9a5004fe9ad27b4266b08002842cd5d48eac57235d3d57f8405bbc38feec719f`

[元パスと識別情報](manifest.json)。最終学習の選択epochは62、formatはspatial_cube_estimator_v1です。前回の4種類の重み構成とは異なります。モデル単体でシミュレータが動くわけではなく、初期状態・カスタム環境などは別途必要です。
