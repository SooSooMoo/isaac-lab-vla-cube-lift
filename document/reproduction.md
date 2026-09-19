# 再現・更新手順

## 既存Podへの更新

配布ZIPと `update_portfolio_on_pod.py` を `/workspace/step3/` に置き、次を実行します。

```bash
/isaac-sim/python.sh /workspace/step3/update_portfolio_on_pod.py
```

更新ツールは、既存の `isaac-lab-vla-cube-lift_2` 全体（.gitを含む）を先にアーカイブし、ハッシュを照合します。必要ファイル・容量を確認してから内容を更新し、モデル4ファイル、検証JSON、3種類の動画を取り込みます。原実験フォルダは変更しません。再学習・再シミュレーションはしません。

## 検証済みPodで評価を再実行する場合

```bash
bash code/evaluation/verify_xy_attenuation_above.sh
bash code/evaluation/verify_xy_attenuation1300.sh
bash code/evaluation/repeat_both_xy_attenuation.sh
```

スクリプトは成功時の `/workspace/step3/` のモデル・学習データ・ラッパー・Isaac Lab環境に依存します。リポジトリだけを新PCへコピーして動く構成ではありません。自動更新処理ではこれらを実行しません。

ソース確認：`python code/tools/check_package.py`。Podで取り込み後の確認：`python code/tools/check_package.py --require-imported`。

最終モデル構成の言語ペア再検証、クリーン環境からの復元、未知配置への評価は別途必要です。
