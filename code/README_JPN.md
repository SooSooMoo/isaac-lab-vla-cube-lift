# コード一覧と実行ワークフロー

## このディレクトリの役割

卓上・手首カメラ、ロボット状態、言語指示から動作を生成し、同じモデル構成で次の2タスクを実行するためのコードです。

- `move above the cube`：cube上方の目標位置へ移動・保持
- `pick up the cube`：cubeを把持して持ち上げる

現行構成は、基礎モデル・視覚アダプター・開閉補正・XY減衰の4ファイルを組み合わせます。両タスクで同じ構成を使います。モデルの識別情報は [モデルmanifest](../result/model/manifest.json)、数値結果は [検証記録](../result/evidence/README.md) を参照してください。

**以下は成功時のPodに依存する実験スクリプト集です。上から全ファイルを実行する手順ではありません。** 学習データ、過去の実験出力、Isaac Labのカスタム環境など、リポジトリ外のファイルも必要です。各スクリプト内の固定パス・ハッシュを確認してください。

## 1. コードファイル一覧

パスはすべて `code/` からの相対パスです。説明用READMEを除く、コード・設定ファイルを掲載しています。

### モデル・推論

| ファイル名 | 概要 | 詳細・実行タイミング |
|---|---|---|
| [model/language_vla/model.py](model/language_vla/model.py) | 基礎VLAモデル | 2カメラ画像・状態・言語を処理し、腕とグリッパーの出力を生成。モデル構造を読む際の入口。最終構成には追加補正が必要です。 |
| [model/language_vla/__init__.py](model/language_vla/__init__.py) | Pythonパッケージ定義 | モデルをモジュールとして配置するためのファイル。単独実行は不要です。 |
| [inference/shared_loader.py](inference/shared_loader.py) | 最終構成の共通ローダー | 重み・依存ハッシュを確認し、視覚残差、開閉補正、特徴に応じたXY減衰を組み込みます。画像正規化と状態生成も担当。検証スクリプトから抽出した参照実装で、単独CLIではありません。 |

評価スクリプトは、それぞれ内部に検証時のローダーを保持しています。`inference/shared_loader.py` の編集だけで全評価スクリプトが変更される構造ではありません。

### シミュレータ実行基盤

| ファイル名 | 概要 | 詳細・実行タイミング |
|---|---|---|
| [runtime/evaluate.py](runtime/evaluate.py) | 評価ループ | 環境の起動、観測取得、推論、動作実行、軌道記録、終了判定を担当。最終構成の確認は後述の `evaluation/` から行います。 |
| [runtime/evaluate_above_wrapper.py](runtime/evaluate_above_wrapper.py) | 評価前の検査・起動 | 初期入力やPyTorch設定などを確認して評価を起動するラッパー。検証スクリプトは元Podの対応するラッパーを参照・加工します。 |
| [runtime/smoke_utils.py](runtime/smoke_utils.py) | 共通ユーティリティ | 言語トークン化、観測取り出し、エピソード保存、初期状態比較、方策ロードなど。最終評価ではロード処理を共通モデル構成に置き換えます。 |
| [runtime/collect.py](runtime/collect.py) | 教師エピソード収集 | 指定指示・初期状態で教師制御を実行し、観測と行動を保存する初期開発用コード。学習済みモデルの自律成功評価とは区別します。 |
| [runtime/pack.py](runtime/pack.py) | 2指示データの統合 | pick/aboveのHDF5について指示、初期状態、行動前観測の条件を確認し、統合データを作成。教師データ準備時に使用します。 |
| [runtime/language_vla/model.py](runtime/language_vla/model.py) | 実行基盤用モデル | `model/language_vla/model.py` と同じモデルソースを実行用の配置に保持しています。 |
| [runtime/language_vla/dataset.py](runtime/language_vla/dataset.py) | 学習データ読込 | HDF5から画像・状態・言語・行動のサンプルを構成し、バッチ化する処理。データ形式を確認・拡張するときに参照します。 |
| [runtime/language_vla/__init__.py](runtime/language_vla/__init__.py) | Pythonパッケージ定義 | 実行用 `language_vla` のパッケージ配置。単独実行は不要です。 |

### 追加データ収集

| ファイル名 | 概要 | 詳細・実行タイミング |
|---|---|---|
| [data_collection/collect_current_arm_hold623.sh](data_collection/collect_current_arm_hold623.sh) | 静止・閉鎖の教師データ収集 | 623ステップの接近後、20ステップの静止待ちと40ステップの閉鎖保持を収集。腕保持アダプター開発時の教師介入実験です。 |
| [data_collection/collect_visual_inputs13.sh](data_collection/collect_visual_inputs13.sh) | 視覚変動の観測収集 | 基準実行と補正を加えた実行から13観測ずつ保存。視覚整合性学習用で、新しい専門家の行動ラベルを付ける処理ではありません。 |
| [data_collection/collect_retention_inputs700.sh](data_collection/collect_retention_inputs700.sh) | 開閉保持の入力収集 | 700ステップの特徴と行動を保存。614以降の86観測には閉鎖を維持する診断介入の目標を使用。開閉補正の学習前に実施します。 |
| [data_collection/collect_xy_correction879.sh](data_collection/collect_xy_correction879.sh) | XY補正の入力収集 | 先頭800観測の行動保持と、以後79観測のXY抑制目標を保存。XY減衰の学習用です。介入中の成功を自律成功とは扱いません。 |

### 学習

| ファイル名 | 概要 | 詳細・実行タイミング |
|---|---|---|
| [training/train_current_hold_adapter.sh](training/train_current_hold_adapter.sh) | 腕保持アダプターの学習 | 教師の静止・閉鎖データと既存行動を保持する観測を利用。後続の視覚補正で使用するアダプターの開発段階です。 |
| [training/train_visual_consistency13.sh](training/train_visual_consistency13.sh) | 視覚整合性の学習 | 13観測ペアを再利用し、手首画像の変化に対する出力差を抑えます。既存アダプターから学習し、最終構成の `visual_adapter.pt` を作成します。 |
| [training/train_retention_gripper.sh](training/train_retention_gripper.sh) | 開閉出力の補正学習 | 融合特徴から線形補正を学習。先頭614観測・aboveの開閉出力を保持し、86観測の閉鎖目標を学習します。同一入力に対する腕出力は変更しません。 |
| [training/train_xy_attenuation.sh](training/train_xy_attenuation.sh) | XY減衰の学習 | 融合特徴から0～1の減衰量を出すよう学習し、XY指令に適用。保存観測上で接近・aboveの補正をゼロに保ち、79観測のXY抑制目標を学習します。 |

学習の合格は、保存した入力上の条件を満たしたという意味です。新しい重みはシミュレータで再評価する必要があります。既存の最終モデルで動画や評価を再現するだけなら、これらの再学習は不要です。

### 原因調査・診断

| ファイル名 | 概要 | 詳細・実行タイミング |
|---|---|---|
| [diagnostics/diagnose_xy_suppression800.sh](diagnostics/diagnose_xy_suppression800.sh) | 横指令の介入実験 | 800以降のXY指令をゼロにして影響を比較。横移動の原因を調べる実験であり、採用方策にはこの固定ステップ介入を使いません。 |
| [diagnostics/probe_xy_feature_separation.sh](diagnostics/probe_xy_feature_separation.sh) | 特徴の識別可能性を調査 | 保存特徴から補正対象領域と保持領域を線形に識別できるか確認。軌道内のブロック分割を使用し、独立した汎化評価とは区別します。 |

### 最終評価・再現性

| ファイル名 | 概要 | 詳細・実行タイミング |
|---|---|---|
| [evaluation/verify_xy_attenuation_above.sh](evaluation/verify_xy_attenuation_above.sh) | aboveの最終評価 | 全補正を含む共通モデル構成で、30 mm以内・50ステップの位置保持を検証。最終重みを選んだ後に実施します。 |
| [evaluation/verify_xy_attenuation1300.sh](evaluation/verify_xy_attenuation1300.sh) | pickの最終評価 | 最大1300ステップで、高さ約121 mm以上を10ステップ、横移動50 mm以内を検証。成功すると早期終了します。 |
| [evaluation/repeat_both_xy_attenuation.sh](evaluation/repeat_both_xy_attenuation.sh) | 両タスクの再実行比較 | 成功時と同じ構成で再実行し、行動・手先位置/姿勢・cube位置・指位置の7配列を基準軌道と比較。両タスクの合格後に実施します。 |

### 動画

| ファイル名 | 概要 | 詳細・実行タイミング |
|---|---|---|
| [video/record_both_policy_videos.sh](video/record_both_policy_videos.sh) | 元動画の録画 | 同じ方策を再実行し、卓上・手首カメラを左右に配置。50fpsで録画し、検証軌道との一致と動画の復号を確認します。 |
| [video/record_both_continued3s_v2.sh](video/record_both_continued3s_v2.sh) | 成功後3秒の継続録画 | 成功後も同じモデルの推論・制御を150ステップ継続。姿勢固定や静止画追加ではありません。保持が崩れる可能性も含めて観察します。 |
| [video/trim_final_videos.sh](video/trim_final_videos.sh) | 掲載動画の末尾編集 | 延長版からaboveは495フレーム、pickは930フレームを残し、別ファイルへ保存。元動画は保持。既に編集版が存在する場合は上書きせず停止します。 |

### 設定・パッケージ検査

| ファイル名 | 概要 | 詳細・実行タイミング |
|---|---|---|
| [configs/success.json](configs/success.json) | 成功条件と識別情報 | 基礎モデルSHA、進行度の分母、時間刻み、above/pickの数値基準を記録。評価スクリプトの値を自動変更する設定ローダーではありません。 |
| [configs/artifact_import.json](configs/artifact_import.json) | 成果物の取り込み定義 | モデル・検証JSON・動画の元パス、配置先、必須項目、ハッシュやフレーム数を定義。Pod更新プログラムが使用します。 |
| [tools/check_package.py](tools/check_package.py) | 配布内容の整合性確認 | Python・JSON・Markdownリンク・シェル内Python・ファイルハッシュを検査。`--require-imported` で取り込み済み成果物も確認。資料更新後・公開前に使用します。 |

## 2. プロジェクトのワークフローと実行タイミング

| 順番 | 工程 | 実行するタイミング | 次へ進むための確認 |
|---:|---|---|---|
| 1 | 環境・入力の確認 | 新Podへの移行時、環境変更時 | カスタムタスク、依存ファイル、初期観測、モデルハッシュが一致すること |
| 2 | 教師・補正データ収集 | 動作上の問題を特定し、修正目標を決めたとき | 観測と行動の対応、介入前の軌道一致、保存データの整合性 |
| 3 | 学習と保存入力上の検査 | 対象データが揃ったとき | 有限値、既存行動の保持、対象出力の改善、重みの保存・再読込 |
| 4 | 共通構成で両タスク評価 | 新しいモデル候補を採用する前 | aboveの位置保持とpickの高さ・横移動条件を両方満たすこと |
| 5 | 再実行による比較 | 両タスクが数値基準を満たした後 | 同じ条件での軌道一致。固定条件の再現性として記録すること |
| 6 | 録画・視聴 | 検証済みモデルをデモにするとき | 録画時の軌道一致、cubeの見え方、揺れ、終了位置を人が確認 |
| 7 | 必要に応じて末尾編集 | 動画の採用範囲が決まった後 | 元映像を保持し、編集内容を記録。編集で制御性能が改善したとは扱わない |
| 8 | 成果物取り込み・公開準備 | モデル・結果・動画が確定した後 | バックアップ、ハッシュ、資料と数値の一致、リンクの確認 |

今回の採用構成は工程4～6を終え、動画の採用範囲を決めています。成功後3秒間の継続ではaboveの位置ずれとpickの落下が発生したため、掲載版の末尾を編集しました。長時間保持の解決を意味するものではありません。

### 開発時の学習依存関係

```text
基礎モデル ＋ 既存の教師データ・保持用観測
    ↓ 腕保持アダプターの学習
既存アダプター ＋ 13観測ペア
    ↓ 視覚整合性の学習
視覚アダプター ＋ 開閉保持の収集データ
    ↓ 開閉補正の学習
視覚アダプター ＋ 開閉補正 ＋ XY介入の収集データ
    ↓ 特徴の識別確認・XY減衰の学習
最終の共通構成 → 両タスク評価 → 再実行 → 録画
```

これは依存関係の説明です。各段階のパスは採用した実験出力に固定されており、再学習すると新しい出力先になります。そのため、この図だけで最初から自動再構築できるわけではありません。

## 3. よく使う実行コマンド

検証済みPodで実行します。パスが固定された元実験データも必要です。

### 資料・配布ファイルだけ確認する

```bash
cd /workspace/step3/isaac-lab-vla-cube-lift_2
/isaac-sim/python.sh code/tools/check_package.py --require-imported
```

学習・シミュレーションは実行しません。README変更時は、リポジトリ直下の `source_inventory.json` にあるハッシュも更新する必要があります。

### 現行モデルで両タスクを再評価する

```bash
cd /workspace/step3/isaac-lab-vla-cube-lift_2
bash code/evaluation/verify_xy_attenuation_above.sh
bash code/evaluation/verify_xy_attenuation1300.sh
```

この操作はシミュレータを動かし、新しい結果ファイルを作成します。重みを再学習する操作ではありません。

### 同じ構成の再実行一致を確認する

```bash
cd /workspace/step3/isaac-lab-vla-cube-lift_2
bash code/evaluation/repeat_both_xy_attenuation.sh
```

比較先も採用時の記録に固定されています。モデルを変更した場合は、その候補用の評価・比較手順が必要です。

## 4. 実行前に確認すること

- 再現に必要な環境：[環境説明](../document/environment.md)、[再現手順](../document/reproduction.md)
- 学習・録画・再実行は保存容量を使用します。Podの容量制限に注意し、不要な再実行を避けてください。
- データ収集・診断には教師介入があります。最終評価に介入を混ぜないでください。
- 同じ入力で腕や開閉の出力を保っても、動作後の観測が変われば将来の行動は変わります。学習誤差だけで成功判定しません。
- 最終構成は進行度を入力に含みます。未知配置・指示の言い換えへの汎化や、接触力に基づく把持確認は今回の検証範囲外です。
