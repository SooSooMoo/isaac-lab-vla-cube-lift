# データセット（日本語）

[English](README_ENG.md) · [元パス・SHA-256・依存一覧](dependency_manifest.json)

## 収録範囲

外部保管の教師HDF5を含む依存一覧として、視覚整合性の2ファイル、開閉保持、XY介入に加え、腕保持の教師データ、3種類の保持用観測、言語ペアの目標値、検証・学習JSONを整理しました。ファイル名が重複するNPZは役割が分かる名前に変更してコピーしています。元実験ファイルは変更していません。

下表のhold/visual/gripper/xyに対応する学習スクリプトは、依存一覧のconsumersで確認できます。JSONは単なるログとは限らず、教師ラベルやハッシュ検査に使用されるものもあります。

| ファイル（datasets/からの相対パス） | 内容 | 利用スクリプト |
|---|---|---|
| `demonstrations/two_instruction.hdf5` | 教師デモ（外部保管） | train_current_hold_adapter |
| `visual_consistency/base_observations13.npz` | 基準となる画像観測 | train_visual_consistency13 |
| `visual_consistency/tenth_observations13.npz` | 変化後の画像観測 | train_visual_consistency13 |
| `retention/retention_inputs700.npz` | 700観測・86閉鎖目標 | train_retention_gripper |
| `xy_attenuation/xy_correction_inputs879.npz` | 879観測・79介入目標 | train_xy_attenuation |
| `hold/settle_then_close623.npz` | 教師による静止待ち・閉鎖の60観測 | train_current_hold_adapter |
| `anchors/above442.npz` | aboveの既存出力を保持する442観測 | train_current_hold_adapter, train_visual_consistency13, train_retention_gripper, train_xy_attenuation |
| `anchors/base_pick623.npz` | 基礎方策の接近時623観測 | train_current_hold_adapter |
| `anchors/adapter_pick526.npz` | アダプター適用方策の接近時526観測 | train_visual_consistency13 |
| `language/teacher_pairs.json` | 言語ペア目標24レコード | train_current_hold_adapter |
| `metadata/base_training_protocol.json` | 学習スクリプトが参照する設定 | train_current_hold_adapter |
| `metadata/hold_collection.json` | 教師データ収集の検証記録 | train_current_hold_adapter |
| `metadata/above_anchor_collection.json` | 保持用観測の収集検証記録 | train_current_hold_adapter |
| `metadata/base_pick_anchor_collection.json` | 保持用観測の収集検証記録 | train_current_hold_adapter |
| `metadata/adapter_pick_anchor_collection.json` | 保持用観測の出所 | train_visual_consistency13 |
| `metadata/current_hold_training.json` | 中間学習の結果記録 | train_current_hold_adapter |
| `metadata/visual_training.json` | 最終ローダーが参照する結果・保持用観測のハッシュ | shared_loader |

## 教師データの内容

`demonstrations/two_instruction.hdf5`：demo_1はmove above the cube（413サンプル）、demo_0はpick up the cube（743サンプル）。各デモのlanguage_instruction属性に指示を保存。200×200の卓上・手首RGB画像、手先位置、xyzw姿勢、指位置、7成分の行動を含みます。観測は行動前です。方策入力にはstep/1200を追加します。

![aboveの教師例](samples/above_dataset.png)
![pickの教師例](samples/pick_dataset.png)

[抽出した指示・形状・行動例](samples/examples.json)。画像例は最終方策の評価動画ではありません。

## データと重みの区別

中間アダプターのローカル配置先は `../result/model/development/current_hold_adapter.pt`、最終評価用の4つの重みの配置先は `result/model/` です。これらの `.pt` はGitHubに含まれていません。利用前にプロジェクトのバックアップから復元し、記録済みのハッシュと照合してください。

## 再現の範囲

同梱する4本の学習スクリプトの直接参照データと、中間重みを整理したものです。基礎チェックポイントをゼロから作った過去の全学習履歴の完全収録ではありません。Isaac Lab環境・外部ラッパー等も必要です。既存スクリプトは元Podパスを参照しており、コピー先を自動的に読むようには変更していません。

13観測ペア・86閉鎖目標・79XY介入目標は再利用した開発データです。独立した汎化テストや自律成功の証明ではありません。介入収集と最終自律評価を区別します。

GitHubには上表のNPZ 8ファイルを公開しています。別ディレクトリの検証用4ファイル・旧版1ファイルを合わせて、公開済みNPZは13ファイルです。上表の教師HDF5は現在のGitHubに含まれていないため、プロジェクトのバックアップから復元してください。依存一覧は整理済みPodパッケージを記録したもので、全依存ファイルのGitHub配布を保証するものではありません。既存のgitignoreは維持しており、追跡済みNPZは公開済みです。
