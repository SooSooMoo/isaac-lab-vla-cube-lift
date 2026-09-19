# データセット（日本語）

[English](README_ENG.md) · [元パス・SHA-256・依存一覧](dependency_manifest.json)

## 収録範囲

教師HDF5、視覚整合性の2ファイル、開閉保持、XY介入に加え、腕保持の教師データ、3種類の保持用観測、言語ペアの目標値、検証・学習JSONを整理しました。ファイル名が重複するNPZは役割が分かる名前に変更してコピーしています。元実験ファイルは変更していません。

下表のhold/visual/gripper/xyに対応する学習スクリプトは、依存一覧のconsumersで確認できます。JSONは単なるログとは限らず、教師ラベルやハッシュ検査に使用されるものもあります。

| File (relative to datasets/) | Contents | Used by |
|---|---|---|
| `demonstrations/two_instruction.hdf5` | teacher demonstrations | train_current_hold_adapter |
| `visual_consistency/base_observations13.npz` | reference visual observations | train_visual_consistency13 |
| `visual_consistency/tenth_observations13.npz` | perturbed visual observations | train_visual_consistency13 |
| `retention/retention_inputs700.npz` | 700 observations; 86 closure targets | train_retention_gripper |
| `xy_attenuation/xy_correction_inputs879.npz` | 879 observations; 79 intervention targets | train_xy_attenuation |
| `hold/settle_then_close623.npz` | 60 teacher settle/close observations | train_current_hold_adapter |
| `anchors/above442.npz` | 442 above preservation observations | train_current_hold_adapter, train_visual_consistency13, train_retention_gripper, train_xy_attenuation |
| `anchors/base_pick623.npz` | 623 base-policy approach observations | train_current_hold_adapter |
| `anchors/adapter_pick526.npz` | 526 adapter-policy approach observations | train_visual_consistency13 |
| `language/teacher_pairs.json` | 24 language-pair target records | train_current_hold_adapter |
| `metadata/base_training_protocol.json` | protocol read by training script | train_current_hold_adapter |
| `metadata/hold_collection.json` | teacher collection verification | train_current_hold_adapter |
| `metadata/above_anchor_collection.json` | anchor collection verification | train_current_hold_adapter |
| `metadata/base_pick_anchor_collection.json` | anchor collection verification | train_current_hold_adapter |
| `metadata/adapter_pick_anchor_collection.json` | anchor provenance | train_visual_consistency13 |
| `metadata/current_hold_training.json` | intermediate training report | train_current_hold_adapter |
| `metadata/visual_training.json` | report and anchor hashes read by final loader | shared_loader |

## 教師データの内容

`demonstrations/two_instruction.hdf5`：demo_1はmove above the cube（413サンプル）、demo_0はpick up the cube（743サンプル）。各デモのlanguage_instruction属性に指示を保存。200×200の卓上・手首RGB画像、手先位置、xyzw姿勢、指位置、7成分の行動を含みます。観測は行動前です。方策入力にはstep/1200を追加します。

![aboveの教師例](samples/above_dataset.png)
![pickの教師例](samples/pick_dataset.png)

[抽出した指示・形状・行動例](samples/examples.json)。画像例は最終方策の評価動画ではありません。

## データと重みの区別

中間アダプターは `../result/model/development/current_hold_adapter.pt` に配置。最終構成の4ファイルはresult/modelにあります。重みは教師データとは分けています。

## 再現の範囲

同梱する4本の学習スクリプトの直接参照データと、中間重みを整理したものです。基礎チェックポイントをゼロから作った過去の全学習履歴の完全収録ではありません。Isaac Lab環境・外部ラッパー等も必要です。既存スクリプトは元Podパスを参照しており、コピー先を自動的に読むようには変更していません。

13観測ペア・86閉鎖目標・79XY介入目標は再利用した開発データです。独立した汎化テストや自律成功の証明ではありません。介入収集と最終自律評価を区別します。

GitHub公開用の配布方法は未設定です。大容量データと重みの既存gitignoreは維持しています。
