# Datasets

本プロジェクトでは、赤・青Cubeの指定把持に用いる教師エピソードを収集し、模倣学習に使用しました。
最終学習では50エピソード（学習46・検証4）を使用しています。

- 観測：テーブル／手首カメラ画像、手先位置・姿勢、指位置
- 指示：`pick up the red cube` / `pick up the blue cube`
- 動作：相対移動3成分、相対回転3成分、グリッパー開閉
- 修正データ：失敗軌道の再生による観測履歴と、教師による修正動作。再生区間は正解動作の学習対象から除外。

**元の学習用HDF5・画像データは、このリポジトリには同梱していません。**
このディレクトリはデータの説明用です。リポジトリ単体では同じ学習を再実行できません。

学習条件と監査記録は [result/evidence](../result/evidence/)、収集・監査コードは [code](../code/) を参照してください。

## HDF5形式の説明範囲

以下は、公開されている[収集コード](../code/collect_step5_color_replay_recovery_red1_v1.py)、[監査コード](../code/audit_step5_color_replay_recovery_v1.py)、[学習コード](../code/train_step5_color_replay_recovery_v1.py)と[監査結果](../result/evidence/recovery_audit.json)に基づく説明です。非公開の元HDF5を直接検査した仕様書ではありません。

HDF5は、グループ（フォルダーに相当）、データセット（配列）、属性（メタデータ）を1つのファイルに保存する形式です。本プロジェクトでは、基本的に条件別の `episode.hdf5` に1エピソードを保存し、`data/demo_0` を読み込みます。50エピソード全体は、各ファイルを参照する `training_manifest.json` で管理します。

ここでは `T` を教師の正解動作があるステップ数、`C` を修正データの再生履歴長、`L` を指示文のトークン数とします。最終採用の修正データでは **T=565、C=500** です。通常エピソードのTは可変です。

### 1. データの種類

| 種類 | ルート属性 `schema` | 用途 |
|---|---|---|
| 通常の教師データ | `color_teacher_v1` | 教師の観測・正解動作を学習する |
| 再生履歴付き修正データ | `color_recovery_teacher_v1` | 再生履歴でGRUの状態を準備し、後続の教師動作を学習する |
| 学習モデルの評価・収録データ | `color_policy_evaluation_v1` | モデル動作を評価・記録する。学習用教師データではない |

以下のツリーは修正データを中心に示します。通常の教師データには `recovery_context` は必要ありません。既存49件は監査済み一覧として引き継がれるため、古い収集版の補助グループ・属性がすべて同一とは限りません。

### 2. グループ構成

```text
episode.hdf5
├── [ルート属性] schema, step_dt, action_scale, ...
├── data/
│   └── demo_0/
│       ├── [属性] language_instruction, layout, success, num_samples, ...
│       ├── obs/
│       │   ├── table_cam           (T, 200, 200, 3)
│       │   ├── wrist_cam           (T, 200, 200, 3)
│       │   ├── eef_pos             (T, 3)
│       │   ├── eef_quat            (T, 4)
│       │   └── gripper_pos         (T, 2)
│       ├── actions                 (T, 7)
│       ├── language_tokens         (L,)
│       └── reset_state_before_settle/
│           ├── articulation/robot/...
│           └── rigid_object/object, object_blue/...
├── diagnostics/
│   ├── target_post_pos             (T, 3)
│   ├── other_post_pos              (T, 3)
│   ├── joint_pos                   (T, 9)
│   └── phase                       (T,)
└── recovery_context/               [修正データのみ]
    ├── obs/
    │   ├── table_cam               (C, 200, 200, 3)
    │   ├── wrist_cam               (C, 200, 200, 3)
    │   ├── eef_pos                 (C, 3)
    │   ├── eef_quat                (C, 4)
    │   └── gripper_pos             (C, 2)
    └── replayed_actions            (C, 7)
```

`[属性]` は配列データセットではなくHDF5の属性です。`reset_state_before_settle` のツリーは概略で、シミュレータの状態辞書を再帰的に保存します。

### 3. 観測・行動・言語のデータセット

パスは `data/demo_0/` からの相対表記です。

| パス | 形状 | 型・単位 | 内容 |
|---|---|---|---|
| `obs/table_cam` | `(T,200,200,3)` | `uint8`、0〜255 | テーブルカメラのRGB画像。軸順は時間・高さ・幅・色 |
| `obs/wrist_cam` | `(T,200,200,3)` | `uint8`、0〜255 | 手首カメラのRGB画像。同じ軸順 |
| `obs/eef_pos` | `(T,3)` | 浮動小数点、m | 手先位置 `(x,y,z)` |
| `obs/eef_quat` | `(T,4)` | 浮動小数点、無次元 | 手先姿勢の単位クォータニオン `(x,y,z,w)` |
| `obs/gripper_pos` | `(T,2)` | 浮動小数点、m | 左右の指関節位置。開閉指令そのものではない |
| `actions` | `(T,7)` | 浮動小数点 | 相対IK用の移動・回転指令6成分＋開閉指令1成分 |
| `language_tokens` | `(L,)` | `int64` | エピソード全体の指示トークン。ステップごとには重複保存しない |

**浮動小数点の格納精度は、元の配列のdtypeを引き継ぎます。** 収集コードは全数値配列を一律にfloat32へ変換して保存する実装ではありません。正確な保存dtypeは実ファイルの `dataset.dtype` で確認してください。学習コードでは状態と正解動作をfloat32へ変換します。

#### 座標・姿勢・画像の扱い

- `eef_pos` は環境が生成する手先位置観測です。監査では、環境原点を差し引いたCube座標との距離を比較します。行動指令の座標系と無条件に同一と扱わず、元の観測設定・IK設定を合わせてください。
- `eef_quat` は `xyzw` 順で、監査ではノルムが1に近く、最後の成分wが非負（数値誤差を許容）であることを確認します。
- `reset_state_before_settle` に保存する姿勢はシミュレータの内部表現です。観測用の `xyzw` 規約を、その内部配列へ一律に適用しないでください。
- 保存画像はRGB・HWC形式です。学習時にCHW形式へ並べ替え、255で割り、平均 `[0.485,0.456,0.406]`・標準偏差 `[0.229,0.224,0.225]` で正規化します。

#### 行動7成分の意味

```text
actions[t] = [dx/scale, dy/scale, dz/scale,
              rx/scale, ry/scale, rz/scale, gripper]
scale = action_scale = 0.5
```

`dx,dy,dz` は相対位置指令（m）、`rx,ry,rz` は軸角形式の回転ベクトル（rad）です。回転3成分はEuler角ではありません。これらはIK制御へ渡す指令であり、実際に達成された位置差・関節角ではありません。収集教師はロボット基準のIKフレームで誤差を計算し、動作を生成します。

`gripper=+1` は開く、`gripper=−1` は閉じる指令です。グリッパー成分には `action_scale` を掛けません。教師の監査では、1ステップの移動指令のノルムが約0.01 m以下、回転指令のノルムが約0.08 rad以下であることも検査します。

#### 言語トークン

指示は `language_instruction` 属性に文字列、`language_tokens` に整数列として保存します。対象の2指示はいずれも5単語なので `L=5` です。

収集・監査のトークン化は、小文字化・ピリオドとカンマの除去・空白分割を行い、先頭16単語までについて次のIDを計算します。

```python
1 + int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), 'little') % 4095
```

これは本実装固有の単語ハッシュ方式です。一般的な大規模言語モデルのトークナイザではなく、整数IDだけから単語を復元する形式でもありません。

### 4. 観測と行動の時刻対応

ルート属性 `observation_timing='pre_action'` は、次の記録順序を示します。

```text
obs[t] を取得 → actions[t] を記録 → env.step(actions[t])
             → diagnostics[t] に動作後の値を記録 → obs[t+1]
```

したがって、学習の組は `(obs[t], instruction, actions[t])` です。`diagnostics[t]` は動作後なので、次の `obs[t+1]` と時間的に対応します。観測と正解動作を1ステップずらさないことが重要です。最後の動作後の観測が、追加の `obs[T]` として保存されるとは限りません。

最終学習コードが要求する `step_dt` は **0.02秒（50 Hz）**、`action_scale` は **0.5** です。`step_dt` は制御ステップ間隔であり、物理シミュレーション内部のサブステップ間隔とは区別します。

### 5. 属性（メタデータ）

#### ファイルのルート属性

| 属性 | 値・意味 |
|---|---|
| `schema` | 上記のデータ形式識別子 |
| `training_eligible` | 学習候補かどうか。修正収集では成功判定に応じて設定。これだけで監査合格を意味しない |
| `teacher_only` | 教師データではtrue。修正データの再生区間まで教師の正解動作という意味ではない |
| `training_performed` | 収集・評価ではfalse。このファイルを生成する処理で学習していないことを示す |
| `observation_timing` | `pre_action` |
| `step_dt` | 制御ステップ間隔。最終学習では0.02秒 |
| `action_scale` | 相対IKのスケール。最終学習では0.5 |
| `quaternion_order` | 観測姿勢は `xyzw` |
| `uses_progress_input` | false。工程・進捗をモデル入力に含めない |
| `action_space` | 7次元相対IKと開閉符号の説明文字列 |
| `result_json` | 成否・条件・件数などの結果を格納するJSON文字列 |

修正データでは、さらに `replay_source`、`replay_source_sha256`、`replay_steps=500`、`context_is_teacher=false`、`context_usage` を保存します。元の絶対パスは出所の記録であり、リポジトリ内のダウンロード先ではありません。

#### `data/demo_0` の属性

| 属性 | 値・意味 |
|---|---|
| `language_instruction` | 赤または青を指定する英文 |
| `layout` | `red_yminus_blue_yplus` または `blue_yminus_red_yplus` |
| `target_color` | `red` または `blue` |
| `dx_mm`, `dy_mm` | 基準配置からのXYオフセット（mm）。2個のCubeを一緒に移動 |
| `success` | エピソードの合否 |
| `num_samples` | 教師区間のT。修正データでは再生履歴Cを含まない |
| `settle_physics_steps` | 公開修正収集コードでは90 |
| `camera_warmup_renders` | 同コードでは12 |

### 6. 診断情報とリセット状態

`diagnostics` は学習モデルへ渡す観測とは別に保存します。

| パス | 形状・型 | 内容 |
|---|---|---|
| `diagnostics/target_post_pos` | `(T,3)`、浮動小数点、m | 動作後の指定Cube位置。ワールド位置から環境原点を差し引く |
| `diagnostics/other_post_pos` | `(T,3)`、浮動小数点、m | 動作後の指定外Cube位置。同じ座標表現 |
| `diagnostics/joint_pos` | `(T,9)`、浮動小数点 | 腕7関節（rad）と指2関節（m）の動作後の位置 |
| `diagnostics/phase` | `(T,)`、`int8` | 教師工程：0=接近、1=下降、2=閉じる、3=持ち上げる |

Cube座標や `phase` は、教師生成・成功判定・監査のための情報です。最終VLAモデルには入力しません。

`data/demo_0/reset_state_before_settle` は、安定化処理前のシミュレータ状態です。ロボットの `joint_position`、Cubeの `root_pose` などを含む状態辞書を保存します。修正データでは、これは**500ステップの再生を始める前の状態**であり、教師が引き継ぐ時点の状態ではありません。内部のキー・形状・姿勢規約は環境実装に依存するため、同じ環境の `reset_to` に渡す用途で扱います。

### 7. 修正データの500ステップ履歴

公開監査結果では、修正エピソードは次の構成です。

| 区間 | ステップ数 | 保存先 | 学習での扱い |
|---|---|---|---|
| 失敗軌道の再生履歴 | C=500 | `recovery_context/obs` | GRUの内部状態の準備のみ。損失・勾配なし |
| 再生に用いた行動 | 500 | `recovery_context/replayed_actions` | 再現検証用。正解ラベルにしない |
| 教師の修正動作 | T=565 | `data/demo_0/obs` と `actions` | 正解動作の学習対象 |

概念上は観測500＋565=1,065ステップを連続して扱いますが、正解動作は565件です。`actions` に再生行動500件を結合して、1,065件の教師ラベルとして学習させてはいけません。

[学習条件](../result/evidence/training_protocol.json)では、学習46エピソード・正解動作40,301件、検証4エピソード・正解動作3,534件です。500ステップの再生履歴はこれらの正解動作件数に含まれません。検証条件はXYオフセット `(0,+10)` mmで、修正エピソードは学習側に入ります。

### 8. 保存方法と評価HDF5との差

公開修正収集コードは、時系列データセットの先頭軸を伸長可能にし、チャンクを1ステップ単位で作成します。`append()` で保存する配列にはgzipレベル9の可逆圧縮を使用します。`language_tokens`、`replayed_actions`、リセット状態は別の `create_dataset()` で作成するため、全データセットが同じ圧縮設定ではありません。この設定を既存49件の全ファイルへ一律に当てはめることもできません。

[評価コード](../code/evaluate_step5_color_replay_recovery32_v1.py)と[収録コード](../code/record_step5_color_portfolio8_v1.py)のHDF5は、次の点で異なります。

- `schema='color_policy_evaluation_v1'`、`teacher_only=false`、`training_eligible=false`。
- `obs` に保存するのは手先位置・姿勢・指位置で、毎ステップのRGB画像は保存しません。推論自体は毎ステップ画像を使用します。
- 行動は教師の正解ではなく、学習モデルの出力です。
- 教師の `phase` と `recovery_context` はありません。
- 時系列配列の圧縮はgzipレベル1です。画像は別のPNG、収録時はMP4にも保存します。

したがって、評価HDF5を教師HDF5の代わりにそのまま学習へ渡すことはできません。修正収集が参照する過去の失敗記録も、この最終評価コードと同じ画像保存仕様だとは限りません。

### 9. 実ファイルの構造を確認する例

HDF5を別途保持している環境で実行してください。リポジトリ内に対象ファイルがあることを前提とする例ではありません。

```python
import sys
import h5py

# Usage: python inspect_hdf5.py /path/to/episode.hdf5
with h5py.File(sys.argv[1], 'r') as f:
    def show(name, obj):
        label = '/' + name if name else '/'
        if isinstance(obj, h5py.Dataset):
            print(label, 'shape=', obj.shape, 'dtype=', obj.dtype,
                  'chunks=', obj.chunks, 'compression=', obj.compression,
                  'compression_opts=', obj.compression_opts)
        else:
            print(label, '(group)')
        for key, value in obj.attrs.items():
            print('  attr:', key, '=', value)

    show('', f)
    f.visititems(show)

    demo = f['data/demo_0']
    print('action steps:', len(demo['actions']))
    print('instruction:', demo.attrs['language_instruction'])
    if 'recovery_context' in f:
        print('context steps:', len(f['recovery_context/replayed_actions']))
```

この例は読み取り専用で構造を表示します。データ監査や学習適格性の判定は行いません。

### 10. 参照資料

- [修正データの収集コード](../code/collect_step5_color_replay_recovery_red1_v1.py)
- [HDF5と観測・行動の整合性監査](../code/audit_step5_color_replay_recovery_v1.py)
- [最終学習コード](../code/train_step5_color_replay_recovery_v1.py)
- [修正データの監査結果](../result/evidence/recovery_audit.json)
- [最終学習条件](../result/evidence/training_protocol.json)

追記内容は2026年10月24日に確認した公開コード・記録に基づきます。
