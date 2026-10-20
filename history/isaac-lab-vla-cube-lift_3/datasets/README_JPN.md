# データセット

HDF5には動作前のカメラ画像・ロボット状態・7次元動作を保存します。`instruction` 属性は `pick up the cube` です。教師修正データでは履歴を保持するためポリシーの接近部分も保存し、`teacher_mask` で教師ラベルを区別します。

今回の資料ZIPにはHDF5本体が含まれていません。全学習データの依存関係と配布方法の整理は未完了です。旧位置推定用データを今回の学習データとして案内しません。


[Dependency manifest](dependency_manifest.json)

評価の初期状態は `baseline_inputs/datasets/lift_robomimic_language.hdf5` の `data/demo_0/initial_state` と先頭のEEF観測から読みます。これは全学習データを示すものではありません。
