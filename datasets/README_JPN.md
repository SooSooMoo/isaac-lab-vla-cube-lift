# データセット

[English](README_ENG.md)

## 保存先と公開範囲
HDF5本体はGitHubに同梱していません。以下は元RunPodの絶対パスです。23本すべてが学習対象ではありません。[依存一覧](dependency_manifest.json)に各HDF5と別保存ラベルのSHA-256、分割を記録しています。

| ID | XY mm | Split | Samples | HDF5 on RunPod |
|---:|---|---|---:|---|
| 0 | [0.0, 0.0] | train | 893 | `/workspace/step4/direct_teacher_data_amrq3ejy/x0_y0/training_episode.hdf5` |
| 1 | [10.0, 0.0] | train | 901 | `/workspace/step4/direct_teacher_data_amrq3ejy/x10_y0/training_episode.hdf5` |
| 2 | [0, 0] | train | 322 | `/workspace/step4/takeover_pick_s4r3ef77/x0_y0/training_episode.hdf5` |
| 3 | [10, 0] | train | 324 | `/workspace/step4/takeover_pick_s4r3ef77/x10_y0/training_episode.hdf5` |
| 4 | [0, 0] | validation | 322 | `/workspace/step4/center_takeover_dcvrc4uk/x0_y0/training_episode.hdf5` |
| 5 | [0, 0] | train | 322 | `/workspace/step4/det_center_takeover_v7h811lb/x0_y0/training_episode.hdf5` |
| 6 | [0, 20] | train | 414 | `/workspace/step4/premature_close_data_saotojo5/x0_y20/training_episode.hdf5` |
| 7 | [-20, 20] | recovery_check | 536 | `/workspace/step4/premature_close_data_saotojo5/x-20_y20/training_episode.hdf5` |
| 8 | [20, -20] | heldout | 431 | `/workspace/step4/premature_close_data_saotojo5/x20_y-20/training_episode.hdf5` |
| 9 | [0, -10] | train | 946 | `/workspace/step4/successful_policy_data_x9j8_vz7/x0_y-10/training_episode.hdf5` |
| 10 | [20, 0] | train | 908 | `/workspace/step4/successful_policy_data_x9j8_vz7/x20_y0/training_episode.hdf5` |
| 11 | [-20, 20] | recovery_check | 620 | `/workspace/step4/candidate_recovery_extended_0se75c9v/x-20_y20/training_episode.hdf5` |
| 12 | [20, -20] | heldout | 410 | `/workspace/step4/candidate_recovery_extended_0se75c9v/x20_y-20/training_episode.hdf5` |
| 13 | [0, -10] | validation | 394 | `/workspace/step4/candidate_recovery_extended_0se75c9v/x0_y-10/training_episode.hdf5` |
| 14 | [0, -20] | train | 406 | `/workspace/step4/candidate_recovery_extended_0se75c9v/x0_y-20/training_episode.hdf5` |
| 15 | [0.0, 0.0] | train | 893 | `/workspace/step4/cube_vision_full_data_5a28urn1/position_00/vision_episode.hdf5` |
| 16 | [10.0, 0.0] | train | 901 | `/workspace/step4/cube_vision_full_data_5a28urn1/position_01/vision_episode.hdf5` |
| 17 | [0.0, 20.0] | train | 896 | `/workspace/step4/cube_vision_full_data_5a28urn1/position_02/vision_episode.hdf5` |
| 18 | [0.0, -10.0] | train | 892 | `/workspace/step4/cube_vision_full_data_5a28urn1/position_03/vision_episode.hdf5` |
| 19 | [20.0, 0.0] | train | 910 | `/workspace/step4/cube_vision_full_data_5a28urn1/position_04/vision_episode.hdf5` |
| 20 | [0.0, -20.0] | train | 891 | `/workspace/step4/cube_vision_full_data_5a28urn1/position_05/vision_episode.hdf5` |
| 21 | [-20, 20] | train | 1300 | `/workspace/step4/smoothed_vision_capture_ndgerrvl/trial_01/diagnostic_observations.hdf5` |
| 22 | [30.0, -30.0] | train | 918 | `/workspace/step4/oracle_vision_xp30ym30_d986vh_p/position_00/vision_episode.hdf5` |

## 内容
卓上・手首RGB画像、EEF位置・xyzw姿勢、指位置を入力にし、Cube−EEFを教師位置とします。HDF5の行動ラベルを今回の位置ヘッドが直接学習するわけではありません。古いエピソードの正解位置はNPZ、追加教師軌道はHDF5内ラベル、失敗記録はtrue_cube_preを使います。

モデルは[約9.46MBの単一pt](../result/models/README_JPN.md)で、データ本体とは別です。初期状態HDF5や古い特徴抽出器の重みも再学習・再評価の外部依存です。保存先を一覧化したことは、それらがGitHubで配布されていることを意味しません。
