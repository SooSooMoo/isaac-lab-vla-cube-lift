# Third-Party Notices

This repository contains original project code and materials licensed under the BSD 3-Clause License in the root `LICENSE` file, except where a file or component states otherwise.

Third-party software, source-derived files, trademarks, and assets remain subject to their respective licenses and copyright notices.

## Isaac Lab

This project uses and adapts examples, configuration patterns, and APIs from NVIDIA Isaac Lab.

- Project: Isaac Lab
- Repository: https://github.com/isaac-sim/IsaacLab
- Primary license: BSD 3-Clause
- Additional components: Some Isaac Lab components, including certain mimic-related components, may use Apache License 2.0 or other terms identified by the upstream project.

Copyright notices and SPDX identifiers present in files derived from Isaac Lab must be retained. The root license of this repository does not replace the upstream license notices.

## Robomimic

This project uses Robomimic for behavior-cloning training and policy evaluation.

- Project: Robomimic
- Repository: https://github.com/ARISE-Initiative/robomimic
- License: MIT License

Robomimic is an external dependency and is not relicensed by this repository.

## NVIDIA Isaac Sim and Omniverse Kit

The simulation, rendering, and sensor workflows require NVIDIA Isaac Sim and Omniverse Kit.

- License information: https://docs.isaacsim.omniverse.nvidia.com/latest/common/legal.html

Isaac Sim and Omniverse Kit include components distributed under NVIDIA and third-party license terms. They are not redistributed in this repository. Users are responsible for obtaining Isaac Sim and accepting the applicable NVIDIA license agreements.

## Other Dependencies

The environment also uses Python and GPU software packages such as PyTorch, Torchvision, Gymnasium, NumPy, HDF5/h5py, MoviePy, and their transitive dependencies. Each package remains under its own license. The requirements snapshot is provided for reproducibility and does not modify those licenses.

## Generated Data and Media

The included HDF5 sample, evaluation summaries, action trace, screenshots, and MP4 files were generated as outputs of this project workflow. They may contain rendered representations produced using Isaac Sim assets and software. Use of the underlying NVIDIA software and assets remains governed by the applicable NVIDIA terms.

## Trademarks

NVIDIA, Isaac Sim, Omniverse, Isaac Lab, Franka, Robomimic, and other names may be trademarks of their respective owners. Their use here is for identification and technical description only and does not imply endorsement.
---

# 第三者ソフトウェアに関する通知（日本語）

本リポジトリ独自のコードと成果物には、別記がない限りルートの英語版 `LICENSE` にあるBSD 3-Clause Licenseを適用します。第三者コード、依存関係、商標、アセットには、それぞれのライセンスと著作権表示が適用されます。

## Isaac Lab

本プロジェクトはIsaac Labのサンプル、設定パターン、APIを使用・参考・改変しています。

- リポジトリ: https://github.com/isaac-sim/IsaacLab
- 主なライセンス: BSD 3-Clause
- 一部コンポーネントにはApache License 2.0など別条件が適用される場合があります。

Isaac Lab由来ファイルの著作権表示とSPDX識別子は保持します。本リポジトリのライセンスは上流の表示を置き換えません。

## Robomimic

Behavior Cloningの学習と評価にRobomimicを使用しています。

- リポジトリ: https://github.com/ARISE-Initiative/robomimic
- ライセンス: MIT License

Robomimicは外部依存関係であり、本リポジトリでは再ライセンスしません。

## NVIDIA Isaac SimおよびOmniverse Kit

シミュレーション、レンダリング、センサー処理にはIsaac SimとOmniverse Kitが必要です。

- ライセンス情報: https://docs.isaacsim.omniverse.nvidia.com/latest/common/legal.html

これらの本体は本リポジトリに含めません。利用者は別途入手し、適用されるNVIDIAライセンスへ同意する必要があります。

## その他

PyTorch、Torchvision、Gymnasium、NumPy、HDF5/h5py、MoviePyなどには各ライセンスが適用されます。requirements snapshotは再現性のための情報であり、各ライセンスを変更しません。

HDF5サンプル、評価結果、行動列、PNG、MP4は本プロジェクトで生成した出力です。ただし、Isaac Simのソフトウェアとアセットの使用には該当するNVIDIAの条件が適用されます。

NVIDIA、Isaac Sim、Omniverse、Isaac Lab、Franka、Robomimicなどの名称は、識別と説明のために使用しており、各組織による推薦を意味しません。