## 参考：模倣学習手法の比較

「視覚運動模倣学習」は、次の2つを組み合わせた考え方です。

- **視覚運動学習（Visuomotor Learning）**：画像などの視覚情報からロボット動作を生成する
- **模倣学習（Imitation Learning）**：教師の行動をまねて学習する

| 種類 | 概要 | 今回との関係 |
|---|---|---|
| Behavior Cloning（BC） | 教師のObservationとActionを使用して教師あり学習を行う | 今回の基本方式 |
| BC-RNN | 過去のObservationを内部状態として保持し、次のActionを予測する | 今回使用 |
| BC-GMM | Actionを単一値ではなく、複数の確率分布として予測する | 今回使用 |
| DAgger | 学習方策を動かし、失敗しそうな状態で教師Actionを追加する | 本プロジェクトの前段階で試行。今回の主成果では未採用 |
| Interactive Imitation Learning | 実行中に人や教師が介入し、修正Actionを教える | 未採用 |
| Inverse Reinforcement Learning | 教師行動から報酬関数を推定する | 未採用 |
| GAIL | 敵対的学習によって教師らしい行動を生成する | 未採用 |
| Diffusion Policy | 拡散モデルを使用して行動列を生成する | 未採用 |
| Transformer / ACT | Transformerを使用して複数stepの行動列を予測する | 未採用 |

### 今回使用した方式

```text
模倣学習
└── Behavior Cloning
    └── BC-RNN
        └── GMMによるAction分布
            └── 画像を使用するVisuomotor Policy
```

今回の学習方式は、次のように表現できます。

> 2視点のRGB画像とロボット状態を入力とする、BC-RNN-GMMによる視覚運動模倣学習

