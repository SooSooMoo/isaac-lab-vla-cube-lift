## Reference: Comparison of Imitation-Learning Methods

Visuomotor imitation learning combines the following two concepts:

- **Visuomotor learning:** generating robot actions from visual observations
- **Imitation learning:** learning to reproduce expert behavior

| Method | Overview | Relationship to this project |
|---|---|---|
| Behavior Cloning (BC) | Uses supervised learning with expert observations and actions | Core training approach |
| BC-RNN | Maintains recurrent state from observation history and predicts the next action | Used in this project |
| BC-GMM | Predicts a mixture distribution over possible actions instead of a single action value | Used in this project |
| DAgger | Runs the learned policy and adds expert actions in failure-prone states | Explored before the main experiment; not used for the primary result |
| Interactive Imitation Learning | A person or expert intervenes during execution and supplies corrective actions | Not used |
| Inverse Reinforcement Learning | Infers a reward function from expert behavior | Not used |
| GAIL | Uses adversarial learning to generate expert-like behavior | Not used |
| Diffusion Policy | Uses a diffusion model to generate action sequences | Not used |
| Transformer / ACT | Uses a Transformer to predict multi-step action sequences | Not used |

### Method Used in This Project

```text
Imitation Learning
└── Behavior Cloning
    └── BC-RNN
        └── GMM action distribution
            └── Image-based visuomotor policy
```

The method used in this project can be summarized as:

> Visuomotor imitation learning using a BC-RNN-GMM policy with two RGB camera observations and robot proprioception.

