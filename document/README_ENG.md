# Architecture and evaluation criteria

A shared ResNet18 processes two 200×200 RGB images. Nine robot-state values and language features from embeddings and a GRU are fused into 256 features. A 128-unit causal GRU outputs six arm values and one gripper value. Hidden state resets per episode and updates in temporal order.

The only instruction is `pick up the cube`. Neither phase labels nor step numbers enter the policy.

Criteria include lifting at least 100 mm above the settled cube height and holding for three seconds after success, with horizontal displacement within 50 mm during pick and hold. Joint limits and finite values are also checked. The control interval is 0.02 seconds and the acquisition horizon is 1300 steps.

The final adjustment solved gripper output weights using preservation constraints on successful trajectories and closure labels obtained through diagnostics. Action-overridden diagnostic successes are distinguished from autonomous learned-policy successes.
