# Model

**Model weights are not published.** This repository provides implementation, evaluation evidence, and videos. Inference requires a separate checkpoint and cannot run from the public files alone.

The evaluated checkpoint is 49,396,299 bytes (approximately 47.1 MiB), including parent-model weights and the GRU action decoder. The original remains on RunPod.

This directory publishes metadata and three Python modules; `candidate.pt` is not included. If the checkpoint is obtained separately, place it here. The modules are hash-checked by the loader and should remain unchanged.

[Manifest](manifest.json)
