# Data collection

[日本語](README_JPN.md)

## Two sources
- Oracle control uses true position to record a successful approach-to-lift-and-hold trajectory. Oracle success is not counted as learned-perception success.
- Failed visual runs record the actual observations and synchronized true Cube position. Ground truth supplies labels only, not control.

Final training includes a failed (-20,+20 mm) run and a complete successful oracle (+30,-30 mm) run. The failed (+30,-30 mm) capture used by a rejected candidate is excluded from final training.

## Alignment and schemas
Observations and labels precede the same `env.step`; targets are `cube_pre − eef_pre`. Standard observations use `data/demo_0/obs`; complete oracle labels use `data/demo_0/labels/cube_minus_eef_env_m`. Failure captures use `obs` and `true_cube_pre`. Older episodes reference separate NPZ labels.

See the [23-episode inventory](../../datasets/README_ENG.md) and [dependency manifest](../../datasets/dependency_manifest.json). Collection-only scripts and HDF5 bodies are not included in this published snapshot. This folder documents the method.
