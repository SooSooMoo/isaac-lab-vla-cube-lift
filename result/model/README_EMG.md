# Current Model

[日本語](README_JPN.md)

[manifest.json](manifest.json) records original paths and SHA-256 hashes for the four files in the final stack:

- Base model
- Visual adapter
- Gripper correction
- XY attenuation

The base model alone does not reproduce the current results. All four components are required.

Weights are copied and hash-checked during the Pod update. `.gitignore` excludes weights from ordinary additions, so GitHub may not contain all required weights. Verification scripts that use the existing Pod refer to the original experiment paths.
