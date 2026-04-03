# Asset Inventory

This register tracks bundled robot models and geometry assets present in the public ACTOROB working
tree. It is intended as a release-facing inventory of what ships with the repository and which
artifacts are generated at runtime.

## Status Legend

- `Bundled`: source asset intentionally included in the public repository.
- `Generated`: artifact is produced during local runs and should not be committed as a source asset.

## Bundled Asset Families

| Path | Purpose | Status | Current evidence |
| --- | --- | --- | --- |
| `robots/dog/dog.xml` | Default quadruped MJCF used by trajectory and inverse-design examples. | Bundled | Public release model used by the default examples and tests. |
| `robots/dog/meshes/*.stl` | Quadruped geometry referenced by `robots/dog/dog.xml`. | Bundled | Geometry shipped with the public quadruped example. |
| `robots/humanoid_pry/humanoid_pry.xml` | Humanoid example MJCF. | Bundled | Public in-repo humanoid example. |
| `robots/humanoid_pry/humanoid_pry_merged.xml` | Preprocessed/merged humanoid model. | Bundled | Stored in-repo for experimentation; not part of the active Python examples. |
| `robots/humanoid_pry/meshes/*.stl` | Geometry for `humanoid_pry`. | Bundled | Geometry shipped with the public humanoid example. |
| `robots/humanoid_ryp/humanoid_ryp.xml` | Humanoid example MJCF. | Bundled | Public in-repo humanoid example. |
| `robots/humanoid_ryp/humanoid_ryp_merged.xml` | Preprocessed/merged humanoid model. | Bundled | Stored in-repo for experimentation; not part of the active Python examples. |
| `robots/humanoid_ryp/meshes/*.stl` | Geometry for `humanoid_ryp`. | Bundled | Geometry shipped with the public humanoid example. |
| `robots/humanoid_yrp/humanoid_yrp.xml` | Humanoid example MJCF. | Bundled | Public in-repo humanoid example. |
| `robots/humanoid_yrp/humanoid_yrp_merged.xml` | Preprocessed/merged humanoid model. | Bundled | Stored in-repo for experimentation; not part of the active Python examples. |
| `robots/humanoid_yrp/meshes/*.stl` | Geometry for `humanoid_yrp`. | Bundled | Geometry shipped with the public humanoid example. |
| `robots/pendulum/pendulum.xml` | Simple pendulum example used for local demos. | Bundled | Minimal in-repo example model without external meshes. |

## Public-Tree Notes

- The public working tree is treated as the release source of truth for bundled robot assets.
- This inventory stays at the asset-family level and does not treat internal file names or MJCF body
  names as provenance metadata.
- The `_merged.xml` humanoid variants are currently stored in-repo but are not referenced by the
  active Python example paths.

## Generated Artifacts

The following artifacts are runtime outputs and should remain untracked:

- `outputs/*.pkl`
- `outputs/*.html`
- `logs/trajectories/*`
- generated inverse-design MJCF variants such as `robots/dog/dog_custom_open_invdes_*.xml`

## Audit Notes

- As of this audit, the active examples and tests reference `robots/dog/dog.xml`, while the
  `_merged.xml` humanoid variants are not referenced by the current Python code paths.
- Keep this inventory aligned with the public asset set whenever files are added, renamed, or
  removed.
