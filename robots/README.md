# Robot Assets

This directory contains robot MJCF models, related geometry assets, and a small local pendulum
example used by ACTOROB examples and tests.

## Families

- `dog/`: default quadruped model used by the trajectory optimizer and inverse-design examples.
- `humanoid_pry/`, `humanoid_ryp/`, `humanoid_yrp/`: humanoid model variants stored in-repo for
  experimentation and demos.
- `pendulum/`: a simple in-repo example model used by `examples/dummy_pendulum.py`.

## Important Release Note

See [`../docs/ASSET_PROVENANCE.md`](../docs/ASSET_PROVENANCE.md) for the public asset inventory and the list
of generated files that should remain out of version control.

The inventory is intentionally maintained at the asset-family level. Internal file names or MJCF
body names should not be treated as provenance metadata.

## Generated Files

Some workflows generate temporary MJCF variants next to source assets, for example
`dog_custom_open_invdes_*.xml`. These files are generated artifacts and must not be treated as
source models.
