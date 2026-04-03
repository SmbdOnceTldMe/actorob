# Changelog

All notable repository changes intended for public consumers should be recorded in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses a
simple `Added` / `Changed` / `Fixed` structure to keep release notes readable for research users.

## [Unreleased]

## [0.1.0] - 2026-03-25

### Added

- A minimal quadruped example covering the `walk`, `upstairs`, and `jump_forward` tasks.
- Public open-source preparation materials: contribution guidelines, citation metadata, issue
  templates, release checklist, reproducibility notes, and asset provenance tracking.
- CI validation for tests, lint checks, packaging sanity, wheel installation, and a minimal
  inverse-design smoke test across the release matrix.
- Lazy-import regression coverage for public modules that should not pull optional heavy dependencies
  at import time.
- A fresh-workspace README quickstart validation script for the documented setup path.

### Changed

- The README now documents a canonical `pixi`-based setup path and a lightweight optional `pip`
  extra flow for reporting/progress helpers.
- Project metadata now points to the public project page, repository, DOI, and preferred paper
  citation.
- The bundled asset register now reflects the anonymized public working tree and generated-artifact
  boundaries.
- Public package entry points defer optional heavy imports until the relevant feature is actually
  used.
