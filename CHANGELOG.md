# CHANGELOG

<!-- version list -->

## v1.0.5 (2026-10-02)

### Bug Fixes

- **release**: Include declared licenses in source distributions
  ([`f475e17`](https://github.com/DDonnyy/GenPlanner-lib/commit/f475e173e7e87ca3f2da1f97812909e343572ff5))

Include the Russian license explicitly in maturin source archives.

Validate declared license files before tagging and publishing a release.

Build and inspect the source archive in CI before triggering release.


## v1.0.4 (2026-10-02)

### Bug Fixes

- Stabilize Voronoi splitting and large territory generation
  ([`b89c98f`](https://github.com/DDonnyy/GenPlanner-lib/commit/b89c98f6c5150ad42ab36f3928e1b7372b5739e5))

Maintain the Voronoi core locally and return recoverable errors for degenerate sites and
  intersections instead of propagating native panics.

Preserve split block geometry when clip indices have gaps, deduplicate roads and exclusions, and
  validate generated zones before returning them.

Bound optimization attempts and generation time, use process workers for parallel tasks, and avoid
  the expensive global clip during final road splitting.

Add offline regressions for Voronoi failures, geometry preservation, time limits, parallel
  generation, and public API behavior.

- **release**: Run Semantic Release with current Rust toolchain
  ([`b9172e1`](https://github.com/DDonnyy/GenPlanner-lib/commit/b9172e1852e05d77ef410e01f18a721e95863077))

Execute Python Semantic Release on the Ubuntu runner so the source distribution build uses Cargo
  with 2024 edition support.

Preserve release detection and artifact outputs while keeping the build before the version commit
  and tag.

- **release**: Stamp TOML package versions and locate Rust manifest
  ([`6e2c362`](https://github.com/DDonnyy/GenPlanner-lib/commit/6e2c3622ff1c2398b5834c0ed7977c20489bad51))

Use structured TOML version updates so Semantic Release changes the project and crate versions
  without rewriting the pyo3 dependency.

Point the source distribution build at rust/Cargo.toml from the repository root.

### Code Style

- Format generation regression tests with Black
  ([`a861695`](https://github.com/DDonnyy/GenPlanner-lib/commit/a8616951105d535f7a5639adbaf9c106628140fa))

Apply Black 25.12.0 formatting so the CI format check passes.

### Continuous Integration

- Modernize quality, release, and Pages workflows
  ([`4cecbc4`](https://github.com/DDonnyy/GenPlanner-lib/commit/4cecbc4dacb8b6e57f45e5545097ae384fc19a83))

Run lint, Python compatibility, and coverage checks with locked uv dependency groups.

Release after successful main-branch tests, publish distributions through PyPI Trusted Publishing,
  and deploy documentation with GitHub Pages Actions.

Add Semantic Release configuration, changelog and citation metadata, contributor guidance, and Git
  hygiene for generated artifacts.

### Documentation

- Refresh guides and add repository agent instructions
  ([`9b7c2f2`](https://github.com/DDonnyy/GenPlanner-lib/commit/9b7c2f27cc267fa747c31466b67f4d662a2ab275))

Update the API pages, examples, Python support notes, configuration reference, and citation
  guidance.

Add autosummary templates and repository-specific commands, architecture, testing, and release
  guidance in AGENTS.md.

Correct documentation references and formatting in the public zone and relation modules.


## v1.0.3 (2026-04-09)

Release published before automated changelog generation was enabled.
