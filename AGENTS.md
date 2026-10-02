# GenPlanner repository guide

Guidance for coding agents and contributors working in this repository. Commands
below use the project CLI and do not depend on a particular editor or agent.

## Project overview

GenPlanner generates territorial zones and roads from polygonal territory inputs.
The public Python package is in `src/genplanner`; its Voronoi optimizer is a Rust
extension built with maturin. Python 3.11 and 3.12 are supported. Dependency
groups and their resolved versions live in `pyproject.toml` and `uv.lock`.

## Commands

```bash
uv sync --all-groups       # install Python dependencies and build the extension
make test                  # Python regression tests
make coverage-xml          # coverage.xml for CI
make format-check          # isort and black, without edits
make lint                  # pylint
make docs                  # strict Sphinx build
make build                 # source distribution and wheels
make version-next          # dry-run Semantic Release version

cargo test --manifest-path rust/Cargo.toml
```

Use `make format` for formatting. The source and tests follow a 120-character
line length; isort skips `__init__.py`. Run relevant tests after editing Python
or Rust. A Rust change also needs a rebuilt extension before Python tests can
exercise the new native code.

## Architecture

```text
src/genplanner/
  __init__.py                  public imports
  _config.py                   logger and generation defaults
  main/genplanner.py           GenPlanner API, task queue, process workers
  main/init_validation.py     input preparation, exclusions, roads, fixed zones
  tasks/                      territory, block and polygon subdivision
  utils/geom_utils.py         geometry normalization and road splitting
  zones/                      zone definitions and default sets
  zone_relations/             adjacency and forbidden-neighbor rules
  errors/                     public exceptions
rust/src/
  lib.rs                      PyO3 entry point and optimization loop
  voronoi2.rs                 Voronoi geometry and loss calculation
  voronoi_core.rs             locally maintained Voronoi core
tests/                        small, offline regression tests
docs/source/                  Sphinx source and API pages
```

`GenPlanner` prepares input GeoDataFrames in a projected CRS, removes excluded
areas, splits by input roads, and accounts for existing zones and fixed points.
`features2terr_zones()` and `features2terr_zones2blocks()` enqueue subdivision
tasks. `split_queue()` runs them serially or with `ProcessPoolExecutor` according
to `parallel` and `parallel_max_workers`. The task layer calls
`genplanner._rust.optimize_territory_zoning()` to place Voronoi sites; final
geometries and roads are returned in the original CRS.

The Voronoi core was brought into `rust/src/voronoi_core.rs` so failures can be
fixed here. Keep its attribution in `rust/third_party/del-msh-core-LICENSE`.
Python exceptions should carry actionable input context; never let a native
panic cross the PyO3 boundary.

## Tests and data

`tests/test_generation_regressions.py` covers geometry preservation, duplicate
roads, bounded generation, parallel execution, and degenerate Voronoi input.
`tests/test_public_api.py` checks the package surface and version alignment.
The large 1551/1552 reproductions are local diagnostics, not part of the test
suite. `reports/scenario_1552/` contains local visualizations and is ignored by
Git. Do not commit input data, generated maps, logs, or benchmark outputs.

## CI, documentation, and releases

`quality.yml` runs lint, tests, and coverage. `coverage.yml` posts coverage for
fork pull requests. A successful main-branch test run starts `release.yml`;
Semantic Release updates the version, changelog and tag, builds distributions,
and publishes to PyPI through Trusted Publishing. `docs.yml` builds pull request
previews and deploys released documentation with GitHub Pages Actions. Pages
uses **Source: GitHub Actions**; there is no `gh-pages` deployment branch.

The version fields in `pyproject.toml`, `rust/Cargo.toml`, and `CITATION.cff`
are maintained by Semantic Release. Follow `CONTRIBUTING.md` for Conventional
Commits. Separate points in commit bodies with blank lines so release notes
render correctly. Do not change version fields or create release tags manually.
Do not commit or push unless the user explicitly asks; when asked to commit,
do not add an agent co-author trailer.
