# Contributing to GenPlanner

GenPlanner uses Conventional Commits and releases from the `main` branch of
`DDonnyy/GenPlanner-lib`. Python 3.11 and 3.12 are supported. The package has a
Rust extension, so local installation also needs a Rust toolchain.

## Local workflow

Install [uv](https://docs.astral.sh/uv/) and run `uv sync --all-groups`.
The main checks are:

| Task | Command |
| --- | --- |
| Tests | `make test` |
| Coverage XML | `make coverage-xml` |
| Formatting | `make format-check` |
| Lint | `make lint` |
| Documentation | `make docs` |
| Build | `make build` |
| Next release preview | `make version-next` |

`uv.lock` records the resolved dependency set. Regenerate it with `make lock`
after changing `pyproject.toml`.

## Commits and releases

Use a branch such as `fix/short-description` and a Conventional Commit title.
For squash merges, the PR title becomes the release-driving commit title:

- `feat:` raises the minor version;
- `fix:` or `perf:` raises the patch version;
- `feat!:` or a `BREAKING CHANGE:` footer raises the major version;
- `docs:`, `test:`, `ci:`, and `chore:` do not trigger a release.

Separate each point in a squash commit body with a blank line. Semantic Release
joins adjacent lines in one paragraph when it writes the changelog and release
notes.

After `Tests and Coverage` succeeds on `main`, the `Release` workflow updates
`pyproject.toml`, `rust/Cargo.toml`, `CITATION.cff`, and `CHANGELOG.md`, creates a
tag and GitHub release, builds wheels for Linux, Windows, and macOS, and publishes
to PyPI through Trusted Publishing. The `Docs` workflow then deploys through
GitHub Pages Actions. Do not change the version or create release tags manually.

The PyPI Trusted Publisher is configured for repository `DDonnyy/GenPlanner-lib`
and workflow `release.yml`. GitHub Pages should retain **Source: GitHub Actions**.
The coverage action stores badge data
on its own `python-coverage-comment-action-data` branch.

The release job must be allowed to push its version commit and tag to `main`.
If branch protection requires a pull request for every push, configure a
release credential with a ruleset bypass before enabling automated releases.
