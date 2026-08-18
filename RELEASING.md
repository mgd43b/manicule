# Releasing manicule

Versions are decided by [release-please](https://github.com/googleapis/release-please) from the
[Conventional Commits](https://www.conventionalcommits.org/) on `main`, and published to PyPI by
[`.github/workflows/release.yml`](.github/workflows/release.yml) over **Trusted Publishing** —
OIDC, no API token, nothing to leak from this repository's secrets.

Nothing is triggered by pushing a tag. A tag is how a release is *recorded*; making it the
trigger would let a hand-pushed tag publish a version no release PR ever reviewed.

## Cutting a release

1. Land changes on `main` as Conventional Commits. This repository squash-merges, so **the pull
   request title is the commit** — which is why `.github/workflows/pr-title.yml` checks it. A
   title release-please cannot parse is a change that ships without appearing in the changelog.
2. release-please opens and maintains a PR titled **`chore(main): release X.Y.Z`**, carrying the
   version bump in `pyproject.toml` and the `CHANGELOG.md` entry. Read it: the version is
   computed from the commit types, and `feat` → minor, `fix` → patch, `!` → breaking.
3. **Merge it.** That tags `vX.Y.Z`, creates the GitHub Release, and the publish job builds and
   pushes to PyPI.

Full CI runs on the release PR like any other, which is where the two release-specific checks
land — see *When a minor bump breaks the plugins* below.

## One-time setup

Both steps are required before the first release, and the order matters.

1. **GitHub:** Settings → Environments → New environment, named exactly `pypi`. Optionally add
   required reviewers, which makes each publish pause for a human.
2. **PyPI:** register a *pending* publisher at <https://pypi.org/manage/account/publishing/> →
   GitHub, matching the workflow exactly:

   | | |
   |---|---|
   | PyPI project name | `manicule` |
   | Owner | `mgd43b` |
   | Repository | `manicule` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The first successful run consumes the pending publisher, creates the project and converts it
   to an ordinary trusted publisher.

The manifest baselines at `0.1.0` in [`.release-please-manifest.json`](.release-please-manifest.json),
so the first release PR proposes a bump from there. To publish exactly `0.1.0` instead, put
`Release-As: 0.1.0` in the body of a commit on `main`.

## When a minor bump breaks the plugins

Every built-in plugin declares `core_version=">=0.1,<0.2"`. A release-please PR bumping the
project to `0.2.0` therefore produces a manicule whose own parsers, storage and embedder refuse
to load — each one reported as an incompatible plugin, on the one commit nobody rehearses.

`tests/test_packaging.py::test_the_builtin_plugins_admit_the_running_version` fails on that PR,
which is the point: widen the pins **in the release PR itself**, so the bump and the ranges move
in one reviewed commit. The declarations are in `src/manicule/*/plugin.py`.

## What the publish job checks before it publishes

Worth knowing, because these are the failures that stop a release halfway rather than after it:

- `uv build --package manicule` — explicitly the one package. `packages/*` holds two plugin
  distributions that are test fixtures and must never reach PyPI.
- The `dist/` contents are exactly the expected wheel and sdist for the tag.
- The wheel installs into a clean environment with `[all]` and `manicule --version` runs.

The same checks run on every pull request as the `dist` job in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), plus one the release job does not repeat:
that a bare install — no extras — prints an install hint and exits 1 instead of raising. That is
the defect the job was written for.

## What is not published

- **No container image.** [`Dockerfile`](Dockerfile) and [`compose.yaml`](compose.yaml) are built
  and smoke-tested by CI on every pull request, and thrown away. Publishing means pushing ~3.4 GB
  per tag — mostly baked model weights — and owning a tag-retention policy, and nobody has asked.
  It is one job to add when someone does.
- **No Homebrew formula.** Not a decision: Homebrew builds Python dependencies from **sdists**,
  and `onnxruntime` and `lancedb` publish wheels only. There is no sdist to build, so the formula
  cannot exist. `uv tool install` is the equivalent, and `brew install uv` is the only part of
  this Homebrew can help with.
