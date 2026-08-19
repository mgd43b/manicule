# Releasing manicule

Versions are decided by [release-please](https://github.com/googleapis/release-please) from the
[Conventional Commits](https://www.conventionalcommits.org/) on `main`, and published to PyPI by
[`.github/workflows/release.yml`](.github/workflows/release.yml) over **Trusted Publishing** —
OIDC, no API token, nothing to leak from this repository's secrets.

Nothing is triggered by pushing a tag. A tag is how a release is *recorded*; making it the
trigger would let a hand-pushed tag publish a version no release PR ever reviewed.

## Two distributions, one version

| | License | What it is |
|:---|:---|:---|
| [`manicule`](https://pypi.org/project/manicule/) | MIT | The program |
| [`manicule-mlx`](https://pypi.org/project/manicule-mlx/) | GPL-3.0-or-later | The Metal backend for Apple silicon |

They are separate distributions because they are separately licensed — `manicule-mlx` links
`mlx-embeddings`, which is GPL-3.0 — and that is the whole reason the second package exists
rather than being an extra. See [README §License](README.md#license).

They are versioned **in lockstep**: one release PR bumps both, and release-please rewrites
`packages/manicule-mlx/pyproject.toml` through the `extra-files` entry in
[`release-please-config.json`](release-please-config.json). `manicule_mlx`'s plugin manifest pins
a `core_version` range against manicule, so one number answering "which mlx goes with which
manicule" is worth more than an independent cadence for a package that has no independent
purpose.

## Cutting a release

1. Land changes on `main` as Conventional Commits. This repository squash-merges, so **the pull
   request title is the commit** — which is why `.github/workflows/pr-title.yml` checks it. A
   title release-please cannot parse is a change that ships without appearing in the changelog.
2. release-please opens and maintains a PR titled **`chore(main): release X.Y.Z`**, carrying the
   version bumps and the `CHANGELOG.md` entry. Read it: the version is computed from the commit
   types, and `feat` → minor, `fix` → patch, `!` → breaking.
3. **Merge it.** That tags `vX.Y.Z`, creates the GitHub Release, and the publish job builds and
   pushes both distributions.

Full CI runs on the release PR like any other, which is where the release-specific check lands —
see *When a minor bump breaks the plugins* below.

## One-time setup

Three steps, and the order matters.

1. **GitHub:** Settings → Environments → New environment, named exactly `pypi`. Optionally add
   required reviewers, which makes each publish pause for a human.
2. **PyPI:** register a *pending* publisher for **`manicule`** at
   <https://pypi.org/manage/account/publishing/> → GitHub:

   | | |
   |---|---|
   | PyPI project name | `manicule` |
   | Owner | `mgd43b` |
   | Repository | `manicule` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. **PyPI, again:** register a second pending publisher, identical except for the project name —
   **`manicule-mlx`**. Easy to skip, and skipping it fails the release *after* `manicule` has
   already been published, which is the half-released state that cannot be undone by re-running:
   PyPI refuses a re-upload of a version it already has, so the fix is a new patch version.

The first successful run consumes both pending publishers, creates both projects and converts
them to ordinary trusted publishers.

The manifest baselines at `0.1.0` in [`.release-please-manifest.json`](.release-please-manifest.json),
so the first release PR proposes a bump from there. To publish exactly `0.1.0` instead, put
`Release-As: 0.1.0` in the body of a commit on `main`.

## When a minor bump breaks the plugins

Every plugin in this repository declares `core_version=">=0.1,<0.2"` — the six built-ins under
`src/manicule/*/plugin.py`, `manicule-mlx`, and both fixture plugins. A release-please PR bumping
to `0.2.0` therefore produces a manicule whose own parsers, storage, embedder and Metal backend
all refuse to load, each reported as an incompatible plugin, on the one commit nobody rehearses.

`tests/test_packaging.py::test_every_plugin_admits_the_running_version` fails on that PR, which
is the point: widen the pins **in the release PR itself**, so the bump and the ranges move in one
reviewed commit.

## What the publish job checks before it publishes

Worth knowing, because these are the failures that stop a release halfway rather than after it:

- `uv build --package manicule` and `--package manicule-mlx` — explicitly, one at a time.
  `packages/` also holds two plugin distributions that are test fixtures, one of which is
  deliberately hostile, and naming the packages is what keeps them off PyPI.
- The four artifacts are exactly the expected wheel and sdist for the tag, per distribution.
- Both wheels install into a clean environment with `[all]` and `manicule --version` runs.

The same checks run on every pull request as the `dist` job in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), plus two the release job does not repeat:
that a bare install — no extras — prints an install hint and exits 1 instead of raising, and that
`manicule-mlx` actually claims the `mlx` slot in the entry-point group. A backend that installs
without registering is invisible at run time and raises nothing.

## What is not published

- **No container image.** [`Dockerfile`](Dockerfile) and [`compose.yaml`](compose.yaml) are built
  and smoke-tested by CI on every pull request, and thrown away. Publishing means pushing ~3.4 GB
  per tag — mostly baked model weights — and owning a tag-retention policy, and nobody has asked.
  It is one job to add when someone does.
- **No Homebrew formula.** Not a deferral: Homebrew builds Python dependencies from **sdists**,
  and `onnxruntime` and `lancedb` publish wheels only. There is no sdist to build, so the formula
  cannot exist. `uv tool install` is the equivalent, and `brew install uv` is the only part of
  this Homebrew can help with.
- **Neither fixture plugin.** `manicule-plugin-example` and `manicule-plugin-hostile` exist to be
  installed by the dev group. `tests/test_packaging.py` fails if a workspace member is ever
  neither published nor deliberately withheld.
