# syntax=docker/dockerfile:1
#
# manicule in a container.
#
# **Everything the running container needs arrives with the image.** The grammars, the model
# weights and the Python environment are all resolved while the build has a network, and the
# final stage proves it by doing a whole index-and-search with the network switched off. That
# is manicule's own rule applied to the image: pre-seed, never lazy-load. A container that
# downloads two gigabytes the first time somebody types `index` looks broken for ten minutes
# and then works, which is the worst of both.
#
# The image runs as an unprivileged user and its data directory is `0700`. That is not
# decoration either: with retained source bytes the data directory holds the corpus itself
# (docs/storage.md §7.1), and `manicule doctor` fails on a data directory anybody else can
# read. The build runs `doctor` and will not finish if it does not pass.
#
# No port is exposed and none should be. manicule serves MCP over stdio, which opens no
# socket at all.

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.9.7

# --- the environment, the grammars and the weights ------------------------------------------
#
# One builder stage, because all three want the same thing: a network, and manicule importable
# so that what is fetched is decided by manicule's own code rather than by a list copied into
# this file and left to drift.

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM python:${PYTHON_VERSION}-slim-bookworm AS build

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/manicule/venv \
    UV_FROZEN=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# The extras a running installation needs, and no more. `rerank` is deliberately absent: it is
# torch, it is gigabytes, and the retrieval profiles reach a cross-encoder through a seam that
# is simply unfilled without it. `embeddings` resolves to onnxruntime here — mlx-embeddings is
# marked for Apple Silicon in pyproject.toml and does not install on Linux at all.
ARG EXTRAS="--extra storage --extra embeddings --extra parsers --extra retrieval --extra generation --extra connectors --extra ingest --extra serve"

WORKDIR /src

# Dependencies before source, so that editing a docstring does not re-resolve the world.
# `packages/` is copied because it holds uv workspace members named by uv.lock; `--no-dev`
# means they are not installed.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages ./packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project ${EXTRAS}

COPY src ./src
COPY tools ./tools
# `--no-editable`, which is not the default. `uv sync` installs the workspace root as an
# editable install pointing at `/src`, and `/src` does not exist in the final stage — so the
# image would carry a `.pth` file naming a directory that is not there and fail at
# `import manicule`. A built wheel is what gets copied out.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable ${EXTRAS}

ENV PATH=/opt/manicule/venv/bin:$PATH

# --- tree-sitter grammars, as an installable offline bundle ---
#
# `tree-sitter-language-pack` ships no grammars in its wheel; they are fetched from a release
# bundle on first use. A container that did that would need egress to parse a `.py` file, and
# would parse it differently on a host that had no egress. `tools/build_grammar_bundle.py`
# pre-seeds the declared set, copies each library, and records the pack release, the platform
# and a SHA-256 per library — so what lands in the image is described rather than assumed.
#
# `--package` writes an installable `manicule-grammars` distribution — the module, the bundle
# as package data, and the packaging metadata. This file supplied that metadata itself until
# #62; it no longer does, and it should not: the version and description name the pack release
# and the platform the bundle is valid for, which are facts the builder knows and a Dockerfile
# would have to be told. The point of taking this route rather than the
# `MANICULE_GRAMMAR_BUNDLE` environment variable is that an installed distribution is what
# `grammar_bundle.locate()` finds with nothing configured, and nothing configured is one fewer
# thing for a deployment to get wrong.
#
# **Where the libraries come from is the build's choice, and both answers build the same
# bundle.** Until #80 there was only one: download the release, here, on every build. That made
# every merge in the repository depend on a third-party host being reachable at that moment,
# and when it was not — a transfer dropping part-way, three times in an hour — nothing could be
# merged. So a caller that already has the pack may stage it into the context and the build will
# use it, which is what CI does; `.github/workflows/ci.yml` restores it from a cache keyed on
# the pack release, so the release is fetched once per version rather than once per pull request.
#
# The staged directory is named for the pack release it holds, and this looks for the release
# *this image installs*. That is the whole safety argument: a cache left over from another
# version is not a directory this can find, so it falls through to the download rather than
# packaging libraries under a version they are not. It is the same reason the pack's own cache
# carries a version in its path.
#
# Neither branch can produce a partial bundle, which is the property worth more than the cache.
# `build_grammar_bundle.py` downloads only when asked with `--prefetch`, so the staged branch is
# a copy and a hash; and a staged directory missing a language fails naming it rather than
# writing a bundle without it. An image shipping twenty-three of twenty-four grammars would
# build, pass every check below, and silently parse the twenty-fourth as plain text.
COPY .ci/grammars /build/grammar-cache
RUN <<'GRAMMARS'
set -eu
installed=$(python -c 'from manicule.parsers import grammars; print(grammars.pack_version())')
staged="/build/grammar-cache/${installed}"
# Libraries, not merely a directory. A staged directory can exist and hold nothing this image
# can load — most obviously one staged on a developer's macOS machine, which is `.dylib` — and
# "there is nothing here I can use" and "there is nothing here" deserve the same answer: fetch
# it. What this deliberately does *not* do is second-guess a directory that does hold Linux
# libraries but is short of a language; that one goes to the builder and fails naming it.
if ls "${staged}"/*.so >/dev/null 2>&1; then
  echo "grammars: building from the staged pack for ${installed}; nothing is downloaded"
  python tools/build_grammar_bundle.py --output /build/grammars --package --source "${staged}"
else
  echo "grammars: nothing usable staged for ${installed}; fetching the release"
  python tools/build_grammar_bundle.py --output /build/grammars --package --prefetch
fi
GRAMMARS
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/manicule/venv/bin/python /build/grammars

# --- embedding weights ---
#
# BAAI/bge-m3, the configured model, as its ONNX export: about 2.3 GB. Baked in rather than
# left to a cache volume, and the trade is worth stating. Baking costs image size and a long
# `docker build`; the alternative costs a first `manicule index` that appears to hang while it
# fetches, on a machine where nobody yet trusts the software. A build is where a long step
# belongs, and the result is an image that needs no network at all.
#
# The file patterns are the union of what `manicule.embedding.artifacts` asks for from the
# ONNX export and what `manicule.embedding.cards` reads to learn the model's pooling — the
# latter imported rather than copied, because a card file added upstream must not become a
# container that fetches at run time.
ENV HF_HOME=/opt/manicule/models
RUN --mount=type=cache,target=/tmp/hf \
    HF_HOME=/tmp/hf python -c "\
from huggingface_hub import snapshot_download; \
from manicule.embedding.cards import CARD_FILES; \
print(snapshot_download('BAAI/bge-m3', allow_patterns=['onnx/*', '*.json', *CARD_FILES]))" \
    && mkdir -p "${HF_HOME}" && cp -a /tmp/hf/. "${HF_HOME}/"

# --- tiktoken vocabularies ---
#
# BPE vocabularies, none of them shipped in the `tiktoken` wheel: `cl100k_base`, the chunker's
# stand-in counter, and `o200k_base`, which both the context fitter and the generation budget
# measure prompts with. They are fetched from a Microsoft-hosted blob — which once meant a
# container that indexed happily offline and then failed on the first *search*, because that
# is the first thing to build a context.
#
# This is the same call an operator makes on any air-gapped host, and the encodings come from
# `required_encodings()` — one definition, read here, by CI and by the bundle builder, so an
# image cannot end up with the chunker's vocabulary and not the fitter's. The pre-seed asserts
# the cache afterwards rather than trusting the fetch, so a build that wrote nothing fails
# here instead of shipping an image that cannot answer.
ENV TIKTOKEN_CACHE_DIR=/opt/manicule/tiktoken
RUN mkdir -p "${TIKTOKEN_CACHE_DIR}" && python -c "\
from manicule import vocabularies; \
wanted = vocabularies.required_encodings(); \
print('tiktoken vocabularies seeded:', vocabularies.prefetch(wanted)); \
missing = vocabularies.missing_vocabularies(wanted); \
assert not missing, f'vocabularies still missing after pre-seed: {missing}'"

# --- the image ------------------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="manicule" \
      org.opencontainers.image.description="Self-hosted document search and answers, with citations that resolve." \
      org.opencontainers.image.source="https://github.com/mgd43b/manicule" \
      org.opencontainers.image.licenses="MIT"

# A fixed uid, so a bind-mounted host directory has one predictable owner to match. `--system`
# is not used: this account owns a home directory, because the grammar cache lives in it.
RUN groupadd --gid 10001 manicule \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/manicule --shell /usr/sbin/nologin manicule

COPY --from=build /opt/manicule/venv /opt/manicule/venv
COPY --from=build /opt/manicule/tiktoken /opt/manicule/tiktoken
# Owned by the account that reads them. `huggingface_hub` keeps a small resolution cache
# beside the weights and rewrites it on every load; root-owned, that becomes a "corrupted tree
# cache file … Permission denied" warning on the front of every single command, which is a
# scary-looking line about nothing.
COPY --from=build --chown=manicule:manicule /opt/manicule/models /opt/manicule/models

# `MANICULE_EMBEDDING__PROVIDER=onnx` is set here even though `manicule init` would choose it
# anyway, so that the image is correct before anybody has run `init`. It is worth knowing that
# the environment outranks the config file in manicule's settings sources, so this cannot be
# changed with `manicule config set` from inside the container — which is the right way round:
# `mlx` is Apple silicon and there is no Linux container in which it is a valid answer.
ENV PATH=/opt/manicule/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/manicule \
    HF_HOME=/opt/manicule/models \
    HF_HUB_OFFLINE=1 \
    TIKTOKEN_CACHE_DIR=/opt/manicule/tiktoken \
    MANICULE_DATA_DIR=/data \
    MANICULE_CACHE_DIR=/data/cache \
    MANICULE_CONFIG_FILE=/data/config.toml \
    MANICULE_EMBEDDING__PROVIDER=onnx

# `0700`, set here rather than left to the daemon's umask, and owned by the account that will
# write to it. A named volume mounted over this path inherits both. docs/deployment.md says
# what ends up inside it.
RUN install --directory --mode=0700 --owner=manicule --group=manicule /data

USER manicule
WORKDIR /home/manicule

# Seed the grammars out of the installed bundle, and prove the whole thing works offline.
#
# `--network=none` is the assertion, not a precaution: this step reaches for grammars, model
# weights, a database and a vector index, and it runs with no route to anything. A build that
# passes it cannot be an image that quietly fetches on first use — which is exactly the
# failure a passing build would otherwise hide.
#
# `doctor` is in here because it is the acceptance test for everything above it: it fails, not
# warns, on a data directory that is group- or world-readable, so an image that ran as root
# and left a `0755` data directory would stop here.
RUN --network=none <<'SMOKE'
set -eux

# `manicule doctor` exits 0 whenever it managed to *produce* a diagnosis, whatever the
# diagnosis says — that is the exit-status contract in docs/surfaces.md §2, and it is right:
# the operation succeeded. It also means `manicule doctor` on its own asserts nothing. The
# state has to be read out of the envelope, or this whole step is a build that passes while
# reporting `overall: failing` in its own log.
assert_healthy() {
  manicule doctor
  manicule --json doctor | python -c "
import json, sys
checks = json.load(sys.stdin)['data']['checks']
bad = [check for check in checks if check['state'] == 'failing']
if bad:
    raise SystemExit(f'doctor is failing inside the image: {bad}')
"
}

# Seeding the grammars out of the installed bundle, through the command an operator would run
# rather than through a call into manicule's internals. This step used to reach for `prefetch`
# directly because nothing shipped called it; `doctor --fix` is now that caller, so the image
# is built the way an air-gapped host is repaired. With `--network=none` in force, the grammars
# that land in the cache can only have come out of the installed distribution.
#
# The state is read out of the envelope, and `ok` specifically: a missing grammar is *degraded*
# — an installation with no code in its corpus is fine as it is — so `assert_healthy` below
# would not catch an image that shipped without them, and this image is meant to have them.
#
# The vocabularies `doctor --fix` also seeds need no equivalent line here, and the reason is
# the severity: a missing vocabulary is *failing*, because no corpus can be searched without
# one, so `assert_healthy` catches an image that shipped without them on its own.
manicule --json doctor --fix | python -c "
import json, sys
checks = {check['name']: check for check in json.load(sys.stdin)['data']['checks']}
grammars = checks['grammars']
print('grammars:', grammars['detail'])
if grammars['state'] != 'ok':
    raise SystemExit(f'the image did not seed its grammars from the bundle: {grammars}')
"
# Against the real `/data` first, before anything is redirected: this is the check that the
# directory the image ships — the one a named volume inherits its ownership and mode from —
# is one this account owns and nobody else can read. `manicule` creates a data directory
# `0700` itself, so what this catches is a `/data` that already existed with a looser mode,
# which is what `mkdir -p /data` in a Dockerfile leaves behind. Emptied afterwards so the
# image ships an empty data directory rather than a database.
assert_healthy
find /data -mindepth 1 -delete

export MANICULE_DATA_DIR=/tmp/smoke/data
export MANICULE_CACHE_DIR=/tmp/smoke/cache
export MANICULE_CONFIG_FILE=/tmp/smoke/config.toml
mkdir -p /tmp/smoke/corpus
printf 'def retry(attempts: int) -> None:\n    """Retry up to `attempts` times with a fixed backoff."""\n' > /tmp/smoke/corpus/retry.py
printf '# Retries\n\nThe retry policy waits one second between attempts.\n' > /tmp/smoke/corpus/retries.md
manicule init
assert_healthy

# A write command goes to the server, so the smoke test starts one — which is the point of
# running it this way rather than a workaround for it. `manicule index` used to run in this
# shell and take the data directory itself; a served manicule holds that directory for its
# whole life, so indexing here is now a proxied command over the control socket. If the socket
# does not bind inside this image, or the proxy cannot reach it, this step fails at the point
# an operator would hit it rather than at the point somebody reads the documentation.
#
# `--transport http` on loopback rather than `stdio`, because a stdio server reads the protocol
# from its own stdin and would exit immediately in a build step. `--network=none` is still in
# force: a loopback socket opens no route out of the container.
manicule serve --transport http --port 8765 &
SERVER=$!

# Waiting on the socket rather than on a sleep. The socket is the same liveness signal the
# command line proxies on, so this waits for exactly the condition the next command needs.
SOCKET_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}}/manicule-$(id -u)"
for _ in $(seq 1 100); do
  if [ -n "$(ls -A "$SOCKET_DIR" 2>/dev/null)" ]; then break; fi
  sleep 0.1
done
[ -n "$(ls -A "$SOCKET_DIR" 2>/dev/null)" ] || {
  echo "the server never opened its control socket in $SOCKET_DIR" >&2
  exit 1
}

manicule index /tmp/smoke/corpus
# `--json` so the assertion is on the result rather than on prose that wraps in a terminal
# box. A search that returns nothing exits 0 with an empty list, so the count is the check.
#
# `search` deliberately runs while the server is still up: it takes no writer lock and must
# keep answering with one holding the data directory, which is the half of this arrangement
# that would be worst to get wrong.
manicule --json search "retry policy" --top 2 | python -c "
import json, sys
hits = json.load(sys.stdin)['data']['hits']
print('search hits:', [hit['title'] for hit in hits])
if not hits:
    raise SystemExit('the image indexed two documents and then matched neither')
"
manicule stop
wait "$SERVER" 2>/dev/null || true
rm -rf /tmp/smoke
SMOKE

VOLUME ["/data"]

# No `EXPOSE`, and no default that binds anything. The default command serves MCP over stdio,
# which opens no socket. `manicule start --transport http` serves the HTTP API on
# `security.transport.port`; publishing it is an operator's decision, and docs/deployment.md
# §4 says what it requires of you.
ENTRYPOINT ["manicule"]
CMD ["--help"]
