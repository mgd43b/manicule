# Contributing

## Definition of Done

A change is done when every one of these is true. Not most of them.

1. **`ruff check .` and `ruff format --check src tests packages` pass.**
2. **`pyright` passes in strict mode.** No new `# pyright: ignore` without a comment on the
   same line saying why the checker is wrong.
3. **`pytest` passes**, with tests at the levels described below.
4. **CI is green on the pull request.** Not "green except the flaky one" — a test that fails
   intermittently is a bug in the test, and it is fixed in the same change.
5. **`/code-review high` has been run against the pull request and every finding is
   resolved in that same pull request.** Not noted, not filed — fixed. If the review is
   unavailable, run `/security-review` or `/review` instead and say in the pull request
   which one you actually ran. A review that did not happen is never reported as clean.
6. **No new `TODO`, `FIXME` or `XXX`.** Ruff fails the build on them. If it is worth
   marking, it is worth doing; if it is not worth doing, delete the comment.
7. **The documents still describe the code.** A change that makes a document wrong fixes the
   document.
8. **The pull request title is a Conventional Commit.** `fix: keep the previous revision
   searchable after a failed refresh`, not `Fix the refresh bug`. This is not a style rule:
   the repository squash-merges, so the title *is* the commit on `main`, and release-please
   reads it to decide the next version and write the changelog. A title it cannot parse is a
   change that ships without being mentioned anywhere. `feat` bumps the minor, `fix` the
   patch, a trailing `!` marks a breaking change; `.github/workflows/pr-title.yml` checks it
   and re-runs when you edit the title. See [`RELEASING.md`](RELEASING.md).

Run the whole gate locally before pushing:

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check src tests packages
uv run pyright
uv run pytest
```

Three suites need a machine resource that is not in the repository, and all three **skip**
without it rather than failing — right on a laptop, wrong in CI, so CI pre-seeds and then
requires them:

```bash
uv run manicule doctor --fix                      # grammars, and the BPE vocabularies
uv run tools/prefetch_embedding_models.py --mlx   # add --full for BAAI/bge-m3, ~4.6 GB
```

`doctor --fix` seeds two things because two libraries ship neither: the tree-sitter grammars,
and the 5.3 MB of BPE vocabulary `tiktoken` fetches from a blob store on first use. manicule
never downloads a vocabulary while answering a question, so without the pre-seed the retrieval
suites refuse rather than fetching — which is the same refusal an operator gets, and the
reason it is an install step rather than a surprise at the first query.

`REQUIRE_EMBEDDING_MODELS` names the models that must be present rather than merely welcome —
a comma-separated list, or `all`. CI sets it to exactly what it pre-seeded. If you are touching
embeddings, run `REQUIRE_EMBEDDING_MODELS=all uv run pytest` at least once: a skipped
conformance suite reports green while checking nothing.

`REQUIRE_GRAMMAR_BUNDLE` is the same switch for the grammars. The offline-bundle suite builds a
real bundle out of the pre-seeded cache and proves an air-gapped install can parse code; with an
empty cache it skips. Set it to anything non-empty — CI does — and a missing grammar fails
instead. If you are touching parsing, run `REQUIRE_GRAMMAR_BUNDLE=1 uv run pytest` at least once.

To produce a bundle for a host with no network access, run this on a machine that has one, and
copy the directory over:

```bash
uv run tools/build_grammar_bundle.py --output dist/grammars              # a directory to copy
uv run tools/build_grammar_bundle.py --output dist/pkg --package         # an installable one
MANICULE_GRAMMAR_BUNDLE=/path/to/grammars uv run manicule doctor --fix   # seeds from the copy
uv pip install /path/to/pkg                                              # or install it, and
uv run manicule doctor --fix                                             # need no variable
```

A bundle is valid for one platform and one `tree-sitter-language-pack` release, and manicule
refuses one built for anything else rather than loading it. See
[`docs/parsing.md`](docs/parsing.md#811-the-offline-bundle).

`REQUIRE_VOCABULARY_BUNDLE` is the third switch, for the BPE vocabularies. Its suite builds a
real bundle out of the pre-seeded `tiktoken` cache and proves an air-gapped install can
*answer* — with the cache redirected and the network cut, separately and together. The
vocabulary bundle is platform-independent, so one build serves every host:

```bash
uv run tools/build_vocabulary_bundle.py --output dist/vocabularies   # or --package
MANICULE_VOCABULARY_BUNDLE=/path/to/vocabularies uv run python -c \
  "from manicule import vocabularies; print(vocabularies.prefetch(vocabularies.required_encodings()))"
```

It is not a `MANICULE_` variable, and that is deliberate — the test environment clears that
whole namespace before each test, so a switch living inside it is deleted before it is read.

### `.test_durations`, and when it needs regenerating

CI runs the Python 3.14 suite in three shards, balanced on the committed
`.test_durations`. **You do not need it to run the tests** — `uv run pytest` ignores it
entirely, and it is read only when `--splits` is passed.

It is checked in rather than measured per run because the measurement costs a full suite, and
it goes stale slowly and harmlessly: a test it has never heard of is given the average duration
rather than dropped, so a stale file costs *balance* and can never cost coverage. Regenerate it
when the shards have visibly drifted apart — each job prints its own `--durations=25`, so a
shard that has become the slow one says so — or after a change that moves a lot of test time,
such as adding a corpus:

```bash
uv run pytest --store-durations          # rewrites .test_durations from a full local run
```

Relative cost is what balances the shards, so a laptop's numbers are fine; they need not match
the runner's. `tests/test_ci_test_sharding.py` holds the shard count in the workflow and its
matrix to the same number, because a `--splits` raised without the matrix runs part of the
suite and reports green.

### Measuring candidate CI shard widths

Refreshing `.test_durations` tells `pytest-split` how to balance the suite; it does not tell us
how many GitHub runners make the pull-request critical path shortest. Run the manual **CI shard
benchmark** workflow after a refresh to measure the complete Python 3.14 test job at widths 3,
4, 6, and 8. It uses the same dependency sync, grammar/vocabulary pre-seeding, coverage, and
least-duration split as ordinary CI, so its job durations are comparable rather than optimistic
local stopwatches.

For each width, compare the slowest job's total time (the PR wait) and the sum of all job times
(the runner cost). Each job summary also records pytest-only seconds, which separates test work
from setup and collection. The workflow is manual because one benchmark dispatch launches 21
jobs; it must not make routine pull requests more expensive. Only after choosing a width should
the regular matrix and `--splits` value move together.

## Nothing is deferred

**Everything in scope happens in this change.** Adjacent bugs, dead code, stale
documentation and review findings are fixed now, in the pull request that touched them.

The bar for filing a follow-up ticket instead of fixing something is not "this is a separate
concern". It is **"this is genuinely infeasible within this change's scope"**. A follow-up
ticket is an admission that something could not be done — it is not a way of routing work to
later.

One clarification, because it is otherwise applied backwards. A pull request that changes
only *design documents* legitimately files implementation tickets, because implementation
cannot happen inside a document. A pull request that changes *code* and files a ticket for
something it could have fixed is exactly the failure this rule exists to prevent.

## What tests are required

Tests carry the reasoning. A test name says what is true; its docstring says why anyone
should care. `test_parse_returns_blocks` is a description of the code — prefer
`test_a_table_survives_chunking_whole`, and explain in one line what breaks if it does not.

| Level | Required for | What it looks like |
|---|---|---|
| **Unit** | Every type with a validation rule, every function with a branch | Direct, in `tests/test_<module>.py` |
| **Contract** | Every implementation of a protocol in `manicule.core.protocols` | Import the suite from `manicule.testing` and run it against your component |
| **Negative** | Every guard | A fake that breaks the rule, and proof the guard catches it. See `tests/fakes.py` |
| **Integration** | Every subsystem boundary | Real components through the container, not mocks of your own code |
| **Boundary** | Any new dependency | `tests/test_import_boundary.py` must still pass |

Three obligations are worth stating separately, because each guards a failure that is silent
by nature — nothing raises, and every answer is quietly wrong.

**Anchors round-trip.** Every parser must pass `assert_parser_contract`, which resolves each
block's anchor and checks it returns the text that block claims. A citation pointing at a
page that does not exist looks exactly like one that does.

**Nothing hardcodes a vector dimension.** Every vector store must pass
`assert_vector_store_is_dimension_agnostic`, which exercises it at two unusual dimensions,
and `assert_vector_store_rejects_foreign_vectors`, which checks it refuses a *different model
of the same size*. A dimension check alone passes that case and ruins the index.

**Nothing embeds text the model will not read.** Every path that embeds stored chunks calls
`require_within_context` first, and must pass `assert_refuses_oversized_chunks`. This is
aimed at re-embed, which reads stored `embed_text` without re-chunking, so the chunker's own
budget refusal never runs — and the embedding fingerprint is unchanged by a sequence limit
that *fell*, so nothing else fires either. Past the limit the input is dropped silently, and
the stored vector describes an opening fragment while the chunk still claims all of its text.

**Retrieval stages are uniform and pure.** Candidates in, candidates out, input untouched, a
new list returned. `assert_retrieval_stage_contract` checks all four. A stage that mutates
its input makes the pipeline order-dependent, and an order-dependent pipeline cannot be
compared with another one — which is the whole basis of the evaluation harness.

## Architecture rules

**Core carries no implementation dependencies.** `import manicule` must not pull in a vector
store, a model runtime, an HTTP client or a web framework. Implementations arrive as plugins
and their imports live inside the factories that build them.
`tests/test_import_boundary.py` fails the build otherwise.

**Built-in components register through the public entry-point path.** There is no shorter
internal route. The extension mechanism is the one manicule itself uses, so it cannot quietly
stop working while everything still runs.

**No god function.** The container is assembled from what plugins register. Adding a
component means writing a plugin, not editing a startup routine. If you find yourself adding
a branch to `manicule/container/container.py` for one feature, that is the signal.

**Configuration is declarative and rejects what it does not understand.** A setting that
appears to be in force and silently is not is worse than one that fails at startup.

**Misconfiguration fails before construction.** Anything configuration names must be checked
against what is installed, at startup, with the alternatives listed. Not at the first
document that needs it, by which point a corpus has already been indexed differently on this
machine than on another.

**Two types are locked.** `Anchor` is locked once a corpus has been ingested — changing it
invalidates every stored citation. `RetrievalStage` **is** locked: the evaluation harness
exists (`docs/evaluation.md`) and stores the stage list in every preference record, so widening
it invalidates every recorded result. Both are marked in the source, and a change to either
says so loudly in the pull request.

## Writing a plugin

Start from `packages/manicule-plugin-example`. It is the smallest complete plugin, and CI
builds it, installs it and loads it, so it cannot go stale.

- Advertise an entry point in the `manicule.plugins` group. The entry-point name and
  `manifest.name` must match.
- Declare `core_version` as a PEP 440 range you have tested. A mismatch is refused at
  startup, with both versions named.
- Register factories, not instances. Keep heavy imports inside the factory so an installed
  plugin nobody has configured costs one cheap import.
- Parsers declare their media types at registration, not only on the class. Routing a
  document reads the declaration, so choosing one parser does not construct the rest — and
  a parser that disagrees with its own declaration is caught the first time it is used.
- Declare a `config_model`. Settings written for a component with no model are rejected
  rather than ignored.
- Run the conformance suites from `manicule.testing` against your components.

### Plugins run with full privileges

A plugin is imported into the manicule process and runs with everything that process has:
the network, the filesystem, the environment, the lot. There is no sandbox, no isolation
boundary, and **no `permissions` field** — manicule cannot enforce one, and a guarantee
nothing enforces is worse than an absent one, because it gets believed.

Install plugins you would be willing to run as yourself, because that is what happens.

## Reviewing a dependency update

Dependabot is configured in [`.github/dependabot.yml`](.github/dependabot.yml). Two things
about its pull requests are not obvious from the diff.

**A group name beginning `index-affecting-` means the cost is a re-ingest, not a review.**
Most bumps risk a regression and CI catches them. These change what is *in* the corpus:

| Group | What moves | What it costs |
|---|---|---|
| `index-affecting-embedding` | The stored vectors | Re-embed everything |
| `index-affecting-chunking` | Where chunks begin and end | Re-chunk and re-embed what it touches |
| `index-affecting-extraction` | The text a document was reduced to | Re-parse the affected documents |

There is machinery here — `EmbedFingerprint`, `ChunkFingerprint` and `ParseFingerprint`
refuse output built with something else, and the macOS backend parity job compares MLX
against ONNX within a stated tolerance — so a genuinely divergent bump should turn CI red.
Red is the signal that the corpus needs rebuilding, not a reason to widen a tolerance.

The three costs are not the same size, and the group name is what tells them apart. An
embedding bump re-embeds everything. A chunking bump re-chunks and re-embeds what it touches.
An extraction bump is the narrowest: `documents.parse_fp` records which parser version
produced each document, so change detection re-parses exactly the documents that library
produced and `manicule document reindex --stale` selects the same set without waiting for a
sync. Adding a
library that decides stored text means adding it to `manicule.parsers.versions.PARSERS` *and*
to the `index-affecting-extraction` patterns; `tests/parsers/test_versions.py` fails if the
two disagree.

**Nothing here checks licenses, and this project has rejected dependencies over them.**
`manicule` is MIT. Dependabot reports versions; it says nothing about the terms a new version
ships under, and a relicense lands in a routine-looking bump. So the license is checked by a
person, at selection time, and again if a bump crosses a major version.

**A copyleft dependency of `manicule` itself is refused.** Not as a preference — the whole
point of the package split is that MIT means MIT, with nothing in the dependency closure that
contradicts it. The decisions on record ([`docs/parsing.md`](docs/parsing.md) §12 has the full
reasoning):

- **PyMuPDF is AGPL-3.0 and is rejected.** Its network clause would land on anyone who *runs*
  manicule rather than on us, and a condition we would be imposing on operators is not ours to
  accept on their behalf. `pypdfium2` is permissive and already does the job.
- **`extract-msg` is GPL-3.0 and is rejected** — for the plain reason again. It was briefly
  admissible during the period when manicule was itself GPL-3.0-or-later; that period ended
  when the MLX backend moved to its own distribution, and `.msg` support goes back to being
  either a permissively licensed reader or a hand-written MAPI property reader
  ([#21](https://github.com/mgd43b/manicule/issues/21)).
- **`selectolax` bundles an LGPL-2.1 engine** alongside a permissive one, and only the
  permissive one is imported. Unchanged by any of the above; see §12.

**`packages/manicule-mlx` is the one exception, and it is a separate distribution precisely so
that it is not an exception to anything.** It is GPL-3.0-or-later because `mlx-embeddings` is,
and a copyleft dependency is ordinary *there*. Adding one to `manicule` is not. If you find
yourself wanting a GPL library in core, the shape of the answer is another package, not another
relicense.

## Commit and pull request conventions

- Branch from `origin/main`. Never push to `main`, never self-merge.
- One pull request per ticket. Reference the issue number.
- Write commit messages in the imperative — "Add the anchor round-trip check", not "Added".
- Say **why** in the body. What changed is in the diff; why it changed is not.
- Do not add `Co-Authored-By` trailers.
- The pull request description carries a **"Decisions that need review"** section listing
  every design call not settled by the documents — especially anything touching `Anchor` or
  `RetrievalStage`, since both are expensive to change later.

## Style

- Python 3.14+, 100-column lines, `ruff format`.
- Type everything. `Any` needs a reason on the same line.
- Errors are actionable: name what was wrong, what was expected, and what to do about it.
  `f"no parser for {media_type!r}. Installed: {available}"` beats `"parser not found"`.
- Comments explain why, never what. A comment restating the line above it is noise; a
  comment explaining why the obvious approach was rejected is the most valuable line in the
  file.
- Docstrings on anything public. One line saying what it is, then the reasoning if the
  reasoning is not obvious.
