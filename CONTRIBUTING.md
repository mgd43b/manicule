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

Run the whole gate locally before pushing:

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check src tests packages
uv run pyright
uv run pytest
```

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
invalidates every stored citation. `RetrievalStage` is locked once the evaluation harness
exists — widening it invalidates every recorded result. Both are marked in the source. Widen
them now if at all, and say so loudly in the pull request.

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

- Python 3.12+, 100-column lines, `ruff format`.
- Type everything. `Any` needs a reason on the same line.
- Errors are actionable: name what was wrong, what was expected, and what to do about it.
  `f"no parser for {media_type!r}. Installed: {available}"` beats `"parser not found"`.
- Comments explain why, never what. A comment restating the line above it is noise; a
  comment explaining why the obvious approach was rejected is the most valuable line in the
  file.
- Docstrings on anything public. One line saying what it is, then the reasoning if the
  reasoning is not obvious.
