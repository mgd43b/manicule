"""The model-weights cache key, held against the files that decide what it holds.

The `embeddings` job caches `~/.cache/huggingface` so that a parity run does not fetch several
hundred megabytes from a third-party host on every push. What decides which bytes belong in
that cache is spread across three files, and the cache key named only one of them:

- ``tools/prefetch_embedding_models.py`` chooses *which repositories* to seed,
- ``src/manicule/embedding/artifacts.py`` pins *which commit* of each, and
- ``src/manicule/embedding/cards.py`` lists *which files* to take from the declaration.

**The way this drifts is silent and green, which is why it is a test.** Re-qualifying a model
edits a sha in ``artifacts.py`` and nothing else. With that file outside the key, the key does
not move; an unchanged key is an exact hit, so ``actions/cache`` restores the snapshot for the
*previous* commit — and then skips its post-run save, because saving is what a primary-key miss
triggers. The pre-seed downloads the new revision, the job throws it away at teardown, and the
next run repeats it. The job still passes. It simply stops being cached, forever, and the only
symptom is a slow job and a dependency on somebody else's uptime — the exact dependency #80
removed for the grammar pack and this cache exists to remove for the weights.

Read out of the workflow and out of the seeding script's own imports rather than from a list
kept here. A fourth copy of "what decides the fetch" would drift from the other three, and the
drift would be invisible, which is the defect itself.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
SEEDER = REPO_ROOT / "tools" / "prefetch_embedding_models.py"


def _cache_key() -> str:
    """The `key:` of the step that caches the Hugging Face directory.

    Found by the path it caches rather than by the step's name, because the name is prose and
    the path is the thing under discussion.
    """
    import yaml  # noqa: PLC0415 - a test-only dependency, kept out of this module's import cost

    workflow = cast(dict[str, Any], yaml.safe_load(WORKFLOW.read_text()))
    keys = [
        cast(str, cast(dict[str, Any], step["with"])["key"])
        for job in cast(dict[str, dict[str, Any]], workflow["jobs"]).values()
        for step in cast(list[dict[str, Any]], job.get("steps") or [])
        if "actions/cache" in str(step.get("uses", ""))
        and "huggingface" in str(cast(dict[str, Any], step.get("with", {})).get("path", ""))
    ]
    assert len(keys) == 1, (
        f"expected exactly one step caching a Hugging Face directory, found {len(keys)}. "
        "This test reads that step to check its key; two of them means the key it checks is "
        "not necessarily the key that matters."
    )
    return keys[0]


def _seeding_inputs() -> set[str]:
    """The repository files the seeding script reads its answers out of.

    Taken from the script's own module-scope imports, so that a fourth source of truth added to
    it — a new module naming repositories, revisions or file lists — is covered here without
    anybody remembering to add it to a list.
    """
    tree = ast.parse(SEEDER.read_text())
    modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("manicule")
    }
    assert modules, (
        "tools/prefetch_embedding_models.py imports nothing from manicule at module scope, so "
        "this test is reading the wrong file or the script has been restructured."
    )
    return {f"src/{module.replace('.', '/')}.py" for module in modules}


def test_the_cache_key_covers_every_file_that_decides_what_is_fetched() -> None:
    """Everything the seeding script reads its answers from is hashed into the key.

    ``artifacts.py`` is the one this was written for — it holds the pinned commits, and editing
    a commit is the whole of what re-qualifying a model looks like — but it is asserted through
    the script's imports rather than by name, so ``cards.py`` and anything added later are
    covered by the same sentence.
    """
    key = _cache_key()
    hashed = set(re.findall(r"'([^']+)'", key))

    assert "tools/prefetch_embedding_models.py" in hashed, (
        "the model cache key does not hash the seeding script itself. Changing which "
        "repositories are seeded would then reuse a cache built for the previous set."
    )

    missing = sorted(_seeding_inputs() - hashed)
    assert not missing, (
        f"the model cache key does not hash {missing}, which "
        f"tools/prefetch_embedding_models.py reads to decide what to download. A change there "
        f"leaves the key unchanged, so actions/cache reports an exact hit, restores the "
        f"previous weights and skips its save — and the job re-downloads them on every run "
        f"from then on, green the whole time.\n"
        f"Add the file to the hashFiles(...) call in the `cache model weights` step."
    )


def test_the_seeded_revisions_are_pinned_commits() -> None:
    """What the key protects is a pin, so the pins have to be pins.

    The cache key argument above is only worth making while the seeded artifacts are identified
    by commit. If ``artifacts.py`` ever answered with a branch name, the cache would hold
    whatever that branch meant on the day it was filled and the key would have nothing to say
    about it — the same class of failure the Dockerfile had when it fetched ``BAAI/bge-m3`` at
    HEAD while the runtime asked for a pinned sha.
    """
    from tools.prefetch_embedding_models import FULL_MODEL, PARITY_MODEL  # noqa: PLC0415

    from manicule.embedding.artifacts import (  # noqa: PLC0415
        builtin_model_revision,
        builtin_revision,
    )

    commit = re.compile(r"[0-9a-f]{40}")
    pinned = {
        (model, kind): revision
        for model in (PARITY_MODEL, FULL_MODEL)
        for kind, revision in (
            ("declaration", builtin_model_revision(model)),
            ("onnx", builtin_revision(model, "onnx")),
            ("mlx", builtin_revision(model, "mlx")),
        )
    }

    unpinned = sorted(
        key for key, revision in pinned.items() if not commit.fullmatch(revision or "")
    )
    assert not unpinned, (
        f"these seeded artifacts are not identified by a commit: {unpinned}. The seeding step "
        f"would fetch whatever HEAD is on the day it ran, so the cached copy would be undated "
        f"and the key above would have nothing to say about which bytes it holds."
    )
