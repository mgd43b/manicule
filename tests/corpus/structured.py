"""Fixtures for the structured-data parser: JSON, YAML and TOML.

Two constraints shape every one of these, and both come from the round-trip suite rather than
from taste.

No block's text may appear inside another block's resolved text (``docs/parsing.md`` §3.3,
assertion 3), so no top-level entry here repeats a line that also appears at the top level
somewhere else. A configuration file full of ``enabled: true`` would fail a parser that is
behaving perfectly.

And the hostile cases are the point of the JSON fixtures. Duplicate keys and ``NaN`` are both
valid JSON that the mark-bearing YAML reader cannot describe, so each is a file that indexes
fine and would lose its anchors silently (§11).
"""

from __future__ import annotations

from pathlib import Path

TYPICAL_YAML = """# Ingest configuration for the platform workspace.
version: 3

connectors:
  confluence:
    base_url: https://example.invalid/wiki
    spaces:
      - ENG
      - PLATFORM
  filesystem:
    roots:
      - /srv/corpus/handbook

chunking:
  max_tokens: 512
  overlap_tokens: 64

retrieval:
  pipeline:
    - dense
    - lexical
    - fusion
  top_k: 20
"""

NORWAY_YAML = """# Country codes, which YAML 1.1 would read as booleans.
no: Norway
off: Offaly, a county rather than a switch
yes: an affirmative given as a word
on: the preposition, spelled out here so it is unmistakable
"""
"""The Norway problem, as a fixture.

Under YAML 1.1 the keys ``no``, ``off``, ``yes`` and ``on`` all resolve to booleans, so the
symbols on these blocks would read ``False`` and ``True`` where the document plainly says
``no`` and ``yes``. That is a citation reproducing something the document does not say, which
is why the parser reads YAML 1.2.
"""

NESTED_YAML = """observability:
  metrics:
    exporters:
      prometheus:
        endpoints:
          scrape:
            path: /metrics
            interval_seconds: 15
            labels:
              tier: platform
              region: eu-west
  traces:
    sampling:
      head_ratio: 0.05

alerting:
  routes:
    critical:
      receivers:
        - pager
        - bridge
"""

MULTI_DOC_YAML = """apiVersion: v1
kind: ConfigMap
metadata:
  name: ingest-settings
---
apiVersion: v1
kind: Secret
metadata:
  name: ingest-credentials
stringData:
  token: placeholder-value-for-a-fixture
"""

UNTERMINATED_QUOTE_YAML = """summary: "a quoted scalar that is never closed, in the way a
copied spreadsheet cell arrives
next: 2
"""

TYPICAL_JSON = """{
  "schemaVersion": 4,
  "workspace": {
    "id": "ws-platform",
    "displayName": "Platform Engineering"
  },
  "embedder": {
    "modelId": "bge-small-en-v1.5",
    "dimension": 384,
    "maxSequenceLength": 512
  },
  "sources": [
    {"name": "handbook", "kind": "filesystem"},
    {"name": "wiki", "kind": "confluence"}
  ]
}
"""

DEEP_JSON = """{
  "root": {
    "level1": {
      "level2": {
        "level3": {
          "level4": {
            "level5": {
              "leaf": "the value at the bottom of five nested objects",
              "count": 5
            }
          }
        }
      }
    }
  },
  "sibling": "a second top-level key, so the file has more than one block"
}
"""

COMPACT_JSON = '{"alpha":1,"beta":{"gamma":2},"delta":[3,4,5]}\n'
"""Every key on one line, with no space after the colon.

A line range cannot address a fraction of a line, so this must come back as one block. A
parser that emitted one block per key would give them all the same anchor while claiming
different text, and each would resolve to the whole line.
"""

DUPLICATE_KEYS_JSON = """{
  "retries": 3,
  "timeoutSeconds": 30,
  "retries": 5
}
"""
"""Legal JSON — the last value wins — and a hard error in the YAML reader that supplies
positions. The document must still be indexed, with its anchors declined by name."""

NAN_JSON = '{"observed": 12.5, "expected": NaN, "note": "NaN is JSON that json.loads accepts"}\n'
"""``json`` reads a float; the mark-bearing reader reads the string ``"NaN"``. The two
disagree about what the document contains, so the positions are not trustworthy."""

TYPICAL_TOML = """# Workspace settings.
title = "Platform corpus"
schema_version = 4

[chunking]
max_tokens = 512
overlap_tokens = 64

[storage.sqlite]
path = "/var/lib/manicule/index.db"
busy_timeout_ms = 5000

[[connector]]
name = "handbook"
kind = "filesystem"

[[connector]]
name = "wiki"
kind = "confluence"
"""

FLAT_TOML = """# No table headers anywhere: one block, one exact whole-file span.
title = "A flat configuration"
enabled = true
retries = 4
description = "Every key sits at the top level, which is a shape TOML files often have"
"""

TRICKY_TOML = '''# A table header inside a multi-line string is not a table header.
[service]
name = "ingest"
banner = """
[not a table]
this text merely looks structural
"""

[service.limits]
max_documents = 10000
'''

INVALID_TOML = """title = "unterminated
[section]
"""


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    for name, text in (
        ("typical.yaml", TYPICAL_YAML),
        ("norway.yaml", NORWAY_YAML),
        ("nested.yaml", NESTED_YAML),
        ("multi-document.yaml", MULTI_DOC_YAML),
        ("unterminated-quote.yaml", UNTERMINATED_QUOTE_YAML),
        ("typical.json", TYPICAL_JSON),
        ("deep.json", DEEP_JSON),
        ("compact.json", COMPACT_JSON),
        ("duplicate-keys.json", DUPLICATE_KEYS_JSON),
        ("nan.json", NAN_JSON),
        ("typical.toml", TYPICAL_TOML),
        ("flat.toml", FLAT_TOML),
        ("tricky.toml", TRICKY_TOML),
        ("invalid.toml", INVALID_TOML),
    ):
        (dest / name).write_text(text, encoding="utf-8")
    (dest / "empty.yaml").write_bytes(b"")
    (dest / "empty.json").write_bytes(b"")
    (dest / "structured-large.yaml").write_text(_large_yaml(), encoding="utf-8")


def _large_yaml() -> str:
    """A generated document past the size cap, whose top-level keys are worth splitting.

    One key holds far more lines than the parser's block budget, so the split-at-the-next-level
    path runs on real marks rather than on a contrived two-line example.
    """
    lines = ["# A generated inventory, large enough to exercise the split path.", "hosts:"]
    for number in range(1, 900):
        lines.extend(
            [
                f"  host-{number:04d}:",
                f"    address: 10.{number // 256}.{number % 256}.1",
                f"    role: {_ROLES[number % len(_ROLES)]}",
                f"    rack: r{number % 37:02d}",
            ]
        )
    lines.extend(["", "summary:", "  counted: 899", "  generated_by: tests/corpus/structured.py"])
    return "\n".join(lines) + "\n"


_ROLES = ("ingest", "retrieval", "storage", "scheduler", "gateway")
