"""One structured memory, written as a person or an assistant would write it.

Shared by every test that needs a memory so they all describe the same document: the reader, the
authoring door, the pipeline and the surfaces each assert something different *about* it, and a
memory rebuilt by hand in four places is four slightly different memories.
"""

from __future__ import annotations

SLUG = "reference_vpn_gateway_failover"
LOCATOR = "https://example.slack.com/archives/C0123/p1700000000000100"
SECOND_LOCATOR = "https://jira.example.com/browse/OPS-12"
QUOTE = "failover is manual since the cert rotation: nothing flips on its own"

EVIDENCE = f"""\
evidence:
  - kind: slack
    locator: {LOCATOR}
    author: Jane Doe
    said_at: 2026-03-02T14:05:00Z
    quote: "{QUOTE}"
"""

SECOND_SOURCE = f"""\
  - kind: jira
    locator: {SECOND_LOCATOR}
    said_at: 2026-04-01T09:00:00+02:00
    quote: |
      Confirmed: automatic failover was removed.
      The manual runbook applies.
"""

BODY = """\
# VPN gateway failover is manual

## Problem

The primary gateway fails and nothing flips to the second one.

## Answer

Run the failover runbook; the second gateway does not take over by itself.

## Applicability

Production only, since March 2026. The staging pair still fails over on its own.
"""


def memory(
    *,
    front: str = "",
    evidence: str = EVIDENCE,
    knowledge_type: str | None = "constraint",
    body: str = BODY,
) -> str:
    """A memory, with whatever extra front matter ``front`` adds.

    ``knowledge_type=None`` omits the marker, which is what makes the result an ordinary
    document; ``evidence`` replaces the whole evidence block, so a test can break one field
    without restating the rest.
    """
    marker = f"knowledge_type: {knowledge_type}\n" if knowledge_type is not None else ""
    return (
        f"---\nname: {SLUG}\ndescription: Gateway failover is manual.\nmetadata:\n"
        f"  node_type: memory\n  type: reference\n{marker}{evidence}{front}---\n\n{body}"
    )
