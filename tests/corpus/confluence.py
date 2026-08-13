"""Fixtures for the Confluence storage-format parser, generated rather than committed.

Synthetic throughout. Every page, space key, user and URL here was invented for this
repository: storage format is the format organisations keep their internal runbooks in, and a
fixture corpus is published with the project.

The four kinds ``docs/parsing.md`` §3.5 requires, and what each is here to catch:

**Typical** — a runbook of the shape most pages have: headings, prose, a table, a code macro
with a declared language, a warning panel, a task list, and links to a page and to the outside.

**Structurally hard** — a repeated heading path where only one of the two carries an ``anchor``
macro; macros nested inside a table cell and inside a list item, which is where a flattening
that used ``text(deep=True)`` would leak a parameter; a panel inside a panel; a five-deep list;
and a code macro whose body is itself storage-format markup.

**Degenerate** — zero bytes; a macro with no body; a macro with no name at all; a task with no
body; a table with no rows; and a heading with nothing under it. Each breaks a different
assumption about something being present.

**Hostile** — the load-bearing one, and it is not about malformed markup. A storage-format body
is authored by anyone with write access to the page, so it carries content that *looks like
instructions*: a CDATA body holding a ``<script>`` element, a parameter value holding one, DOT
source holding one, an ``onerror`` attribute, a mention carrying an account id that must never
be indexed, and a Jira macro whose JQL query is configuration rather than text. Plus malformed
UTF-8, which must be declined rather than indexed as replacement characters, and storage XHTML
that is simply broken — unclosed macros, a stray parameter outside any macro, an **unterminated
CDATA section** whose body genuinely cannot be recovered, and a task list that ends mid-task.

Two rules constrain every line here, both from the round-trip contract rather than from taste:

- **No block's text may repeat, or contain another block's text.** The discrimination assertion
  (``docs/parsing.md`` §3.3) compares each block's text against every other anchor's resolved
  text, and two blocks that read identically cannot be told apart by anything, including a
  person reading the citation.
- **A repeated heading path needs something to tell the two apart**, or both sections are
  honestly unlocatable. ``structure.storage`` has exactly one such pair, on purpose.
"""

from __future__ import annotations

from pathlib import Path

ACCOUNT_ID = "557058:11111111-2222-3333-4444-555555555555"
"""The opaque directory identifier a mention carries.

Exported so the suite can assert its absence by the same constant the fixture is written with —
a test that spelled it out a second time would keep passing after somebody changed the fixture.
"""

JQL_QUERY = "project = ORDERS AND status = Open ORDER BY created DESC"
"""A Jira macro's query: configuration, and one of the three values that were being indexed as
though the page had said them."""

SCRIPT_PAYLOAD = "<script>alert('storage')</script>"
"""Content that looks like an instruction, placed everywhere content can arrive from.

Not a hypothetical: ``recover_cdata`` escapes a CDATA body precisely so this is not *promoted*
from inert text to a live element on the way in, and parameter values come from the same page
and the same author."""

ESCAPED_SCRIPT = SCRIPT_PAYLOAD.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
"""The same payload as storage format actually carries it outside a CDATA section.

Storage format is XHTML, so a body that means the *characters* ``<script>`` escapes them; only
inside CDATA do they appear raw. Writing the raw form into a parameter would make it a genuine
element rather than a value — which is a different case, tested separately below, and not the
one a parameter value normally is."""

TYPICAL = """<h1 id="rotating-a-signing-key">Rotating a signing key</h1>
<p>This runbook covers the quarterly rotation. Read it before the maintenance window opens,
because step four cannot be undone once the old key is revoked.</p>
<ac:structured-macro ac:name="warning">
  <ac:parameter ac:name="title">Do not skip the dry run</ac:parameter>
  <ac:rich-text-body><p>A rotation performed without the rehearsal has twice taken the
  checkout service down for the length of a cache lifetime.</p></ac:rich-text-body>
</ac:structured-macro>
<h2 id="before-you-begin">Before you begin</h2>
<p>Confirm with <ac:link><ri:user ri:username="asha.patel"/></ac:link> that no release is in
flight, then open <ac:link><ri:page ri:content-title="Maintenance calendar"/></ac:link> and
claim the window.</p>
<table>
  <thead><tr><th>Environment</th><th>Key age limit</th><th>Owner</th></tr></thead>
  <tbody>
    <tr><td>Staging</td><td>30 days</td><td>Platform</td></tr>
    <tr><td>Production</td><td>90 days</td><td>Security</td></tr>
  </tbody>
</table>
<h2 id="running-the-rotation">Running the rotation</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">python</ac:parameter>
  <ac:parameter ac:name="title">rotate.py</ac:parameter>
  <ac:plain-text-body><![CDATA[from signing import Keyring

keyring = Keyring.load("production")
issued = keyring.issue(algorithm="ed25519")
keyring.retire(issued.predecessor, grace_hours=24)]]></ac:plain-text-body>
</ac:structured-macro>
<p>The grace period is what lets a service that cached the old key finish its requests. See
<a href="https://example.test/runbooks/signing">the published runbook</a> for the rationale.</p>
<h2 id="afterwards">Afterwards</h2>
<ac:task-list>
  <ac:task><ac:task-id>1</ac:task-id><ac:task-status>complete</ac:task-status>
    <ac:task-body>Announce the rotation in the platform channel</ac:task-body></ac:task>
  <ac:task><ac:task-id>2</ac:task-id><ac:task-status>incomplete</ac:task-status>
    <ac:task-body>Delete the retired key material from the escrow bucket</ac:task-body></ac:task>
  <ac:task><ac:task-id>3</ac:task-id><ac:task-status>incomplete</ac:task-status>
    <ac:task-body>Record the new fingerprint against the quarterly audit</ac:task-body></ac:task>
</ac:task-list>
"""

TOPOLOGY_DOT = """digraph orders {
  rankdir = LR;
  checkout -> "order.created" [label="publish"];
  "order.created" -> warehouse;
  "order.created" -> billing;
  warehouse -> "shipment.booked";
}"""
"""The valid diagram's source, exported so the suite can demand it back character for character.

Asserting on a substring would let a parser that reflowed the whitespace pass, and in DOT the
line structure is what a person reads. "Preserve the exact source" is only a claim if something
compares the whole of it."""

TOPOLOGY = (
    """<h1>Queue topology</h1>
<p>How the order events reach the warehouse, drawn from the deployment manifest.</p>
<ac:structured-macro ac:name="graphviz">
  <ac:parameter ac:name="engine">neato</ac:parameter>
  <ac:plain-text-body><![CDATA["""
    + TOPOLOGY_DOT
    + """]]></ac:plain-text-body>
</ac:structured-macro>
<h2>Diagram that does not compile</h2>
<p>Kept deliberately, because a diagram nobody can render is still what somebody wrote and is
exactly what a person debugging the page needs to find.</p>
<ac:structured-macro ac:name="graphviz">
  <ac:plain-text-body><![CDATA[digraph broken {
  alpha -> beta;
  gamma -> delta;]]></ac:plain-text-body>
</ac:structured-macro>
"""
)

STRUCTURE = """<h1>Capacity review</h1>
<p>Two sections below share a heading path. Only one of them publishes an address, which is
what makes the other honestly unlocatable rather than wrongly cited.</p>
<h2>Region</h2>
<h3>Configuration</h3>
<p>The first configuration section describes the shard allocator and nothing else.</p>
<ac:structured-macro ac:name="anchor">
  <ac:parameter ac:name="">retention-configuration</ac:parameter>
</ac:structured-macro>
<h3>Configuration</h3>
<p>The second configuration section describes retention windows and cold tiering.</p>
<h2>Macros in awkward places</h2>
<table>
  <tbody>
    <tr><th>Check</th><th>Command</th></tr>
    <tr>
      <td>Replica lag</td>
      <td><ac:structured-macro ac:name="code">
        <ac:parameter ac:name="language">sql</ac:parameter>
        <ac:plain-text-body><![CDATA[SELECT lag FROM replicas]]></ac:plain-text-body>
      </ac:structured-macro></td>
    </tr>
    <tr>
      <td>Ticket queue</td>
      <td><ac:structured-macro ac:name="jira">
        <ac:parameter ac:name="jqlQuery">project = CAPACITY</ac:parameter>
      </ac:structured-macro></td>
    </tr>
  </tbody>
</table>
<ul>
  <li>An item whose nested macro must not contribute its parameters:
    <ac:structured-macro ac:name="code">
      <ac:parameter ac:name="language">bash</ac:parameter>
      <ac:plain-text-body><![CDATA[df -h]]></ac:plain-text-body>
    </ac:structured-macro></li>
  <li>Depth one
    <ul><li>Depth two
      <ul><li>Depth three
        <ul><li>Depth four
          <ul><li>Depth five, which is as deep as this corpus goes</li></ul>
        </li></ul>
      </li></ul>
    </li></ul>
  </li>
</ul>
<h2>Panel within a panel</h2>
<ac:structured-macro ac:name="note">
  <ac:rich-text-body>
    <p>An outer note that contains its own warning.</p>
    <ac:structured-macro ac:name="warning">
      <ac:rich-text-body><p>The inner warning must keep its own severity.</p></ac:rich-text-body>
    </ac:structured-macro>
  </ac:rich-text-body>
</ac:structured-macro>
<h2>A code macro whose body is markup</h2>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">xml</ac:parameter>
  <ac:plain-text-body><![CDATA[<ac:structured-macro ac:name="info">
  <ac:rich-text-body><p>This is an example, not a macro.</p></ac:rich-text-body>
</ac:structured-macro>]]></ac:plain-text-body>
</ac:structured-macro>
"""

UNSUPPORTED = f"""<h1>Macros with no reader here</h1>
<p>Each macro below is unsupported. None of them may vanish silently.</p>
<ac:structured-macro ac:name="jira">
  <ac:parameter ac:name="jqlQuery">{JQL_QUERY}</ac:parameter>
  <ac:parameter ac:name="maximumIssues">25</ac:parameter>
</ac:structured-macro>
<p>An expand macro carries real prose behind a disclosure triangle, and the prose is not
unsupported merely because the container is.</p>
<ac:structured-macro ac:name="expand">
  <ac:parameter ac:name="title">Why the threshold is ninety seconds</ac:parameter>
  <ac:rich-text-body><p>Because the upstream load balancer gives up at one hundred and twenty,
  and a probe that outlives its caller reports success nobody receives.</p></ac:rich-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="toc"><ac:parameter ac:name="maxLevel">3</ac:parameter>
</ac:structured-macro>
<ac:structured-macro ac:name="roadmap-planner">
  <ac:parameter ac:name="timeline">quarters</ac:parameter>
  <ac:plain-text-body><![CDATA[{{"lanes": ["platform", "billing"]}}]]></ac:plain-text-body>
</ac:structured-macro>
"""

HOSTILE = f"""<h1>Content that looks like an instruction</h1>
<p>Everything below arrives from a page anyone with write access can edit. None of it may be
rendered, executed, or promoted from text into markup.</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">html</ac:parameter>
  <ac:plain-text-body><![CDATA[{SCRIPT_PAYLOAD}]]></ac:plain-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="jira">
  <ac:parameter ac:name="jqlQuery">{ESCAPED_SCRIPT}</ac:parameter>
</ac:structured-macro>
<ac:structured-macro ac:name="graphviz">
  <ac:parameter ac:name="engine">{ESCAPED_SCRIPT}</ac:parameter>
  <ac:plain-text-body><![CDATA[digraph exfiltrate {{
  label = "{SCRIPT_PAYLOAD}";
  a -> b;
}}]]></ac:plain-text-body>
</ac:structured-macro>
<p>A raw script element, which is not what a parameter value looks like but is what a space
administrator can put directly on a page. It is dropped with its contents.</p>
<script>alert('page')</script>
<p>A mention must be a display reference and never a directory identifier.</p>
<p>Reviewed by <ac:link><ri:user ri:account-id="{ACCOUNT_ID}"/></ac:link> on the second of
the month.</p>
<p>An image whose attachment is named but must never be fetched during parsing.</p>
<ac:image ac:alt="Throughput over the window"><ri:attachment ri:filename="throughput.png"/>
</ac:image>
<p onerror="alert('attribute')">A paragraph carrying an event-handler attribute.</p>
<ac:structured-macro ac:name="warning">
  <ac:parameter ac:name="title">{ESCAPED_SCRIPT}</ac:parameter>
  <ac:rich-text-body><p>A panel title is the one parameter a reader actually sees, so it is
  the one that becomes text — which is exactly why it has to be inert.</p></ac:rich-text-body>
</ac:structured-macro>
"""

MALFORMED = """<h1>Storage format that is simply broken</h1>
<p>An export truncated mid-write, which a parser must still get something out of.</p>
<ac:parameter ac:name="orphan">A parameter outside any macro at all.</ac:parameter>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">go</ac:parameter>
  <ac:plain-text-body><![CDATA[func main() { println("unterminated macro follows") }]]>
</ac:structured-macro>
<ac:structured-macro ac:name="noformat">
  <ac:plain-text-body><![CDATA[An export truncated inside a CDATA section. It never closes, so
recovery leaves it exactly as it found it rather than guessing where the author meant it to end,
and this text genuinely does not reach the index.
</ac:structured-macro>
<ac:structured-macro ac:name="info">
  <ac:rich-text-body><p>An info panel whose closing tags never arrive.
<ac:task-list>
  <ac:task><ac:task-id>9</ac:task-id><ac:task-status>incomplete</ac:task-status>
    <ac:task-body>A task the document ends in the middle of
"""

DEGENERATE = """<h1>Nothing much here</h1>
<h2>A heading with nothing under it</h2>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">rust</ac:parameter>
</ac:structured-macro>
<ac:structured-macro><ac:parameter ac:name="stray">A macro with no name attribute.</ac:parameter>
</ac:structured-macro>
<ac:task-list><ac:task><ac:task-id>4</ac:task-id><ac:task-status>incomplete</ac:task-status>
  <ac:task-body></ac:task-body></ac:task></ac:task-list>
<table></table>
<ac:structured-macro ac:name="graphviz"><ac:plain-text-body><![CDATA[]]></ac:plain-text-body>
</ac:structured-macro>
"""

ASTRAL = """<h1>Unicode past the basic plane 🛰️</h1>
<p>A heading and a body carrying astral-plane characters, which is where anything counting
bytes where it means characters comes apart: a clef 𝄞, a CJK extension B ideograph 𠀋, and
an emoji 🚚 mid-sentence.</p>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="language">python</ac:parameter>
  <ac:plain-text-body><![CDATA[LABELS = {"运输": "shipping",
          "🚚": "in transit"}]]></ac:plain-text-body>
</ac:structured-macro>
"""

MOJIBAKE = (
    b"<h1>Bytes that are not UTF-8</h1>\n<p>The next byte is invalid: \xff\xfe and the "
    b"sentence continues past it.</p>\n"
)

_LARGE_TOPICS = (
    "shard rebalancing",
    "cold tier promotion",
    "replica lag alarms",
    "checkout retries",
    "escrow expiry",
    "audit sampling",
    "queue draining",
    "cache invalidation",
)


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    _write(dest / "typical.storage", TYPICAL)
    _write(dest / "topology.storage", TOPOLOGY)
    _write(dest / "structure.storage", STRUCTURE)
    _write(dest / "unsupported.storage", UNSUPPORTED)
    _write(dest / "hostile.storage", HOSTILE)
    _write(dest / "malformed.storage", MALFORMED)
    _write(dest / "degenerate.storage", DEGENERATE)
    _write(dest / "astral.storage", ASTRAL)
    _write(dest / "empty.storage", "")
    (dest / "mojibake.storage").write_bytes(MOJIBAKE)
    _write(dest / "handbook-large.storage", _large())


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _large() -> str:
    """A page past the size cap, to exercise the streaming path.

    Every section and every paragraph is distinct by construction rather than by inspection: the
    number is zero-padded so section one's heading is not a prefix of section eleven's, and each
    paragraph names both its section and its own position within it.
    """
    parts: list[str] = ["<h1>Operations handbook</h1>\n"]
    for index, topic in enumerate(_LARGE_TOPICS, start=1):
        parts.append(f"<h2>Chapter {index:02d}: {topic}</h2>\n")
        for step in range(1, 11):
            body = " ".join(
                f"Step {step:02d} of chapter {index:02d} on {topic} records observation "
                f"{number:03d}, what the operator saw and what they did about it."
                for number in range(1, 16)
            )
            parts.append(f"<p>{body}</p>\n")
        parts.append(
            f'<ac:structured-macro ac:name="code">\n'
            f'  <ac:parameter ac:name="language">bash</ac:parameter>\n'
            f"  <ac:plain-text-body><![CDATA[manicule inspect --chapter {index:02d} "
            f"--topic {topic.replace(' ', '-')}]]></ac:plain-text-body>\n"
            f"</ac:structured-macro>\n"
        )
    return "".join(parts)
