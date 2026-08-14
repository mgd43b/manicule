"""Fixtures for the HTML parser, generated rather than committed.

Generation keeps the repository small, makes every fixture's structure reviewable as code
rather than as an opaque blob, and lets the hostile cases exist without being stored at all.

The four kinds ``docs/parsing.md`` §3.5 requires, and what each is here to catch:

**Typical** — an ordinary documentation page: a ``<title>``, headings with ``id=``, a table
with a ``<thead>``, a ``<pre><code class="language-…">`` block and a nested list.

**Structurally hard** — a list nested five deep; a table; a code block; an inline ``<br>``,
which is the one element that contributes a character to ``text`` that is not in any text node
of the source; a heading addressed by the empty anchor element before it rather than by an
``id`` of its own; and two sibling headings with the same title and no published address, which
is the case that has to come back :class:`~manicule.core.anchors.Unlocated` rather than pick one
of them.

The break is in the corpus rather than only in a focused test because it is what puts a newline
*inside* a block's text, and the round-trip assertions are where that has to survive:
:func:`~manicule.testing.normalize.normalize` reduces it to a space on both sides of every
comparison, so containment and tightness hold across it or the fixture says so.

**Degenerate** — zero bytes, a heading with nothing under it, and a file with no trailing
newline.

**Hostile** — malformed UTF-8, which must be declined rather than indexed as replacement
characters; unclosed tags, which the parser has to survive because half the web is like that;
a ``<script>`` whose body must not reach the index; and astral-plane text in a heading.

Every block's text in every generated file is distinct, and no block's text contains
another's. That is a requirement rather than tidiness: the discrimination assertion
(``docs/parsing.md`` §3.3) compares each block's text against every other anchor's resolved
text, and two blocks that read identically cannot be told apart by anything, a person reading
the citation included. In HTML a heading's text is exactly its title, so a title that also
occurs in a neighboring sentence is the same collision — which is why no heading here
repeats a word of its own body.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["LARGE_SECTIONS", "build"]

LARGE_SECTIONS = 12
"""Sections in the over-cap fixture. Few and long rather than many and short: the streaming
path is exercised by the byte count, while the assertions that compare every block against
every other are quadratic in the block count."""

_TYPICAL = """<!doctype html>
<html lang="en">
<head><title>Ledger service handbook</title></head>
<body>
<main>
  <h1 id="running-the-service">Running the service</h1>
  <p>Three machines share one journal volume and take turns holding the write lease.</p>
  <h2 id="starting-up">Starting up</h2>
  <p>Each process reads its settings once, then announces itself on the bus.</p>
  <pre><code class="language-bash">ledgerctl start --wait
ledgerctl status</code></pre>
  <h2 id="watching-it">Watching it</h2>
  <p>The dashboard shows one row per machine and one column per measurement.</p>
  <table>
    <thead><tr><th>Signal</th><th>Alarm above</th></tr></thead>
    <tbody>
      <tr><td>lease age</td><td>90 seconds</td></tr>
      <tr><td>journal depth</td><td>4096 records</td></tr>
    </tbody>
  </table>
  <ul>
    <li>green means the lease is fresh</li>
    <li>amber means it was renewed late
      <ul><li>which is normal during a rolling restart</li></ul>
    </li>
  </ul>
  <img src="topology.png" alt="Three machines around one shared volume">
</main>
</body>
</html>
"""

_STRUCTURE = """<!doctype html>
<html lang="en">
<head><title>Fabric reference</title></head>
<body>
<article>
  <p>An opening paragraph, written before any heading and after the title element.</p>
  <h1 id="fabric">Fabric</h1>
  <p>Every rack holds eight machines, one spare, and a pair of switches.</p>
  <h2 id="cabling">Cabling</h2>
  <p>Each machine takes two uplinks,<br/>and no two uplinks share a switch.</p>
  <pre><code class="language-python">def uplinks(machine: str) -> tuple[str, str]:
    return (machine + "-a", machine + "-b")</code></pre>
  <table>
    <thead><tr><th>Port</th><th>Speed</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>25G</td></tr>
      <tr><td>2</td><td>25G</td></tr>
    </tbody>
  </table>
  <ol>
    <li>outermost step
      <ol><li>second level
        <ol><li>third level
          <ol><li>fourth level
            <ol><li>fifth level, as deep as this corpus goes</li></ol>
          </li></ol>
        </li></ol>
      </li></ol>
    </li>
  </ol>
  <a id="legacy-power"></a>
  <h2>Power</h2>
  <p>Two feeds per rack, drawn from separate boards on separate floors.</p>
  <h2>Overview</h2>
  <p>The first sibling with this title, which the page leaves unaddressable.</p>
  <h2>Overview</h2>
  <p>The second, indistinguishable from it by anything the markup publishes.</p>
</article>
</body>
</html>
"""

_HEADING_ONLY = """<!doctype html>
<html lang="en"><head><title>Stub</title></head>
<body><h1>A stub page that never got written</h1></body></html>
"""

_NO_TRAILING_NEWLINE = (
    '<!doctype html>\n<html lang="en"><head><title>Terse</title></head>\n'
    "<body><h1>Terse</h1><p>One paragraph, and no newline where a file usually ends.</p>"
    "</body></html>"
)

_ASTRAL = """<!doctype html>
<html lang="en">
<head><title>Launch</title></head>
<body>
  <h1 id="launch">🚀 Checklist for 𠀋 builds</h1>
  <p>The identifier 𡃁 appears in the body as well as in the heading above it.</p>
  <ul><li>🛰 confirm the relay answers</li><li>𠮟 confirm the reviewer signed off</li></ul>
</body>
</html>
"""

_UNCLOSED = """<!doctype html>
<html lang="en">
<head><title>Ragged</title></head>
<body>
<div class="content">
  <h1 id="ragged">Tags that were never closed</h1>
  <p>A paragraph opened and abandoned, which the parser has to survive
  <ul>
    <li>an item without its closing tag
    <li>and another one after it
  </ul>
  <script>var tracking = {"page": "ragged"}; report(tracking);</script>
  <style>.content { color: rebeccapurple; }</style>
  <p>Text after the script, which must still be indexed.</p>
</div>
</body>
"""

_MOJIBAKE = (
    b'<!doctype html>\n<html lang="en"><head><title>Broken</title></head>\n'
    b"<body><h1>A heading that decodes</h1>\n"
    b"<p>And a paragraph that does not: \xff\xfe\x00\xc3\x28</p></body></html>\n"
)

_LARGE_TOPICS = (
    "quorum loss",
    "disk pressure",
    "clock drift",
    "certificate expiry",
    "network partition",
    "memory ballooning",
    "queue backlog",
    "replica lag",
    "cache stampede",
    "log rotation",
    "index rebuild",
    "credential rotation",
)


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    _write(dest / "typical.html", _TYPICAL)
    _write(dest / "structure.html", _STRUCTURE)
    _write(dest / "heading-only.html", _HEADING_ONLY)
    _write(dest / "no-trailing-newline.htm", _NO_TRAILING_NEWLINE)
    _write(dest / "empty.html", "")
    _write(dest / "astral.html", _ASTRAL)
    _write(dest / "unclosed.html", _UNCLOSED)
    (dest / "mojibake.html").write_bytes(_MOJIBAKE)
    _write(dest / "manual-large.html", _large())


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _large() -> str:
    """A page past the size cap, every paragraph of it distinct.

    Distinct by construction rather than by inspection: the section number is zero-padded so
    that section one's heading is not a prefix of section eleven's, and every paragraph names
    both its section and its own position within it.
    """
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><title>Incident handbook</title></head><body><main>',
    ]
    for index, topic in enumerate(_LARGE_TOPICS, start=1):
        parts.append(f'<h2 id="runbook-{index:02d}">Runbook {index:02d}: {topic}</h2>')
        for step in range(1, 10):
            body = " ".join(
                f"Step {step:02d} of runbook {index:02d} for {topic} continues with "
                f"observation {number:03d}, which records what the operator saw and what "
                f"they did about it."
                for number in range(1, 21)
            )
            parts.append(f"<p>{body}</p>")
    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"
