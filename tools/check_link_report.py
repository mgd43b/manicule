#!/usr/bin/env python
"""Decide whether a lychee run actually checked the documentation.

    python tools/check_link_report.py lychee-report.json

**This exists because a link check that checks nothing exits 0.** Measured, not assumed: point
lychee at a glob matching no files and it prints one `[WARN] No files found for this input
source` among thousands of lines and returns success. A job that only propagates lychee's exit
code therefore reports the same green tick for "every link resolves" and "there were no links",
and the second is the state a broken glob, a renamed directory or a bad `--exclude` leaves
behind. This repository's signature defect is a check whose name is broader than what it
verified, and that is exactly what an unguarded link check becomes the day its inputs stop
matching.

So the *report* is the gate rather than the exit status. That has a second property worth as
much as the first: the report is an artefact only a real run can produce. A setup step that
failed and was somehow not fatal — a `continue-on-error` added in good faith, an install that
wrote a broken binary — leaves no report, and no report is a failure here. The check cannot be
disabled by breaking it; breaking it is loud.

Three questions, each a different way of checking nothing:

- **Is there a report at all?** No file means lychee did not run to completion.
- **Did it look at any links?** ``total`` counts every link found in every input. Zero means the
  inputs matched nothing.
- **Did it verify any of them?** ``successful`` counts the ones actually resolved. A run where
  everything was *excluded* has a healthy-looking ``total`` and has confirmed nothing — which is
  one `--exclude` away at any time, and would otherwise read as a pass.

Errors are reported too, and named rather than counted, because the exit status upstream says
only that there were some.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["EMPTY_REPORT", "NOTHING_VERIFIED", "check", "main"]

EMPTY_REPORT = "the link checker ran and found no links at all"
"""Said when ``total`` is zero. The wording matters: this is not "the documentation has no
links", it is "the checker was pointed at nothing", and the fix is the input glob rather than
the documents."""

NOTHING_VERIFIED = "the link checker found links and verified none of them"
"""Said when every link was excluded, redirected or otherwise never resolved. A report like that
passes an errors-only check while proving nothing at all."""


def check(report: dict[str, Any]) -> list[str]:
    """Everything wrong with ``report``, as sentences. Empty means the check really passed.

    All of them rather than the first, because "no links were checked" and "these links are
    broken" have different fixes and a run can only tell you about one per attempt otherwise.
    """
    problems: list[str] = []
    total = int(report.get("total", 0))
    successful = int(report.get("successful", 0))
    errors = int(report.get("errors", 0))

    if total == 0:
        problems.append(
            f"{EMPTY_REPORT}. Nothing was verified and the job would otherwise have reported "
            f"success. Check the input glob in the workflow before believing the documentation"
        )
    elif successful == 0 and errors == 0:
        # Only when there is nothing else wrong. A run where every link is *broken* also
        # verified none of them, and saying so there would bury the twenty broken links under a
        # sentence about exclusions. This clause exists for the run that would otherwise pass.
        problems.append(
            f"{NOTHING_VERIFIED} ({total} found, all excluded or unresolved). An exclusion that "
            f"grew to cover everything leaves a report that looks busy and asserts nothing"
        )
    if errors:
        problems.append(f"{errors} broken link(s):\n{_broken(report)}")
    return problems


def _broken(report: dict[str, Any]) -> str:
    """The broken links, as ``file → target: reason`` lines.

    lychee's ``error_map`` carries a timing block and a byte span per entry, and printing it
    raw buries two useful strings under forty lines of nanoseconds. What somebody fixing a link
    needs is which document, which target, and why.
    """
    lines: list[str] = []
    error_map: dict[str, Any] = report.get("error_map") or {}
    for source in sorted(error_map):
        for entry in error_map[source]:
            status: dict[str, Any] = entry.get("status") or {}
            reason = status.get("text") or status.get("details") or "no reason given"
            span: dict[str, Any] = entry.get("span") or {}
            where = f":{span['line']}" if "line" in span else ""
            lines.append(f"  {source}{where} → {entry.get('url')}: {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Check the report named on the command line. Returns a process exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("report", type=Path, help="The JSON report lychee wrote with --output.")
    args = parser.parse_args(argv)

    path = Path(args.report)
    if not path.is_file():
        # The "it never ran" case, and the reason this is a file check rather than a flag: an
        # install that failed leaves no report, so there is nothing here to be satisfied by.
        print(
            f"error: no link report at {path}. The link checker did not run to completion, so "
            f"nothing about this repository's links has been established",
            file=sys.stderr,
        )
        return 1
    try:
        report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read the link report at {path}: {exc}", file=sys.stderr)
        return 1

    problems = check(report)
    print(
        f"link check: {report.get('total', 0)} links, {report.get('successful', 0)} verified, "
        f"{report.get('excludes', 0)} excluded, {report.get('errors', 0)} broken"
    )
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover - a script entry point
    raise SystemExit(main())
