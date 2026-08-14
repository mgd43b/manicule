"""Generated `.ipynb` fixtures: typical, structurally hard, degenerate and hostile.

Notebooks are JSON, so they are written as literal dictionaries — the structure of every
fixture is the thing on the page, which is the whole argument for generating fixtures rather
than committing them.

The version numbers are the point of several of these. Cell ids arrived in **nbformat 4.5**, so
a 4.4 notebook has no fragment and its heading path is the only address it has; a 4.4 notebook
whose heading path repeats has no address at all, and that is what the ``Unlocated`` reason has
to explain. Both are here, and neither is converted on the way in: converting would generate
the ids the file does not have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["build"]

_Json = dict[str, Any]
"""A notebook JSON object.

``Any`` because notebook JSON is heterogeneous by definition: ``source`` is a string or a list
of strings, an output's ``data`` maps media types to either, and a fixture that flattened that
would stop testing what the parser actually receives.
"""

_KERNEL: _Json = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    _write(dest / "notebook_typical.ipynb", _typical())
    _write(dest / "notebook_structurally_hard.ipynb", _structurally_hard())
    _write(dest / "notebook_degenerate_heading_only.ipynb", _heading_only())
    _write(dest / "notebook_degenerate_no_cells.ipynb", _no_cells())
    (dest / "notebook_degenerate_zero_bytes.ipynb").write_bytes(b"")
    _write(dest / "notebook_hostile_astral.ipynb", _astral())
    _write(dest / "notebook_hostile_no_cell_ids.ipynb", _no_cell_ids())
    _write(dest / "notebook_hostile_repeated_path.ipynb", _repeated_path())
    _write(dest / "notebook_hostile_version_three.ipynb", _version_three())
    (dest / "notebook_hostile_not_json.ipynb").write_text("this is not JSON", encoding="utf-8")
    _write(dest / "notebook-large.ipynb", _large())


def _typical() -> _Json:
    return _notebook(
        4,
        5,
        [
            _markdown("intro", "# Retry budget analysis\n\nEstablishing where the retries go.\n"),
            _code(
                "load",
                "frame = read_metrics('retries.parquet')\nframe.shape\n",
                outputs=[_stream("Loaded 41 820 rows of retry telemetry\n"), _result("(41820, 7)")],
            ),
            _markdown("by-service", "## Retries by service\n\nCheckout dominates the tail.\n"),
            _code(
                "group",
                "frame.groupby('service').retries.sum().sort_values()\n",
                outputs=[_result("search 118\ncheckout 9 402\nName: retries")],
            ),
        ],
    )


def _structurally_hard() -> _Json:
    """One markdown cell holding two sections, a deep list, an image output, and an error."""
    return _notebook(
        4,
        5,
        [
            _markdown(
                "two-sections",
                "# Capacity model\n\nThe model runs weekly.\n\n"
                "## Inputs\n\nHeadroom, growth rate and the retention window.\n",
            ),
            _markdown(
                "deep-list",
                "## Assumptions\n\n"
                "- Level one assumption about traffic\n"
                "  - Level two assumption about caching\n"
                "    - Level three assumption about eviction\n"
                "      - Level four assumption about cold starts\n"
                "        - Level five assumption about the tail\n",
            ),
            _markdown(
                "fenced",
                "## Fenced\n\nThe fence below is code, and its hash is not a heading.\n\n"
                "```python\n# not a heading, a comment\nmodel.fit()\n```\n",
            ),
            _code(
                "plot",
                "chart = plot_headroom(model)\nchart\n",
                outputs=[_image()],
            ),
            _code(
                "boom",
                "model.extrapolate(years=200)\n",
                outputs=[
                    _error("OverflowError", "horizon beyond the fitted range", ["Traceback line"])
                ],
            ),
            _raw("front-matter", "subtitle: headroom projection\nauthor: platform team\n"),
        ],
    )


def _heading_only() -> _Json:
    """A single heading and nothing else."""
    return _notebook(4, 5, [_markdown("only", "# Nothing but a heading\n")])


def _no_cells() -> _Json:
    """A valid notebook with no cells. Zero blocks, and that is not a failure."""
    return _notebook(4, 5, [])


def _astral() -> _Json:
    return _notebook(
        4,
        5,
        [
            # Ambiguous-character warnings are suppressed deliberately: the astral codepoints
            # are the fixture, and a citation has to reproduce them exactly.
            _markdown("astral", "# 解析 𝔄nalysis\n\nGlyphs above the BMP: 🜄 𠀋 𝕆\n"),  # noqa: RUF001
            _code("emoji", "label = '🌐 global'\nlabel\n", outputs=[_result("'🌐 global'")]),
        ],
    )


def _no_cell_ids() -> _Json:
    """nbformat 4.4: no cell ids, so the heading path is the only address there is."""
    return _notebook(
        4,
        4,
        [
            _markdown(None, "# Legacy notebook\n\nSaved before cell ids existed.\n"),
            _code("", "legacy_total = 7\nlegacy_total\n", outputs=[_result("7")]),
            _markdown(None, "## Legacy appendix\n\nA second section with a distinct path.\n"),
        ],
    )


def _repeated_path() -> _Json:
    """nbformat 4.4 whose heading path repeats **non-contiguously**, which is the ambiguous case.

    Two adjacent cells under one heading are one section and are addressed perfectly well by its
    path. The address only becomes ambiguous when the same path opens twice with something else
    in between, because then it names two places — so the fixture puts a section between them.

    Deliberately outside the six-assertion harness: identical heading text means one section's
    resolved span contains the other's whole heading block, which assertion 3 of
    ``docs/parsing.md`` §3.3 cannot distinguish from a misplaced anchor.
    ``tests/parsers/test_notebook.py`` asserts the reason directly instead.
    """
    return _notebook(
        4,
        4,
        [
            _markdown(None, "## Setup\n\nFirst installation walkthrough.\n"),
            _markdown(None, "## Teardown\n\nHow to remove it again.\n"),
            _markdown(None, "## Setup\n\nSecond installation walkthrough.\n"),
        ],
    )


def _version_three() -> _Json:
    """nbformat 3, which this parser declines rather than converting."""
    return {"nbformat": 3, "nbformat_minor": 0, "metadata": {}, "worksheets": [{"cells": []}]}


def _large() -> _Json:
    """Enough cells to exercise the streaming path."""
    cells: list[_Json] = [_markdown("start", "# Long run log\n\nOne section per hour.\n")]
    for hour in range(1, 41):
        cells.append(
            _markdown(
                f"hour-{hour:02d}",
                f"## Hour {hour:02d}\n\nWhat window {hour:02d} covered, in one line.\n",
            )
        )
        cells.append(
            _code(
                f"run-{hour:02d}",
                f"summarize(window={hour})\n",
                outputs=[_stream(f"window {hour} totaled {hour * 311} events, {hour} retried\n")],
            )
        )
    return _notebook(4, 5, cells)


def _notebook(major: int, minor: int, cells: list[_Json]) -> _Json:
    return {
        "cells": cells,
        "metadata": dict(_KERNEL),
        "nbformat": major,
        "nbformat_minor": minor,
    }


def _markdown(cell_id: str | None, source: str) -> _Json:
    cell: _Json = {"cell_type": "markdown", "metadata": {}, "source": source}
    if cell_id is not None:
        cell["id"] = cell_id
    return cell


def _raw(cell_id: str, source: str) -> _Json:
    return {"cell_type": "raw", "id": cell_id, "metadata": {}, "source": source}


def _code(cell_id: str, source: str, *, outputs: list[_Json] | None = None) -> _Json:
    cell: _Json = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": 1,
        "source": source,
        "outputs": outputs or [],
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def _stream(text: str) -> _Json:
    return {"output_type": "stream", "name": "stdout", "text": text}


def _result(plain: str) -> _Json:
    return {
        "output_type": "execute_result",
        "data": {"text/plain": plain},
        "metadata": {},
        "execution_count": 1,
    }


def _image() -> _Json:
    """A display output that is only an image: no text, so nothing to index."""
    return {
        "output_type": "display_data",
        "data": {"image/png": "iVBORw0KGgoAAAANSUhEUg=="},
        "metadata": {},
    }


def _error(name: str, value: str, traceback: list[str]) -> _Json:
    return {
        "output_type": "error",
        "ename": name,
        "evalue": value,
        "traceback": traceback,
    }


def _write(path: Path, notebook: _Json) -> None:
    """Write the notebook with stable key order, so the bytes are the same on every run."""
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
