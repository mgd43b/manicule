#!/usr/bin/env python3
"""Extract user-visible capabilities from OpenDocuments.

Prose summaries of a 33k-line codebase lose things, so this extracts rather than
describes. It emits **what a user can do** — commands, tools, endpoints, file
types, sources, settings.

It deliberately does NOT emit internal structure. Exported symbols, database
columns and class names are OpenDocuments' implementation, and reproducing those
would be cloning rather than building something better. `--internals` dumps them
anyway, as a reference for checking whether a behaviour was considered — never as
a specification.

Usage:
    uv run tools/extract_surface.py /path/to/OpenDocuments > CAPABILITIES.md
    uv run tools/extract_surface.py /path/to/OpenDocuments --internals
"""

# /// script
# requires-python = ">=3.10"
# ///

import re
import sys
from pathlib import Path


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def cli_commands(root: Path) -> list[str]:
    """Commander.js command names and their flags."""
    out = []
    for f in sorted((root / "packages/cli/src/commands").glob("*.ts")):
        src = read(f)
        for name in re.findall(r"\.command\(\s*['\"]([^'\"]+)['\"]", src):
            out.append(f"`{name}`")
        for flag in re.findall(r"\.option\(\s*['\"]([^'\"]+)['\"]", src):
            out.append(f"{f.stem} — option `{flag}`")
    return out


def mcp_tools(root: Path) -> list[str]:
    src = read(root / "packages/server/src/mcp/server.ts")
    return [f"`{n}`" for n in re.findall(r"name:\s*'([a-z_]+)'", src)]


def http_endpoints(root: Path) -> list[str]:
    out = []
    for f in sorted((root / "packages/server/src/http/routes").glob("*.ts")):
        src = read(f)
        for verb, path in re.findall(
            r"app\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]", src
        ):
            out.append(f"`{verb.upper():6} {path}`")
    return out


def db_columns(root: Path) -> list[str]:
    """Every column of every table, from the migration SQL."""
    out = []
    for f in sorted((root / "packages/core/src/storage/migrations").glob("*.sql")):
        src = read(f)
        for m in re.finditer(r"CREATE TABLE (?:IF NOT EXISTS )?(\w+)\s*\((.*?)\n\s*\);", src, re.S):
            table, body = m.group(1), m.group(2)
            for line in body.splitlines():
                line = line.strip().rstrip(",")
                if not line or line.upper().startswith(
                    ("PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT")
                ):
                    continue
                col = line.split()[0].strip('"`')
                if col:
                    out.append(f"`{table}.{col}`")
    return out


def plugin_methods(root: Path) -> list[str]:
    src = read(root / "packages/core/src/plugin/interfaces.ts")
    out = []
    for m in re.finditer(r"export interface (\w+)[^{]*\{(.*?)\n\}", src, re.S):
        iface, body = m.group(1), m.group(2)
        for line in body.splitlines():
            line = line.strip()
            sig = re.match(r"(\w+)\??\s*[(:]", line)
            if sig and not line.startswith("//"):
                out.append(f"`{iface}.{sig.group(1)}`")
    return out


def parser_types(root: Path) -> list[str]:
    out = []
    for f in sorted(root.glob("plugins/parser-*/src/index.ts")):
        src = read(f)
        m = re.search(r"supportedTypes\s*=\s*\[(.*?)\]", src, re.S)
        if m:
            for ext in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)):
                out.append(f"`{ext}` ({f.parent.parent.name})")
    for f in sorted((root / "packages/core/src/parsers").glob("*.ts")):
        src = read(f)
        m = re.search(r"supportedTypes\s*=\s*\[(.*?)\]", src, re.S)
        if m:
            for ext in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)):
                out.append(f"`{ext}` (core/{f.stem})")
    return out


def config_keys(root: Path) -> list[str]:
    """Top-level and nested keys from the config defaults."""
    src = read(root / "packages/core/src/config/defaults.ts")
    return [f"`{k}`" for k in sorted(set(re.findall(r"(\w+):\s*[\{'\"\d\[]", src)))]


def core_exports(root: Path) -> list[str]:
    src = read(root / "packages/core/src/index.ts")
    out = []
    for m in re.finditer(r"export\s*\{([^}]*)\}", src, re.S):
        for name in m.group(1).split(","):
            name = name.strip().removeprefix("type ").split(" as ")[0].strip()
            if name:
                out.append(f"`{name}`")
    return out


# Subsystems whose behaviour is deliberately not ported.
SKIP = ("web/src/lib/i18n",)  # internationalisation is out of scope


def behaviours(root: Path) -> list[str]:
    """Every exported symbol in every source file.

    The interface extractors above read 46 of ~194 source files. The rest —
    chunking, the RAG engine, the ingest pipeline, connector management — is
    where the behaviour lives, and where every gap found so far has been.

    This does not capture semantics. It captures existence, so that nothing is
    invisible: every behaviour gets a row that must be ticked or struck.
    """
    out = []
    roots = [
        root / "packages/core/src",
        root / "packages/server/src",
        root / "packages/cli/src",
        root / "packages/client/src",
    ]
    for base in roots:
        pkg = base.parent.name
        for f in sorted(base.rglob("*.ts")):
            rel = f.relative_to(base).with_suffix("")
            if ".test" in f.name or any(sk in str(f) for sk in SKIP):
                continue
            src = read(f)
            syms = re.findall(
                r"^export\s+(?:async\s+)?(?:function|class|const|interface|type|enum)\s+(\w+)",
                src,
                re.M,
            )
            for sym in syms:
                out.append(f"`{pkg}/{rel}` → `{sym}`")
    for f in sorted(root.glob("plugins/*/src/index.ts")):
        src = read(f)
        for sym in re.findall(r"^export\s+(?:default\s+)?class\s+(\w+)", src, re.M):
            out.append(f"`{f.parent.parent.name}` → `{sym}`")
    return out


# What a user can do. This is the parity target.
CAPABILITIES = [
    ("CLI", cli_commands, 8),
    ("MCP tools", mcp_tools, 8),
    ("HTTP endpoints", http_endpoints, 11),
    ("File types", parser_types, 4),
    ("Settings", config_keys, 1),
]

# Implementation detail. Reference only — NOT a to-do list.
INTERNALS = [
    ("Database columns", db_columns, 2),
    ("Plugin interface", plugin_methods, 1),
    ("Core exports", core_exports, 1),
    ("Exported symbols per module", behaviours, 1),
]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    if not (root / "packages/core").is_dir():
        sys.exit(f"not an OpenDocuments checkout: {root}")

    internals = "--internals" in sys.argv
    sections = INTERNALS if internals else CAPABILITIES

    if internals:
        print("# OpenDocuments internals — reference only\n")
        print(
            "**This is not a to-do list.** It is OpenDocuments' implementation "
            "structure, kept so a behaviour can be checked against it — *did they "
            "handle X, and how?* Reproducing this list would be cloning.\n"
        )
    else:
        print("# Capabilities\n")
        print(
            "What a user can do, extracted from the OpenDocuments source rather than "
            "summarised. This is the capability floor manicule should meet or "
            "deliberately decline.\n"
        )
        print(
            "Tick when built. Strike through when dropped, with a reason. These are "
            "user-visible capabilities — *how* they are implemented is open, and in "
            "several cases should differ.\n"
        )
        print("Regenerate:\n")
        print("```bash\nuv run tools/extract_surface.py ../OpenDocuments > CAPABILITIES.md\n```\n")

    total = 0
    counts = []
    body = []
    for title, fn, ticket in sections:
        items = sorted(set(fn(root)))
        total += len(items)
        counts.append((title, len(items), ticket))
        body.append(f"\n## {title} — {len(items)}\n")
        body.append(f"Ticket: #{ticket}\n")
        for i in items:
            body.append(f"- [ ] {i}")

    print("| Area | Items | Ticket |")
    print("|---|---:|---|")
    for title, n, ticket in counts:
        print(f"| {title} | {n} | #{ticket} |")
    print(f"| **Total** | **{total}** | |")
    print("\n".join(body))


if __name__ == "__main__":
    main()
