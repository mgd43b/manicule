# Documents in a Git repository: the `git-docs` connector

Design and behavior of the `git-docs` connector, which indexes the Markdown and text in a Git
repository (including a repository of BMAD output) and keeps the index current.

> **Status: implemented.** The code for each section is named in §9. The `bmad-v7` layout is read
> from the 7.x **preview alpha**, which is still changing; §6 says what that costs.

[`filesystem`](filesystem.md) can already index a checkout, and [`git-site`](web-crawler.md) can
index a checkout that is a website. `git-docs` is for the other case, which is the common one: a
repository whose *documents* are wanted and whose *code* is not. Its three differences are what
this page is about:

- it reads **documents only**, so a large source tree costs one directory listing (§2);
- it can **reach a remote** itself, so the host running manicule needs no checkout and a push is
  seen at the next sync (§3); and
- it **recognizes BMAD's output folders**, so a citation says it is a PRD or an epic's
  retrospective and not `prd-checkout.md` (§6).

It ends at `RawDocument`, like every connector. The normal pipeline parses, chunks, embeds and
publishes; the Markdown and plain-text parsers are the existing ones.

## 1. Configuration

```toml
[connectors.planning]
type = "git-docs"
schedule_s = 900                 # a served manicule syncs this every fifteen minutes
reconcile_interval_s = 86400     # and looks for documents deleted upstream daily

[connectors.planning.options]
repository = "https://git.example.org/team/planning.git"   # or a path to a checkout
revision = "main"
token_env = "PLANNING_GIT_TOKEN"   # the variable's name; a private repository only
```

`GitDocsConfig` is frozen and rejects any other key. Periodic change detection is not part of
this connector: `schedule_s` and `reconcile_interval_s` are the scheduler's, as for every source
(`docs/ingest.md` §11, `docs/deployment.md`).

| Field | Default | Meaning |
|---|---|---|
| `repository` | required | An `https` address, or the path of a local repository. Credentials in an address are refused. |
| `revision` | `HEAD` | Branch, tag or commit. Resolved to one commit at the start of every sync. |
| `content_root` | `.` | A folder bounding what is read. Documents are still identified by their path from the repository root. A root with no files in it is an error, not an empty corpus. |
| `include` | `**/*.md`, `**/*.markdown`, `**/*.txt` | Path globs admitted. |
| `exclude` | dependency folders and agent-framework folders (§2) | Path globs refused. Setting it **replaces** the default. |
| `layout` | `auto` | `auto`, `bmad-v6`, `bmad-v7` or `none` (§6). |
| `token_env` | none | Name of the variable holding a token. Only for an `https` repository. |
| `username` | `x-access-token` | The user name sent with the token. |
| `blob_url` | derived | Where a reader opens a document, with `{commit}` and `{path}` (§5). |
| `max_bytes` | 2 MiB | Largest document read. A larger one is skipped and counted. |
| `fetch_timeout_s` | 300 | Longest one fetch may take. |

Everything that is wrong is refused when configuration is loaded, naming the setting and never
echoing the value, because the field somebody pastes a token into by mistake is `repository`.

## 2. What is read

The tree of one commit is listed, and a path is admitted when `include` matches it and `exclude`
does not. Only admitted blobs are ever read; code is never fetched from Git's object store at all.

The default `exclude` removes `node_modules`, `vendor`, `third_party`, `.venv` and the folders an
agent framework installs into a project: `_bmad`, `.claude`, `.agents`, `.cursor`, `.codex` and
`.gemini`. BMAD alone writes hundreds of Markdown files into `_bmad/` and the tool folders, and every
one is a prompt for a tool rather than a document about the project. Admitting them would bury the
project's own words, which is why this is a default and not a recommendation.

Three kinds of path are admitted by a glob and still not read, and each is **counted** on the
watermark (`skipped_oversized`, `skipped_unsupported`) and in a warning, and left out of
reconciliation too, so a document that grows past `max_bytes` leaves the index instead of lingering
at its last small version:

- a blob over `max_bytes`;
- a suffix with no media type (`.md`, `.markdown`, `.mdx` and `.txt` have one, matched
  case-insensitively), so `include = ["**/*"]` does not hand a PNG to a text parser; and
- a path that is not printable or is longer than a citation field can carry.

Symbolic links and submodules are never followed, because the reader admits only ordinary blobs.

## 3. Reaching a remote

`PinnedGitReader` reads a repository that is already on disk and never fetches, deliberately. For an
`https` repository `git-docs` adds the other half, [`git_mirror.py`](../../src/manicule/connectors/git_mirror.py):
a **bare** mirror under the cache directory, refreshed before every sync, whose tip is fetched into a
ref of its own and handed to the reader as a path and a ref name.

- **The mirror is a cache.** It holds nothing the remote does not. Deleting
  `<cache>/git-mirrors/<digest>` costs one fetch. Two sources naming one address share one mirror
  and take turns fetching it.
- **Only the tip is fetched** (`--depth 1`): the commit and the blobs in its tree, which is all a
  source that indexes one commit can use. A blobless clone would fetch less and was rejected,
  because a lazy fetch is one round trip per document and what a server does for it varies.
- **The credential is never on disk, in the URL or on a command line.** It is sent as
  `http.<origin>/.extraHeader` through `GIT_CONFIG_*` in the child's environment, scoped to the
  origin so a redirect to another host carries nothing. Git's own error text is kept because it
  finds the problem, with the token removed in both forms it can take.
- **Only `https` is spoken** (`GIT_ALLOW_PROTOCOL`), hooks point nowhere, credential helpers are
  disabled and nothing can prompt. The commit a fetch brought is resolved to its id while the
  mirror's lock is still held, and the reader is given that id and never the ref, which names
  whatever the next fetch brings. SSH is absent on purpose: it needs a key file and a trusted
  `known_hosts`, which is two more things to provision into a container and two more ways for a
  fetch to hang.
- **One ref per revision.** Two sources on one remote share its mirror and may name different
  revisions, so each lands in a ref of its own (`refs/manicule/<digest of the revision>`). One
  shared ref would let the later fetch replace the commit the earlier source was about to read.
- **A private certificate authority** is trusted through the environment, not through Git's own
  configuration, which is deliberately not read: set `SSL_CERT_FILE` (or `SSL_CERT_DIR`) for the
  manicule process, and Git's TLS library honors it.
- **A fetch is bounded.** `fetch_timeout_s` stops the whole process tree, not only its parent, so
  `git-remote-https` cannot outlive the sync that started it.

`setup()` does not touch the network, because the container awaits it before it hands the connector
to anyone, and a network that is down at startup must not stop an installation from serving what it
has already indexed. The first fetch is the first sync.

A **local path** is read as it stands; whatever keeps it current is the operator's. It needs no
mirror and rejects `token_env`.

## 4. One commit, start to finish

The revision is resolved to a commit when a sync starts. Discovery, every fetch and the
reconciliation that follows read that commit's objects, never a working tree, so a push or a
checkout while a sync runs cannot hand it half of two states.

- **Identity is the path from the repository root.** A rename is a deletion and an addition, which
  is what `reconcile` and content-hash deduplication are for.
- **The change token is the blob's object id** (plus the BMAD facts, the policy version and the
  citation address's template, §5 and §6). It
  is **not** the commit and not the address, both of which move with every push; a token that did
  would re-embed the whole corpus each time anybody touched a line of code. A commit that changed no
  document changes no token, and nothing is fetched.
- **The watermark is the commit, and it is how a quiet repository costs nothing.** It is recorded
  after a complete run and handed back to the next one. If the revision still resolves to that
  commit, under the same scope and the same version of this connector's own rules, discovery
  **lists nothing and fetches nothing**: a check every few minutes of a repository nobody has
  pushed to is one `rev-parse`, and for a remote the fetch that precedes it, which transfers
  nothing when nothing moved. Any other commit lists the admitted tree, and each document's blob-id
  token decides whether its bytes are read. The rules are compared as well as the commit so that
  a release that changes how a document is described re-ingests at a commit nobody has pushed
  to, and the token covers the citation address's template (never the address itself, which holds
  the commit), so correcting `blob_url` re-cites documents that did not change. A diff against the previous commit would list less and was not used: a depth-1 mirror
  usually no longer has the previous commit.
- **An empty manifest is not an empty source.** Because a quiet run lists nothing, the connector
  does not declare that its manifests are full inventories. It reports instead, after each run,
  whether that run listed the tree (`walked_full_inventory`), as the mail connectors do, so only a
  run that did is evidence for deletion and coverage.
- **A reference survives a restart.** The durable pipeline can resume an interrupted run and fetch
  its journaled references without discovering again. A reference names its commit, so a process
  that holds no inventory for it rebuilds one from that commit's tree, from the mirror or the local
  repository, with no fetch and no lookup of the revision. If the repository no longer holds the
  commit, the error says to sync again.
- **A reference is checked against its inventory** before a blob is read. The inventory, not the
  reference, is the authority; a forged one is refused.
- **Reconciliation** inside a sync reuses that sync's inventory. Anywhere else (a scheduled pass, a
  manual command, a pass after a restart) it pins a fresh one, which for a remote **fetches first**.
  A fetch that fails raises, and a pass that raises deletes nothing (`docs/ingest.md` §11.1): the
  paths seen so far would be a prefix, not the truth. Beyond that, the usual guards apply: a
  deletion ceiling, and soft deletes only.

## 5. Citations

`DocRef.uri` and the citation's canonical address come from `blob_url`. Unset, an `https` repository
gets `<repository>/blob/{commit}/{path}`, which GitHub, GitHub Enterprise and GitLab serve; set it for
a forge that spells it differently (Gitea and Forgejo use `/src/commit/{commit}/{path}`). A local
path has no address, so it cites by its repository path and a `file:` location, and the record's
`canonical_uri` is left empty rather than made to claim a `file:` address is the publication.

The address names the commit **at which the document last changed**, since the stored record is only
rewritten when its token moves. The bytes there are the bytes indexed, so the link is exact; it just
does not move forward with every unrelated push.

The citation also carries the title (the file name without its suffix), the blob id as the version,
and the directories above the file as the section path.

## 6. BMAD

`layout = "auto"` recognizes BMAD's output folders by their **shape**, at any depth, and records
what a document is, as [`bmad_layout.py`](../../src/manicule/connectors/bmad_layout.py) defines. It
reads no file and consults no configuration, so a v7 store that is a repository of its own, with the
output folder as the root, is recognized exactly as one under `_bmad-output/`.

| Layout | Recognized by | Facts recorded |
|---|---|---|
| `bmad-v6` | `planning-artifacts/`, `implementation-artifacts/`, `sprint-status.yaml` | `prd`, `architecture`, `epic`, `ux-design`, `product-brief`, `project-context`, `story` (with its epic number), `retrospective`, `tech-spec`, and others |
| `bmad-v7` | `initiative-<slug>/initiative-<slug>.md`, `tickets/tickets.toml`, a `backlog/` beside an initiative | the same kinds, plus the initiative and epic slugs, `story-plan` and `memlog` |

They are stored under the connector's own keys (`bmad_layout`, `bmad_kind`, `bmad_initiative`,
`bmad_epic`), not on the general provenance record, which has no field for any of them and gained
none. The layout found is also on the watermark and in the sync's log line.

**This is enrichment, never a gate.** It changes what is *known* about a document and never what is
*admitted*, so a path that matches no rule is indexed like any other, and `layout = "none"` indexes
the same set of documents without the facts. A rule that stops matching after BMAD renames a folder
costs citation detail and nothing else. Changing the facts changes the token, so documents are
re-ingested with the new description instead of keeping one nobody would now produce.

**What is not done.** `tickets.toml` and `sprint-status.yaml` carry epic and story status, and are not
Markdown or text, so the defaults do not admit them. Add `**/*.toml` or `**/*.yaml` to `include` to
read them; the structured parser handles both. Both layouts are matched by path, not by reading
BMAD's own configuration, which is deliberate: that file is the thing most likely to move.

**v7 is a preview.** Its layout was read from the BMAD-METHOD `main` branch at commit
`a867e160` (November 2026), whose own README says to expect breaking changes, and the release was
pending. `bmad migrate method` moves a v6 tree into the v7 layout; to this connector that is a mass
rename, which is a mass deletion and a mass addition. The reconciliation ceiling will refuse the
deletion and surface a proposal for a person to confirm, which is the right behavior for a move
that large. The v6 shapes were read from the 6.12 branch and the upgrade guide.

## 7. What is not here

- **SSH, and plain `http`.** See §3.
- **Credentials in the address.** Refused at configuration.
- **Submodules.** Their contents are not in the commit's tree.
- **A working tree.** An uncommitted draft is not a pinned object; a source that indexed one would
  have the consistency problem `git-site` describes, and `filesystem` is the connector for it.
- **Permission awareness.** As for every source, the index is not permission-aware: a private
  repository's documents are readable by everyone who can search the workspace.

## 8. Operation

- **A sync that fails** reports the failure and the scheduler tries again at the next interval.
  A wrong token, a missing revision and an unreachable host each say so, naming the address.
- **A source nobody has synced** has nothing to reconcile, so it is not asked to.
- **Moving a token to another variable, or changing the timeout, is not a new source.** Everything
  that decides which documents there are, or what is said about them, is part of the scope; how
  the remote is reached is not.

## 9. Where it lives

```text
src/manicule/connectors/git_docs.py     inventory, discovery, fetch, reconciliation
src/manicule/connectors/git_mirror.py   the bare mirror, the fetch, the credential
src/manicule/connectors/git_remote.py   address and revision validation (no Git imported)
src/manicule/connectors/bmad_layout.py  path to BMAD facts
src/manicule/connectors/config.py       GitDocsConfig
src/manicule/connectors/plugin.py       registration and the factory (reads token_env)
tests/connectors/test_git_docs.py       what is read, one commit, identity, deletion, BMAD, a remote
tests/connectors/test_git_mirror.py     the fetch, the credential, the allowlist, the timeout
tests/connectors/test_bmad_layout.py    both layouts, and the folders that only share a name
```
