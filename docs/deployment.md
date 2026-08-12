# Deployment

What running manicule puts on disk, who can read it, how to copy it safely, and what a network
bind will cost you when there is something to bind. Written for the person who has to answer
"where did the documents go and who else can see them" — not for the person choosing a colour
scheme.

Two facts decide most of this document, and both are consequences of decisions taken
elsewhere:

- **The data directory is the corpus**, not a derived index of it (§1).
- **The index is not permission-aware**, so what one user can search, every user can search
  (§1.2).

---

## 1. What the data directory contains

`<data_dir>` defaults to `$XDG_DATA_HOME/manicule` — `~/.local/share/manicule` on a machine
with no `XDG_DATA_HOME` set — and is overridden by `data_dir` in the config file or
`MANICULE_DATA_DIR` in the environment. Inside it:

| | |
|---|---|
| `manicule.db` | SQLite: documents, chunks, versions, workspaces, API key digests, conversations |
| `vectors/` | The LanceDB table the dense leg searches |
| `blobs/sha256/…` | **The retained original bytes of every ingested document** |

That third row is the one to read twice.

### 1.1 It is a complete, verbatim copy of everything indexed

`docs/storage.md` §7 makes `original_ref` point at the source bytes as they were fetched, so
that fixing a parser bug means re-parsing rather than re-crawling. The consequence is
immediate and is stated here rather than left to be discovered: **every PDF, every attachment,
every wiki page body is in `<data_dir>/blobs`, byte-identical to the original.** Before
retention the directory held extracted text and vectors, and reconstructing the sources from
it would have been lossy and partial. It is not that any more.

So the directory is not "an index that can be rebuilt". It is a second copy of the corpus, and
it should be given whatever the corpus itself is given — the same disk encryption, the same
backup handling, the same answer to "may this leave the building".

Two exceptions, both narrow and both recorded rather than silent. A document above 256 MiB is
not retained — `manicule.storage.blobs.MAX_ORIGINAL_BYTES`, a constant rather than a setting —
and its `original_ref` is `NULL` with `original_omitted_reason` saying why. And
`storage.retain_source_bytes = false` turns retention off entirely, at the cost of making every
re-parse a re-fetch.

### 1.2 The index is not permission-aware

Content is fetched as whatever account the connector is configured with, so the index holds
everything **that** account can see, and anyone who can search the index can retrieve any of
it. `docs/connectors/confluence.md` §9 says this for Confluence and it is a property of the
whole system: source-side restrictions do not travel with the content, and they do not travel
with the retained bytes either.

The operational form of that: **the sync account is the blast radius.** Point a connector at
an account with access to exactly what the index is meant to hold. An admin token indexes
everything an admin can see, into a directory that is a verbatim copy, on a machine whose file
permissions are now the only access control left.

---

## 2. Filesystem permissions

`manicule.storage.engine.prepare_data_dir` creates `<data_dir>`, `vectors/` and `blobs/` mode
`0700`, and the blob store writes retained bytes `0600`. The modes are set explicitly rather
than left to the process `umask`, because a default that varies with the invoking shell is not
a default.

**`manicule doctor` fails — not warns — when `<data_dir>` carries any group or other
permission bit**, and names the directory and the mode:

```
   failing  permissions  /srv/manicule carries group or other permissions (055), so it is
            reachable by accounts other than the one running manicule. It holds the retained
            source bytes of every indexed document, which makes this an exposure of the
            corpus rather than a tidiness problem. Run `chmod 0700 /srv/manicule`.
```

Failing rather than degraded is deliberate. Nothing stops working when the directory is
world-readable; the corpus has simply been published to every account on the machine, and
nothing else in the system will ever mention it. A warning is the wrong shape for a fact that
has already happened.

The check is on the directory and only on the directory. POSIX gates every read on the modes
of every ancestor, so a directory nobody else can enter is one nobody else can read *through*,
whatever the files inside it say — and walking the blob store would make a diagnostic cost one
`stat` per retained document.

That distinction is load-bearing rather than theoretical: **`manicule.db` and its `-wal` and
`-shm` siblings are created `0644`**, because SQLite creates them and manicule does not chmod
them afterwards. Unreachable inside a `0700` directory, and worth knowing before copying one
of those files somewhere with a different parent. The copies manicule makes itself are the
exception: `backup` writes its snapshot database `0600` and `export` writes every file in an
archive `0600` (§3), because those copies are made to be moved.

**Do not branch a script on `manicule doctor`'s exit status.** It exits **0** whenever it
managed to produce a diagnosis, whatever the diagnosis says — which is the exit-status
contract in `docs/surfaces.md` §2 working as written, because producing the diagnosis is the
operation and it succeeded. To gate on health, read the envelope:

```bash
manicule --json doctor | jq -e '[.data.checks[] | select(.state == "failing")] | length == 0'
```

Note the position of `--json`: it is an option of `manicule`, not of `doctor`.

**The common way to fail this is to point `data_dir` at a directory you made yourself.**
`mkdir /srv/manicule` under the usual `umask 022` produces `0755`, and manicule will use it as
it found it. `chmod 0700` is the whole fix.

Running manicule as root is the other way, and it is worth saying plainly: the account that
runs manicule owns the corpus. Give it its own account, and the data directory to go with it.

---

## 3. Backup

`manicule backup --output <dir>` takes a consistent snapshot: the SQLite database through
SQLite's own online backup API, then the vector table, then the blob store — in that order, so
the derived stores can only ever be *ahead* of the database snapshot, never behind it. A
manifest records the schema revision, the embedding and chunking fingerprints and an inventory.
`manicule backup --restore <dir>` puts one back.

The output directory is created `0700`, and the snapshot database and manifest inside it
`0600`.

**A backup is a second copy of the corpus, and it is the copy most likely to end up somewhere
careless** — a shared drive, a `/tmp` scratch, an object store with a permissive bucket policy.
Everything §1 says about the data directory applies to it unchanged, so `backup` applies §2's
rule to where you send it:

```console
$ manicule backup --output /srv/share/manicule
backup failed: InsecureTargetError
backup target /srv/share/manicule carries group or other permissions (055), so what
manicule writes into it would be readable by accounts other than the one running
manicule. … Run `chmod 0700 /srv/share/manicule`, choose a target only this account
can read, or pass --allow-insecure-target to write it there knowingly.
```

The refusal names the path and the mode, and `chmod 0700` on that path is the whole repair.
It covers a directory that was already there, which is the one that matters: a target manicule
creates is `0700` because it made it, and a target you created last month is whatever your
`umask` said at the time.

`--allow-insecure-target` writes the backup anyway, unchanged and unhidden — for a volume
whose permissions are somebody else's decision, where the protection is the volume rather than
the mode. It is a consent, not a repair: nothing about the copy is different, and it is a
complete copy of every document indexed.

Two habits are still worth having:

- Prefer a target only the manicule account can read, rather than one manicule was told to
  accept. The refusal checks the directory it writes into; it has no opinion about a volume
  that is exported over NFS, replicated to an object store, or backed up by something else.
- Treat an off-machine backup as an export of the corpus, because that is what it is.

### 3.1 `export` is held to the same rule

`manicule export --output <dir>` is a different thing and is not a backup: it writes retained
bytes and metadata but never chunks or vectors, so the importing machine re-derives both with
its own fingerprints. It is how a corpus moves between installations. It is also a complete
copy of the source documents — which is why it refuses a group- or world-readable target in
the same words, with the same `--allow-insecure-target` to override it, from the same function
that decides for `backup`.

It is held to a **stricter** rule about the files inside, not a looser one. The archive
directory is `0700`, its `blobs/` shard `0700`, and **every file in it `0600`** — the manifest
and every retained document alike. A backup is written to sit still; an archive is written to
be carried, and a file copied out of a `0700` directory takes its own mode with it, not the
directory's. Until [#68](https://github.com/mgd43b/manicule/issues/68) an export asked for no
mode at all, so a fresh archive of the entire corpus landed `0755`/`0644` under the usual
`umask` — weaker than the backup path even before that was fixed.

### 3.2 `upgrade` takes one first, and you do not choose where

`manicule upgrade` takes a snapshot before it tells you how to upgrade. It writes to a sibling
of the data directory — `<data_dir>-backups/pre-upgrade-<unix-seconds>`, so
`~/.local/share/manicule-backups/…` for a default install — and reports the path it used:

```console
$ manicule --json upgrade | jq -r .data.backup
/home/manicule/.local/share/manicule-backups/pre-upgrade-1786539202
```

Beside the data directory rather than inside it, because `backup` refuses to snapshot a
directory into itself — the copy would include itself — and for a while that refusal met a
caller that asked for exactly that, so every `manicule upgrade` failed unless `--skip-backup`
was passed ([#66](https://github.com/mgd43b/manicule/issues/66)). Two consequences worth
knowing:

- **These accumulate.** One directory per upgrade, each a full copy of the corpus, each
  `0700`. Nothing prunes them; delete the old ones when you are satisfied the new version
  works.
- **`doctor` does not look there.** Its permissions check is about `<data_dir>`. The snapshot
  directory is created and verified `0700` by `backup` itself, but if you move one somewhere
  else, §1 applies to it from then on.

---

## 4. Binding a port

`manicule start` serves MCP over **stdio** by default, which opens no socket at all, and that
is still the ordinary way to run it. `manicule start --transport http` serves the HTTP API
([#11](https://github.com/mgd43b/manicule/issues/11)) on `security.transport.port`, 8765 by
default, **and the browser surface at `/ui`**
([#12](https://github.com/mgd43b/manicule/issues/12)) on the same socket. There is no separate
UI server and no second port.

The browser surface is for the loopback, single-operator installation. A browser cannot attach a
header to a page load and this build has no session cookie, so with `security.auth.mode` set to
anything but `none` a page load carries no credential and is refused — with a page saying so.
That is deliberate rather than a gap; [`web.md` §5](web.md#5-what-a-browser-cannot-present-said-plainly)
explains it, and an interactive login belongs to
[#13](https://github.com/mgd43b/manicule/issues/13). The practical consequence for a deployment
is that publishing a port gets you the API and the widget, and the pages will refuse — which is
the safe direction for the surface that renders the corpus.

**Publish to host loopback, with authentication on.**

```bash
docker run -p 127.0.0.1:8765:8765 …     # reachable from this machine
docker run -p 8765:8765 …               # reachable from the network. Not this.
```

The image still publishes nothing by default and `compose.yaml` still declares no `ports:`,
because serving HTTP is something an operator asks for rather than the default posture.

The bare form binds `0.0.0.0` on the host, which means every interface the machine has,
including the one facing the office network. A search index over a verbatim copy of the corpus
is precisely the service that must not be reachable by anyone who can route a packet to it.

manicule refuses the software half of this on its own. `manicule.app.bind.resolve_bind` wants
three separate things before it will bind anywhere but loopback: a non-loopback host somebody
wrote into configuration, `--allow-public-bind` on the command line where no config file can
supply it, and `security.auth.mode` set to something other than `none`. Any one missing is a
refusal, and `manicule doctor` reports a non-loopback bind as failing when authentication is
off.

**In a container the two decisions compose, and both are yours.** A loopback bind *inside* a
container is reachable only from inside it — `-p` forwards to the container's routable
address, not to its `127.0.0.1` — so publishing anything at all means the container-side bind
was already widened, which means manicule's three refusals were already satisfied and
authentication is already on. What `-p` then decides is which of the **host's** interfaces see
it, and `-p PORT:PORT` decides all of them. The guard inside the container cannot make that
choice for you and does not try to.

---

## 5. The container

[`Dockerfile`](../Dockerfile) and [`compose.yaml`](../compose.yaml) build an image that
carries everything it needs: the Python environment, the tree-sitter grammars as an installed
offline bundle, and the `BAAI/bge-m3` ONNX weights. The final build stage runs `doctor`,
`init`, an index and a search **with `--network=none`**, so an image that would have fetched
anything on first use fails to build.

What that buys, and what it costs:

- **No network at run time.** `HF_HUB_OFFLINE=1` is set in the image and the grammar bundle is
  installed as a distribution, so nothing reaches out mid-ingest.
- **About 3.4 GB**, of which 2.3 GB is the model. Weights are baked in rather than mounted
  from a cache volume so that the download happens during `docker build`, where a long step is
  expected, rather than inside a first `index` that appears to hang.
- **Unprivileged.** uid 10001, `/data` created `0700` and owned by it, `cap_drop: ALL` and
  `no-new-privileges` in the compose file. A named volume mounted at `/data` inherits the
  ownership and mode from the image, which is what keeps `doctor` passing.
- **No published port.** The image `EXPOSE`s nothing and the compose file declares no
  `ports:`, because the default command serves MCP over stdio. Publishing one is an operator's
  decision, and §4 is what it costs.

The container puts its config file in the data directory too — `MANICULE_CONFIG_FILE` is
`/data/config.toml` — so one named volume carries the whole installation. `MANICULE_CACHE_DIR`
is `/data/cache` for the same reason; that subdirectory is regenerable and is the one thing
under `/data` that is safe to delete.

**The backend is ONNX.** MLX is Apple silicon and no Linux container can use it. That changes
throughput and does not change output — the `backend parity (macOS)` CI job compares vectors
from both backends precisely so that this sentence can be a guarantee rather than a hope.

**Run MCP natively rather than in the container.** MCP over stdio means the client owns the
process, and putting `docker compose run` in that position adds failure modes the assistant
sees as a tool that will not start. Use the container for the CLI and for batch ingest. If a
native install and a container share a data directory they are the same index — but not at the
same time: SQLite's locking is per-file and a concurrent ingest from two processes is not
something this design has been exercised against.

### 5.1 Redistributing an image

manicule is **GPL-3.0-or-later**, and that reaches an image built from it. Publishing an image
to a registry others can pull is distribution: the corresponding source has to be available on
the same terms, including any changes made to it, and the licence text has to travel with it.
Building an image for your own machines is not distribution and none of this applies. See
[`LICENSE`](../LICENSE), and take advice before publishing an image containing modifications
you intend to keep private.

---

## 6. Still open

Not settled here, and deliberately:

- **Encryption at rest** and its key management
  ([#19](https://github.com/mgd43b/manicule/issues/19)). Today the answer is the volume's, not
  manicule's: full-disk encryption, or an encrypted filesystem under `<data_dir>`.
- **Whether retention is opt-out per connector**, and the split between retaining a live
  document's bytes and retaining prior versions' (`docs/storage.md` §7).
