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
of those files somewhere with a different parent.

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

The output directory is created `0700` and its manifest `0600`.

**A backup is a second copy of the corpus, and it is the copy most likely to end up somewhere
careless** — a shared drive, a `/tmp` scratch, an object store with a permissive bucket policy.
Everything §1 says about the data directory applies to it unchanged. Two habits are worth
having:

- Write backups somewhere only the manicule account can read, and check with `ls -ld` rather
  than assuming; manicule sets the mode of the directory it creates and has no opinion about
  the volume you created it on.
- Treat an off-machine backup as an export of the corpus, because that is what it is.

`manicule export --output <dir>` is a different thing and is not a backup: it writes retained
bytes and metadata but never chunks or vectors, so the importing machine re-derives both with
its own fingerprints. It is how a corpus moves between installations. It is also a complete
copy of the source documents.

---

## 4. Binding a port

There is nothing to publish today. `manicule start` serves MCP over **stdio**, which opens no
socket at all; the HTTP API and the web UI are not built
([#11](https://github.com/mgd43b/manicule/issues/11)). The principle is recorded now so that it
is not invented under time pressure later.

**Publish to host loopback, with authentication on.**

```bash
docker run -p 127.0.0.1:PORT:PORT …     # reachable from this machine
docker run -p PORT:PORT …               # reachable from the network. Not this.
```

`PORT` is a placeholder on purpose: nothing here has a port yet, and putting a number in this
document would be inventing one before the surface that owns it exists.

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
- **No published port**, because there is nothing to publish. §4 applies when there is.

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
