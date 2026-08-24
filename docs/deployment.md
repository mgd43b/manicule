# Deployment

What running manicule puts on disk, who can read it, how to copy it safely, and what a network
bind will cost you when there is something to bind. Written for the person who has to answer
"where did the documents go and who else can see them" — not for the person choosing a color
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
`storage.retain_source_bytes = false` changes the installation default, at the cost of making
every re-parse a re-fetch. A connector may set `retain_source_bytes = true` or `false` beside
`enabled`; omitting it inherits the storage default. This is useful when remote sites need an
offline copy but a durable local Git checkout should remain the only copy of its own bytes.

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
manicule doctor --json | jq -e '[.data.checks[] | select(.state == "failing")] | length == 0'
```

`--json` goes on either side of the command name; `manicule --json doctor` is the same
invocation. Select on `.name`, which is the stable identifier — never on `.detail`, which is
prose and is free to be reworded. Each check also carries `.facts` for the numbers behind that
sentence and `.remedy` for the suggested fix, so a gate that reports *what to do* does not have
to parse English to find it:

```bash
manicule doctor --json | jq -r '.data.checks[] | select(.state == "failing") | "\(.name): \(.remedy)"'
```

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
([#12](https://github.com/mgd43b/manicule/issues/12)) **and MCP at `/mcp/`** on the same socket.
There is no separate UI server, no separate MCP server and no second port.

MCP served that way carries the **read-only tools only** — every mutating tool is absent from it
rather than refused on it, for the reason `docs/surfaces.md` §6.1 gives. Over stdio, where one
client talks to one process down a pipe, the whole surface is offered.

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

### 5.1 An install with no network, outside the container

The image carries what it needs because the build put it there. A native install on an
air-gapped host has to be given the same two things, and neither of them ships in a wheel.

**Grammars** are [`parsing.md`](parsing.md) §8.1.1's offline bundle, built with
`tools/build_grammar_bundle.py` on a machine that has network access and the same platform.

**BPE vocabularies** are the same story arriving through `tiktoken`, and the failure it caused
is worth knowing because it does not look like a missing artifact: the host **indexes
perfectly**, which reads as a working install, and then fails at the first question with a
connection error naming a blob storage host. Three things fix it, in increasing order of
effort:

```sh
# 1. On a host that has a network — the ordinary case, and `manicule init` already did it.
#    5.3 MB, once. The same command repairs an install that was made before it had a route.
manicule doctor --fix

# 2. On a host that never will. Build the bundle where there is a network:
python tools/build_vocabulary_bundle.py --output dist/vocabularies
#    copy `dist/vocabularies` across, point manicule at it, and seed from it:
export MANICULE_VOCABULARY_BUNDLE=/opt/manicule/vocabularies
manicule doctor --fix

# 3. Or build it into a distribution, so it installs like any other dependency
#    and needs no environment variable at all:
python tools/build_vocabulary_bundle.py --output build/pkg --package
```

`init` and `doctor --fix` seed the vocabularies through the same call, alongside the grammars,
so an install and a repair cannot come to mean different things. A pre-seed that could not
complete is a **note in the report, not a refusal to finish**: the configuration is written,
and an air-gapped host with no bundle can still install software it is perfectly able to run
once the bundle arrives. `doctor` then reports it as **failing** — not degraded, which is what
a missing grammar is, because a corpus of Markdown works fine without a grammar and nothing
works without a vocabulary.

**Where the vocabularies live, and why it is not where `tiktoken` would put them.** With
nothing set, `tiktoken` caches under the system temporary directory — and a temp sweep that
removes 5 MB turns a working install into one that refuses every question, weeks later, with
nothing having changed and nothing having said so. macOS sweeps `$TMPDIR` on a schedule. So
manicule's default is `$XDG_CACHE_HOME/manicule/data-gym-cache`, which nothing reclaims on a
timer, and it points `TIKTOKEN_CACHE_DIR` at that directory for the duration of every call
that reads or writes the cache — because `tiktoken` is the reader, and a manicule that merely
*answered* differently would seed one directory and leave the library looking in another.

**`TIKTOKEN_CACHE_DIR` still wins wherever it is set**, and remains how a deployment moves the
cache: the image sets it to `/opt/manicule/tiktoken`, and a read-only deployment can point it
straight at a bundle's `vocab/` directory, which is laid out as a `tiktoken` cache and needs no
copy. `manicule doctor` reports the cache as **degraded** when it resolves to somewhere under
the system temporary directory — which, after this default, takes somebody having put it
there.

manicule does not redistribute these files and will not: OpenAI publishes them with no SPDX
license expression, so manicule cannot state the terms under which it would be handing them
on. The bundle manifest records the URL every file came from, so whoever carries one can see
exactly what is in it.

**Model weights** are the third, and they are a different kind of artifact: 2.3 GB of ONNX
export, not something anyone carries in a manifest. They are pre-seeded rather than bundled —
`uv run tools/prefetch_embedding_models.py`, or point `embedding.model` (or a backend's
`weights`) at a local directory — and `HF_HUB_OFFLINE=1` is what stops the hub looking at all.
Set it on any host that has pre-seeded: a first search on a machine with a slow route to
huggingface.co otherwise spends half a minute in what looks like a hang. A model that is
neither cached nor reachable now refuses naming the pre-seed rather than surfacing the hub's
own exception mid-query.

**Nothing on the query path downloads silently.** A vocabulary that was never seeded is a
refusal while retrieval is being assembled — naming the encoding, the cache that was read and
where a bundle was looked for — rather than a download at the first question. A model that is
not on the machine says which command supplies it. And the model table the generation provider
library fetches at import is switched off: manicule sets `LITELLM_LOCAL_MODEL_COST_MAP` unless
you have, so both an air-gapped host and a networked one read the table pinned in the lockfile
instead of whichever revision a GitHub repository is serving today.

### 5.2 Redistributing an image

manicule is **MIT**, so publishing an image built from it obliges you to little: keep the
copyright notice and the license text with it. Modifications need not be published. The
`Dockerfile` here builds an MIT image — it installs the `onnx` embedding backend, and nothing in
its dependency closure is copyleft.

**Two things in an image can still carry terms of their own, and neither is manicule.**

- **`manicule-mlx`, if you add it.** The Metal backend is GPL-3.0-or-later, and it is Apple
  silicon only — so it has no place in a Linux image and the shipped `Dockerfile` does not
  install it. An image that did would be a GPL-3.0 distribution: corresponding source on the
  same terms, including modifications.
- **Bundled third-party binaries.** `pypdfium2`'s wheels ship a `BUILD_LICENSES/` directory
  that must be carried through, and `selectolax`'s wheel contains an LGPL-2.1 engine alongside
  the permissive one manicule imports. Both are ordinary redistribution conditions and both
  predate any of this.

See [`LICENSE`](../LICENSE), and take advice if you are packaging manicule together with
anything copyleft.

---

## 6. Running manicule as a server

`manicule serve` is the long-lived writer. It takes the data directory's lock for its whole life
(§2, and `docs/ingest.md` §6.4), runs whatever schedule configuration asks for, holds any
captured Confluence session in memory, and answers write commands from the command line over a
Unix domain socket. `manicule start` is the same command under its older name; both work.

**What this changes for somebody using manicule, in one table.**

| | With a server running | With none |
|---|---|---|
| `search`, `ask`, `doctor`, `document list`, `connector snapshot/verify`, `rebuild plan/status` | work | work |
| `connector sync`, `index`, `document reindex`, `rebuild execute`, the repair verbs | run **in the server** | refused, naming `manicule serve` |
| `connector login` | captures here, hands the session to the server | refused before it opens a browser |
| A Confluence sync | uses the session the server holds | there is no session to use |

The refusal is deliberate and it is not a fallback. A Confluence session lives in the server's
memory and nowhere else, so a sync in a process that is not the server has no credential and
never will; falling back to a local run would mean falling back to a run that cannot
authenticate, reported as a sync that failed rather than as an arrangement nobody set up. And
**nothing starts a server for you**. A command line that spawned a background writer you did not
ask for would hold the lock, outlive your terminal, and be found later by somebody wondering what
has the data directory.

`manicule init` is the one write that does not need a server, because it writes the
configuration file a server reads at startup — requiring one first would be a cycle.

### 6.1 The control socket

A `0600` Unix domain socket, inside a `0700` directory owned by you, in this machine's runtime
directory: `$XDG_RUNTIME_DIR` on Linux, `$TMPDIR` on macOS, `/tmp/manicule-<uid>` otherwise. It
is named for a digest of the data directory it serves, so one data directory has one socket and
two have two.

**Not the HTTP API**, for two reasons that are worth keeping separate. A route that started a
sync would cross the line `docs/surfaces.md` draws and `tests/api/test_routes.py` asserts by
name — the destructive operations have no route, and #113 refused one for corpus-wide reparse
because an unattended caller could hold the accelerator for an hour. And an HTTP control channel
would drag the bind policy (§4) and the authentication model into a question a socket with no
network does not raise. **The bind policy is unchanged by any of this**, and the socket is not a
way around it: it is unreachable from the network, and what protects it is the filesystem.

**It is not in the data directory**, which is where the pid file is, and the reason is measured
rather than stylistic: `sockaddr_un.sun_path` is 104 bytes on macOS and an ordinary data
directory under a per-user temporary path is already past that before a filename is added.

The client checks the owner and the mode before it writes a byte. A socket anybody else can
reach is refused rather than used.

### 6.2 Scheduled syncs

Per source, beside the source:

```toml
[connectors.handbook]
type = "confluence"
enabled = true
schedule_s = 3600          # sync this source hourly

[connectors.runbooks]
type = "filesystem"
schedule_s = 600           # and this one every ten minutes
retain_source_bytes = false # use the durable local tree instead of copying its bytes
```

Four things about it, each of which is a decision rather than an accident:

- **Only a server runs it.** An unserved installation has no process to run a schedule in.
- **A source with no `schedule_s` is not scheduled**, which is every configuration written
  before this existed.
- **`enabled = false` is honored.** A disabled source is never scheduled, whatever its
  `schedule_s` says — a schedule is exactly where a source you turned off would come back to
  life without anybody typing anything.
- **The first run is one interval after startup, not at startup.** Restarting is how a session
  is re-taken, so it is something you will do deliberately and often; a server that swept every
  scheduled source on startup would turn each of those into a full corpus sync nobody asked for.

A source never overlaps itself: the loop awaits its own sync before waiting again, so a source
whose sync takes longer than its interval runs back to back. A sync that fails does not stop the
schedule — the failure is reported and the next interval is tried, because an instance that was
down at ten past may be up at twenty past, and a session that expired needs a person rather than
a stopped loop.

`schedule_s` existed before and was deleted in #98, because it configured a scheduler that did
not exist and would have been cited as evidence that one did. It is back because one does.

### 6.3 Running it under launchd

`tools/launchd/com.manicule.server.plist` is a template. Four paths in it contain `REPLACE` —
the `manicule` executable, the two log files and the working directory — so `grep REPLACE` on
your copy is how you know you have finished editing it. Copy, edit, check, load:

```bash
mkdir -p ~/Library/Logs/manicule
cp tools/launchd/com.manicule.server.plist ~/Library/LaunchAgents/
$EDITOR ~/Library/LaunchAgents/com.manicule.server.plist   # `which manicule` gives the first
grep REPLACE ~/Library/LaunchAgents/com.manicule.server.plist   # silence means it is finished
plutil -lint ~/Library/LaunchAgents/com.manicule.server.plist   # and that it is still a plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.manicule.server.plist
```

An **absolute** path to `manicule` matters more than it looks: launchd does not read your
shell's profile, so `PATH` is the system default and a bare `manicule` is a job that fails to
start with nothing useful in the log. So does `WorkingDirectory` — launchd's default is `/`, and
a relative `[connectors.<name>] root` would resolve against the root of the disk.

`launchctl print gui/$(id -u)/com.manicule.server` says whether it is running and what it last
exited with. To stop it: `launchctl bootout gui/$(id -u)/com.manicule.server`.

**A user agent, not a system daemon.** manicule installs per user with no privileged component:
the data directory is in your home, the sessions are yours, and the control socket is `0600`
owned by you. A root daemon would put a verbatim copy of everything you indexed (§1.1) behind an
account you do not log in as.

**Restarting does not fight the lock, and a refusal says which kind it is.** The data directory
has one writer (§2), so during a restart the outgoing process may still hold it when the incoming
one starts. That is not a fault and it fixes itself. A wrong setting is a fault and does not. So
`manicule serve` exits with one of two statuses:

| Status | Meaning | What is in the way | What to do |
|---|---|---|---|
| **75** `EX_TEMPFAIL` | try again later | another manicule holds the data directory, or something answers on the control socket | nothing — it clears when that process goes |
| **78** `EX_CONFIG` | somebody has to fix it | a bind this configuration forbids, a plugin that will not import, a data directory that cannot be opened, a runtime directory another account owns or others can read | read the message; it names the setting or the path |

**launchd restarts the job either way, and it is worth being exact about that** because the
opposite is widely assumed. `launchd.plist(5)`'s `KeepAlive` conditions are `SuccessfulExit`,
`PathState`, `OtherJobEnabled` and `Crashed` — **none of them keys on an exit status** — so there
is no way to tell launchd "stop retrying this one". What bounds the loop is `ThrottleInterval`:
one attempt every thirty seconds rather than a spin, for both.

The status is therefore a *diagnostic*. `launchctl print gui/$(id -u)/com.manicule.server` reports
the last exit status, so a service that keeps coming back tells you in one line whether it is
waiting for a process to let go (75, and it will stop on its own) or waiting for you (78, and it
will not). A repeating 78 with no repair is a server that will never come up; the log line beside
it names the setting.

A claim you will find repeated elsewhere — that launchd declines to restart a job exiting
`EX_CONFIG` — is not in the manual page, and nothing here depends on it.

The lock itself needs no cleanup. It is an `flock` held by an open file description, so the
kernel releases it however the process died — including `SIGKILL` and a power cut. The lock
*file* is left behind and is a diagnostic rather than a lock; nothing has to delete it.

**A restart signs you out, and that is the design rather than a bug.** A Confluence session lives
in the server's memory and nowhere else, so a crash, a logout or a reboot ends it. What manicule
guarantees is that you find out in the two places you would look rather than from a sync that
quietly stopped working:

- **The scheduler says so, in its own sentence.** A scheduled sync of a source with no session
  does not report a failure that reads like the instance being down. It says the server holds no
  session, that the instance has not been contacted, that this is expected after a restart, and
  it names `manicule connector login <name> --browser`.
- **`doctor` reports it.** The `sessions` check asks the running server — over the control
  socket, because a `doctor` typed at a terminal is a different process and holds no sessions —
  and reports `degraded` naming the source and the same command. It also tells the two apart:
  no server at all is its own state with its own remedy.

So the morning after a reboot, `manicule doctor` is enough:

```text
sessions   degraded   the manicule server holds no Confluence session for 'handbook', so the
                      next scheduled sync of that source stops without contacting the instance.
                      … Sign in again with `manicule connector login handbook --browser`.
```

**Stopping it.** `launchctl bootout` sends `SIGTERM`, and manicule handles it itself rather than
letting the HTTP server handle it: the scheduler stops first so no new sync starts, whatever was
syncing drains its ingest stages within `ingest.shutdown_grace_s`, the control socket waits for
the write commands already in flight, and the MCP sessions and the HTTP server close last. A
second `SIGTERM` — or a second `Ctrl-C` — stops waiting. `docs/surfaces.md` §6.2 has the order
and why it is that order.

### 6.4 Keeping the vectors honest

Vectors are the one part of the data directory that no amount of metadata can vouch for. A
chunk's text is content-addressed, a blob has a digest, a document has a fingerprint — but a
vector is a thousand floats whose only distinguishing property is being those floats. Change one
of them into another perfectly valid float and every identity, fingerprint, dimension and
publication check in manicule still passes.

So every vector manicule writes carries a SHA-256 over the exact `float32` values that reach
disk, recomputed on every read. `docs/storage.md` §6.2.5 has the contract; what an operator needs
is the four-way distinction it sits inside, because these get called "integrity" interchangeably
and only one of them is what a green check means:

| You want to know | What answers it | What it costs |
|---|---|---|
| Are the stored numbers the numbers that were written? | the vector checksum — `manicule vector-checksum --verify` | a hash per row; ~18,000 rows/s at `D = 1024` |
| Is this vector this chunk's, made by this model? | the embedding-input identity and the fingerprint, on every read | a comparison per row |
| Did the model produce the *right* vector for the text? | re-embedding, and nothing else | a forward pass per row — hours over a large corpus |
| Could somebody have changed a vector without leaving a trace? | nothing manicule ships | out of scope |

The last row is not a hedge. The checksum is unkeyed and stored beside the vector it describes,
so anything with write access to the data directory can change both. It detects **accident** —
bit rot, a partial write, a storage layer that rewrote a block — and says nothing about an
actor. `<data_dir>` permissions (§2) are what keep actors out.

**Reading the report.** `manicule index` with no path and `manicule doctor` both report
coverage as two counts, and the distinction between them is the whole point:

```text
vector integrity   412 of 1,204 row(s) checksummed
  unverified       792 row(s) predate checksums; run manicule vector-checksum --yes
```

`unverified` is not damage. It is a corpus indexed before this version, whose rows carry no
checksum because none existed when they were written. They are readable, searchable and correct
as far as anything knows; what is true is that nothing has vouched for their numbers.

**After upgrading.** Run `manicule vector-checksum --yes` until `remaining` reaches zero. Each
pass reads and rewrites one bounded page (`storage.checksum_backfill_batch`, default 512),
hashes vectors already on disk, and contacts nothing — no model, no connector, no source system.
It is safe to interrupt: the next pass selects rows that still record neither half of the
checksum pair, so it resumes without a cursor and cannot duplicate or skip a row. Running it once
more after it finishes reads nothing. Rows it will never touch — a vector it cannot hash, or a row
holding one half of the pair and not the other — are reported as `failed` rather than backfilled,
because writing over them would erase what they are telling you.

**What a backfill cannot do.** It records a checksum over the bytes as they are *now*. A row that
was already corrupted gets a digest over the corruption and verifies from then on. That is not
fixable from inside — nothing on disk records what the vector used to be — so if you have reason
to suspect *past* corruption, the answer is `manicule rebuild plan <snapshot>` followed by
`manicule rebuild execute <snapshot>`, which re-derives vectors from the retained source bytes
(`docs/storage.md` §7) rather than trusting what is in the vector table. A backfill protects the
future; a rebuild is what re-establishes the past.

**`failing` means something else entirely.** A recorded checksum that no longer matches its
vector is a row whose numbers changed after they were written. `manicule vector-checksum
--verify` reports how many and of what kind without printing any of them; the response is a
rebuild of the affected generation and a look at the disk underneath it, not a backfill.

## 7. Still open

Not settled here, and deliberately:

- **Encryption at rest** and its key management
  ([#19](https://github.com/mgd43b/manicule/issues/19)). Today the answer is the volume's, not
  manicule's: full-disk encryption, or an encrypted filesystem under `<data_dir>`.
- **Whether retention is opt-out per connector**, and the split between retaining a live
  document's bytes and retaining prior versions' (`docs/storage.md` §7).
