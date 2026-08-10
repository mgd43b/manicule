# manicule-plugin-hostile

Parsers that misbehave on purpose, installed so that manicule's isolation can be *proven*
rather than asserted.

Two of the three ways a document takes down an ingest run are not exceptions, and neither can
be demonstrated by a well-behaved parser:

| Parser | Media type | What it does |
|---|---|---|
| `hanging` | `text/x-hangs` | Blocks forever. Nothing in the parent can interrupt it; only killing the process ends it. |
| `greedy` | `text/x-greedy` | Allocates until something stops it. |
| `crashing` | `text/x-crashes` | Exits the interpreter mid-parse, the way a segfault in a native extension does. |

They live in a plugin, and register through the public entry-point group, because that is how
the parse workers build parsers. A misbehaving parser reachable by some other route would
exercise a path production does not have — and the point of these is to exercise the path it
does.

Not published, and not a dependency of anything but the test suite.
