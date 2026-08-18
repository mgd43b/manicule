# manicule-mlx

The Metal-native embedding backend for [manicule](https://github.com/mgd43b/manicule), on
Apple Silicon.

```bash
uv pip install manicule manicule-mlx
```

Then select it:

```toml
[embedding]
provider = "mlx"
```

## License — read this first

**This package is GPL-3.0-or-later. manicule itself is MIT.** They live in one repository and
they are not under one license.

| | License |
|---|---|
| `manicule` — everything under `src/manicule/` | **MIT** |
| `manicule-plugin-example`, `manicule-plugin-hostile` | **MIT** |
| **`manicule-mlx` — everything under this directory** | **GPL-3.0-or-later** |

The reason is one dependency. `mlx-embeddings` is GPL-3.0, and a backend that links it is very
likely a derivative work. Rather than make the whole project copyleft to obtain one accelerated
backend — which is what manicule did until this split — the backend that carries the obligation
is packaged on its own. MIT code may be incorporated into a GPL work freely; the reverse is not
true, which is why the boundary falls exactly here.

What that means in practice:

- **Installing manicule alone** gets you an MIT program with no GPL anywhere in its dependency
  closure. The `onnx` backend is the default and runs everywhere.
- **Installing both** produces a combined work on your machine that is GPL-3.0. Running it
  imposes nothing on you — the GPL's obligations attach to *distribution*. If you redistribute
  the combination, they attach to you.
- **Writing a plugin** for manicule does not make it GPL. Writing one that imports
  `manicule_mlx` very likely does.

Nothing here decides it for you. Take advice if you intend to distribute either manicule or a
plugin under other terms.

## Why it exists

On Apple Silicon this backend is roughly **4–5× faster than onnxruntime** on the indexing path.
Measured on an M4 Max, `BAAI/bge-m3`, 512-token chunks, batch 32, five interleaved repetitions:

| Chunk length | MLX | onnxruntime | Ratio |
|---:|---:|---:|---:|
| 128 tokens | 136.7 chunks/s | 24.5 chunks/s | 5.58× |
| 256 tokens | 64.1 chunks/s | 12.3 chunks/s | 5.23× |
| 512 tokens | 25.6 chunks/s | 6.0 chunks/s | 4.28× |
| ~24 tokens (a query) | 98.7 chunks/s | 80.0 chunks/s | 1.23× |

The gap is on indexing, not on queries. A corpus of 270 000 chunks is about 3 hours here and
about 12 on onnxruntime; a single query differs by roughly two milliseconds, which vanishes
under the generation call that follows it.

**The vectors are the same either way.** Cosine agreement between the two backends is
0.99999998 with a largest component difference of 2.1 × 10⁻⁵, retrieval ranking is identical,
and the two write byte-identical canonical fingerprints. `backend` is excluded from
`EmbedFingerprint` identity precisely so that installing or removing this package is never a
re-embed. `tests/test_parity.py` in this package is what licenses that claim, and it is the
reason the test lives here rather than in manicule: the plugin asserts parity against the
reference, in the same CI run.

## Requirements

Apple Silicon. `mlx-embeddings` is marked `sys_platform == 'darwin' and platform_machine ==
'arm64'`, so on any other platform this distribution installs and the backend refuses at setup
with a message naming `onnx` — rather than failing to install and taking the rest of an
environment with it.
