# Incremental GPT decoding with a KV cache

`GptDecoder` implements [#50](https://github.com/tninesling/tnsr/issues/50):
per-layer projected keys and values are retained while new tokens advance the
learned position offset. CPU and generated Tile IR/PTX support batched prompt
prefill, single-token decoding, chunked appends and reset. The GPT example uses
cached greedy generation; the existing `Gpt::forward` and `Gpt::generate` APIs
remain available for full-context execution.

## API

```rust,ignore
use tnsr::Backend;

let mut decoder = gpt.decoder(Backend::Cpu, 1)?;
let prompt_logits = decoder.decode(&[12, 34, 56])?;
let next_logits = decoder.decode(&[78])?;
assert_eq!(decoder.len(), 4);
decoder.reset()?;
```

With CUDA enabled, select `Backend::Ptx` or `Backend::Cuda`. Initialization
failure returns an error. `decoder_with_capacity(backend, batch, capacity)`
limits memory/context to a positive capacity no larger than the model maximum.

Inputs are flat row-major `[batch, new_tokens]` IDs. All batch rows advance by
the same positive number of tokens, and returned logits are row-major
`[batch, new_tokens, vocabulary]`. Past tokens are not supplied again. Each
query attends to the prior prefix and its own causal prefix within the chunk.
Positions start at `len()` and advance only after successful execution.

Bad tokens, nonrectangular/empty chunks and capacity overflow return errors
without consuming the prefix. Parameter revision checks reject stale cached
states after a model weight change; reset before decoding with the updated
weights. An execution failure requires reset. Sessions retain up to eight
recently used graphs/programs. CUDA compiles by chunk size and reuses each program as the live prefix changes; CPU reference
graphs are keyed by chunk size and live length. Reset retains those entries.

`generate_cached(backend, prompt, max_new_tokens)` provides greedy generation.
When the context fills, it rebuilds the retained window with positions starting
at zero, matching the existing generator. Evicting an old K/V entry alone would
retain hidden states computed with different learned positions and history.
This boundary therefore loses the incremental benefit. The explicit decoder
returns an overflow error instead of silently sliding.

## Execution and storage

Only new positions run through embeddings, normalization, Q/K/V projections,
feed-forward layers and the language-model head. Each layer has fixed f32 K/V
storage shaped `[batch, capacity, width]`.

CPU attention slices the exact live cache prefix and concatenates it with the
current chunk. Its ordinary matmul/softmax graph uses a causal mask over those
positions. CPU cache appends still copy the full backing parameter buffer.

CUDA attention reads cached and newly projected K/V directly. Generated
online-normalized scans compute scores, softmax and weighted values without
allocating concatenated K/V, full attention scores/probabilities, or a dense
mask. The loop bound is the live prefix length plus the query's causal offset;
unused cache slots are never read. A u32 length/bounds feed changes that bound
without recompiling the model for each position.

Below 128 total tokens, one scan handles each query/head. At 128 or more,
the live context is divided into 16 contiguous parts. Blocks produce normalized
partial vectors and log-sum-exp normalizers, then a second scan merges them.
Empty parts contribute zero; nonfinite weighted sums still propagate. These
fixed 16 partials bound workspace independently of the context capacity and
provide more parallel work for single-token decoding. The runtime selects the
path without changing the compiled chunk shape. Per layer, partial vectors and
normalizers require `16 * batch * heads * chunk * (head_width + 1) * sizeof(f32)`
bytes in addition to the attended output. Prefill therefore needs more workspace
than a single scan; dedicated prefill tiling remains a separate optimization.

The scan reuses the compiler's generic normalized-reduction scheduler and lowers
to ordinary scalar loads, arithmetic, shared memory, guards and loops in Tile IR.
`LoadIndex` loads the runtime u32 index; PTX has no attention-specific instruction.
Heads up to width 128 use a cooperative block with 32-score tiles and 132 bytes
of shared memory. Wider heads use the existing scalar schedule. This is a scalar
f32 implementation; tensor-core attention and dedicated prefill tiling remain
future work.

The CUDA owner keeps a private ordered stream, resident model weights and K/V,
and fixed-binding programs for the surrounding model stages. Contiguous stage
outputs share their existing allocations; other views materialize through
generated copy kernels. Projected K/V are appended after model execution. Only
new token IDs, positions, live bounds and returned logits cross the host boundary.
The owner fences work before freeing any bound storage or compiled programs.
Reset clears K/V; changed parameters refresh existing device allocations.

Model projections use strict f32 matmuls, and surrounding reductions use
deterministic f32 trees. Attention uses online maximum/denominator/numerator
rescaling, so equality with the materialized reference uses floating-point
tolerances. This cache targets the existing f32 GPT API; it does not add mixed
dtype, training-time caching, ragged batch lengths or serialized decoder state.
Cache storage alone is `2 * layers * batch * capacity * width * sizeof(f32)`
bytes; compiled programs, model weights and workspaces are additional. The
measured configuration uses 256 KiB of K/V storage.

## Measurements

The [initial cache measurements](benchmarks/kv-cache.json) record the previous
materialized attention implementation. The live-prefix follow-up is measured
separately in `benchmarks/kv-cache-live-attention.json` using
`examples/kv_cache_bench.rs` and the saved initial benchmark binary.

Both versions use deterministic identical model weights and token prefixes:
two layers, width 64, four heads, FFN width 256, vocabulary 256, batch one and
capacity 256. GPU: NVIDIA RTX 4080. The materialized full-context reference uses
strict f32 matmuls and deterministic tree reductions. The new CUDA cache uses
online-normalized f32 attention; numerical errors are checked before timing.

Each process retains 32 alternating cached/full-context pairs after three
excluded warmups. Cached timings append one token after prompt prefill; reset
and prefill are excluded. Timings include staging, ordinary kernel submissions,
cache writes, logits download and synchronization. First use, including
compilation/JIT with driver disk caches left intact, is reported separately.
These are warm single-token latencies, not whole-generation timings.

The final comparison runs the saved previous binary and the new binary in
before/after/after/before process order for each backend. Each version contributes
64 warm samples per case. The table pools those samples; comparisons between
versions are separate process runs rather than paired observations.

| Backend | Prompt tokens | Previous cached median | Live-prefix cached median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 16 | 2.004 ms | 0.461 ms | 4.34× |
| CPU | 64 | 2.025 ms | 0.801 ms | 2.53× |
| CPU | 128 | 2.056 ms | 1.293 ms | 1.59× |
| CUDA | 16 | 0.273 ms | 0.192 ms | 1.43× |
| CUDA | 64 | 0.272 ms | 0.207 ms | 1.32× |
| CUDA | 128 | 0.272 ms | 0.195 ms | 1.39× |

CUDA latency falls 24–30% in these cases; CPU latency falls 37–77%. A separate
32-sample-per-version run checks the longer context at capacity 1024:

| Prompt tokens | Previous cached median | Live-prefix cached median | Speedup |
| --- | ---: | ---: | ---: |
| 16 | 0.298 ms | 0.191 ms | 1.56× |
| 128 | 0.298 ms | 0.196 ms | 1.52× |
| 512 | 0.306 ms | 0.198 ms | 1.55× |

The 512-token case reduces latency by 35%. Measured CPU logits match exactly;
CUDA maximum absolute logit error stays below 1.3e-6 in these fixtures.
Raw samples, errors, first-use timings, process order,
source/binary hashes and check summaries are in the
[follow-up report](benchmarks/kv-cache-live-attention.json). Desktop graphics and
unrelated CPU work remain running; GPU snapshots show no unrelated compute job.
Benchmark processes run sequentially after correctness jobs finish.

For the three capacity-256 cases, the medians of two process-first-use runs give
CUDA prefill 47.8/46.0/44.5 ms and first one-token decode 42.4/29.9/29.4 ms.
Those values include compilation/module setup with existing driver caches; they
do not represent a clean-driver-cache cold start. Warm prefill and complete
request latency need their own benchmarks.

## Reproduction and validation

```sh
cargo build --release --features cuda --example kv_cache_bench
target/release/examples/kv_cache_bench --backend cpu --repeats 32
target/release/examples/kv_cache_bench --backend cuda --repeats 32
target/release/examples/kv_cache_bench --backend cuda --repeats 32 --capacity 1024 --prompts 16,128,512
```

The example writes JSON to stdout and short timing summaries to stderr. CPU-only
builds can omit `--features cuda` and use `--backend cpu`.

The full suites pass: 210 CPU tests and 429 CUDA tests, with one opt-in GEMM sweep
skipped. CPU/CUDA checks, rustfmt, all-target/all-feature Clippy and documentation
tests pass. Decoder tests compare appended logits against full-sequence decoding
for batches, chunk boundaries, reset/reuse, model mutation, invalid inputs and
capacity limits. Growing-context tests cross 32- and 64-token boundaries using
one compiled chunk shape, switch scan paths across 128 tokens, and repeat after
reset.

Direct scan tests compare to an independent f64 reference for head widths
4/33/128/129, batches of two, chunks of one/three and changing prefixes through
65 tokens in both single-block and split-context modes. Unused cache slots
contain NaNs; a separate overflow/nonfinite case checks zero-weight infinity
and NaN propagation through partial merges. An allocation test checks that
capacity 70 versus 700 has identical compiled workspace for the same chunk and
that reset and changed live lengths retain one compiled program. Existing
online-normalization tests also cover masking, nonfinite values, escaping
intermediates and training gradients.
