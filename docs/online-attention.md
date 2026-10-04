# Online normalization and automatically fused attention

The PTX compiler can rewrite the primitive stable-softmax/weighted-matmul
computation into an online normalized contraction. There is no new attention
operation in `TensorExpr` or `TensorGraphNode`, and the `nn` API is unchanged.
The compiler matches an exponential normalizer, independently matches compatible
consumers, and inlines eligible scalar/dot-product producers. It emits ordinary
scalar expressions, memory accesses, guards, collectives, and loops with explicit
carried state. The PTX backend has no attention or online-normalization instruction.
These are bounded algebraic rules with a selected schedule; unrestricted
algebraic and loop search remains future work.

Enable it explicitly for f32 inference:

```rust
use tnsr::ptx::{PtxExecutor, PtxReductionMode, types::F32};

let mut executor =
    PtxExecutor::<F32>::try_new_with_reduction_mode(PtxReductionMode::Online)?;
```

`Online` allows online normalization and deterministic reduction trees. The
existing default, `Strict`, preserves its reduction policy and does not apply
this rewrite. Online normalization changes floating-point ordering; its outputs
are checked against the primitive reference with tolerances.

## Recognition and scheduling

The small normalization rule matches:

```text
E = exp(S - broadcast(reduce_max(S, last_axis)))
L = reduce_sum(E, last_axis)
```

It captures one broadcast, and can lower `L` alone. A separate consumer rule
recognizes division by the invariant `L` and moves it outside a homogeneous sum:

```text
P = E / broadcast(L)
O = P @ V                         # matrix contraction
O = reduce_sum(P * W, last_axis)   # elementwise weighted sum
```

Both consumer forms use the same normalizer rule. The second works without any
matmul, and the first does not require a matmul score producer. Matching the
second broadcast belongs to the consumer rule: it proves that the denominator
is invariant along the sum, rather than being necessary to discover normalization.
Whole attention is never a tensor-IR pattern or operation.

`S` can be an input or a bounded arithmetic expression containing a matrix
product. This applies to hand-written primitive expressions as well as
`nn::scaled_dot_product_attention` and attention inside `TransformerBlock`.
Score and value matrices currently require identical batch prefixes. The pass
checks external consumers, whole physical-region replacement, input
availability, and output ordering before contracting the graph. Unsupported
shapes, dtypes, patterns, and region boundaries retain the existing plan.
Exposed scores or probabilities prevent contraction. Training graphs whose
backward computations consume intermediates also retain the reference path;
autograd continues to operate on the primitive graph.

For weighted output widths up to 128, one block handles each score row. Its
first warp produces 32 scores, sharing the tile across weighted-output lanes.
The running maximum, normalization sum, and weighted accumulator are rescaled
once per tile. Each score tile is consumed before the shared storage is reused;
the complete score/probability matrices are never written to global memory.
Scalar memory indices enter the existing index optimizer, which simplifies them
and hoists expressions to the scope containing their dependencies. This replaces
the previous backend's bespoke affine dot-product address hoisting.
Wider outputs currently use a serial schedule per output element, which can
repeat score production and is not performance-tuned.

`OnlineRegion` remains transient compiler analysis metadata. It is eliminated
before backend lowering; the old `Stmt::OnlineRegion` and dedicated PTX emitter
are removed. `ScalarLoop` initializes carried scalars before the loop, including
empty loops, and its ordinary assignments update the state. Body-local values
cannot escape their scope. Loads, floating-point operations, and barriers are
never moved by index optimization.

The normalization merge computes `m_new = max(m, tile_max)` and
`a = exp(m - m_new)`, then rescales the carried denominator by `a`.
Each compatible weighted-sum consumer rescales its numerator by the same `a`.
Tile contributions are accumulated relative to `m_new`; final division occurs
after traversal. Leading negative-infinity scores preserve the identity, while
all-masked outputs and nonfinite weights retain the reference's NaN behavior.
The current schedule uses scalar f32 arithmetic, without a custom kernel library
or tensor cores.

This is closer to composing normalization and fusion rules, but it still chooses
a fixed 32-score schedule and directly recognizes compatible sum consumers.
The compiler does not yet search equivalent scalar loops, tilings, or schedules.

## Attention-only measurements

The original implementation is preserved at commit `17d7438`. The table below
records that checkpoint; current generic-lowering measurements are linked below.

Run the same primitive expression through three reduction policies:

```sh
cargo run --release --features cuda --example attention_profile -- strict 64 32 full
cargo run --release --features cuda --example attention_profile -- cooperative 64 32 full
cargo run --release --features cuda --example attention_profile -- online 64 32 full
```

Arguments are policy, sequence length, head width, and `full` or `causal`.
The example profiles one rank-two attention head with fixed Q/K/V values,
matching head and value widths, and strict-f32 matmul operands in every policy.
It validates resident output against CPU. Resident timings upload and allocate
once, then measure 20 CUDA-event samples of 32 executions after warmup.
End-to-end timings measure 20 samples of five compiled executor calls, including
transfers and buffer management. Reported p10/p90 values are sample percentiles,
not confidence intervals. Compilation/JIT is excluded from execution timing
and reported separately; its cache state varies between processes.

Measured sequentially on an RTX 4080, driver 580.178.04, on 2026-10-04:

| Sequence / width / mask | Strict resident (us) | Online resident (us) | Strict end-to-end (us) | Online end-to-end (us) |
| --- | ---: | ---: | ---: | ---: |
| 64 / 32 / full | 28.510 | 16.768 | 63.236 | 45.055 |
| 128 / 64 / full | 40.576 | 37.920 | 82.446 | 71.185 |
| 256 / 64 / full | 85.632 | 86.144 | 136.507 | 133.000 |
| 512 / 64 / full | 250.368 | 184.320 | 340.158 | 269.562 |
| 128 / 64 / causal | 38.016 | 34.624 | 91.125 | 76.401 |
| 256 / 64 / causal | 78.784 | 86.560 | 168.756 | 168.804 |

The reference plans launch 13 kernels for full attention and 14 for causal
attention. All six online cases launch one kernel and report zero intermediate
materialized bytes. The largest measured absolute error versus CPU is 2.31e-7.
Checkpoint strict, cooperative, and online results, including timing spreads and
compilation metrics, are in
[results.csv](../benchmarks/attention-online/results.csv).

These are comparisons against the existing strict-f32 implementation, not
against cuBLAS, handwritten FlashAttention, or default TF32 schedules. At the checkpoint, the
256-token causal case is slower in resident execution, and end-to-end medians
are effectively unchanged. The online policy is experimental and does not yet
cost-select between online and reference schedules.

Causal masks remain ordinary dense constants: their host storage and uploads
are still quadratic. Zero intermediate materialization does not imply zero
input memory or transfer cost. Procedural mask lowering, tensor-core score/value
tiles, native half storage, and general loop/reduction rewrite search remain
follow-up compiler work.


The refactored compiler is profiled with the same six cases in
[generic-results.csv](../benchmarks/attention-online/generic-results.csv).
Both weighted output forms and standalone normalizers have CUDA correctness
coverage; a separate generic recurrence test exercises nested updates and
zero-trip identities without normalization.

A direct comparison rebuilt checkpoint `17d7438` in an isolated directory and
alternated checkpoint/refactored online runs on the same RTX 4080. Each cell is
the median of three process runs, each using the sampling procedure above:

| Sequence / width / mask | Checkpoint resident (us) | Generic resident (us) | Speedup |
| --- | ---: | ---: | ---: |
| 64 / 32 / full | 16.792 | 12.634 | 1.33x |
| 128 / 64 / full | 37.948 | 29.984 | 1.27x |
| 256 / 64 / full | 86.176 | 69.856 | 1.23x |
| 512 / 64 / full | 184.352 | 158.016 | 1.17x |
| 128 / 64 / causal | 34.650 | 26.976 | 1.28x |
| 256 / 64 / causal | 86.591 | 69.912 | 1.24x |

All compared online plans still launch one kernel and materialize zero
intermediate bytes. These six cases show 1.17–1.33x faster resident execution
with generic lowering; this is not a claim about arbitrary attention shapes.
Raw runs, timing spreads, end-to-end timings, and compilation/index-optimization
metrics are in [checkpoint-comparison.csv](../benchmarks/attention-online/checkpoint-comparison.csv).
