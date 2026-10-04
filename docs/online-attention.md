# Online normalization and automatically fused attention

The PTX compiler can rewrite the primitive stable-softmax/weighted-matmul
computation into an online normalized contraction. There is no new attention
operation in `TensorExpr` or `TensorGraphNode`, and the `nn` API is unchanged.
The compiler recognizes the algebra, inlines eligible scalar/dot-product score
producers, and generates PTX for the resulting scheduling region. This is an
initial algebraic rewrite, not unrestricted equality-saturation discovery.

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

The recognized computation is:

```text
E = exp(S - broadcast(reduce_max(S, last_axis)))
P = E / broadcast(reduce_sum(E, last_axis))
O = P @ V
```

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
Affine dot-product access maps are evaluated outside the contraction loop.
Wider outputs currently use a serial schedule per output element, which can
repeat score production and is not performance-tuned.

The scheduling region and score-expression representation live in the compiler
IR; they do not extend the mathematical tensor operation set. The initial GPU
schedule uses scalar f32 arithmetic, not tensor cores. Shared-memory scoring and
normalization are expressed by generated PTX, without a custom kernel library.

## Attention-only measurements

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
Raw strict, cooperative, and online results, including timing spreads and
compilation metrics, are in
[results.csv](../benchmarks/attention-online/results.csv).

These are comparisons against the existing strict-f32 implementation, not
against cuBLAS, handwritten FlashAttention, or default TF32 schedules. The
256-token causal case is slower in resident execution, and end-to-end medians
are effectively unchanged. The online policy is experimental and does not yet
cost-select between online and reference schedules.

Causal masks remain ordinary dense constants: their host storage and uploads
are still quadratic. Zero intermediate materialization does not imply zero
input memory or transfer cost. Procedural mask lowering, tensor-core score/value
tiles, native half storage, and general loop/reduction rewrite search remain
follow-up compiler work.
