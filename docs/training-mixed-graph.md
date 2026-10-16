# Explicit mixed graph execution and autograd

`TensorExpr<f32>::cast(ValueDType::{BF16,F32})` now introduces an observable
storage boundary in `TensorGraph`. `ptx::mixed::MixedExecutor` executes these
graphs using physical two-byte BF16 allocations alongside f32 sources,
intermediates and parameter gradients. Its backward comes from ordinary graph
autograd, extending the earlier [standalone linear owner](training-bf16-linear.md).
This implements part of [#114](https://github.com/tninesling/tnsr/issues/114).
The forty-five production PPO executor and precision policy remain unchanged.

## Semantics

Sources are f32. Views preserve their source dtype; arithmetic promotes to f32
if any floating operand is f32. Constants are explicitly f32, without weak
scalar typing. Every BF16 arithmetic result rounds once to its storage dtype.
This is an explicit graph API, not a Torch autocast policy.

Autograd converts each incoming contribution to its primal's dtype before
accumulation. A shared BF16 value with two distinct f32 promotions therefore
narrows the two contributions separately, sums them in BF16, and promotes the
result on the way to an f32 master. Separate casts of the same master instead
meet in f32 at that master. Tests use BF16 halfway values that distinguish
these rules. The gradient seed also has the scalar loss's logical dtype.

Expression rewriting preserves graphs containing casts, including cast sharing.
Gradient pruning preserves forward cast identities. The existing split-K pass
skips BF16 contractions: ordinary BF16 partials would round before the sum.
Supporting split BF16 contractions requires explicit f32 partial storage.

The CPU executor provides a semantic oracle by rounding BF16 logical values
inside f32 host vectors. The homogeneous CUDA executor rejects mixed casts with
an actionable error. Mixed execution must use `MixedExecutor` explicitly.

## CUDA execution

The fixed-shape owner compiles generated Tile IR/PTX programs, retains every
buffer/function through capture, and synchronizes its private stream before
destruction. Source uploads validate all shapes before changing device values.
Each run refreshes casts from current f32 sources. Uploads, readbacks and
compilation are outside replay and event timing. Identity casts of an already
matching storage dtype alias their source; narrowing followed by promotion
retains its rounding. Casts through virtual views materialize directly into
the destination dtype in one kernel.

Contiguous f32 regions use the existing fusion planner. Values with consumers
outside a region become explicit sinks, retaining saved forward intermediates
for backward. BF16 arithmetic boundaries separate programs, so f32 fusion
cannot erase their rounding. `new_unfused` provides a diagnostic comparator;
`new_fused_f32` uses the existing whole-graph f32 planner for a fair baseline.
Only materialized nodes and virtual views of retained storage are readable.

Supported arithmetic includes unary/binary operations, matmul, reductions,
comparison/mask and narrow, plus reshape/flatten/transpose/permute/broadcast
views. Exact-index lookup/loss, concat and convolution are not yet wired into
this mixed owner. It also lacks resident optimizer bindings, allocation reuse,
and shared cast caching across distinct operators.

An operator policy still needs explicit boundaries. In particular, casting
both inputs of `matmul + bias` to BF16 rounds GEMM before bias. Torch linear
needs bias added to the f32 accumulator before its single BF16 store. The
standalone linear owner already implements that behavior; a graph operator
must preserve it before enabling the forty-five linear policy. For SiLU, the
probe promotes the BF16 primal, performs SiLU in f32, and narrows its result
once; its derivative likewise rounds at those explicit boundaries.

## Profile and validation

The [raw evidence](benchmarks/training-parity/mixed-graph.json) retains all
240 samples per path, oracle errors, GPU state, worker hashes, and trace
attribution. The graph is two bias-free projections, SiLU, a nonuniform f32
scalar objective, and graph-generated gradients for input and both weights.
The f32 reference uses whole-graph fusion and current eligible split-K
gradients. The same external forty-five training process occupied the GPU;
these are conditional contended measurements.

| Shape: rows × input → hidden → output | Fused f32 median | Mixed median | Median paired speed ratio |
| --- | ---: | ---: | ---: |
| 4096 × 683 → 256 → 683 | 2248.26 µs | 1658.24 µs | 1.358× |
| 512 × 683 → 256 → 683 | 184.19 µs | 135.29 µs | 1.361× |
| 33 × 65 → 35 → 17 | 21.63 µs | 25.47 µs | 0.850× |

All four paths run in one process, with every permutation of execution order
balanced over 240 samples. Every private-stream warmup is fenced before
timing. CUDA events cover eight complete executions per sample, including
all casts, both projections, SiLU, objective reductions and backward. Upload,
readback and optimizer time are excluded. BF16 wins 207/240 large and 240/240
512-row comparisons; the tiny graph should retain f32. These subgraph results
do not establish a complete PPO update speedup.

Profiling drove the f32-region fusion change. Before it, the large mixed graph
was roughly tied with fused f32. A separate node-level Nsight trace attributed
32% of mixed kernel time to pointwise/view work, 14% to casts, and 52% to GEMM.
Fusing f32 regions reduced the mixed graph from 56 to 23 kernels, and its
retained allocation payload from 253.1 MB to 111.9 MB. The fused f32 comparator
retains 83.2 MB, so mixed execution currently uses more memory despite physical
BF16 storage. The paired fusion gain is 1.363× at 4096 rows and 1.283× at 512.

The final trace independently verifies 23 mixed kernels and six programs with
native BF16 MMA instructions for the large case. Pointwise/view work falls to
about 5% of its captured kernel time; GEMM is now 69%, casts 19%, and reductions
8%. Both traces verify 33 launches per graph (validation, eight warmups and 24
timed calls), with every timed call after the final owner's warmup fence.
Kernel sums and CUDA-event durations remain separate measurements.

Native timings finish before importing Torch. Default CUDA autocast forward
and all three gradients pass the existing primitive sanity thresholds:
normalized maximum 2e-5 for f32 and 1/128 for BF16. The largest BF16 normalized
maximum is 0.007463. Torch's BF16 reduced-precision reduction setting is
recorded and unchanged. These thresholds are not model/PPO acceptance criteria;
full loss feedback, clipping, resident Adam and policy calibration remain open.

Validation also passes 198 CPU and 407 CUDA-required all-feature tests, both
build configurations, all-target/all-feature Clippy, formatting and doc tests.

The evidence also retains a rejected ReLU experiment. Its small f32 forward
errors and much larger gradient errors are consistent with sensitivity near
ReLU's derivative discontinuity; the exact cause still needs a dedicated
diagnosis. Default BF16 cuBLAS comparisons also failed gradient checks. It is
not used as accepted parity evidence. The SiLU fixture follows forty-five's
continuous activation and passed without changing either tolerance.

Reproduce with:

```bash
cargo build --release --features cuda --example mixed_graph_probe
/home/taylor/code/forty-five/.venv/bin/python \
  integrations/forty_five/profile_mixed_graph.py \
  --worker target/release/examples/mixed_graph_probe \
  --output target/forty_five_mixed_graph/reproduce
```

Next integration work is graph-native fused linear/bias semantics, exact-index
operators and resident f32 master/gradient bindings. Those permit the actual
forty-five mixed policy and complete PPO/Adam parity and throughput comparison.
