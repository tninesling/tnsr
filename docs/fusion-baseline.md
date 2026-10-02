# Fusion Baseline

This document records the unfused PTX baseline used to evaluate the general
fusion implementation. Run it with:

```sh
cargo bench --bench fusion --features cuda -- --noplot
```

## Observability

`PtxExecutor` exposes separate compile and execution measurements:

- graph nodes and generated kernels;
- generated PTX bytes and lowering/JIT duration;
- actual CUDA kernel launches, including each batched matmul launch;
- materialized values and logical materialized bytes;
- intermediate materialized bytes; and
- explicit host/device and device/device copies.

`PtxExecutionPlan` records the baseline topological order, input edges, physical
action, materialization intent, release points, and normalized virtual outputs.
`describe_plan` renders uploads, kernels, materialization copies, and
virtual views for inspection.

## Initial Measurements

Measured on 2026-08-09 with an NVIDIA GeForce RTX 4080 and driver 580.65.06.
Criterion reports 95% confidence intervals.

| Graph | CPU | Static CUDA | Tile PTX |
| --- | ---: | ---: | ---: |
| Pointwise diamond, `[256, 1024]` | 632.04-683.02 us | 343.20-345.18 us | 346.08-347.75 us |
| Reduction chain, `[256, 1024] -> [256]` | 2.5492-2.6306 ms | 275.07-275.34 us | 267.56-270.68 us |

These timings include source upload and final result download, matching current
executor behavior. Later milestones must report both end-to-end time and the
plan metrics so launch or materialization reductions remain visible when copies
dominate small graphs.

## Stage Status

- Stage 0 is complete for the PTX backend: metrics, plan rendering, exact tests,
  pointwise/reduction benchmarks, and existing transformer/convolution coverage.
- Stage 1 has an unfused one-node-per-step `PtxExecutionPlan`. Runtime consumes
  its order, dependencies, and release decisions, establishing the boundary
  used by virtual values and future fused regions.
- Stage 2 implements structural iteration domains, index maps, predicates, and
  normalized virtual tensors. Reshape and flatten alias storage directly.
  Trailing transpose, permutation, and broadcast paths remain virtual through
  host output materialization, while paths feeding opaque kernels conservatively
  retain their existing materialization until consumer fusion is available.

Stage 2 verification on the same GPU measured Tile PTX pointwise at
337.86-340.53 us and reduction at 271.37-272.62 us. Criterion classified the
pointwise change as an improvement and the reduction change as within noise. A
constant -> reshape -> flatten plan now reports zero kernel launches, zero
device-to-device copies, and one materialized value.

- Stage 3 introduces backend-independent scalar SSA regions for unary, binary,
  comparison, and mask operations. Maximal same-shape pointwise components are
  contracted into physical plan steps with deterministic inputs and multiple
  escaping outputs. Regions are capped at 64 operations, and a quotient-cycle
  check conservatively falls back to unfused steps for non-convex candidates.
  Internal values remain in registers and are absent from plan liveness.

Stage 3 verification measured the fused Tile PTX pointwise diamond at
312.79-314.45 us, down from the Stage 2 range of 337.86-340.53 us. Criterion
reported a 7.39-8.22% improvement. The reduction chain measured
265.35-265.79 us, a further 2.27-2.61% improvement from pointwise producer
fusion before the still-materialized reduction.

- Stage 4 binds each region input to a logical virtual tensor and a physical
  base buffer. Region lowering composes reshape, permutation, and broadcast
  maps into source addresses. Views remain virtual only when every consumer can
  use the map; opaque consumers force materialization. The initial analytical
  cost guard also materializes maps above 64 index operations or non-identity
  views feeding more than two distinct regions.

Stage 4 measured bias broadcast plus ReLU at 324.64-329.05 us for Tile PTX,
versus 365.81-369.20 us for static CUDA. A transposed pointwise residual path
measured 408.60-410.08 us versus 411.85-414.12 us for static CUDA; its
non-coalesced source access largely offsets the removed materialization. A
repeat pointwise-diamond run measured 311.16-313.10 us, showing no regression
for identity access maps.

- Stage 5a adds strict-order `ReductionRegion` kernels. One thread owns each
  output fiber, evaluates full-shape producer SSA in an increasing-index runtime
  loop, accumulates sum/mean/max without reassociation, and evaluates
  reduced-shape epilogues before storing. Escaping producer values may be stored
  during the loop as additional full-shape outputs. This supports arbitrary
  reduction axes while preserving the existing PTX serial reduction order.

The `relu -> exp -> sum -> log` reduction benchmark now executes as one kernel
at 257.83-258.81 us, compared with Stage 4's 263.64-265.04 us. At the Stage 5a
boundary, the stable-softmax graph used the reduction region for shifted
exponential plus sum and measured 843.62-845.63 us; broadcast-back epilogue
fusion was added in Stage 5b.

- Stage 5b contracts compatible reduction-result broadcasts and evaluates
  full-shape epilogues in a second per-fiber loop with pure producer
  recomputation. This reduces small-width softmax to two kernels and supports a
  complete one-kernel RMS-like normalization. LayerNorm's variance reduction,
  inverse-standard-deviation chain, broadcast, and affine output are one region;
  its first mean remains separate.
- The serial full-epilogue schedule is limited to reduction axes of at most 32
  elements because one thread performs both per-fiber loops; beyond that width,
  recomputation and lost parallelism can cost more than the eliminated launch
  and materialization. Larger axes therefore retain a parallel pointwise
  epilogue. Non-identity virtual inputs to large serial reductions are
  materialized to avoid repeated index decoding inside the reduction loop.

At width 1024, the guarded softmax path measured 838.48-839.03 us and LayerNorm
measured 1.2379-1.2451 ms. An unguarded serial full epilogue measured about
2.005 ms for softmax and 7.44 ms for LayerNorm, demonstrating why launch-count
reduction is not sufficient as a profitability criterion. Parallel warp/block
reduction schedules are the next reduction milestone.

Stage 5b is complete within its intended scope. Its reduction regions have one
anchor and preserve strict serial accumulation order; they do not yet provide
warp-, block-, or persistent-reduction schedules. The next Stage 5 work is to
add those cooperative schedules, select them by reduction extent and target
limits, and keep the serial schedule available when numerical ordering requires
it.

## Stage 5b Landing Validation

Measured on 2026-10-01 with an NVIDIA GeForce RTX 4080 and driver 580.178.04.
Criterion reports 95% confidence intervals.

| Graph | Static CUDA | Tile PTX |
| --- | ---: | ---: |
| Pointwise diamond | 321.50-329.04 us | 307.58-311.11 us |
| Reduction chain | 264.42-266.84 us | 252.25-253.69 us |
| Bias broadcast region | 349.84-363.17 us | 321.91-323.07 us |
| Permutation region | 402.72-405.46 us | 397.12-397.82 us |
| Softmax, `[256, 1024]` | 536.51-538.98 us | 805.81-810.08 us |
| LayerNorm, `[256, 1024]` | 716.52-723.71 us | 842.54-848.18 us |

The pointwise, view, and reduction-chain paths show no regression. The
width-1024 normalization cases remain slower than static CUDA because the Tile
PTX reduction itself is serial per output fiber; the Stage 5b guard prevents an
additional serial full-epilogue regression but does not replace the need for a
cooperative reduction schedule.

The same run measured warm Tile PTX transformer forward at 379.59-383.06 us
(tiny), 391.78-395.76 us (small), and 565.89-567.20 us (medium). Cold compile/JIT
measured 2.2368-2.3078 ms, 2.6260-2.6708 ms, and 3.2633-3.3733 ms respectively.
These are improvements over the pre-fusion warm baselines recorded in issue
#36 and keep compilation within the previously observed range.
