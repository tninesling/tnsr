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

## Stage 5c Cooperative Warp Reductions

Stage 5c adds an opt-in `DeterministicTree` reduction policy while retaining
strict serial accumulation as the default. Eligible axes of at least 32
elements use one warp per output fiber. Every lane accumulates a strided subset,
the 32 lane-local partials are combined through shared memory, and lane zero
evaluates and stores the reduced epilogue. The schedule supports sum, mean, max,
partial warps, and arbitrary reduction axes. Regions with fused full-shape
epilogues currently remain serial.

The execution policy and selected schedule are part of compilation identity,
and schedule-specific launch geometry keeps all cooperative lanes active. The
PTX model also includes predicate-aware warp shuffle instructions for a later
tree-reduction refinement; the initial production path deliberately uses the
shared-memory partial fold validated here.

Measured on 2026-10-01 with the same RTX 4080. These are end-to-end Criterion
95% confidence intervals:

| Graph | Strict Tile PTX | Cooperative Tile PTX | Midpoint improvement |
| --- | ---: | ---: | ---: |
| Reduction chain | 248.23-249.12 us | 159.93-160.84 us | 35.6% |
| Softmax, `[256, 1024]` | 791.33-802.47 us | 442.21-448.96 us | 43.9% |
| LayerNorm, `[256, 1024]` | 833.30-837.48 us | 460.19-474.07 us | 43.8% |

Cooperative softmax and LayerNorm now outperform the Stage 5b static-CUDA
ranges. The next reduction work is to replace the lane-zero partial fold with a
validated shuffle tree, support cooperative full-shape epilogues, and add a
multi-warp block schedule for wider or lower-fiber-count reductions.

## Stage 5d Block Reductions and Cooperative Full Epilogues

Stage 5d adds a 128-thread block schedule for axes of at least 256 elements and
for profitable broadcast-back epilogues. Threads accumulate strided local
partials, then combine them through a deterministic power-of-two shared-memory
tree. The final reduced value is broadcast through shared memory so the entire
block can evaluate a full-shape epilogue in parallel.

The planner fuses cooperative full epilogues containing at most two pointwise
operations. More complex epilogues remain separate pointwise regions: this
guard preserves LayerNorm parallelism while still fusing the simple final
division in softmax. Schedule metadata uses backend-neutral subgroup widths;
the PTX planner supplies CUDA's 32-lane subgroup and 128-thread block sizes.

Measured on 2026-10-02 with the same RTX 4080:

| Graph | Stage 5c cooperative | Stage 5d cooperative | Change |
| --- | ---: | ---: | ---: |
| Reduction chain | 159.93-160.84 us | 152.81-153.33 us | 5.0% faster |
| Softmax, `[256, 1024]` | 442.21-448.96 us | 418.07-422.22 us | 5.1% faster |
| LayerNorm, `[256, 1024]` | 460.19-474.07 us | 449.18-452.09 us | 4.0% faster at interval midpoints |

The remaining reduction work is a validated warp-shuffle finalization path and
target-aware tuning beyond the initial CUDA thresholds.

## Stage 5e Warp-Shuffle Finalization

Stage 5e replaces the subgroup schedule's lane-zero shared-memory fold with a
five-stage CUDA warp shuffle tree. Predicate-aware shuffle results select the
reduction identity for lanes beyond each stage's valid source range, preserving
correct sum, mean, and max behavior for non-power-of-two extents. Shared memory
is reduced from 32 partial values to one value used only to broadcast the final
result before epilogue evaluation. The block schedule continues to use its
128-thread shared-memory tree.

The generated PTX is checked for the five offsets 1, 2, 4, 8, and 16, while
CUDA correctness coverage exercises widths 32, 33, and 127 on a middle axis.
The benchmark uses shape `[2048, 128]`, preserving the 262,144-element workload
of the existing `[256, 1024]` reduction benchmark while selecting the subgroup
schedule.

Measured on 2026-10-02 with the same RTX 4080:

| Path, `[2048, 128]` | Time |
| --- | ---: |
| Static CUDA | 213.98-214.36 us |
| Strict Tile PTX | 200.27-200.81 us |
| Cooperative Tile PTX | 156.96-157.64 us |

The shuffle schedule is 21.6% faster than strict Tile PTX at interval
midpoints and 26.6% faster than static CUDA. Remaining work is target-aware
schedule tuning and broader reduction strategy selection rather than another
missing CUDA reduction primitive.

## Stage 6a Matmul Epilogue Fusion

Stage 6a introduces a dedicated matmul anchor region for rank-2, nonzero-inner-
dimension matrix multiplications. Same-shape pointwise descendants are evaluated
inside the existing 16x16 scalar or TF32 matmul kernel, so the raw matmul result
does not need a global intermediate or a separate pointwise launch. External
epilogue inputs retain composed virtual indexing; the common vector-bias pattern
therefore remains a virtual broadcast and is loaded directly by the matmul
epilogue.

The first slice deliberately excludes batched matmul, zero-K matmul, and
pointwise producers on either matrix operand. Those cases keep their existing
execution paths until batch pointer binding and predicated tiled producer
evaluation are represented explicitly. Both scalar and tensor-core plans are
covered, including partial 16x16 edge tiles.

Measured on 2026-10-02 with the same RTX 4080. The graph is a `[128, 128]`
matmul followed by a `[128]` vector bias and ReLU:

| Backend | Time |
| --- | ---: |
| CPU | 503.92-577.63 us |
| Static CUDA | 93.753-95.091 us |
| Tile PTX, fused | 39.726-40.162 us |

The fused Tile PTX path uses one physical kernel, materializes no raw matmul
intermediate, and is 57.6% faster than static CUDA at interval midpoints. The
next anchor work is batched epilogue binding, tiled operand-producer fusion, and
the equivalent convolution epilogue model; target-aware matmul schedule metadata
is the next tile-selection prerequisite.

## Stage 6b Target-Aware TF32 Gating

Stage 6b queries the CUDA compute capability once when the PTX executor is
created and carries it through compilation. Tensor-core TF32 matmul is selected
only for SM80-or-newer targets; older targets retain the scalar FP32 plan. The
generated module header now names the actual target instead of always declaring
SM80, and compute capability is part of the compilation signature so cached code
cannot be reused across incompatible targets.

Synthetic SM75 and SM80 codegen tests verify both the PTX header and the absence
or presence of TF32 conversion and WMMA instructions. On the RTX 4080, the
`[128, 128]` fused matmul epilogue remeasured at 40.270-40.624 us. Criterion
classified the 1.6% midpoint movement from Stage 6a as within its noise
threshold. This establishes safe target plumbing; multiple tile shapes and
resource-aware schedule selection remain follow-on work.

## Stage 6c Target-Aware Schedules and Batched Epilogues

`MatMulSchedule` now separates the logical matrix shape, block tile, instruction
shape, compute-warp topology, operand and accumulator layouts, operand dtype,
and pipeline depth. The execution plan selects one schedule for each standalone
matmul and each fused region; tile lowering, WMMA instruction staging, and launch
geometry consume that same descriptor. The initial schedules retain synchronous
16x16x16 staging with one pipeline stage. TF32 uses 16x16x8 instructions and one
compute warp, while all eight block warps participate in staging. Scalar F32
assigns one output element to each thread.

Capability profiles before SM80, including SM75, use scalar F32 for these F32
operands. SM80+ may select TF32 at the existing shape/work threshold. An explicit
`MatMulPrecision::StrictF32` policy disables TF32 operand rounding on every
target; `AllowTf32` remains the default. Changing precision invalidates the
executor's compilation signature. Call `execute` to recompile automatically,
or call `compile_owned` before the next `execute_compiled`.

Nonzero-K batched matmuls with pointwise epilogues now use one launch, with
`grid.z` selecting a flattened output batch. Input addressing decodes that
batch coordinate and drops broadcast dimensions independently for each matrix.
Bias and residual inputs retain composed virtual indexing across the entire
output shape. This supports two-sided batch broadcasts, rank-two operands,
partial tile dimensions, and aliased matrix/epilogue operands. Zero-K and empty
outputs retain their existing correctness paths; batches above CUDA's 65,535
`grid.z` limit retain the per-matrix fallback. Operand producers still
materialize before matmul. Shared-memory swizzling, asynchronous copies,
multiple pipeline stages, and warpgroup execution remain future schedules.

Run the representative shape benchmark with:

```sh
cargo bench --bench fusion --features cuda -- batched_matmul_epilogue \
  --warm-up-time 0.5 --measurement-time 1 --sample-size 10
```

Measured on 2026-10-03 with the RTX 4080 (SM89), driver 580.178.04, without
concurrent GPU tests. Each workload evaluates `relu(A @ B + bias) + residual`;
B is shared across batches. Times include input uploads and output download.
Criterion's elements/s is configured as FLOPs/s using `2 * batch * M * N * K`.
Compile times below are individual calls with the CUDA driver JIT cache warm;
they are not cold compilation estimates.

| Workload, batch/M/K/N | Policy | End-to-end time (95% CI) | Throughput (GFLOP/s, estimate) | Compile (us) | Max absolute error vs CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| Transformer projection, 2/64/128/384 | TF32 | 119.36–120.39 us | 105.05 | 364 | 5.07e-4 |
| Transformer projection, 2/64/128/384 | Strict F32 | 120.29–121.47 us | 104.02 | 338 | 3.58e-7 |
| Attention scores, 8/64/32/64 | TF32 | 70.813–71.936 us | 29.424 | 353 | 2.72e-4 |
| Attention scores, 8/64/32/64 | Strict F32 | 72.152–72.622 us | 29.001 | 336 | 2.38e-7 |
| GPT projection, 2/128/256/768 | TF32 | 385.28–388.69 us | 259.81 | 391 | 4.88e-4 |
| GPT projection, 2/128/256/768 | Strict F32 | 390.60–392.26 us | 257.08 | 345 | 5.96e-7 |

Every case records **one launch and zero intermediate materialized bytes**.
For either precision, the transformer, attention, and GPT cases respectively
materialize 656,896, 336,128, and 2,624,512 bytes including inputs and outputs.
Their estimated global kernel traffic is 3,735,552, 917,504, and 27,525,120 bytes.
The traffic estimate counts valid operand loads repeated per output block,
per-element bias/residual reads, and output stores; it excludes host transfers
and does not measure cache behavior or physical DRAM transactions.

These initial TF32 and scalar schedules have similar end-to-end throughput;
the interface and removal of batched intermediates establish a foundation for
later tile and pipeline tuning, rather than a claim of peak tensor-core speed.
Synthetic SM61/SM75/SM80/SM89 codegen coverage checks precision gating and one
kernel entry. GPU tests cover edge dimensions, both batch broadcast directions,
strict-policy recompilation, shared operands, zero-K, and empty batches.

## Stage 6c GPT Projection Profiling

A direct comparison with previous commit `d4bb9b5` found a roughly 10% end-to-end
regression for `relu(A @ B + bias) + residual` at B/M/K/N = 2/128/256/768.
The dedicated `matmul_profile` example uploads and allocates resident buffers
once, launches the compiled plan using its physical bindings, checks numerical
parity, and measures repeated GPU work with CUDA events. Its normal executor
measurement continues to include transfers and buffer management. Run it with:

```sh
cargo run --release --features cuda --example matmul_profile -- \
  batch 3 2 128 256 768
cargo run --release --features cuda --example matmul_profile -- \
  flat 3 2 128 256 768
```

Arguments are layout, epilogue stage, B, M, K, N, and optional `trace` mode.
Stages 0 through 3 evaluate matmul, bias, ReLU, and residual incrementally.
`flat` merges batch and rows into a rank-two matrix and is equivalent because
these workloads share B across batches. Resident timings use 20 samples of 32
repetitions; executor timings use 20 samples of 5 repetitions. Reported spreads
are sample p10/p90, rather than confidence intervals. The harness uses the
initial 16x16 launch geometry and supports these shared-weight forward graphs;
it is not a general execution backend.

Measured on the RTX 4080 on 2026-10-03, using both revisions and identical data:

| Variant | Previous resident GPU time (us) | Current resident GPU time (us) |
| --- | ---: | ---: |
| Batched matmul alone | 22.432 | 23.201 |
| Batched matmul + bias | 36.096 | 66.016 |
| Batched matmul + bias + ReLU | 29.440 | 66.144 |
| Batched matmul + bias + ReLU + residual | 29.728 | 66.268 |
| Flattened matmul alone | 19.264 | 19.680 |
| Flattened full epilogue | 23.328 | 23.680 |

Full-epilogue executor medians were 357.924 us before and 394.761 us after.
An Nsight Systems trace independently measured the previous two matmuls at
10.80 us each and its pointwise region at 7.07 us, versus 65.89 us for the new
fused batched kernel. Transfer durations were essentially unchanged.

The slowdown comes from batch index division in the K loop. Static `ptxas`
and `nvdisasm` inspection shows two general `__cuda_sm20_div_u64` calls per K
iteration in the fused batched kernel: one for division by 1, and one for
quotient extraction for modulo 2. The flattened kernel has neither call in
its K loop. Both fused kernels use 40 registers, 3,072 bytes of shared memory,
and zero spill loads/stores with the local CUDA 12.9 assembler. These static
counts are evidence against a spill explanation; they are not dynamic
occupancy measurements. Nsight Compute hardware counters could not be
collected because this account lacks NVIDIA performance-counter access.

A PTX-only experiment replaced division by 1 with a move and power-of-two
division with a right shift. Without changing arithmetic or buffers, resident
GPU time fell from 66.268 to 25.632 us (61.3% lower), with identical maximum
absolute error versus CPU (4.88e-4). The prototype's executor measurement still
uses the original generated module, so only its resident result evaluates the
edited PTX. The production lowering still needs the corresponding correction.
[Issue #61](https://github.com/tninesling/tnsr/issues/61) tracks a shared `egg`
index-optimization mechanism, expression extraction, and loop-invariant
placement to replace local simplifications and the PTX-only workaround.
Removing these divisions and hoisting invariant batch offsets are the next
implementation changes; a shape-based fusion fallback would conceal this
lowering inefficiency.

The 27-shape sweep covers B = 1/2/8, K = 128/256/512, N = 384/768/1536 at M=128.
Current fused batched resident execution regressed at every B=2 and B=8 shape;
B=1 improved. The flattened control was faster than the previous batched plan
at every shape. At B=2 and N=768, current batched GPU time grew from 36.128 us
at K=128 to 66.272 us at K=256 and 126.398 us at K=512, consistent with division
cost repeated in the K loop. Numerical parity was checked throughout.

Raw measurements are retained in
[ablations.csv](benchmarks/matmul-issue60/ablations.csv) and
[sweep.csv](benchmarks/matmul-issue60/sweep.csv). Set `TNSR_PROFILE_PTX` to save
compiled PTX, or `TNSR_PROFILE_OVERRIDE_PTX` to load a counterfactual module for
resident timing. Each process should run alone on the GPU. Use separate build
directories for baseline and current code to prevent Cargo artifact reuse
across snapshots.
