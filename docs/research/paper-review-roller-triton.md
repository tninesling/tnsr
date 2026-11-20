# ROLLER and Triton: techniques applicable to tnsr

Review date: 2026-10-31. Research only: no production code, GPU runs, or new
performance measurements. The papers were read from the checked-in PDFs;
section references below use one-based PDF pages (including ROLLER's cover).

- [ROLLER, OSDI 2022](../references/roller-osdi-2022.pdf): especially Sections
  3.1-3.3 and 4, PDF pages 4-9 (proceedings pages 235-240), plus the small-shape
  and model-validity experiments in Section 5.1, PDF page 12.
- [Triton, MAPL 2019](../references/triton-mapl-2019.pdf): Sections 5.1-5.3,
  PDF pages 5-7; convolution examples in Section 6.2, PDF pages 7-8.

## Main finding

The useful next step is to improve how existing schedule families use resources
and are selected. Larger register tiles, operand reuse, two-stage asynchronous
staging, and shared layout choices already exist. The clearest additional
technique is **shared-memory lifetime reuse between the GEMM mainloop and its
tensor-core epilogue exchange**. A second experiment is a small register-fragment
pipeline inside the implemented 4x4 SIMT kernel, not another global-to-shared
double-buffering toggle. ROLLER supplies a bounded way to rank these choices
without building a general learned autotuner first.

## What is already implemented

| Paper technique | Current implementation and status |
| --- | --- |
| Hierarchical block/warp/thread tiles | `src/tile/matmul_schedule.rs:96` represents instruction/block tiles and warp fragments. The default scalar register tile has four outputs/thread; native BF16 has larger multi-fragment schedules. |
| Two-dimensional register outer products | `src/tile/matmul_schedule.rs:121` implements the opt-in 64x64x16 CTA with 4x4 outputs/thread. `src/ptx/graph/matmul.rs:223` loads four A and four B values and updates sixteen accumulators. It is already implemented and qualified for selected workloads; see [parallel PPO prototypes](../ppo-parallel-prototypes.md). |
| Reusing tensor-core operand fragments | `src/ptx/graph/matmul.rs:4` loads the A/B fragment sets once per instruction-K slice, then reuses each across output fragments. |
| Global-to-shared prefetching | `src/tile/graph.rs:937` implements two shared stages and asynchronous prefetch of the next K tile. Pipeline legality and vector alignment are checked in `src/tile/matmul_schedule.rs:329`. |
| Layout-aware coalescing | `src/tile/graph.rs:1468` selects operand staging coordinates; schedule layout selection supports padding, swizzling, and orientation at `src/tile/matmul_schedule.rs:234`. |
| Device resource guards | `src/tile/matmul_schedule.rs:487` uses resource limits queried by `src/ptx/target.rs:80`; these guards are conservative rather than a calibrated throughput model. |
| Graph-level transpose/view simplification | Existing graph rewrites and virtual operand maps already address this category. Triton Section 5.1.2's double-transpose example is not a missing architectural optimization. |

The older [GEMM replacement proposal](ada-gemm-replacement.md) describes the
4x4 outer product as future work. Current source and
[prototype results](../ppo-parallel-prototypes.md) supersede that status.
For mixed BF16, [within-20 results](../training-bf16-within20.md) supersede earlier
headroom estimates: repacking unaligned split-K operands already enables
aligned two-operand asynchronous staging. Do not propose that as new work.

## 1. Reuse dead shared storage for the tensor-core epilogue

**Paper basis:** Triton Section 5.2.3 and Figure 7, PDF page 6, compute tile live
intervals and assign storage across nonoverlapping intervals. Section 5.2.4 on
the same page explains RAW/WAR hazard tracking for safe synchronization.

**Current gap:** `src/tile/graph.rs:782` allocates A and B, optionally a second
pair, and then an independent full F32 C exchange at line 801. C is not written
until after the K loop, its final async wait, and the block barrier at lines
963-972. `TileIRBuilder::alloc_shared_layout` at `src/tile/builder.rs:42` sums all
allocations; PTX lowering at `src/ptx/graph.rs:590` assigns monotonically
increasing offsets. Mainloop operand lifetime and epilogue lifetime therefore
do not currently reduce the shared-memory footprint of ordinary GEMM.

**Concrete first experiment:** reserve one aligned scratch arena sized for
`max(all_operand_stages, epilogue_exchange)`, expose typed views into it, and
place C at offset zero only after all outstanding async operations and operand
reads finish. Keep the current barriers, arithmetic and output mapping. Begin
with tensor-core GEMMs; scalar register tiles already store directly and have
no C shared exchange to remove.

For the current padded, row-major 64x64x32 BF16 geometry, the static arithmetic is:

| Allocation | Bytes |
| --- | ---: |
| A stage: 64 rows x stride 40 x 2 bytes | 5,120 |
| B stage: 32 rows x stride 72 x 2 bytes | 4,608 |
| Two A/B stage pairs | 19,456 |
| F32 C exchange: 64x64x4 | 16,384 |
| Current disjoint total | 35,840 |
| Hypothetical overlaid total | 19,456 |

This is a 45.7% static shared-allocation reduction for that geometry, not a
predicted time reduction. Registers may still limit resident CTAs, and reduced
shared use can change carveout/cache behavior. Other orientations/layouts need
their own allocation arithmetic. The current retained BF16 report records
35,840 static shared bytes for its repacked GEMMs, consistent with this example.

**Required representation work:** `Stmt::AliasShared` exists and is used in
`src/nn/kv_cache/prefill.rs:201`, but it cannot directly express this overlay.
Its layout validator at `src/tile/layout.rs:111` inherits source dtype and bounds
the alias by one source allocation. An F32 C tile cannot simply alias a smaller
BF16 A tile. Add an explicit byte arena and aligned typed views, or a validated
allocation-offset plan; never bypass the size/dtype checks. Update the schedule
budget functions as well as builder/PTX allocation so their accounting agrees.

**Acceptance:** validate address ranges and nonoverlapping live intervals on CPU;
compare full GPU output bits for tails, batches, transpose views and epilogues;
include zero-K and asynchronous final-prefetch cases. Inspect actual static
shared bytes, registers, local spills and occupancy. Time both eligible resident
GEMMs and complete BF16 updates. Only then consider a general interval allocator.

## 2. Add a bounded register-fragment pipeline to the existing outer product

**Paper basis:** Triton Section 5.1.1 and Listing 7, PDF page 5, hoist the first
loop load and prefetch the next while independent work is available. ROLLER's
hierarchical register tiles (Section 3.2, PDF page 6) identify a separate level
of reuse above shared storage.

**Current gap:** `lower_outer_product_scalar_matmul` at
`src/ptx/graph/matmul.rs:223` loads the current four A/four B scalar operands,
then emits all sixteen products, then advances K. It has no explicit next-K
register-fragment state. This is separate from the already supported two shared
stages in `src/tile/graph.rs`.

**Concrete experiment:** retain the same 64x64x16 CTA and 4x4 ownership. First
compare current lowering against a two-set operand pipeline, loading K+1 before
finishing K's products while each output still consumes K positions in order.
Then, only if generated instructions justify it, try small four-K fragments
with short live ranges. Preserve tails/zero filling and original arithmetic
policy. This is an investigation of issued shared loads and instruction
overlap; it does not assume scalar source expressions survive the driver JIT.

**Risk and screen:** the driver may already reorder/unroll the loop. Explicit
operand liveness can inflate registers and reduce occupancy. Extract actual
SASS before timing to check whether instruction scheduling/load width changed;
reject spills and avoid a register cap that merely hides them. Use large-output
projections first and keep long-K weight-gradient fallbacks. The prior
[four-way-unrolling and async-toggle experiments](../ppo-followup-experiments.md)
were rejected; repeating either alone is not this experiment.

## 3. Rank a small schedule set using reuse, tails and output waves

**Paper basis:** ROLLER Sections 3.1-3.3, PDF pages 5-8, grows tiles according
to traffic saved per extra footprint, respects hardware alignment and shrinks
tiles when the output grid underfills the device. Its implementation profiles
a small top-K set because device-compiler choices remain unpredictable.
Section 5.1, PDF page 12, explicitly shows poor small/irregular-shape behavior
and model error when parallelism/alignment assumptions do not hold. Triton
Section 5.3, PDF page 7, exposes hierarchical tile parameters to a tuner.

**Current gap:** `select_with_resources` at `src/tile/matmul_schedule.rs:487`
chooses by fixed shape thresholds and a 128-register estimate. The opt-in scalar
selector at line 121 likewise uses fixed M/N/K cutoffs. Neither receives the
actual SM count, even though `PtxTarget::multiprocessor_count` exists at
`src/ptx/target.rs:75`. The family can therefore choose different effective
parallelism on different devices without knowing it.

**Concrete experiment:** expose a small list of legal schedules and record:

- logical work versus padded tile work, including N=683 tails;
- per-layer input/output bytes and register-operand reuse;
- actual shared footprint after layout and pipeline selection;
- estimated register use with explicit uncertainty margin;
- output CTA count and waves relative to queried SM count.

Keep arithmetic policy as a hard legality constraint. Rank candidates and
benchmark only a small finalist set offline; initially preserve existing defaults
and emit the reasons for each candidate's acceptance/rejection. Separate the
large-output forward family from small-output/long-K gradients. Calibrate on
fresh actual JIT resources rather than copying ROLLER's V100/K80 96-register
limit or assuming its bandwidth model predicts an Ada issue-limited kernel.

There is also an accounting cleanup to audit: at lines 576-578 the initial
selector includes `block_m * block_n * 4` shared C bytes even for ScalarF32,
where `lower_matmul_impl` allocates no C exchange. Layout/pipeline budget paths
correctly omit it. This conservative difference can reject scalar schedules on
tighter-resource targets; align it with actual allocation before building a
new cost model. A changed guard needs explicit boundary tests, not an assumption
that smaller accounting improves measured speed.

## 4. Co-design shared padding with actual vector loads

**Paper basis:** ROLLER Section 3.1 and Figure 5, PDF page 5, require both
transaction and bank alignment; Triton Sections 5.2.1-5.2.2, PDF page 6, couple
thread ordering to the memory hierarchy and coalescing.

**Current gap:** Scalar `MatMulSharedLayout::Padded` adds one element at
`src/tile/matmul_schedule.rs:257`. The retained stride-33 experiment documented
in [register-tile research](ada-register-tiles.md) lost LDS.128 vectorization.
Adding generic padding or swizzling is therefore already supported and is not
itself a recommendation.

**Concrete experiment:** extend the *existing* forced layout probe to allow
per-operand alignment-preserving skew (K32 stride36 or K16 stride20 for F32),
with a K-contiguous orientation where useful. Compare against contiguous and
current +1 padding at fixed ownership before combining with a new register
pipeline. Check both global-to-shared stores and shared-to-register loads.
Require actual LDS/STS widths and transaction grouping, not a scalar whole-warp
bank-count argument. This extends an already identified repository experiment,
not a newly discovered paper result.

## What not to infer from these papers

- Direct PTX generation does not remove the proprietary driver's register
  allocation and scheduling choices. ROLLER's proposed assembly backend is not
  evidence that our PTX virtual-register count predicts physical occupancy.
- A large tile is not uniformly better. ROLLER's scale-out caveat and Triton's
  shallow-transformer discussion support keeping small-output fallbacks.
- Triton Section 6.1 notes that split reductions can improve shallow GEMMs;
  split-K changes addition grouping and is incompatible with increasing-K
  StrictF32. Mixed BF16 already has a separate accepted split-K path.
- Triton Section 6.2's implicit-GEMM convolution and pointer-increment tables
  are useful later, but current convolution lowering already uses structured
  channel/row/column loops and hoisted offsets. A new tiled convolution family
  needs current workload coverage before it can outrank GEMM work.
- The model should not reward eliminating materialization indiscriminately:
  prior dense-reshape elision regressed, while the latest BF16 work benefits
  from deliberate operand repacking. Count the complete producer/copy/consumer
  path and preserve existing planner guards.

## Recommended sequence

1. Prototype the explicit shared arena/epilogue overlay, initially on native
   BF16 tensor-core GEMMs. This is the strongest distinct technique in this pair.
2. Inspect the existing 4x4 kernel's actual SASS and screen one small
   register-fragment pipeline against it.
3. Add explainable candidate ranking/output-wave accounting around the existing
   schedule families, incorporating actual allocation and measured JIT resources.
4. Use that harness for per-operand alignment-preserving skew and, only after
   those results, larger 64x128/8x4 SIMT schedule families.

The outstanding late power-of-two decode and exact-tile bounds proofs in
[existing research](ada-register-tiles.md) remain sensible small control fixes;
they should not be credited as architectural gains from these papers.
