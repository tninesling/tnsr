# Kernel paper review: applicable techniques

Reviewed 2026-10-31 against checkout
`10b8461cf09d9ed14c3f9cd8d382b72e45188c7b`.
Three subagents read all six [downloaded papers](../references/README.md),
in pairs, and compared them with implementation and retained experiments.
This pass changes research documentation only. No new kernels, builds, GPU
measurements, or performance claims were produced.

## Recommended experiments

| Priority | Experiment | Paper basis | First target |
| --- | --- | --- | --- |
| 1 | Reuse operand shared storage for the tensor-core epilogue exchange | Triton shared-memory liveness; Welder memory reuse | Existing BF16 GEMM family |
| 2 | Compute each online attention weight once and share it across output lanes | FlashAttention-2 reduction of non-matmul work | Existing cooperative Online inference scan |
| 3 | Enumerate a bounded set of legal GEMM schedules with schedule-specific traffic/resource features | ROLLER construction; Ansor sketches; Welder tile traffic | Existing schedule sweeps and opt-in SIMT tile |
| 4 | Co-design vector-aligned shared layouts and register-fragment prefetch | Triton tiling/prefetch; ThunderKittens layouts | Existing 4x4 strict-F32 outer product |
| 5 | Jointly select fusion boundaries, materialization, and tile schedules | Welder tile-graph | One f32 GEMM epilogue family |

These priorities express expected usefulness and implementation scope, not
predicted speedups. Online attention targets inference; it is not an immediate
PPO training optimization. Schedule-feature corrections should precede trusting
an automated chooser. Full tiled attention forward/backward remains a larger,
separate project.

## 1. Overlay shared-memory lifetimes

Current GEMM lowering allocates A, B, optional second A/B stages, and a separate
f32 C tile for tensor-core fragment-to-scalar conversion. The builder sums all
allocations even though C is used only after the operand loop has ended.
The pipelined path already drains the final prefetch and synchronizes before
the epilogue. These are useful prerequisites for a controlled lifetime overlay.

For a padded, row-major 64x64x32 BF16 tile, the existing allocation is:

- Two A stages: `2 * 64 * 40 * 2 = 10,240` bytes.
- Two B stages: `2 * 32 * 72 * 2 = 9,216` bytes.
- C exchange: `64 * 64 * 4 = 16,384` bytes.
- Total: 35,840 bytes; a valid overlay would require
  `max(19,456, 16,384) = 19,456` bytes before any additional alignment overhead.

This is a 45.7% static shared-allocation reduction for that layout, not a latency
estimate. Registers, shared carveout, and grid parallelism can still limit
occupancy. Transposed operands can have different padded sizes.

Start at `src/tile/graph.rs` GEMM allocation/epilogue construction,
`src/tile/builder.rs::alloc_shared_layout`, `src/tile/layout.rs`, and
`src/ptx/graph.rs` allocation lowering. Existing `AliasShared` is insufficient:
it inherits the source dtype and must fit inside one source allocation.
BF16 operand storage reused as a larger f32 C view needs an explicitly sized,
aligned byte arena with validated typed views, or equivalent allocator support.
Do not reinterpret one small BF16 tile using the current alias statement.

Acceptance: prove that all operand reads and async writes are retired before C
overwrites the arena; handle zero-K and tail paths; compare complete outputs,
generated shared allocation, JIT resources, resident kernel timings, and full
updates. Update scheduler/feature resource estimates with the allocator.

## 2. Share online exponential weights

`src/tile/normalized_schedule.rs::lower_scan` already computes a tile of scores
in the first warp and shares those scores. Output lanes then independently
evaluate `exp(score - maximum)` and update their denominator for every key.
Move that identical exponential to the score-producing lanes and share weights
instead of scores, retaining each consumer's existing serial denominator and
weighted-value traversal initially.

The exponential must use the updated **running maximum**, not merely the
current tile maximum. Preserve the `-infinity` zero-weight rule, maximum/alpha
merging, all-masked NaNs, dynamic key bounds, and partial-scan behavior. Avoid
changing denominator summation to a warp tree in this first experiment.
Prove the producer state and publication barrier make weights visible before
consumption, including blocks with fewer than 32 useful output columns.
The safest first version retains score publication and the running-max merge,
then has the first warp overwrite scores with weights and adds a second
publication barrier. Measure that barrier cost. A later version may avoid it,
but the existing warp reduction guarantees the full maximum only in lane zero;
changing the reduction/broadcast tree needs its own nonfinite/signed-zero audit.

Acceptance: exact output comparisons where operation order is retained,
nonfinite/mask/causal-prefix cases, cached decoding and prefill, and generated
exponential/barrier counts before kernel and end-to-end inference timings.
This is a scheduling hypothesis; fewer exponentials may expose another cost.

A separate BF16 prefill experiment is to apply row rescaling directly to mapped
output register fragments. `src/nn/kv_cache/prefill.rs` currently exchanges the
PV accumulator through shared memory when the maximum changes. The existing
`changed` guard already skips this when all scales equal one, so measure its
incidence before adding fragment-layout support. Preserve BF16 conversion
boundaries and MMA order. See the detailed attention review for the larger
register-retained score/probability design.

## 3. Make schedule search small and faithful

Keep the current schedule as a candidate. Enumerate only supported combinations
of block tile, thread/warp fragments, operand layout, pipeline, and traversal.
Use legality/resource checks to prune first, then measure finalists offline.
Preserve arithmetic policy, dtype, view alignment, tails, and output ownership
as hard constraints. Avoid adding tuning to ordinary execution startup.

The current `FusionFeatures` has useful geometry and pressure estimates, but
`matmul_features` does not add schedule-dependent repeated CTA loads alongside
the initial unique-input byte count. Add distinct estimates for global tile
requests, shared-to-register traffic, materialization, tail waste, and grid
waves; do not call estimated requests measured DRAM traffic. Calibrate actual
registers/spills rather than treating the current register estimate as fact.

Also reconcile launch geometry with the execution path: feature extraction
assigns 1,024 threads to legacy standalone elementwise/materialized views,
while fixed-binding linear dispatch uses the 256-thread pointwise helper.
Ordinary execution still has legacy launch paths, so this requires explicit
execution-path context rather than universally changing the estimate.

Reconcile another resource estimate in
`src/tile/matmul_schedule.rs::select_with_resources`: its initial shared budget
includes a full f32 C exchange for scalar schedules, although scalar lowering
does not allocate one. This conservative mismatch can reject a schedule on
tighter targets. Pass actual SM count into candidate ranking as well; current
`MatMulResources` does not carry it, although `PtxTarget` knows it.

Persist measurements by device, compiler signature, arithmetic/storage policy,
shape, operand access maps, and epilogue. Compare held-out shape/layout families
and end-to-end outcomes. Existing learned rewrite models and schedule sweeps
are infrastructure, not evidence that a new learned kernel scheduler will win.

## 4. Improve the existing outer-product pipeline

The 64x64x16 block / 4x4 thread tile already exists behind
`set_experimental_register_tiles(true)`, with documented qualified PPO results.
It is not a new implementation opportunity. The next experiment should retain
that ownership while comparing vector-compatible operand layouts and a small
register-fragment lookahead, preserving each accumulator's increasing-K order.

Use the existing 4x1 tile as a control. Test one operand's aligned skew or swizzle
at a time, then consider combined layouts. The rejected stride-33 experiment
destroyed wide shared loads; reducing nominal bank multiplicity is insufficient.
Audit generated load/store widths, transaction groups, address work, registers,
and spills. A prefetch that lengthens too many live ranges may lose even when
it exposes more overlap. Do not revive blanket unrolling or unchanged two-stage
F32 staging as if the previous negative screens had not happened.

## 5. Let fusion and schedules vary together

The planner already compares fusion alternatives and has a replaceable scorer.
Changing only its score cannot discover a missing kernel architecture. Add a
small candidate set that pairs an existing legal schedule with each fused or
materialized epilogue boundary, and cost the whole plan including copies.
Start with f32: half-storage partitioning can introduce extra narrowing stores
and currently has an explicit legality guard.

Retain the original and materialized candidates. Existing evidence includes
both profitable BF16 operand repacking and harmful broad reshape-copy removal;
fewer kernels or fewer logical bytes do not establish a faster implementation.
Do not combine the already rejected standalone two-kernel SwiGLU schedule with
a generic claim that training fusion is missing.

## Already present or unsuitable for immediate adoption

- Register tiling, operand-fragment reuse, padded/swizzled shared layouts,
  aligned async copies, two stages, and fused GEMM epilogues already exist.
- Packed FFNs and the larger SIMT tile have opt-in implementations and retained
  complete-update evidence in [parallel prototypes](../ppo-parallel-prototypes.md).
- Grouped-M traversal exists as an experimental option. BF16 regressions in
  [the traversal study](../matmul-grid-order.md) argue against blanket enabling.
- BF16 split-K, operand repacking, smaller pointwise launches, and packed warp
  reductions have separate accepted experiments. Keep their arithmetic and
  workload scope separate from strict-F32 proposals.
- Hopper TMA/WGMMA and dynamic register redistribution cannot be transplanted
  directly into the current Ada path. A deeper async ring also needs new
  wait-group/lifetime semantics; changing a stage count is insufficient.
- FlashAttention-2's tiled forward/backward is relevant longer term, but cannot
  be claimed as an arithmetic-preserving replacement for current strict paths.

## Detailed paper reviews

- [ROLLER and Triton](paper-review-roller-triton.md)
- [Welder and Ansor](paper-review-welder-ansor.md)
- [ThunderKittens and FlashAttention-2](paper-review-thunderkittens-flashattention2.md)

Each review distinguishes paper mechanisms, current implementations, and
proposed experiments with paper-section and code references.
