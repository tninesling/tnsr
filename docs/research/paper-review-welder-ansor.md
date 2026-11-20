# Welder and Ansor: applicable techniques for tnsr

Review date: 2026-10-31. Research only: no production changes, builds,
GPU workloads, or new performance measurements. The recommendations below
compare the downloaded papers with the current working tree, including newer
BF16 work rather than only the older strict-F32 research baseline.

Paper references use **printed proceedings pages**; the downloaded PDFs each
include a cover, so PDF page 2 is printed page 701 for Welder and 863 for Ansor.

- [Welder PDF](../references/welder-osdi-2023.pdf), OSDI 2023, pp. 701-718.
- [Ansor PDF](../references/ansor-osdi-2020.pdf), OSDI 2020, pp. 863-879.

## Findings

The most useful new step is to make **fusion boundaries and kernel schedules
joint candidates**, supported by schedule-dependent traffic estimates. tnsr
already has cost-aware fusion and explicit tile schedules, but the current
matmul fusion alternatives retain the original schedule. A more accurate
scorer alone cannot select a fused shared-memory operator chain that the
candidate generator and lowering do not represent.

A second concrete opportunity is **shared-memory lifetime reuse**. The generic
matmul lowering reserves separate operand stages and an output exchange tile,
even though the output exchange occurs after operand consumption. This is a
bounded application of Welder's allocation approach before attempting a
general multi-operator compiler.

Ansor contributes an incremental way to explore these choices: a few legal
schedule families, complete candidates, a small measured finalist set, and
tuning effort allocated by end-to-end impact. Its full evolutionary system is
more infrastructure than we need for a first experiment.

## What the papers establish

| Technique | Paper location | Implication for this project |
| --- | --- | --- |
| Propagate an output tile backwards through tensor access expressions; reject inconsistent connected outputs | Welder §3.1, pp. 704-705 | A consumer's tile requirements can determine producer input and intermediate tiles instead of matching operator names alone. Irregular accesses require conservative bounds. |
| Select reuse edges at register/shared/global levels, then schedule connected tiles | Welder §3.2, Fig. 7, pp. 705-706 | Fusion is also a choice of where an intermediate lives and which common schedule computes it. |
| Rank by tile traffic, reject excessive footprint, penalize uncoalesced accesses and insufficient parallelism, then profile finalists | Welder §§3.2, 4.1, pp. 706-708 | Saved intermediate bytes are insufficient if fusion worsens operand reloads, grid size, or resources. The paper itself uses hardware profiling to choose final configurations. |
| Plan shared buffers from liveness with alignment-aware allocation | Welder §4.2, p. 708 | Dead operand storage can potentially back later exchange/reuse buffers. Async operations and barriers extend lifetimes. |
| Separate high-level sketches from tile sizes and annotations; evaluate complete programs | Ansor §§2-4, pp. 865-869, especially Fig. 3 and Table 1 | Search structure and numeric schedule parameters separately. Poor estimates for partial programs can discard good final schedules. |
| Measure promising programs and update a learned model; retain mutation/crossover provenance | Ansor §5, pp. 869-870 | Persist complete legal schedule measurements; use a model to rank rather than treat static estimates as truth. |
| Allocate tuning budget according to task multiplicity, latency, and recent improvement | Ansor §6, pp. 871-872 | Tune repeated expensive kernels first, then reduce effort when improvement stalls. |

Welder's memory-level independence is an approximation for its search
decomposition under its stated tiling assumptions, not proof that our GPU
schedule's runtime decomposes independently. Register pressure, occupancy,
alignment, instruction scheduling, and cache behavior still couple choices.
Its evaluation is chiefly inference. Ansor's original implementation also
targets static dense shapes and explicitly lists limited special-instruction
support, including tensor cores, in §9 (p. 876). We should borrow search
principles rather than replace our native BF16/TF32 lowering with its backend.

## What tnsr already has

- **Cost-aware region partitioning:** `src/ptx/plan/fusion.rs:135` and `:155`
  compare producer materialization, epilogue splits, a balanced pointwise cut,
  and serial-reduction block sizes. `choose` at `:503` scores complete ordered
  kernel-feature vectors. `FusionScorer` in `src/tile/fusion_cost.rs:65`
  explicitly permits nonadditive scorers. Recommending basic cost-aware fusion
  as a new feature would duplicate existing work.
- **Resource and repeated-work estimates:** `src/tile/fusion_cost.rs:9` records
  launches, traffic, repeated operations, live-value register estimates,
  shared bytes, and reduction geometry. `AnalyticalFusionScorer::score` at
  `:97` includes occupancy and serial working-set proxies. These remain
  estimates, not profiler counters.
- **Several schedule families:** `src/tile/matmul_schedule.rs:487` chooses
  shape/dtype/resource-dependent schedules. Layout and pipeline controls exist
  at `:230` and `:319`; an opt-in 64x64x16 / 4x4-thread SIMT family exists at
  `:119`. New candidate search should include this implementation rather than
  describe that tile as wholly missing.
- **Register-level epilogues and reduction fusion:**
  `src/tile/graph.rs:998` lowers matmul epilogue operations;
  `src/ptx/plan/fusion_features.rs:92` models reduction producer recomputation
  and full-domain revisits. The remaining gap is more general shared-memory
  connections across operator anchors.
- **A SwiGLU training prototype:** `src/tile/swiglu.rs:42` and
  `src/ptx/swiglu.rs:10` generate explicit forward/backward kernels;
  `examples/swiglu_training_probe.rs` exercises them. The packaging function
  explicitly leaves graph selection unchanged. The standalone two-kernel
  schedule was already rejected in
  [parallel prototypes](../ppo-parallel-prototypes.md): generic fusion can
  produce one multi-output kernel when all operands are available. Different
  recomputation boundaries in real training need a new candidate and control;
  integrating the rejected pair unchanged is not a recommendation.
- **Learned ranking research:** `integrations/rewrite_cost_model/train.py` and
  [learning rewrite profitability](../learned-rewrite-cost.md) already retain
  original/alternative graph measurements and hold out shapes and families.
  Graph rewrite ranking is not yet general kernel schedule generation.
- **Kernel-body reuse and replay:**
  [kernel reuse startup](../training-kernel-reuse-startup.md) documents
  normalized-PTX entry deduplication. A new schedule-result cache should reuse
  measured tuning decisions and compilation results, not claim that identical
  function reuse or CUDA Graph replay is absent.

## Prioritized experiments

### 1. Correct and extend the planning features before using them to prune

**Current gap.** `src/ptx/plan/fusion_features.rs:56` counts each physical-step
input tensor once. `matmul_features` at `:311` changes compute, shared bytes,
and active threads but does not account for A/B tiles requested by multiple
CTAs. Its register increment is a fixed 48, so it cannot describe different
accumulator/fragment budgets accurately.

For one ordinary unbatched GEMM, a useful first requested-element estimate is

`ceil(N / BN) * M * K + ceil(M / BM) * K * N + M * N`.

Count valid operand elements and the output separately; use actual storage
bytes, broadcast batches, epilogue inputs, masks/tails, and source orientation.
Keep this distinct from unique tensor bytes and **measured DRAM traffic**:
L2 reuse can satisfy repeated requests, and vector/sector effects can add
transactions. A separate shared-to-register estimate should use the actual
thread/warp fragment reuse. Otherwise a larger CTA's principal reuse benefit
is invisible to the model.

**A verified geometry mismatch to resolve.** Features at
`src/ptx/plan/fusion_features.rs:39-46` assign 1,024 threads to standalone
elementwise/materialized-view steps. Fixed-binding execution routes ordinary
`Kernel | PointwiseRegion` steps through `pointwise_launch_config` at
`src/ptx/linear/program.rs:292`, whose block size is 256 at
`src/ptx/target.rs:8`. The ordinary homogeneous executor still has 1,024-thread
legacy unary dispatch at `src/ptx/executor.rs:1569`. Thus execution flavor must
be represented in features, or both must consume the same launch descriptor.
Do not globally replace the estimate with 256 without auditing all paths.

**Deliverable and gates.** Add plan-only diagnostics for identical GEMMs under
two legal tile/geometry choices. Verify feature counts from explicit tile
enumeration, plus dispatch/feature agreement for both execution paths.
Calibrate on actual JIT resources and measured finalists before enabling new
pruning. This repairs decision evidence; it is not itself a demonstrated
runtime optimization.

### 2. Alias dead operand shared storage with the tensor-core output exchange

**Current gap.** `src/tile/graph.rs:782-807` allocates A/B stages and a separate
F32 C exchange tile. C is used after the K loop at `:970`.
`TileIRBuilder::alloc_shared_layout` at `src/tile/builder.rs:42` sums allocation
sizes, and shared lowering at `src/ptx/graph.rs:592-612` advances a monotonic
offset. This is not a shared-memory liveness allocator.

**Bounded prototype.** Start with a synchronous tensor-core GEMM, explicitly
end A/B lifetimes after the final operand-consumption barrier, then place C at
an aligned overlapping offset. For an unpadded 64x64x32 BF16 tile with two
operand stages, operands use 16 KiB and C uses 16 KiB: their additive capacity
is 32 KiB while their phasewise maximum is 16 KiB, if async retirement is
proven. These are allocation arithmetic examples, not actual JIT occupancy
or speedup measurements. Synchronous and padded variants differ.

**Gates.** Prove barriers, outstanding async writes, and cross-thread readers
retire before reuse. Check PTX shared declarations and actual JIT resources,
all transpose/tail/K=0 cases, epilogues and replay. Recompute scheduler resource
budgets and fusion estimates from the same allocation plan. Reject if added
synchronization erases any latency benefit. This is more narrowly scoped than
a general best-fit shared allocator.

### 3. Generate a small joint fusion-and-schedule candidate set

**Current gap.** Matmul alternatives at `src/ptx/plan/fusion.rs:349-385` split
the epilogue while explicitly keeping `region.schedule`. They do not generate
Welder-style MatMul-to-reduction shared connections or enumerate a different
GEMM schedule for each fusion boundary.

**First increment.** Cross a few already legal GEMM schedules with keep/split
epilogue candidates, including the existing experimental register tile where
legal. Preserve the original plan. Share feature extraction and discard only
hard-invalid resource/layout candidates initially. This needs candidate
generation; substituting another `FusionScorer` is insufficient.

**Next increment.** Implement one explicit shared producer-consumer region
with static tile propagation, such as GEMM plus a row reduction when a complete
row fits a CTA. Include nonfused variants. A whole-row consumer can force a
wider GEMM tile, reduce output-grid parallelism, or increase repeated operand
loads. For Softmax/attention, coordinate with the existing online attention
path rather than materializing a score tile solely to demonstrate fusion.
Start with forward inference or a numerically ordered row operation; training
requires saved outputs and external users to remain observable.

**Gates.** Identical numerical policy, both operand orientations, N=683 tails,
small-output long-K gradients, multiple users, and batch broadcast. Verify
emitted stores/loads and phase lifetimes. Time complete updates with copies,
capture changes and fallback included. Fewer launches alone is insufficient.

### 4. Tune expensive schedule families offline with an explicit budget

Borrow Ansor's sketches/annotations distinction rather than immediately
building full evolutionary search. A candidate can name its family, CTA tile,
thread/warp ownership, layout, stage count, and epilogue partition. Mutations
must produce a complete valid candidate. Keep measured negative examples.

First sweep a small curated family set; then allocate additional trials by
`occurrence_count * measured_latency` and recent improvement per tuning cost.
Use a nonzero exploration budget so an inaccurate early model does not discard
an entire family. Persist target, driver/codegen version, shape/strides,
precision/reduction policy, fusion boundary, candidate provenance, actual
resources, timing uncertainty, and correctness result. Existing normalized
kernel identity can avoid redundant compilation, but tuning identity must
also include launch geometry and binding semantics.

Use the latest workload attribution in
[BF16 within 20%](../training-bf16-within20.md), not the older strict-F32
research graph as the sole objective. Its report separates remaining ordinary
BF16 GEMMs, F32 pointwise work and reductions. Keep StrictF32, TF32x3 and BF16
datasets separate; include non-PPO workloads. Hold out shapes/families and
retain the original plan when expected improvement is within noise.

## Constraints and ideas to defer

- **Numerical legality remains a hard filter.** Half-storage fusion cuts are
  currently excluded at `src/ptx/plan/fusion.rs:168` because an extra store
  changes narrowing. Carry storage-rounding boundaries explicitly before
  extending that search. Ansor's reduction factorization and cross-thread
  reduction rules alter grouping and cannot be added to StrictF32 candidates.
- **Trainable weights are not immutable constants.** Ansor §4.2's compile-time
  weight-layout rewrite is cheap for inference constants. Training repacking
  must include refresh/invalidation after optimizer publication and its copy
  cost. [BF16 within 20%](../training-bf16-within20.md) already materializes
  aligned split-K operands; do not propose that exact optimization as new.
- **Broad copying elimination is not supported by prior results.**
  [Rejected follow-ups](../ppo-followup-experiments.md) include reshape copy
  elision and several individually plausible codegen changes. The newer BF16
  result actually adds copies to improve GEMMs. Model saved bytes together
  with resulting access/layout efficiency.
- **Do not launch a general out-of-core compiler first.** Welder §5.4's host
  memory hierarchy extension solves capacity problems; current performance
  evidence does not establish host spilling as the leading workload need.
- **Do not infer a bottleneck from Welder's workloads.** Its memory-bound
  motivation and inference speedups do not establish our strict-F32 or BF16
  training bottleneck. Use matched trace attribution and actual JIT code;
  existing performance-counter permission limits remain unchanged.
- **Packing already has a qualified opt-in implementation.**
  [Parallel prototypes](../ppo-parallel-prototypes.md) reports packed FFNs and
  combined register-tile experiments. These are existing controls for future
  schedule/fusion work, not new deductions from Welder or Ansor.

## Suggested implementation order

First repair feature/dispatch agreement and report per-tile traffic. In
parallel design work, specify shared-storage phase lifetimes. Then measure a
bounded output-scratch alias prototype and joint candidates built only from
existing schedule families. Add a new shared operator-connection family only
after those controls are credible. Use measured tuning data to decide whether
learned ranking or evolutionary search would justify its complexity.
