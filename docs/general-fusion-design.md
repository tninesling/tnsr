# General Fusion Design

Tracking issue: [#15 Implement operator fusion](https://github.com/tninesling/tnsr/issues/15)

Status: accepted; implementation in progress.

Implemented through Stage 2: PTX execution plans, observability, structural
access maps, and virtual view paths.

Primary reference: [DNNFusion, PLDI 2021](references/dnnfusion-pldi-2021.pdf).

Implementation baseline: [Fusion Baseline](fusion-baseline.md).

## Summary

`tnsr` should replace operator-specific unary-chain fusion with a backend
execution planner built around virtual tensors, composable access maps, and
structured iteration domains. The planner will form multi-operation regions,
prove legality independently from profitability, choose materialization and
schedules for the target GPU, and lower each selected region to one Tile IR
kernel.

DNNFusion's operator mapping classes and transformation impedance are useful
for candidate discovery, but are too coarse to define CUDA legality or predict
tile-level performance. We will derive similar classes from precise access
maps, then combine them with ideas from newer systems:

- TVM Relax post-dominator partitioning for reconvergent graph regions.
- MLIR Linalg, TVM TensorIR, and nvFuser-style iteration and access domains.
- Inductor-style dependency-aware fusion around template/library anchors.
- Welder-style tile traffic and on-chip resource analysis.
- ROLLER-style restricted, hardware-aligned schedule candidates.
- Explicit transformer templates and algorithmic rewrites for operations such
  as FlashAttention that general loop fusion should not be expected to invent.

The semantic `TensorGraph` remains unchanged. Fusion runs after autograd has
produced the executable forward/backward graph and creates a backend-specific
`ExecutionPlan`. This avoids teaching autograd how to differentiate arbitrary
fused regions and allows CPU, static CUDA, and PTX backends to make different
planning decisions.

## Motivation

The PTX backend currently lowers each `TensorGraphNode` to an independent
`TileIR`, then each `TileIR` to an independent PTX entry point. This preserves
logical operation coverage, but it materializes almost every graph edge and
pays a launch for almost every node. Tensor-core matmul reduced the compute
cost of eligible matrix multiplications, while end-to-end transformer timings
remain substantially launch- and memory-traffic-bound.

The existing fusion pass in `src/graph/fusion.rs` is intentionally narrow:

- It recognizes only linear chains of unary operations.
- It requires each intermediate to have one consumer.
- It represents the result as `TensorGraphNode::FusedUnary`.
- It cannot represent binary DAGs, broadcasts, views, reductions, anchor
  epilogues, multiple outputs, or target-specific scheduling choices.

Adding more fused node variants would produce an open-ended catalog of patterns
without solving graph partitioning, index composition, materialization, or GPU
resource planning.

## Goals

- Fuse general pointwise DAGs rather than named unary sequences.
- Compose reshape, permutation, transpose, broadcast, and padding access
  patterns without materializing intermediate tensors where profitable.
- Fuse pointwise producers into reductions and pointwise epilogues out of
  reductions when synchronization permits.
- Support matmul and convolution as schedule anchors with fused prologues and
  epilogues.
- Support reconvergent regions and multiple region outputs where profitable.
- Make materialization an explicit planning decision.
- Keep fusion target-, shape-, layout-, and dtype-aware.
- Preserve exact graph semantics by default, including forward and backward
  graphs, repeated consumers, reduction ordering policy, and indexed effects.
- Expose enough structure for later BF16, dynamic shapes, tile search, and
  transformer-specific kernels.
- Measure launch count, intermediate traffic, resource usage, compile time, and
  end-to-end performance instead of treating fused node count as success.

## Non-goals

- Reimplement all of Inductor, TVM, XLA, MLIR, or nvFuser.
- Adopt a C++ compiler dependency or replace the Rust-native Tile IR/PTX path.
- Expect the general fuser to discover algorithmic kernels such as
  FlashAttention or paged decode attention.
- Implement unrestricted polyhedral optimization.
- Make full autotuning a prerequisite for the first useful fusion pass.
- Fuse every legal region. Materialization can be the best plan.
- Change floating-point reassociation silently.
- Add distributed communication fusion before multi-GPU execution exists.

## Terminology

### Logical tensor

A tensor value defined by graph semantics: shape, dtype, producer, and users.
It does not imply storage or layout.

### Virtual tensor

A logical tensor whose elements are computed by patterned access to another
value. A virtual tensor has no allocation by default. Transpose, reshape,
broadcast, slicing, padding, and an `im2col`-like convolution view can all be
represented this way.

### Physical tensor

A materialized buffer with an address space, storage dtype, physical layout,
strides, alignment, and lifetime.

### Fusion region

A connected set of graph operations selected for one kernel. A region has
external inputs, one or more externally visible outputs, internal virtual
values, an iteration/reduction description, and a chosen schedule.

### Anchor

An operation whose schedule dominates a region, such as matmul, convolution,
softmax, or normalization. Pointwise operations can often adopt an anchor's
schedule; combining two anchors is substantially harder.

### Template or algorithmic primitive

A recognized composite operation with a purpose-built implementation and
schedule family. FlashAttention is an algorithmic primitive, not merely a
large ordinary fusion region.

## Research Findings

### What to retain from DNNFusion

DNNFusion classifies operators into One-to-One, One-to-Many, Many-to-Many,
Reorganize, and Shuffle based on input/output mappings. It uses qualitative
"transformation impedance" to summarize which operation controls the mapping
of a fused result. It also separates clear wins, clear losses, and combinations
requiring profiling.

These ideas remain useful:

- Classify by data mapping rather than operator name.
- Eliminate data movement by changing indices.
- Separate machine-independent candidate filtering from measured
  profitability.
- Perform graph rewriting before fusion to expose better regions.
- Cache generated implementations and profiling results structurally.

### What not to copy directly

DNNFusion targets mobile inference with C++ and OpenCL. Its mapping table mixes
legality and profitability, combines reductions and general many-to-many
operations, and does not model CUDA block ownership, barriers, shared memory,
tensor-core fragments, register pressure, atomics, or backward graphs.

Its fusion planner is greedy, starts from small One-to-One intermediates, and
uses a large profiling database. Its graph rewrites are selected mainly by
FLOP count and assume later fusion makes intermediate size less important. The
paper also treats algebraic associativity more freely than an exact
floating-point compiler should.

We will therefore use mapping classes as summaries and candidate heuristics,
not as proofs or schedules.

### Lessons from newer systems

| System or technique | Relevant lesson |
| --- | --- |
| PyTorch Inductor | Separate graph capture, decomposition, scheduling, template selection, and code generation. Fuse by loop dependencies and retain strong kernels as anchors. |
| nvFuser | Separate logical tensor domains from transformed loop domains. Persistent reduction schedules matter for normalization-heavy transformer graphs. |
| TVM Relax/TensorIR | Use post-dominators for graph regions, explicit spatial/reduction blocks for scheduling, and external/template calls alongside generated kernels. |
| MLIR Linalg/Transform | Indexing maps and iterator kinds provide a compact basis for tile-and-fuse transformations. Scheduling should be separable from semantics. |
| XLA/StableHLO | Make one fusion correspond to one kernel, account for indexing compatibility and duplication, and select among native codegen, templates, and libraries. |
| Welder | Jointly reason about graph fusion, tile shape, recomputation, and traffic at each memory level. Graph grouping alone is not enough. |
| ROLLER | Restrict schedule candidates to hardware-aligned tiles and use a small performance model rather than an unbounded search. |
| Triton/ThunderKittens | A blocked tile program is the appropriate GPU abstraction. Layout, warp ownership, masks, staging, and pipelining must be explicit. |
| FlashAttention | Important transformer fusion can require a new algorithm, not concatenation of existing loops. |
| TensorRT-LLM/FlashInfer | Prefill and decode need different templates, KV-cache layouts, and runtime policies. Serving is not ordinary static graph fusion. |
| Mirage | Multi-level graph and schedule search is promising, but requires a mature IR, verifier, and cost model and is a long-term direction. |

## Design Principles

1. **Semantics before classification.** Precise domains, access maps, and
   effects define legality. Mapping classes are derived summaries.
2. **Legality and profitability are separate.** A legal fusion may reduce
   occupancy or duplicate enough work to be slower.
3. **Logical and physical layouts are separate.** A transpose can be virtual
   even when its source has a concrete row-major layout.
4. **Graph grouping and tile scheduling are separate but connected.** A region
   candidate is incomplete until a viable tile/materialization plan exists.
5. **General fusion and templates coexist.** Templates are not a failure of the
   general mechanism; they encode algorithms or schedules outside its search
   space.
6. **Fusion is an execution optimization.** It must not complicate semantic
   graph construction or autograd.
7. **No silent numerical weakening.** Reassociation and approximation require
   an explicit math policy.
8. **Compilation cost is part of the cost model.** A marginal runtime win does
   not justify unbounded JIT search.

## Rejected Alternatives

### Add specialized fused graph nodes

Variants such as `FusedMatMulBiasRelu` provide a quick implementation for one
pattern but require graph semantics, autograd, lowering, signatures, and every
executor to understand an expanding catalog. They do not solve region
partitioning or physical planning. Composite names remain appropriate for
template recognition, not as the general representation.

### Mutate TensorGraph into the optimized execution graph

This entangles backend decisions with semantic identity, complicates gradients,
and makes CPU and GPU optimization choices interfere. A separate
`ExecutionPlan` preserves the source graph and supports target-specific plans.

### Use DNNFusion's mapping table as legality

Its green/yellow/red table combines legality and expected profitability and
does not describe CUDA synchronization, effects, write collisions, or resource
limits. Derived mapping classes will filter candidates, while access/effect
analysis proves legality.

### Group graph nodes but retain one lowered operation at a time

Placing several opaque operation statements in one container does not establish
a shared iteration domain or eliminate intermediates. General fusion requires
composable scalar/tile expressions and access maps inside one kernel region.

### Lower everything to a low-level affine/polyhedral IR first

This provides formal loop machinery but loses useful operation and algorithm
identity too early, and indirect accesses remain awkward. The proposed IR keeps
structured tensor semantics while making index relations explicit.

### Rely on unrestricted autotuning or superoptimization

The current operation and schedule space does not justify the compile-time and
verification machinery. Analytical pruning and a small hardware-aligned
candidate set come first; broader search remains a later option.

### Discover transformer algorithms through generic fusion

FlashAttention, paged attention, and architecture-specific asynchronous
pipelines change algorithms, state, or schedules beyond ordinary
producer-consumer inlining. They require explicit recognition and template
contracts integrated with, but not generated by, the general planner.

## Compiler Pipeline

```text
TensorGraph
  -> semantic canonicalization and guarded graph rewrites
  -> composite/template recognition
  -> access and effect analysis
  -> candidate region formation
  -> legality filtering
  -> schedule/materialization candidates
  -> analytical costing and optional profiling
  -> ExecutionPlan
  -> one structured Tile IR kernel per selected region
  -> target-specific PTX
  -> guarded compilation/profile cache
```

The plan is built from the final executable graph. For training this means
after `with_gradients` has added backward nodes. The planner may fuse forward
and backward regions independently, but initially will not fuse across their
natural dependency boundary.

## Access IR

### Iteration domains

Each operation or region describes named logical dimensions and their roles:

```rust
pub struct IterDomain {
    pub dimensions: Vec<IterDim>,
}

pub struct IterDim {
    pub extent: Extent,
    pub kind: IterKind,
}

pub enum IterKind {
    Parallel,
    Reduction(ReduceOp),
}
```

Static `usize` extents are sufficient initially. `Extent` should be an enum so
symbolic dimensions and guarded specialization can be added without replacing
the IR.

### Index expressions and maps

Index maps are structural data, not Rust closures. They must be cloneable,
hashable, printable, canonicalizable, and composable.

```rust
pub enum IndexExpr {
    IterDim(usize),
    Symbol(usize),
    Const(i64),
    Add(Box<IndexExpr>, Box<IndexExpr>),
    Sub(Box<IndexExpr>, Box<IndexExpr>),
    Mul(Box<IndexExpr>, Box<IndexExpr>),
    FloorDiv(Box<IndexExpr>, i64),
    Mod(Box<IndexExpr>, i64),
}

pub struct IndexMap {
    pub results: Vec<IndexExpr>,
}
```

Predicates are represented separately so padding and masked tiles remain
explicit. Initial predicates need comparisons and boolean conjunction. Data-
dependent gather indices are represented as indirect accesses rather than
pretending they are affine maps.

### Virtual tensors

```rust
pub struct VirtualTensor {
    pub source: ValueId,
    pub shape: Shape,
    pub dtype: DType,
    pub access: IndexMap,
    pub predicate: Option<IndexPredicate>,
    pub out_of_bounds: Option<ScalarValue>,
}
```

Nested virtual tensors are normalized by composing maps toward the base value.
The normalized representation avoids deep chains of reshape/transpose nodes
and provides a stable cache key.

Virtual tensors are immutable read views. Writes use separate output access
descriptors that record injectivity, collisions, reduction semantics, and
atomic requirements. This is necessary for embedding backward, scatter,
overlapping windows, and future mutable state.

### Scalar and reduction expressions

A fusion region needs SSA-like values for scalar/tile computation rather than
operation-specific fused variants:

```rust
pub enum RegionOp {
    Load { tensor: VirtualTensor },
    Unary { op: UnaryOp, input: ValueId },
    Binary { op: BinaryOp, lhs: ValueId, rhs: ValueId },
    Select { condition: ValueId, yes: ValueId, no: ValueId },
    Reduce { op: ReduceOp, input: ValueId, dimensions: Vec<usize> },
    Store { output: RegionOutput, value: ValueId },
    Anchor(AnchorOp),
}
```

The initial implementation can remain scalar per logical iteration. Tile SSA,
vector values, shared-memory promotion, and fragment values are scheduling
lowerings over the same logical region.

## Derived Mapping Classes

DNNFusion-style classes are derived from access properties:

| Class | Derived property | Examples |
| --- | --- | --- |
| One-to-One | Injective output-to-input map with scalar expression | Unary, same-shape binary |
| Reorganize | Bijective linearized map with rank/shape change | Reshape, flatten |
| Shuffle | Bijective axis/index permutation | Transpose, permute |
| One-to-Many | Non-injective output-to-input map | Broadcast, expand |
| Reduction | Explicit reduction dimensions | Sum, max, normalization statistics |
| Window/Contraction | Parallel plus reduction domains with multiple mapped reads | Matmul, convolution, pooling |
| Indirect | Data-dependent read map | Embedding, gather, paged KV cache |
| Effectful/Scatter | Non-injective write map or externally visible effect | Scatter-add, parameter update |

Unlike DNNFusion, reductions, contractions, indirect reads, and colliding writes
remain distinct because their CUDA legality and schedules differ materially.

Transformation impedance becomes a diagnostic summary of composition cost:

- Scalar expressions have low impedance.
- Bijective views add index complexity but no recomputation.
- Broadcasts can duplicate producer computation.
- Reductions and anchors impose a schedule and synchronization domain.
- Indirect accesses and effects frequently require materialization or templates.

## Execution Plan

The planner must decouple execution from the current one-node/one-kernel model:

```rust
pub struct ExecutionPlan {
    pub steps: Vec<PlanStep>,
    pub graph_outputs: Vec<NodeIndex>,
}

pub enum PlanStep {
    Kernel(FusionRegion),
    View(VirtualValue),
    Copy(Materialization),
    Template(TemplateInvocation),
}

pub struct FusionRegion {
    pub members: Vec<NodeIndex>,
    pub inputs: Vec<RegionInput>,
    pub outputs: Vec<RegionOutput>,
    pub domain: IterDomain,
    pub operations: Vec<RegionOp>,
    pub schedule: RegionSchedule,
}
```

Plan outputs retain original `NodeIndex` identities so liveness, gradients,
debugging, and public output lookup continue to work. Internal graph nodes are
not assigned buffers unless selected for materialization.

`PtxExecutor` will compile and execute plan steps rather than traversing the
original graph and loading one kernel per node. CPU and static CUDA executors do
not need to adopt the PTX plan immediately.

## Candidate Region Formation

Candidate generation runs in stages so high-value structured opportunities are
not consumed by a generic greedy pass.

1. Apply semantics-preserving view canonicalization and exact graph rewrites.
2. Recognize algorithmic and transformer templates.
3. Form anchor-centered vertical candidates.
4. Form horizontal candidates such as shared-input QKV projections.
5. Use post-dominator analysis for remaining pointwise, broadcast, injective,
   and reduction regions.
6. Split candidates that fail legality or resource constraints.

Post-dominators permit diamonds whose branches reconverge, unlike the current
single-consumer chain analysis. A node with an escaping consumer becomes a
region output or a materialization boundary; it does not automatically prevent
the rest of the region from fusing.

The initial selection algorithm will be deterministic and greedy. It should
compare several local alternatives rather than grow one seed irreversibly. Beam
search or dynamic programming can be introduced only when benchmarks show the
greedy planner leaves important opportunities behind.

## Legality

A region is legal only if all conditions hold:

- Every internal read can be expressed from the region's iteration/reduction
  dimensions or from a permitted indirect index.
- Every externally visible value is represented as a region output.
- Writes are injective, an explicit reduction, or use required atomic semantics.
- Thread/block ownership and launch dimensions are compatible.
- Required synchronization can be placed without barrier divergence or cycles.
- Producer recomputation does not change effectful behavior.
- Alias and mutation relationships are preserved.
- Reduction order conforms to the selected numerical policy.
- The target supports every required dtype, instruction, and address space.
- Kernel parameters, shared memory, register estimates, and code size remain
  within target limits.

Legality returns reasons, not a boolean, so planner diagnostics and tests can
explain why an edge or region was split.

## Materialization

Virtual values are materialized when one of these conditions wins:

- Consumers need incompatible schedules or layouts.
- Recomputing a shared producer costs more than storing it.
- A consumer requires a non-composable or indirect access.
- The value escapes the region or must be retained for debugging.
- Register/shared-memory pressure makes the larger region slower or invalid.
- A stable global ordering or atomic boundary is required.
- Compilation/code-size cost exceeds policy limits.

Fan-out is a costed choice among materialization, recomputation, multiple region
outputs, and duplicating a cheap producer into separate regions.

Liveness analysis operates on plan values and materialized outputs. Virtual
intermediates consume no buffer-pool allocation.

## Scheduling and Physical Layout

Region semantics do not prescribe thread or memory layout. Scheduling assigns:

- Grid, block, warp, and lane ownership.
- Tile extents and vector widths.
- Reduction decomposition.
- Register, shared, fragment, and global placement.
- Physical layouts, strides, swizzles, and alignment.
- Predication and edge handling.
- Pipeline stages and asynchronous movement when supported.
- Tensor-core instruction shape and accumulator layout.

Index maps describe logical access. Code generation composes them with physical
layouts to produce addresses. This separation prevents virtual transpose or
broadcast semantics from being confused with concrete row-major storage.

Anchors initially provide a small set of schedule families. ROLLER-style
hardware-aligned candidates can then be enumerated and rejected analytically by
resource limits. A broad autotuner is deferred.

## Profitability Model

The first model is deterministic and analytical:

```text
benefit = launches_removed * launch_cost
        + global_bytes_eliminated / effective_bandwidth
        + allocation_cost_eliminated
        - duplicated_compute / effective_throughput
        - extra_indexing_cost
        - synchronization_cost
        - occupancy_penalty
        - code_size_and_compile_penalty
```

Inputs include:

- Global bytes read/written and intermediate bytes removed.
- Number and size of recomputed values.
- Estimated registers per thread or warp.
- Shared memory per block.
- Blocks per SM under register/shared/thread constraints.
- Number of barriers and reductions.
- Coalescing/vectorization quality.
- Tensor-core utilization for anchors.
- Launch count and graph depth.
- Generated instruction/code-size estimate.

Uncertain top candidates may be profiled. Profile keys include graph structure,
normalized access maps, concrete or bucketed shape, logical/storage/compute
dtypes, physical layout, schedule, target SM, and compiler/PTX version. Profile
misses must never make compilation depend on hours of tuning.

## Numerical Policy and Rewriting

Fusion without reassociation should preserve operation order and ordinary PTX
semantics. Graph rewriting is a separate pass with explicit contracts:

```rust
pub enum MathMode {
    Strict,
    Approximate,
    Fast,
}
```

`Strict` does not apply real-number identities that fail for IEEE floating
point, NaNs, infinities, signed zero, overflow, or function domains. Examples
such as `exp(log(x)) -> x`, reassociation, distributivity, and moving operations
through reductions require guards, proof, or a weaker mode.

The current e-graph extractor uses AST size. A later fusion-aware extractor can
consider FLOPs, materialization, access-map complexity, and whether an
equivalent graph exposes a profitable region. Fusion must not assume every
e-graph rewrite is numerically valid.

## Transformer Strategy

General fusion should handle the broad memory-bound remainder of transformer
graphs:

- Pointwise activation and residual DAGs.
- Bias, scale, mask, comparison, and cast operations.
- Virtual reshape, transpose, head splitting/merging, and broadcast.
- Producers and epilogues around reductions.
- Matmul bias/activation/residual epilogues.
- LayerNorm/RMSNorm scalar work around a reduction schedule.
- Optimizer update DAGs.

Some high-value transformations are structured rewrites or templates:

- Q, K, and V projections can be rewritten to one concatenated GEMM when
  layouts and weights permit, or planned as horizontal fusion otherwise.
- SwiGLU and related gated MLP branches benefit from shared-input horizontal
  fusion or concatenated projection.
- Stable softmax needs a persistent reduction schedule.
- Cross-entropy can remain a recognized stable reduction template.
- FlashAttention uses online softmax and tiled Q/K/V traversal to avoid the
  quadratic score matrix; it is an algorithmic primitive.
- Prefill, decode, causal/sliding attention, grouped-query attention, and paged
  KV cache require distinct schedule/template families.

The first attention template should follow FlashAttention-2 concepts on Ampere/
Ada after BF16 tensor-core support, warp/block reductions, shared layouts, and
pipelined movement exist. FlashAttention-3's Hopper-specific TMA and warp-group
specialization are later target-specific work.

## Dtype Design

Fusion regions carry logical dtype per value. Scheduling separately chooses:

- Storage dtype.
- Load/conversion dtype.
- Compute dtype.
- Accumulator dtype.
- Output dtype.

This is required for BF16 storage with F32 reductions/accumulators and for
future FP8/quantized templates. Casts are scalar value operations, not index
maps. Dtype transitions participate in legality, costing, and cache keys.

The access/region IR should land before broad BF16 kernel work so BF16 paths do
not duplicate the current one-node-per-kernel architecture. BF16 primitives can
be developed in parallel once these interfaces stabilize.

## Staged Implementation

### Stage 0: Baselines and observability

- Record kernel count, materialized bytes, peak live bytes, compilation time,
  PTX size, registers, spills, shared memory, and latency.
- Add benchmark graphs for pointwise diamonds, reductions, transformer blocks,
  GPT training/inference, and LeNet.
- Preserve unfused execution as a correctness/performance baseline.

Acceptance: measurements are reproducible and each executor reports the chosen
plan or a readable unfused schedule.

### Stage 1: ExecutionPlan decoupling

- Introduce plan steps and original-node output mapping.
- Make PTX execution consume a plan rather than assume one kernel per node.
- Initially emit one kernel step for every existing node with identical behavior.

Acceptance: all tests and compilation-cache tests pass with no intended numeric
or performance change.

### Stage 2: Access IR and virtual views

- Implement iteration domains, index expressions/maps, predicates, and virtual
  tensors.
- Lower reshape, flatten, transpose, permutation, and broadcast as virtual plan
  values where no materialization is required.
- Compose and canonicalize nested maps.

Acceptance: view-only graph paths launch no kernels or copies, address mapping
matches CPU references for randomized shapes, and cache signatures are stable.

### Stage 3: General pointwise regions

- Add scalar SSA region operations.
- Form post-dominator-based unary/binary/select DAG regions.
- Support multiple external inputs and outputs.
- Replace and remove `FusedUnary` after parity is established.

Acceptance: randomized pointwise DAGs match unfused CPU/PTX behavior, diamonds
fuse, launch/materialization counts decrease, and resource limits split large
regions predictably.

### Stage 4: Broadcast, reindex, and materialization planning

- Compose virtual access maps into pointwise regions.
- Cost fan-out recomputation versus materialization.
- Add coalescing/index-complexity estimates.

Acceptance: transformer head reshape/permute chains and bias broadcasts are
usually virtual or fused, with no regressions on adversarial layouts.

### Stage 5: Reduction fusion

- Represent parallel/reduction domains and reduction identities.
- Fuse pointwise producers into reductions and pointwise epilogues out.
- Add warp/block and persistent reduction schedule families.
- Enforce numerical-mode reduction ordering.

Acceptance: softmax components, LayerNorm/RMSNorm statistics, and reduction
gradients match references and improve launch/global-byte metrics.

### Stage 6: Anchor fusion and tile-aware costing

- Fuse matmul and convolution prologues/epilogues.
- Carry fragment/tile outputs directly into epilogue scalar expressions.
- Add tile traffic, register, shared-memory, and occupancy estimates.
- Enumerate a small set of target-aware tile candidates.

Acceptance: matmul/conv bias, activation, and residual paths avoid intermediate
buffers; generated PTX uses expected tensor-core instructions; end-to-end
transformer and CNN benchmarks improve.

### Stage 7: Transformer composites and templates

- Add QKV and gated-MLP rewrites/horizontal plans.
- Add stable normalization and softmax templates where generic reduction plans
  are insufficient.
- Implement FlashAttention-2 forward, then backward.
- Add prefill/decode distinction before KV-cache work.

Acceptance: compare against unfused attention for correctness across aligned,
edge, causal, and batched shapes; demonstrate reduced memory complexity and
measured transformer speedup.

### Stage 8: Profiling, dynamic shapes, and advanced targets

- Profile only top analytical candidates and cache results.
- Add symbolic extents, guards, and shape buckets.
- Add architecture-specific asynchronous pipelines.
- Investigate Mirage-style multi-level search only after the schedule space and
  verifier are mature.

Acceptance: bounded compilation overhead, deterministic cache behavior, and no
regression when a profile entry is absent.

## Validation Strategy

### Correctness

- Differential tests between fused and unfused CPU/PTX graphs.
- Property-generated DAGs including fan-out, reconvergence, and multiple outputs.
- Shape boundaries around tile/vector sizes and zero extents.
- NaN, infinity, signed-zero, overflow, and reduction-order tests by math mode.
- Forward and backward graph coverage, including repeated IDs and overlapping
  reductions/scatters.
- PTX assembly and hardware execution on supported targets.

### Planner tests

- Golden region membership and split reasons.
- Access-map composition and canonicalization.
- Injective/non-injective write classification.
- Post-dominator diamonds and escaping users.
- Materialize-versus-recompute decisions.
- Stable structural signatures independent of node allocation order.

### Performance

- Kernel launches and materialized bytes are primary early metrics.
- Register count, spills, shared memory, occupancy, PTX size, and JIT time guard
  against over-fusion.
- End-to-end benchmarks cover transformer forward/backward, GPT generation,
  LeNet, convolution, and representative synthetic regions.
- Report compile/JIT separately from steady-state execution.
- Compare against the unfused Tile PTX backend and static CUDA baseline.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| IR project expands without producing kernels | Land stages with executable acceptance tests; Stage 3 must improve real pointwise graphs before reduction work. |
| Over-fusion reduces occupancy | Resource estimates, hard target limits, split diagnostics, and unfused candidates remain available. |
| Index-map complexity harms codegen | Canonicalization, complexity cost, and explicit materialization boundaries. |
| Graph rewrites change floating-point behavior | Strict default math mode and rewrite-specific proof/guard requirements. |
| Template catalog undermines generality | Templates are selected before generic partitioning and expose explicit contracts; ordinary pointwise/view work remains in the general region IR. |
| Profiling makes JIT unpredictable | Analytical default, small candidate cap, explicit profiling policy, persistent keyed cache. |
| Training fan-out causes recomputation | Cost fan-out using forward/backward consumers and permit multiple outputs/materialization. |
| Current Tile IR cannot express schedules | Stage the structured access/region layer before extending low-level tile scheduling; keep existing monolithic statements as template anchors during migration. |

## Open Questions

1. Should `ExecutionPlan` be shared across backends or begin PTX-specific with a
   backend-neutral interface?
2. Should Stage 3 support multiple outputs immediately, or first model escaping
   values as forced materializations?
3. What exact subset of index expressions is sufficient before symbolic shapes?
4. Should normalized access maps use multidimensional expressions, linearized
   offsets, or retain both forms?
5. Where should indirect gather/scatter access enter the general IR versus
   remaining template anchors?
6. Which register/occupancy model is accurate enough before SASS-level feedback?
7. What compile-time budget should gate optional profiling in interactive JIT,
   training, and ahead-of-time modes?
8. Which floating-point transformations belong in `Approximate` versus `Fast`?
9. Should QKV concatenation be represented as a semantic graph rewrite or as a
   multi-output horizontal anchor plan?
10. How should debug value retention force materialization without changing the
    normal optimized plan cache?

## Recommended Epic Split

Create one epic tracking issue with stage issues that can merge independently:

1. Fusion observability and execution-plan decoupling.
2. Virtual tensors and composable access maps.
3. General pointwise DAG fusion.
4. View/broadcast fusion and materialization planning.
5. Reduction and normalization fusion.
6. Matmul/convolution anchor fusion and tile cost model.
7. Transformer composites and FlashAttention.
8. Profiling cache, dynamic shapes, and advanced target schedules.

Issue #15 should own the epic or point to it. The current unary fusion feature
should remain only until Stage 3 supersedes it.

## References

- Niu et al., [DNNFusion](https://doi.org/10.1145/3453483.3454083), PLDI 2021.
- Ansel et al., [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://doi.org/10.1145/3620665.3640366), ASPLOS 2024.
- Feng et al., [TensorIR](https://arxiv.org/abs/2207.04296), 2022/ASPLOS 2023.
- Roesch et al., [Relax](https://arxiv.org/abs/2311.02103), 2023/ASPLOS 2025.
- MLIR, [Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/) and [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/).
- NVIDIA, [nvFuser](https://github.com/NVIDIA/Fuser).
- Shi et al., [Welder](https://www.usenix.org/conference/osdi23/presentation/shi), OSDI 2023.
- Zhu et al., [ROLLER](https://www.usenix.org/conference/osdi22/presentation/zhu), OSDI 2022.
- Tillet et al., [Triton](https://doi.org/10.1145/3315508.3329973), MAPL 2019.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
- Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691), 2023.
- Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608), 2024.
- Spector et al., [ThunderKittens](https://arxiv.org/abs/2410.20399), 2024.
- Wu et al., [Mirage](https://arxiv.org/abs/2405.05751), 2024/OSDI 2025.
- Ye et al., [FlashInfer](https://arxiv.org/abs/2501.01005), MLSys 2025.
- Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
