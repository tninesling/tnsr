# Memory Allocator Implementation Plan

Tracking issue: [#37 Implement memory allocator for tensors](https://github.com/tninesling/tnsr/issues/37)

Status: implemented for CPU, static CUDA, and PTX executors.

## Background

Both executors keep `values: HashMap<NodeIndex, _>` where every intermediate is
materialized as an owned buffer and never freed, not even between `execute` calls.
Parameters and constants live in the graph itself (`Arc<Mutex<Vec<D>>>` for
parameters) and are cloned into `values` on every execution. Gradient nodes are
identifiable via `WithGrad::param_to_grad`, which defines what must stay pinned
for `get_gradients`. Execution is a single `toposort()` pass, which is the hook
for liveness analysis.

## Design overview

Three new pieces, then mechanical executor changes:

1. **Liveness analysis** (`src/graph/liveness.rs`): computed per `execute()` call, O(V+E).
2. **Buffer pool** (`src/alloc.rs`): size-bucketed reuse, one per executor.
3. **Pooled buffer handle**: `Arc`-based; dropping returns the buffer to the pool.

The final implementation keeps executor-owned buffers directly in the value
map and returns them to the pool at liveness boundaries. This avoids the
locking and reference-counting overhead of an `Arc`-based pooled handle while
preserving the same lifetime semantics.

After toposort, compute each node's last-use position. When execution passes a
node's last use, its buffer returns to the pool and can be handed out for the
next allocation. Pins (below) are exempt.

Pin set (never freed during execution):

- The output node (returned from `execute`)
- Gradient nodes listed in `WithGrad::param_to_grad` (read by `get_gradients`
  after execution)
- Parameters/constants/inputs: these stop being copied at all (see Phase 2), so
  pinning is free

## Phase 0: Baseline instrumentation

Before changing behavior, make memory measurable.

- Add `tracing` counters to both executors: bytes allocated, bytes live, pool
  hits/misses.
- Add a peak-memory test harness using `stats_alloc` (already in dev-deps) on a
  deep chain graph (e.g. 100 chained unary ops): current behavior is O(depth)
  live buffers; target is O(1).
- Record baseline peak RSS on `examples/mnist.rs` for comparison at the end.

Validation: baseline numbers written down; no behavior change.

## Phase 1: Liveness analysis

New file `src/graph/liveness.rs`:

```rust
pub struct Liveness {
    /// Position in topo order of each node's last consumer.
    last_use: HashMap<NodeIndex, usize>,
    /// Nodes that must survive execution (output, gradient nodes).
    pinned: HashSet<NodeIndex>,
}

pub fn analyze<D, G>(graph: &TensorGraph<D, G>, order: &[NodeIndex]) -> Liveness
```

- For each node at topo position `i`, its consumers' positions determine
  `last_use`; a node with no consumers is its own last use.
- If `G = WithGrad`, add all `param_to_grad` values plus the order's last node
  to `pinned`.
- Recompute on every `execute()`: graph rewrites/fusion change shape, so never
  cache across calls. At O(V+E) this is noise.

Validation: unit tests for chain, branch, diamond, and a `with_gradients` graph
(assert forward activations die at their backward consumer, grad nodes stay
pinned).

## Phase 2: Buffer pool + CPU executor

New file `src/alloc.rs`:

```rust
pub struct BufferPool<D> {
    free: HashMap<usize, Vec<Vec<D>>>,  // exact-size buckets
    stats: PoolStats,
}

pub struct PooledBuffer<D> { /* Arc-inner, returns to pool on Drop */ }
```

- Exact-size buckets first. Training loops run the same shapes every step, so
  exact match gives near-perfect reuse. Power-of-two bucketing is a follow-up
  only if fragmentation shows up.
- Pool owned by the executor, `&mut` access only: the execution loop is
  sequential (rayon parallelism is inside ops), so no locking needed.

`SimpleExecutor` changes:

- `values: HashMap<NodeIndex, PooledBuffer<D>>`; ops read through handles as
  `&[D]`.
- After computing the node at topo position `i`, remove every entry whose
  `last_use == i` and isn't pinned; dropping returns buffers to the pool.
- Stop cloning graph-owned data: `Constant`/`Parameter`/`Input` nodes insert
  shared `Arc` references into `values` instead of `data.clone()`. This is the
  "delayed materialization" half of the issue and is a pure win on its own.
- At the start of `execute()`, drain the previous `values` map (returns buffers
  to the pool). Fixes the current leak-across-executions behavior.
- API unchanged: `execute` still returns `Vec<D>` (clone of pinned output),
  `get_gradients` still returns owned vecs.

Validation: `cargo nextest run`; the Phase 0 chain test now shows O(1) peak;
`cargo bench --bench cpu` shows no regression (expect a small win from clone
elimination); clippy/fmt clean.

## Phase 3: Training-loop semantics

The issue's core requirement: intermediates die after their gradients are
computed.

- This falls out of Phases 1 and 2 for free on `WithGrad` graphs: liveness over
  the combined forward+backward graph kills each activation at its backward
  consumer. This phase is mostly proving it: a test asserting peak live bytes
  during a training-step execution scales with the live set, not the node count.
- Define the gradient-lifetime contract explicitly: grad nodes stay pinned until
  the next `execute()` call (or an explicit `release_gradients()`), so
  `SGD::step` can read them. Document on `Executor::get_gradients`.
- Run `examples/mnist.rs` end-to-end; compare peak RSS against the Phase 0
  baseline.

Validation: new peak-memory test passes; mnist trains correctly with measurably
lower peak RSS.

## Phase 4: CUDA executor

Same structure on GPU:

- `CudaBufferPool` holding `CudaSlice<f32>` buckets, one per size, on the
  device's stream.
- cudarc allocations are stream-ordered, so reuse on the same stream needs no
  synchronization. Multi-stream would need events; explicitly out of scope.
- `CudaExecutor::values` becomes pooled handles; same last-use eviction.

Validation: `cargo nextest run --features cuda`; `cargo bench --bench gpu
--features cuda` (pooling should help alloc-heavy graphs); no numeric changes vs
CPU reference.

## Phase 5: PTX executor + docs

- Apply the same treatment to `PtxExecutor`.
- Document the memory model in `lib.rs`/`runtime.rs` docs: what's pinned, when
  things are freed, and that `get_value` on non-pinned intermediates may return
  `None` after execution (behavior change worth calling out; it is currently
  "useful for debugging"). If that matters, add a `retain_values` debug flag on
  the executors.

## Out of scope

- Dead-node skipping (only computing nodes reachable from output + grads): a
  natural follow-up, but separable.
- Cross-executor pool sharing, unified CPU/GPU memory, multi-stream.
- Allocator-aware kernel planning (that is #36/#43 territory).

## PR split

1. Phase 0 + 1 (instrumentation + liveness, no behavior change)
2. Phase 2 (CPU pool, the bulk of the work)
3. Phase 3 (training semantics + mnist numbers)
4. Phases 4 + 5 (GPU + docs)

Each PR is green on `cargo nextest run`, `cargo clippy --all-targets
--all-features -D warnings`, and `cargo fmt --all --check` on its own.

## Open design decisions

1. `get_value` returns `None` for released intermediates. A future
   `retain_values` debug flag can opt back into retaining all values if needed.
2. Pools use exact-size buckets, which match repeated training workloads.
   Power-of-two bucketing remains a follow-up if fragmentation is observed.
