# Exact integer lookup and loss indices

The local implementation of issue #113 separates index sources from floating
tensor storage. `IndexTensor` describes a named input or immutable constant of
`u32` values. `IndexInputs` supplies those named feeds independently of f32,
f16 or bf16 data. `checked_indices` and `IndexTensor::from_i64` reject negative
or oversized signed values without passing through a floating type.

```rust
use std::collections::HashMap;
use tnsr::{Executor, SimpleExecutor};
use tnsr::graph::TensorGraph;
use tnsr::tensor::{IndexTensor, TensorExpr};

let table = TensorExpr::constant(vec![1.0f32, 2.0, 3.0], vec![3, 1]);
let graph: TensorGraph<f32> = table
    .embedding_indices(IndexTensor::input("ids", vec![2]))
    .into();
let values = SimpleExecutor::new().execute_with_indices(
    &graph,
    HashMap::new(),
    HashMap::from([("ids".into(), vec![2, 0])]),
)?;
assert_eq!(values, vec![3.0, 1.0]);
```

Use `indexed_cross_entropy_indices` for exact targets. Integer constants use
the same operations and need no feed. Existing floating-index operations remain
compatible. Exact sources currently support indexed argument slots only; integer
arithmetic, general views and conversion to floating outputs are not exposed.
Graph validation rejects using an integer source in a floating operation or
declaring the same input name with both types. Index sources are nondifferentiable;
table/logits gradients retain their ordinary floating storage and sharing.

## CUDA ownership and replay

CUDA sources use actual `CudaSlice<u32>` allocations. Indexed PTX loads use
`ld.global.u32` and widen to u64 for addresses, rather than converting through
f32. Tile IR and virtual source metadata carry the integer dtype; floating
kernel arguments, allocations and casts remain independent. Integer computation
tiles and scalar floating accesses to integer arguments fail layout validation.

Compilation caches the required source slots, lengths and tightest consumer
bounds. Runtime feeds and current constant payloads are checked before any
integer buffer is overwritten or captured graph launched. This avoids rescanning
the full model to discover index requirements on every microbatch. The compile
signature distinguishes integer sources, names, shapes and edges (schema 15).
Constant contents can change without recompilation, with checked in-place upload.

Resident execution uses `ResidentAdam::accumulate_with_indices`. Each executor
owns stable index buffers on its private stream; staging precedes capture/replay.
Unchanged constants reuse their upload by weak host-source identity. The graph
is destroyed before any referenced integer buffer can be cleared, resized or
destroyed, under the existing stream gate. Invalid IDs preserve the captured
graph and skip its launch, allowing a later valid update to recover. Clearing
the pool invalidates capture before releasing index allocations.

The public executor/runtime `execute_with_indices` API uses checked ordinary
execution. PTX also exposes `execute_compiled_with_indices` and
`compile_and_execute_with_indices`. Ordinary calls retain their existing capture
invalidation behavior; the resident owner manages replay.

## Forty-five integration

The adapter now keeps token, type, phase, position and subset-embedding count
indices integer. Action targets feed the indexed cross-entropy primitive for
selected log probabilities, replacing host-generated floating one-hot selection.
Masks and normalization arithmetic remain f32, including the separate floating
discard-count scale. This is preparation for mixed precision; it does not enable
native BF16 training or change master/gradient/Adam storage.

The half-storage embedding forward kernel also now bounds threads by lookup
output size for every dtype. Its previous half path used vocabulary size, which
could truncate repeated lookups or let excess threads access past a short lookup
output. Regressions cover lookup counts both above and below vocabulary size.

Colocated tests cover signed conversion, scalar/empty sources, shape overflow,
name/type collisions, unsupported graph uses, repeated/shared gradients, exact
targets, a valid CUDA lookup at 16,777,217, and IDs 257/365 with f16/BF16 weights.
The resident test checks six updates across full/partial shape executors, changed
IDs, replaced constants, invalid-ID recovery and recapture after pool clearing,
comparing named clipped gradients with the CPU reference.

## Local validation and regression measurements

All **193 CPU** and **385 all-feature CUDA** tests pass, with CUDA required for
the GPU run. CPU/CUDA builds, formatting and all-target/all-feature clippy pass.
Allocation diagnostics include the separately owned integer buffers and reset
their counters without discarding warm allocations. Shape/name declaration
errors return before compilation or execution rather than relying on a lowering
assertion.

Both actual forty-five critics pass inference, deterministic repeats and saved
checkpoint round-trips. Each also passes two PPO updates in plain strict mode,
strict replay and tree replay, including full/partial shapes, nonzero anchor KL,
schedule changes and every named clipped gradient and updated weight. Maximum
gradient/weight errors stay below **3.7e-7 / 1.3e-5**, under existing tolerances.
Both replay buckets remain active with zero fallbacks.

Two paired ten-sample tree-replay runs compare the preserved pre-index worker
with this local implementation on the same v8.4 archive and seeded 64-transition
batch. Median ms/update is **103.49 -> 103.34** at microbatch 8 and
**81.91 -> 81.69** at 64, effectively unchanged within measurement noise. Worker
GPU memory is unchanged at **746 / 2,510 MiB**. Existing experiments remained
running; builds, tests, checkpoint validation and timings were run separately.
These diagnostics do not establish exclusive-GPU throughput parity.

The exact action-loss primitive adds one physical kernel per microbatch
(22,744 -> 22,752 / 3,081 -> 3,082 per update). Logical intermediate sizes fall
by only about 11 KB per update; there is no meaningful activation-memory saving
from this index-source change. The physical BF16 activation/cast work remains
#114, followed by original-BF16 calibration in #115.

[exact-indices.json](benchmarks/training-parity/exact-indices.json) retains full
paired samples, compact correctness reports, source hashes and binary hashes.
The measured binary predates allocation-diagnostic and invalid-declaration
hardening plus additional regressions; generated PTX and valid-model arithmetic
are unchanged, and the final suites cover those follow-ups. This evidence
describes uncommitted local work; it does not imply a new published revision.
