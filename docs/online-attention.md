# Online normalization

Enable experimental f32 inference fusion with:

```rust
use tnsr::ptx::{PtxExecutor, PtxReductionMode, types::F32};
let mut executor =
    PtxExecutor::<F32>::try_new_with_reduction_mode(PtxReductionMode::Online)?;
```

`Strict` remains the default. `Online` permits changed floating-point ordering
and deterministic reduction trees; comparisons require float tolerances.

The normalization rule matches `E = exp(S - broadcast(max(S)))` and `L = sum(E)`
over the last axis. A separate consumer rule moves division by the invariant
`L` outside a compatible indexed multiply/sum. Matmul and explicit
broadcast/multiply/reduce expressions share the same contraction representation.
The pass checks access maps, external consumers, complete physical-region
replacement, and dependency ordering before fusing. Unsupported patterns and
training graphs that need intermediate values retain the reference plan.

Lowering emits ordinary scalar expressions, memory accesses, guards, collectives,
and `ForLoop` with carried state. The tensor graph and nn API gain no attention
operation, and the backend needs no attention-specific instruction.

One block handles each normalized row for output widths up to 128. A warp
produces 32 scores into shared memory; the normalizer and sum accumulator are
rescaled once per tile before adding contributions. Wider outputs use a serial
schedule per output element. Leading masked scores preserve the identity;
all-masked outputs and nonfinite weights retain NaN behavior. The existing index
optimizer simplifies and hoists pure address expressions without moving loads,
floating-point operations, or barriers.

Run end-to-end comparisons with the existing GPU benchmark suite:

```sh
cargo bench --bench gpu --features cuda -- attention
```

Compilation is outside timing; transfers and buffer management are included.
The benchmark compares Strict and Online with strict-f32 matmuls. CUDA tests
cover equivalent contraction forms, batches, masking, nonfinite values, escaping
intermediates, gradients, and loop identities/scope.

The schedule is fixed scalar-f32 code, without general loop/tiling search or
attention tensor-core scheduling. Causal masks remain dense inputs. Standalone
materialized softmax outputs retain the existing path; standalone exponential
normalizers and normalized weighted sums can fuse without attention.
