# First kernel paper prototypes

Implementation started 2026-10-31 from the
[six-paper review](research/paper-review.md).
Two kernel alternatives are implemented and disabled by default. Planner
diagnostics were extended to support the later schedule-search work.
Initial sandboxed checks saw no `/dev/nvidia*` devices and returned
`CUDA_ERROR_NO_DEVICE`. Rechecking outside the sandbox confirmed a working
RTX 4080 with driver 580.196.07. GPU commands must run outside that restricted
device namespace; the initial failure did not establish that the host lacked
a working GPU. The required CUDA suite subsequently passed outside the sandbox:
16 tests across `matmul_shared_memory_reuse` and `ptx_online`, with zero skipped.

The subsequent [resident benchmark pass](kernel-paper-performance.md) confirms
1.11-1.19x throughput gains on three large BF16 projection kernels, alongside
flat and regressing controls. Online weight sharing improves one large synthetic
case but slows several smaller cases. Both options remain explicit; complete
workload qualification is still required.

## Shared-memory lifetime reuse

`PtxExecutor::set_experimental_shared_memory_reuse(true)` selects an arena
for ordinary tensor-core GEMM operand stages and the later f32 epilogue
exchange. It does not change the selected tile, layout, stage count, arithmetic,
or synchronization. Scalar schedules retain their original storage.

Operand stages have disjoint typed views in a byte arena. Only after the
existing final consumption barrier, and final async-copy drain when applicable,
does the f32 exchange view reuse offset zero. Arena views validate dtype/layout,
alignment, byte ranges, checked extent arithmetic, declared capacity, and
immutable arena identities. The caller still establishes lifetime ordering;
this is not a general automatic liveness allocator.

For the padded row-major 64x64x32 BF16 two-stage schedule, the generated shared
declaration falls from 35,840 to 19,456 bytes. Lower shared use is a resource
result, not a throughput result. Resource guards deliberately keep the original
disjoint budget during selection so this experiment isolates storage reuse.
`MatMulSchedule::shared_mem_bytes()` reports the selected allocation accurately,
including the scalar path's absence of an output exchange.

The option is wired through homogeneous `PtxExecutor`, including its BF16/F16
instances. Fixed-binding mixed/linear programs construct their own executors;
their public constructors do not yet expose this option. Integrate that rollout
only after device qualification of the ordinary GEMM family.

## Shared online exponential weights

Under `PtxReductionMode::Online`, call
`set_experimental_online_weight_sharing(true)` before compilation. The first
warp computes each key weight using the existing updated running maximum,
overwrites the shared score tile, and publishes weights with an additional
uniform block barrier. Consumers keep their original increasing-key denominator
and weighted-sum order. The extra barrier can outweigh saved exponentials on
small cases; selection remains explicit.

The experiment applies only to recognized cooperative online regions with up
to 128 output columns. Wider serial fallback and cached-decoder scan callers
retain the original lowering. It does not implement FlashAttention training,
new BF16 precision boundaries, or a new reduction tree.

Both options participate in compilation identity. Changing either invalidates
compiled execution and graph replay. The signature schema advances to 38.

## Schedule-search diagnostics

`PtxExecutionPlan::kernel_features_for_execution(graph, flavor)` distinguishes
homogeneous legacy dispatch from 256-thread fixed-binding pointwise dispatch.
It does not change which execution path runs.

New fields separate logical unique input/output bytes, repeated global operand
requests across GEMM CTAs, and shared operand consumption for the implemented
4x1 and 4x4 SIMT thread tiles. The tensor-core shared-read estimate is explicitly
unmodeled. These are compiler estimates, not DRAM counters, issued load counts,
or measured bank transactions. Independent predicates, cache reuse, and physical
transaction effects require separate treatment. Existing traffic scoring and
fusion selection are retained; the new fields are diagnostics, not a calibrated
new cost model. `fusion_probe` includes them in its JSON output.

## Validation and reproduction

The [compact evidence](benchmarks/kernel-paper-prototypes.json) records source
hashes, offline resource examples, and pending device checks. Final validation:

- 309 CPU tests passed, none skipped.
- 299 CUDA-feature library tests passed; four manual/ignored tests skipped.
  These are compiler/unit tests, not device qualification.
- With `TNSR_REQUIRE_CUDA=1` outside the sandbox, all 16 tests across the
  shared-memory-reuse and online suites passed, none skipped. This covers
  finite output bits, NaN classification, changed feeds, policy invalidation,
  transpose/broadcast/tail/zero-K GEMMs, and online masks/nonfinite inputs.
- CPU check, CUDA all-target check, strict all-target/all-feature Clippy, and
  rustfmt passed.
- 194 PTX modules assembled successfully for SM89 with CUDA 12.9, with zero
  spilled functions. All 192 GEMM module sizes match their generated manifests.
- The representative two-stage row-major BF16 pair uses 82 registers in both
  offline assemblies, with shared allocation 35,840 versus 19,456 bytes.
  The online pair uses 32 registers and 132 shared bytes in both assemblies.

The CPU scan test executes actual generated scalar IR in lockstep, including
shared-write visibility at barriers. It compares 1,296 baseline/candidate cases
across output widths, key tails, empty scans, running maxima, masks, signed zero,
NaN/Inf and nonfinite value operands. It is not a model of GPU exponential
approximation or an alternative softmax implementation.

The offline GEMM test compares baseline/candidate schedules and emitted memory,
MMA, copy and barrier operations across BF16, F16, TF32 and TF32x3, transposed
broadcast operands, aligned/tail dimensions, zero K, and synchronous/asynchronous
policies. CPU layout tests also cover each existing shared-layout policy and
reject invalid arena extents and pointer rebinding.

Run compiler tests and export PTX without initializing a device:

```sh
cargo nextest run
TNSR_PAPER_PTX_DIR=target/paper-kernels/ptx \
  cargo nextest run --features cuda --lib
python3 integrations/kernel_research/assemble_ptx.py target/paper-kernels/ptx \
  --ptxas /usr/local/cuda/bin/ptxas --arch sm_89 \
  --output target/paper-kernels/assembly.json
```

The assembler report records PTX/cubin hashes and offline assembler resources;
it does not describe the actual driver's JIT, execute kernels, or measure speed.

On a working GPU, require actual CUDA initialization for the prepared output and
cache-invalidation comparisons, rather than counting unavailable-device returns
as qualification:

```sh
TNSR_REQUIRE_CUDA=1 cargo nextest run --features cuda \
  --test matmul_shared_memory_reuse
TNSR_REQUIRE_CUDA=1 cargo nextest run --features cuda \
  --test ptx_online -E 'test(online_weight_sharing)'
```

Before retaining either option automatically, extend the passed targeted output
and nonfinite checks to complete workloads; inspect actual driver resources and generated machine code,
resident-kernel event timings with rotated baseline/candidate order, and
complete workloads including capture/replay and conversions. Shared-memory
reuse should then be checked in the mixed-training path with the original
numerical gates and matched updates.

## Remaining experiments

Advance selective shared-memory reuse through the mixed-program path and matched
complete updates, using the [measured component results](kernel-paper-performance.md)
as candidate evidence rather than a full-model speedup prediction. Keep online
weight sharing experimental after its confirmed smaller-case regressions.
Inspect the current 4x4 SIMT kernel's real
instruction schedule before adding register-fragment prefetch or new padding.
Measure how often BF16 prefill's accumulator-rescale exchange executes before
adding register-fragment row scaling. A bounded joint fusion/schedule generator
should retain the original plan, use existing legal schedule constructors, and
include copy/materialization costs; the new diagnostic fields alone do not
create or validate that search space.
