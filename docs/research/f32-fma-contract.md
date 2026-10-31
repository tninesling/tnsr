# F32 contraction contract: research audit

Research only, 2026-10-30. No production or benchmark code was changed, no builds
were run, and this investigation did not use the GPU.

## Decision

Adding FMA is **not currently supported as the next performance opportunity**:
the retained driver's compiled hot GEMMs already contain `FFMA`. The useful
next step is to distinguish the intended arithmetic contract from the one
actually guaranteed by emitted PTX, add a discriminating test on the register
path, and make any future arithmetic selection explicit. Enforcing separately
rounded products could change performance; it must not silently replace or
be presented as the frozen historical baseline.

## Intended versus guaranteed versus observed

The current `MatMulPrecision::StrictF32` selects scalar F32 accumulation and
disables TF32. Its documentation promises increasing-K scalar accumulation.
The register lowering additionally comments that each product is multiplied
then added separately. It calls `Inst::mul_f32` and `Inst::add_f32` in that order.
See [policy selection](/home/taylor/code/tnsr/src/tile/matmul_schedule.rs:7),
[register lowering](/home/taylor/code/tnsr/src/ptx/graph/matmul.rs:201), and
[small scalar lowering](/home/taylor/code/tnsr/src/ptx/graph.rs:1212).

However, [F32 code generation](/home/taylor/code/tnsr/src/ptx/codegen.rs:419)
emits `mul.f32` and `add.f32`, without explicit rounding modifiers. The module
is passed directly to the CUDA driver with `load_module(Ptx::from_src(...))`;
this call site specifies no option disabling contraction. NVIDIA documents
that unqualified multiply/add sequences may contract, while explicit rounding
modifiers constrain optimization. Explicit `fma.rn.f32` specifies a fused
operation with one final rounding. These are different contracts even though
all inputs, storage and accumulation are F32. [PTX ISA, floating-point
instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul)

The parallel GPU investigation identified a cached driver-JIT ELF containing
the frozen retained PTX byte-for-byte. Its disassembly reports 128 `FFMA` and
zero `FMUL`/`FADD` for each examined entry: `matmul_region_1_4796`,
`matmul_region_5_4800`, `matmul_region_50_4845`, `matmul_region_94_4889` and
`matmul_region_96_4891`. This is actual retained driver code, not just a fresh
offline assembly. Exact cache, PTX and ELF provenance belongs to the companion
[GEMM bottleneck investigation](ppo-gemm-bottlenecks.md).

This proves contraction for those retained entries. It does not establish the
instruction mix of every kernel or every future driver/compiler build, nor a
potential speedup from replacing already-contracted code with explicit FMA.

## Existing tests do not close this gap

`ptx_strict_matmul_preserves_rounded_products_and_k_order` in
[ptx_transformer.rs](/home/taylor/code/tnsr/tests/ptx_transformer.rs:1410)
uses M=17, N=19, K=65. The register tile requires M>=128, N>=256, K>=128, so
that test does not exercise the hot register path. Its near-one factors also
alternate with an exact negative product: each non-exact product starts from
a zero accumulator, allowing fused and separate accumulation to agree.

The large-shape cancellation test in
[matmul_register_ordered.rs](/home/taylor/code/tnsr/tests/matmul_register_ordered.rs:129)
uses a left operand of one. Its products are exact, so it tests accumulation
order but cannot distinguish product rounding from FMA. The general register
test uses a tolerant CPU f64 dot product. The nonfinite test checks NaN and
infinity classes, not a fused-versus-separate overflow/cancellation case.

### Focused future test specification

Use symbolic input feeds with M=129, N=257, K=129, then assert that the actual
selected schedule has `ScalarF32`, a 32x32 output tile and
`ThreadArray { elements: 4 }`. Do not rely on dimensions alone; retain the
schedule assertion if selection heuristics change.

For every output, arrange products in this exact K order:

```text
k=0: -1 * 1
k=1: (1 + 2^-23) * (1 - 2^-23)
k>=2: 0 * 0
```

The second product is exactly `1 - 2^-46` before rounding. An enforced
separate-product policy produces +0; ordered FMA produces `-2^-46`, which is
a normal F32 value. Derive the expected bits analytically, or use a CPU
`f32::mul_add` reference for the fused case and an explicitly noncontracted
reference for the separate case. Do not use a generic optimized CPU expression
as the sole proof that operations remained separate.

Check every output, including M/N tails; repeat with changed feeds,
transposed operands, contiguous/padded/swizzled shared layouts and both
synchronous and supported double-buffered staging. Include signed zeros,
subnormal boundaries, overflow with cancellation, NaN/Inf classes and epilogues
as separate semantic tests. Numerical tolerance qualification is still needed
for complete models; it does not replace this arithmetic-contract test.

## Competitor precision contract

The benchmark's `strict-f32` label currently means F32 contractions/storage with
TF32 and BF16 disabled; it does **not** specify separately rounded products or
a common reduction tree across frameworks. See the sibling
[PPO protocol](/home/taylor/code/tnsr-benchmarks/docs/ppo.md:5).

| Adapter | Verified local controls | What is not established |
| --- | --- | --- |
| PyTorch | CUDA/cuDNN TF32 flags false; `set_float32_matmul_precision('highest')`; F32 model/input tensors | Actual fused-instruction counts and reduction order of each frozen eager/compiled kernel |
| tinygrad | `TC=0`, `ALLOW_TF32=0`, BEAM0, optimizer fusion0 | Exactly which frozen kernels used compiler contraction or explicit MULACC |
| Burn | Pinned compatibility patch disables implicit F32-to-TF32 and Flex32-to-F16 promotion | A no-contraction guarantee or actual selected kernel instruction mix |

PyTorch documents `highest` as F32 internal computation, not a prohibition on
FMA. Its numerical-accuracy documentation does not promise bitwise equivalence
between mathematically equivalent executions. [PyTorch matmul precision](https://docs.pytorch.org/docs/main/generated/torch.set_float32_matmul_precision.html),
[numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)

The installed tinygrad CUDA renderer's NVRTC options do not disable `--fmad`;
its PTX renderer also has an explicit floating MULACC mapping to `fma.rn`.
NVRTC documents contraction enabled by default. This shows permitted backend
behavior, not proof of the frozen PPO kernel mix.
[NVRTC options](https://docs.nvidia.com/cuda/archive/12.9.0/nvrtc/index.html)

The Burn patch's recorded scope is precision promotion, not contraction. Its
register-tile source uses accumulator-plus-product expressions. No claim that
all Burn kernels use FMA, or that none do, follows from these sources alone.

## Concrete optional-policy implementation map

Do not add a second `MatMulPlan::ScalarF32`-like plan just to express contraction:
many layout/dispatch/resource checks treat any non-scalar plan as tensor-core
work. Keep operand precision and accumulator arithmetic separate.

An independent policy such as `MatMulF32Arithmetic` can expose:

- `CompilerDefault`: legacy code generation, preserving the current default
  and frozen behavior, without a guarantee about contraction.
- `OrderedSeparate`: explicit `mul.rn.f32` followed by `add.rn.f32` per K.
- `OrderedFma`: explicit `fma.rn.f32` per K, without TF32, reassociation,
  split-K, saturation or flush-to-zero.

This is a proposed API, not an implemented mode. A future decision to make
enforced separation the default is an arithmetic-contract correction requiring
new validation and performance evidence, not a free hardening change.

| Layer | Required future change |
| --- | --- |
| `src/ptx/instructions.rs`, `codegen.rs` | Add explicit F32 FMA and explicit rounded mul/add forms without globally changing unrelated pointwise instructions. Existing generic `FmaInst<T>` formats integer `mad.lo`, so do not route F32 through that formatter unchanged. |
| `src/tile/matmul_schedule.rs` | Carry an independent arithmetic policy in the schedule via a builder/setter; preserve `MatMulPlan::ScalarF32`, resource selection and all half/tensor-core precision gates. |
| `src/ptx/graph/matmul.rs`, `graph.rs` | Apply the choice consistently to register-tiled and one-output scalar paths; preserve increasing-K dependencies and epilogue order. |
| `src/ptx/executor.rs` | Setter invalidates compiled signatures and replay. Encode the policy in graph signatures and bump signature schema version when encoding changes. Repartitioned final schedules must retain it. |
| `src/ptx/linear/program.rs`, mixed/resident constructors | Thread the policy into every executor/program that recompiles graphs; do not configure only an inference executor. |
| `examples/forty_five/ppo.rs` | Add an explicit setting if PPO prototypes proceed, include it in cached-setting compatibility, configure the resident-created executor near line 516, and report it in result metadata near line 746. The generic worker `runtime()` setting alone does not control resident PPO. |
| Benchmark harness | Reject unsupported combinations, hash/freeze workers and sources, expose arithmetic metadata and create new profile labels. Do not mutate existing historical rows. |

Qualification must keep the existing metrics, gradients, weights and Adam-moment
gates unchanged. Use identical initial weights, state checkpoints before/after
measured windows, matched process pairs at microbatches 64 and 8, and separate
cold/steady/memory scopes. Cross-check LeNet, GPT and AlexNet before promoting a
general default. A meaningful explicit-FMA experiment must first show a code
difference from the retained SASS; otherwise it is a contract/reproducibility
experiment rather than a new arithmetic-throughput optimization.
