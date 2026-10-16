# Corrected TF32 matmuls for forty-five training

`MatMulPrecision::Tf32x3` enables corrected TF32 tensor-core products for f32
matmuls on SM80+ targets. The forty-five trainer exposes this through `--tf32x3`
and JSON `settings.tf32x3=true`. Strict f32 remains the default and reference.
The policy cannot change during an optimizer session.

The kernel splits each f32 operand into `hi = tf32(x)` and
`lo = tf32(x - hi)`, then accumulates `a_hi*b_hi`, `a_hi*b_lo` and
`a_lo*b_hi`. Both synchronous staging and asynchronous copies retain the original
f32 operand bits until the fragment split. Outputs, activations, master weights,
accumulated gradients and Adam state retain f32 storage. Shared layouts, batched
broadcasts, fused epilogues and the existing target/resource limits still apply.
Small/ineligible shapes and targets without TF32 use the scalar fallback.

This policy changes rounding and accumulation order. The omitted `lo*lo` term,
tensor-core accumulation, and subnormal/overflow behavior mean it does not
provide strict scalar-f32 semantics. It also does not implement the original
model's selective BF16 autocast contract. See the
[original BF16 audit](training-bf16-contract.md) for the remaining mixed-storage,
cast/autograd and calibration work.

## Usage

Build the worker, then add the new flag to the existing training command:

```sh
cargo build --release --features cuda --example forty_five_worker
/path/to/forty-five/.venv/bin/python integrations/forty_five/train.py \
  /path/to/forty-five/models/v8.4.onnx \
  --forty-five-root /path/to/forty-five --backend cuda \
  --output target/forty_five_training_tf32x3 \
  --tf32x3 --tree-reductions --cuda-graphs
```

Inference retains its strict-f32 policy. Training responses, validation reports
and benchmark reports name the matmul policy. The training CLI and worker reject
TF32x3 on the CPU backend. CUDA Graph replay remains an independent option.

## Validation

The CPU suite passes 193 tests; the CUDA-required all-feature suite passes 387.
The new GPU checks cover precision-policy recompilation, separate rounded
products/K order in the strict reference, and corrected products on odd/expanded
tiles with contiguous, padded and swizzled shared layouts and synchronous and
asynchronous pipelines. Non-binary operands distinguish corrected arithmetic
from ordinary TF32; those cases reduce maximum error by more than 20 times.
Formatting, CPU/CUDA builds and all-target/all-feature Clippy pass.

Real-model checks use the unchanged CPU-f32 PyTorch PPO oracle and tolerances:
metrics `atol=3e-5, rtol=5e-4`, clipped gradients
`atol=3e-6, rtol=3e-3`, and updated weights `atol=2e-5, rtol=2e-4`.
Each check runs two optimizer updates with 3/3/2 microbatches, nonzero anchor KL,
clipping, all named parameter gradients, saved weights and scheduler steps.
These are short numerical checks, not a learning-quality experiment.

| Real model / execution | Parameters checked | Max gradient error | Max weight error |
| --- | ---: | ---: | ---: |
| Shared critic / tree reductions + replay | 140 | 1.19e-6 | 1.23e-5 |
| Separate critic / tree reductions + replay | 229 | 7.00e-7 | 1.44e-5 |
| Shared critic / strict reductions, no replay | 140 | 1.11e-6 | 1.13e-5 |

Both replay validations keep both microbatch shapes active with zero fallbacks.
The separate-critic validation took 461 seconds end to end, compared with 75
seconds in the earlier strict-f32 evidence. Those are not matched compilation
benchmarks: they also include the oracle, checkpoint I/O and host contention.
The long wait before its first update shows that startup needs further profiling;
the steady-state speedup below does not include that cost.

## Measurements

Measured on an RTX 4080 (SM89), using a fixed 64-transition game-derived batch,
the same initialized archive and optimizer settings, and identical tree
reductions and graph replay in both native workers. The baseline is the local
exact-integer-index worker, so this isolates matmul arithmetic. Complete update
timing includes JSON, input transfers, forward/backward, accumulated gradients,
clipping, Adam and status readback. Two setup/warmup updates are excluded.
First-use compilation and graph setup are recorded separately in the reports.

Ten-sample interleaved medians:

| Microbatch | Strict-f32 baseline | TF32x3 | Transitions/s, before → after | Throughput gain |
| --- | ---: | ---: | ---: | ---: |
| 8 | 106.48 ms | 90.10 ms | 601 → 710 | 18.2% |
| 64 | 81.03 ms | 64.75 ms | 790 → 988 | 25.2% |

Worker GPU memory was 746 → 748 MiB at microbatch 8 and 2510 → 2510 MiB at
microbatch 64. Both workers replayed every timed update with no fallbacks.
The graph still has 3851 nodes and the same physical kernel count; the change
accelerates matmul computation without removing other training work.

The original compiled-BF16 updater measured 244.49 ms at microbatch 8 and
25.74 ms at microbatch 64 in these runs. Native TF32x3 remains about **2.5x
slower** at microbatch 64; parity across the batch envelope is unfinished.
PyTorch timings varied more than native timings under the existing host load.
These policies use different arithmetic; the numerical checks above establish
short-update agreement with the f32 reference, not equivalence to BF16 training.

The initial native-only check alternated worker order across 20 timed samples:
median update time fell from **80.05 ms to 64.45 ms**, or **24.2% higher
throughput** (19.5% less time per update). Its first candidate setup/update took
31.10 s versus 2.62 s for the baseline. In the later microbatch-64 run the first
updates took 3.55 s and 3.70 s respectively, consistent with a cache-sensitive
startup cost. Short jobs must account for setup as well as steady-state speed.
Full matched benchmark and real-model
results are retained in [the evidence](benchmarks/training-parity/tf32x3.json).
Generated full reports are under ignored `target/forty_five_tf32x3/`.

To repeat a matched comparison, the same current binary can serve both roles:
the benchmark enables TF32x3 only in the candidate worker.

```sh
TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_training.py ARCHIVE \
  --forty-five-root /path/to/forty-five \
  --worker target/release/examples/forty_five_worker \
  --baseline-worker target/release/examples/forty_five_worker \
  --output target/forty_five_tf32x3_repeat --microbatch 64 --repeats 10 \
  --tf32x3 --tree-reductions --baseline-tree-reductions \
  --cuda-graphs --baseline-cuda-graphs --compile
```
