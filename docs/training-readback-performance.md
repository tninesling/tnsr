# Gradient readback performance — issue 89

The first implementation after the [parity research](training-parity-research.md)
removes per-element CPU interpretation for affine virtual output layouts. A
checked descriptor derives source offset/strides from the general index map,
proves coordinate bounds/overflow safety once, drops singleton axes, and copies
with one coordinate vector for the entire read. Contiguous trailing axes use
block copies; a full contiguous rank-only reshape returns the original vector.
Non-affine maps keep the existing checked evaluator. Floating-point values are
copied without arithmetic, preserving NaN payloads and signed zero.

This is a general readback optimization, not a model-shape-specific kernel or a
resident optimizer. Parameters still upload and gradients still download per
microbatch; device residency remains issue 91.

## Paired forty-five measurements

Same v8.4 archive, real seeded rollout expanded to 64 transitions, one optimizer
step, f32 native arithmetic, anchor KL disabled, two warmups and three timed
samples per case. Saved baseline binary corresponds to merged research revision
`cc5c69b`. Before/after binaries and original PyTorch cases are interleaved on the
RTX 4080, with the existing training job left running. CUDA validation is run
separately. These are diagnostic samples affected by contention, not an exclusive
GPU performance gate.

| Median seconds | Microbatch 8 | Microbatch 64 |
| --- | ---: | ---: |
| Before: interpreted readback | 5.858 | 1.036 |
| After: compiled strided readback | 1.164 | 0.438 |
| Original PyTorch eager f32 | 0.239 | 0.0311 |
| Original PyTorch compiled-policy bf16 | 0.187 | 0.0324 |

Whole-update improvements are about **5.0x / 2.4x**. Median gradient extraction
falls **5.100 -> 0.437 seconds** at microbatch 8 and **0.622 -> 0.0467 seconds**
at microbatch 64. The native kernel count is unchanged: 52,104 / 29,361 per update.
The remaining gap is still about 5x/14x versus eager f32 in these particular
samples. Dispatch, device kernel efficiency and resident state remain important;
this change does not establish training throughput parity.

Raw timings, stage samples, source/batch and executable hashes are retained in
[readback.json](benchmarks/training-parity/readback.json). Full generated inputs
and reports are in ignored `target/forty_five_readback/paired-micro8` and
`paired-micro64`.

## Validation and reproduction

Seven focused tests cover all rank-three permutations, singleton removal,
non-affine reshape/broadcast fallback, positive/negative/zero strides, bounded
floor/mod arithmetic, empty/scalar outputs, invalid coordinates/storage,
intermediate overflow and exact float bits. Hundreds of generated affine maps
are compared with the original evaluator. The full 179 CPU / 340 all-feature
CUDA-required suites, formatting, clippy and doc tests pass. Full-sized shared
and separate-critic PPO named-gradient/weight parity is checked separately.

The research harness accepts `--baseline-worker PATH` for interleaved binary
comparisons, includes profiler stage results, and records both executable hashes.
Preserve the baseline release binary before building the changed source:

```sh
cp target/release/examples/forty_five_worker /tmp/tnsr-before-worker
cargo build --release --features cuda --example forty_five_worker
TNSR_PROFILE_PPO=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  /path/to/forty-five/.venv/bin/python integrations/forty_five/benchmark_training.py \
  /path/to/native/archive --forty-five-root /path/to/forty-five \
  --baseline-worker /tmp/tnsr-before-worker --transitions 64 --microbatch 8 \
  --compile --output target/forty_five_readback/comparison
```

Repeat with `--microbatch 64` and a fresh output directory. Cold initialization
and compilation stay in excluded warmups. Do not run CUDA validation concurrently
with throughput measurements.
