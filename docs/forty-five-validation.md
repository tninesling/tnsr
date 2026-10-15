# Forty-five port validation — 2026-10-14

The real forty-five model runs inference and PPO through tnsr on CPU and generated
CUDA/PTX. The integration also collects real Rust game rollouts, restores trained
weights, evaluates both seat pairs, and exports the existing serving ONNX format.
This is an f32 compatibility milestone; the short arena runs do not establish
playing-strength improvement or production training throughput.

## Environment and sources

- Forty-five source: `c5f595c4812191aa050458f6efcfcfe2751af846`.
- PyTorch `2.11.0+cu130`; CPU-f32 reference, two CPU threads.
- NVIDIA GeForce RTX 4080, 16,376 MiB, driver `580.178.04`.
- CUDA native matmuls use strict f32. CPU and CUDA are selected explicitly.
- Shared model: `models/v8.4.onnx`, SHA256
  `3bf8cbb5dbba1ecec8d22d0492b4afb5e2e2c017c628ff9f47c4b379c6fa1d85`.
- Separate legal critic: `checkpoints/v9-2x2-20261004/width-256-separate/best.pt`, SHA256
  `3daae76ac6f58e70b0aa25487c3f39a93f7c1a386f101f3a01e436f735cd2463`.
- Both models: width 256, eight heads, six layers, 64 tokens, learned positions,
  SwiGLU, phase/card-conditioned action heads. RoPE, privileged critics and legacy
  non-SwiGLU encoders are explicitly outside this port.

## Inference and checkpoint parity

Probes include real game observations from all four playable phases, phase
embedding IDs 0/5, and all-padding input. Maximum absolute errors against original
PyTorch:

| Model | Backend | Logits | Values | Legal-action probabilities |
| --- | --- | ---: | ---: | ---: |
| v8.4 shared | CPU | 1.63e-5 | 5.22e-7 | 6.56e-7 |
| v8.4 shared | CUDA | 1.41e-5 | 8.50e-7 | 6.56e-7 |
| Separate critic | CPU | 1.24e-5 | 4.48e-7 | 5.37e-7 |
| Separate critic | CUDA | 1.15e-5 | 4.77e-7 | 3.58e-7 |

Recovering v8.4's original named weights from ONNX is checked independently:
restored PyTorch versus ONNX Runtime has maximum logit error 1.44e-5 and value
error 5.97e-7. Every native save reloads exactly into PyTorch weights, and restores
into a fresh native worker. Repeat native inference is deterministic.

## PPO parity

The oracle calls forty-five's existing `ppo_update` in CPU f32. A fixed batch
exercises positive/negative advantages, policy/value clipping, entropy, nonzero
bidding-anchor KL, partial microbatches, gradient accumulation, global clipping,
policy/value learning-rate groups, two Adam updates, and cosine scheduling.
Every named clipped gradient and every updated weight is checked each step:

| Model | Backend | Named parameters | Max gradient error | Max weight error |
| --- | --- | ---: | ---: | ---: |
| v8.4 shared | CPU | 140 | 3.64e-7 | 3.00e-6 |
| v8.4 shared | CUDA | 140 | 3.41e-7 | 3.00e-6 |
| Separate critic | CPU | 229 | 2.84e-7 | 9.35e-6 |
| Separate critic | CUDA | 229 | 1.48e-7 | 1.28e-5 |

The validator uses gradient `atol=3e-6, rtol=3e-3` and updated-weight
`atol=2e-5, rtol=2e-4`; near-zero gradient differences can be amplified by Adam.
These are tolerance-based f32 comparisons, not bitwise-identical training.
Small committed shared/separate fixtures check two updates in ordinary CI without
the sibling project. A separate test confirms actor/value gradient isolation.

## Real game/training/evaluation runs

The final shared-model command is the README training example with
`--final-entropy-coef .004`. Each of two updates collects 62 transitions, spanning
all playable phases, and completes three hands (the collector may overshoot a
requested hand target). Whole-rollout normalization and original hand grouping
are preserved. The seeded first native/PyTorch rollouts have identical tokens,
types, phases, masks, hands and sampled actions. Maximum log-probability error is
6.68e-6; value/GAE errors are below 4.83e-6; returns differ by at most 2.99e-8.

Both updates save/reload correctly. Changing entropy coefficients preserves the
two compiled microbatch shapes. After training, live native predictions match
restored PyTorch and a fresh native worker. Two arena games alternate opposite
seat pairs and finish without truncation; native and restored-PyTorch outcomes
are identical. The separate-critic run also completes a rollout/update with two
frozen-opponent environments and identical native/PyTorch arena outcomes.

The final native archive is restored through forty-five's original checkpoint
loader and ONNX exporter. ONNX checker and ONNX Runtime pass; exported final
logit/value/probability errors versus restored PyTorch are at most
7.63e-6 / 6.71e-7 / 5.37e-7. No serving model is installed or promoted.

## Timing and remaining performance work

These are small synchronous smoke measurements on a workstation also running
other work. Compiler/disk caches and concurrency affect them. They do not compare
native f32 with the original production GPU/bf16 training workload.

| Shared-model run | Rollout seconds | PPO seconds |
| --- | ---: | ---: |
| Initial compile run, update 1 | 11.59 | 69.90 |
| Same worker, update 2 | 1.60 | 5.60 |
| Final validation with existing disk cache, update 1 | 1.97 | 7.54 |
| Same final worker, update 2 | 0.46 | 5.88 |

The matched seeded original CPU-f32 rollout takes 0.55 seconds in the final run.
No matched production GPU throughput claim is made. Final native/PyTorch arena
runs take 7.01/4.84 seconds respectively on the same deals.

The shared model has 24,079,680 bytes of f32 parameters. Each 62-transition update
uses eight microbatches and transfers at least 192,637,440 parameter bytes to
CUDA and the same named-gradient payload back. Inputs/constants, extra result
copies, JSON and allocation overhead are additional. Reports count JSON bytes and
separate rollout/PPO time. Compilation is reused; model data still crosses the
host boundary on every execution. Next performance work should establish a
matched workload, then retain versioned device parameters, accumulate/clip/update
on device, and only then evaluate mixed precision.

## Checks and reproducibility

Passed `cargo check`, `cargo check --features cuda`, 179 CPU nextest tests,
333 all-feature nextest tests with `TNSR_REQUIRE_CUDA=1`, four doc tests,
`cargo fmt --all --check`, and all-target/all-feature clippy with warnings denied.
See [integration commands](../integrations/forty_five/README.md) for rerunning all
checks. Full machine reports/checkpoints from this workstation are retained under
ignored `target/forty_five_validation/`: `v84-inference`, `separate-inference`,
`v84-ppo`, `separate-ppo`, `v84-training`, `separate-training`, and `final-driver`.

Archives/checkpoint handoffs contain model weights/config. They do not resume
Adam moments, RNG, rollout state or a scheduler mid-run. The driver writes fresh
output directories, keeps the frozen bidding anchor, and leaves original
experiments, snapshot/promotion policy and serving files untouched.
