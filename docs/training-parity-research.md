# Training performance parity research

Measured on 2026-10-14 against the real forty-five v8.4 shared actor/critic. The
largest immediate bottleneck is CPU reconstruction of gradient views. Batched
matmul dispatch and device-resident training state are the next runtime/compiler
priorities. These measurements revise the earlier hypothesis that PCIe volume or
host Adam alone explained the gap.

## Matched GPU measurements

RTX 4080, PyTorch 2.11.0+cu130, width 256, six layers, eight heads, sequence 64.
Both implementations start from the same named weight archive. Inputs come from
seeded real Rust game collection; 62 collected transitions are repeated to form
one fixed 64-transition minibatch. All rows are assigned one synthetic hand ID
for the timing experiment so the original updater preserves the same row order.
This affects grouping only; game/observation/reward data remain unchanged.

Each call performs one PPO epoch and one Adam update, including microbatch
accumulation, global clipping, policy/value learning-rate groups and metrics.
Bidding-anchor KL is disabled in this matrix to compare identical work; anchor
parity remains covered by the existing correctness validator. Native calls include
JSON/process communication, input uploads, gradient reads, CPU accumulation/Adam
and finite checks. PyTorch calls use the original updater with its normal input
staging and metric synchronization. For strict f32 only, a copied updater body
has its autocast decision disabled; TF32 is disabled. The bf16 case uses the
original CUDA autocast behavior. The compiled case wraps the policy with
`torch.compile(dynamic=True)` in default mode, matching the original optional
training path; it does not compile the entire Python PPO loop or explicitly fuse
Adam. Original source and updater hashes are recorded where available.

Two warmups are excluded and three synchronized wall-clock samples are taken.
Cases are interleaved. The existing forty-five training job is **left running**;
GPU load changes during the runs. These are diagnostic results, not an exclusive
GPU benchmark or statistically stable performance acceptance result. Profiler
runs are separate from the timed samples.

| 64 transitions / one optimizer step | Microbatch 8, median (range), seconds | Microbatch 64, median (range), seconds |
| --- | ---: | ---: |
| tnsr strict f32 | 6.448 (6.200–6.768) | 1.048 (0.878–1.056) |
| PyTorch eager, f32 with TF32 disabled | 0.211 (0.190–0.230) | 0.0324 (0.0294–0.0344) |
| PyTorch eager, bf16 autocast | 0.243 (0.225–0.252) | 0.0505 (0.0291–0.0579) |
| PyTorch compiled policy, bf16 autocast | 0.192 (0.175–0.208) | 0.0331 (0.0282–0.0360) |

The observed native gap is about 31–34x at microbatch 8 and 32x at microbatch 64.
Larger microbatches help both implementations. Bf16 is not consistently faster
in this small, launch-sensitive workload; operator time falls, but wall-clock
results still contain dispatch overhead and competing GPU work. It should not
be treated as a guaranteed first fix. Default forty-five production settings
allow up to 1024 transitions per microbatch, multiple hand minibatches and four
PPO epochs; this 64-transition experiment does not measure that full workload.

Cold starts are material: the first compiled-PyTorch bf16 calls took 68.7/47.4
seconds at microbatch 8/64, and native's first microbatch-64 call took 45.9 seconds.
These include initialization and compilation/cache effects, and are excluded
from warm throughput. The comparison is of actual configured paths, not the
maximum speed attainable by optimizing PyTorch itself.

Compact raw data, provenance, stage samples and trace totals are committed in
[research.json](benchmarks/training-parity/research.json). Large traces, generated
requests and full reports remain in ignored `target/forty_five_research/`.

## What the native profile says

The opt-in stage timer around `Executor::get_gradients` reveals most of the cost.
On the fixed parity fixture at microbatch 8, two warm updates average:

| Stage | Seconds per update |
| --- | ---: |
| Graph execution, including input/parameter transfers and final result read | 0.660 |
| Named gradient extraction | 4.808 |
| Host gradient accumulation/finiteness checks | 0.051 |
| Norm clipping, grouped Adam and final parameter checks | 0.045 |
| Total | 5.565 |

On the real-rollout batch with detailed readback logging, warm totals are about
6.44 seconds: execution 0.66, extraction 5.65, accumulation 0.073, remaining host
work 0.059. Additional per-read logging and host load affect these timings. The
copy/view breakdown across all three detailed updates averages **0.135 seconds
copying/allocating readback storage versus 5.273 seconds rebuilding host views**.

`PtxExecutor::get_value` downloads a physical source, then
`materialize_host_view` evaluates a virtual index map for every output element.
The general path allocates coordinate vectors and evaluates/maps/validates them
per element. It handles large transposed parameter-gradient matrices repeatedly
on the CPU. The most expensive layouts include `[1,256,683] -> [683,256]` and
`[1,256,768] -> [768,256]`. The singleton leading dimension also prevents the
existing same-shape/identity shortcut from helping. At eight microbatches,
roughly six million parameter-gradient elements are read back eight times per
update, with the large permuted matrices going through this general view path.

This explains why reducing PCIe bytes alone misses the dominant cost. Host Adam
is currently a small fraction. Removing gradient extraction entirely would still
leave roughly 0.7 seconds in this particular warm stage sample, so it would not
by itself establish parity with a 0.2-second PyTorch update.

Nsight Systems independently shows excessive launch fragmentation:

| Native trace, totals divided by three updates | Microbatch 8 | Microbatch 64 |
| --- | ---: | ---: |
| Kernel launches per update | 52,104 | 29,361 |
| Sum of GPU kernel durations, seconds | 0.227 | 0.188 |
| H2D copies per update | 3,200 | 400 |
| D2H copies per update | 1,216 | 152 |

The microbatch-8 trace uses the earlier fixed parity fixture with supplied anchor
inputs; the microbatch-64 trace uses the real-rollout batch. They have the same
4484-node graph structure for a given microbatch shape and runtime coefficients;
inputs affect timing. Trace totals are diagnostic, with profiler overhead and
contention; CPU API time and GPU duration overlap and must not be added together.
For scale, separate PyTorch profiles on the real rollout show 6928 eager-f32 /
7256 compiled-bf16 launches at microbatch 8, and 835 / 838 at microbatch 64. GPU
kernel duration totals are approximately 25.3/16.4 ms and 13.8/5.0 ms respectively.
There is a device-kernel efficiency gap as well as a host/runtime gap.

The fallback `execute_matmul` launches once per leading batch matrix. For
attention this includes the head dimension. The existing `execute_matmul_region`
already dispatches batches in `grid.z`, but `form_matmul_regions` skips anchors
with an empty epilogue. Unfused/backward matmuls therefore lose batched dispatch
precisely when they lack a pointwise epilogue. This is a compiler/dispatch coverage
problem, not a reason to bring back static CUDA or add forty-five-specific kernels.

## How to define parity

Proposed acceptance targets, not speedup predictions:

1. First reach at most **2x original eager CUDA f32 wall time**, retaining current
   loss/gradient/update correctness tolerances. This isolates runtime/compiler
   improvements from changing arithmetic precision.
2. Reach at most **1.25x original compiled-policy bf16 warm PPO wall time** on
   agreed forty-five shared/separate-critic workloads, and at most 1.5x its p90.
   Compare equal epochs, hand groups, transition count, microbatch shapes,
   coefficients, clipping, optimizer updates and data placement. Add both
   training-only and whole rollout/PPO/evaluation throughput metrics.
3. Establish a comparable VRAM envelope and batch capacity; no silent CPU
   offload/fallback. Record active/pool/reserved/peak memory separately. Repeat
   on an exclusive GPU with at least ten samples, randomized case order and
   cold-compilation time reported separately. Use multiple rollout seeds, full
   and partial microbatches, and the larger production batch sizes that fit.
4. Preserve numerical/learning behavior: f32 differential tests for all named
   gradients and optimizer state, then bf16 comparisons with appropriate
   tolerances, finite checks, multistep updates and multi-seed learning curves.
   Evaluate candidates with the existing fixed opponents and balanced seats over
   a meaningful game budget. A two-game smoke result establishes compatibility.

At the current diagnostic microbatch-64 median, the proposed 1.25x target is about
41 ms/update versus 1048 ms observed. That is a useful order-of-magnitude goal,
not an exclusive-GPU baseline or a guarantee about achievable performance.

## Implementation order

### 1. Remove CPU gradient-view reconstruction from the hot path

Add a device-gradient/readback boundary that can produce the requested logical
layout with the existing TileIR copy machinery. Downloads for correctness/debug
should be contiguous copies, with common host reshape/transpose fast paths where
appropriate. The normal training path should be able to keep logical gradient
views on device. Preserve the existing host `get_gradients` API for reference
validation, but stop requiring it for every training microbatch.

Acceptance: the view-copy microbenchmark and whole-model parity remain correct
for rank changes, permutations, overlapping/shared parameter uses and partial
batches; the multi-second per-element index interpreter disappears from warm PPO.
Measure how much time moved to device layout copies instead of assuming it is
free. This is the first change to pursue given the measured profile.

### 2. Make batched contraction dispatch independent of epilogue fusion

Use the existing batch-aware TileIR matmul-region lowering even for matmuls
without an epilogue, or give the generic fallback equivalent batch-grid dispatch.
Keep broadcast operands and backward layouts correct. Where a shared RHS makes
it legal, flatten leading batch/row dimensions into the GEMM M dimension; this
improves matrix size as well as launch count. Use the existing general index-map
optimizer to hoist/simplify address work. Existing matmul profiling in
[fusion-baseline.md](fusion-baseline.md) already demonstrates why simply fusing
more operations can regress if indexing becomes expensive.

Acceptance: rank-3/4 attention and backward GEMMs issue one batch dispatch per
contraction rather than one per matrix/head; full/partial/broadcast/zero-size
cases and f32 gradients pass. Re-measure CPU dispatch, GPU time, materialized bytes
and total PPO time; launch count alone is insufficient. NVIDIA's
[strided-batched GEMM discussion](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/)
provides the relevant execution model: uniform small GEMMs should be submitted
together, with strides/broadcast reuse, instead of many independent launches.
This principle can be implemented in generated TileIR without depending on
cuBLAS. A temporary cuBLAS comparison remains useful as a kernel-quality ceiling.

### 3. Keep parameters, accumulated gradients and optimizer state on device

Introduce an explicit training session with resident buffers keyed by parameter
identity/version. Upload unchanged weights/constants once, retain gradients
across microbatches with the same transition weights, and run global norm
clipping and grouped Adam through generated device operations. Reuse compiled
graphs and stable storage. Synchronize host weights explicitly for checkpointing,
serving handoff or host inspection; define invalidation for `set_values`, archive
loads, other executors and shared parameters. Cached inference must see updated
weights, and saves must never serialize stale host copies.

Move finite checks and index-validation reductions to device or validate known
host inputs once when uploading, retaining safe failure behavior. Return small
metrics/status values after an update instead of downloading all named gradients
and rescanning every parameter. Profile/differential tests can still opt into
full named readback. This removes the view bottleneck structurally and supports
future graph replay; fused Adam by itself only attacks the small current Adam
slice. PyTorch's [Adam documentation](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html)
explains its foreach default on CUDA and the vertical/horizontal fusion offered
by fused implementations.

Acceptance: no parameter/gradient host roundtrip between microbatches; group
learning rates, global clipping, moments, bias correction, schedule timing and
shared-gradient accumulation match multistep references. Explicit save/reload and
live inference/cache coherence tests are essential.

### 4. Reduce backward fragmentation and support fixed-buffer replay

Profile again after the first changes. Improve generic pointwise/reduction
regions for normalization, softmax, SiLU/SwiGLU and loss backward graphs; preserve
fan-out/liveness and numerical policy. Cache physical bindings/workspace and
avoid repeated graph-signature walks, plan clones and fresh host/device staging
where these become significant. Pool reuse alone does not remove dispatch or
host materialization.

Once buffers have stable addresses, capture repeated forward/backward/update
work with CUDA Graphs per supported shape bucket. Upload fresh observations into
long-lived buffers; handle partial batches with explicit masks/buckets. Keep
host validation, pageable transfers, checkpoint reads and uncapturable paths
outside capture. CUDA Graphs primarily reduce CPU dispatch overhead; PyTorch's
[CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html) describes fixed
addresses and static graph constraints, and
[`torch.compile` modes](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)
document the small-batch `reduce-overhead` strategy and its limitations. Capture
must follow stable resident storage rather than freeze today's upload/download
path. Consider graph-safe subsections if the entire update cannot be captured.

Acceptance: compare graph replay with the same uncaptured generated kernels,
verify stable addresses/state updates and reduced host time. Keep a correct
uncaptured fallback and show end-to-end benefit, including memory overhead.

### 5. Add actual mixed-precision training semantics

The backend already has BF16/F16 storage and tensor-core schedules; the model
worker is f32 and the tensor graph is homogeneous in its generic dtype. Converting
the entire model/graph to `bf16` would not reproduce autocast. Introduce per-value
storage/compute policy and cast boundaries: bf16-capable matmuls, f32 sensitive
normalization/loss reductions, f32 master weights, accumulated gradients and Adam
moments. Keep token/action indices exact and separately typed. Current token
indices travel in the graph's floating dtype; BF16 cannot exactly represent all
vocabulary IDs above 256, so a blanket dtype substitution can change lookups.

PyTorch's [AMP reference](https://docs.pytorch.org/docs/2.11/amp.html) applies
precision by operation rather than casting everything uniformly; its CUDA lists
include matmul/linear low-precision eligibility and f32 treatment for operations
such as normalization, softmax and loss calculations. Start from the original
bf16 policy, check each actual forward/backward dtype, and retain the strict-f32
reference. FP16 loss scaling is separate work if that format is added.

Acceptance: compare every named gradient/update against original bf16 training
with calibrated tolerances, ensure finite updates at clipping/underflow edges,
and run the multi-seed learning/arena checks before drawing quality conclusions.
Re-measure optimized tensor-core schedules on representative forward and backward
GEMM shapes, not just square standalone matmuls.

### 6. Optimize attention only when the new profile justifies it

Forty-five currently computes explicit attention scores/masks/softmax/matmul;
it does not call PyTorch SDPA. Sequence length is only 64. Existing tnsr online
normalization fusion is documented as inference-oriented and falls back when
training needs intermediate values. Extend general contraction/online reduction
planning and its backward support before adding a model-specific attention op.

PyTorch [SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)
can select FlashAttention or memory-efficient kernels when inputs permit.
[FlashAttention](https://arxiv.org/abs/2205.14135) reduces attention memory traffic
through tiled IO-aware exact attention. Those are useful designs and a potential
optimized baseline, but neither removes the measured host gradient interpreter.
Prioritize attention work using post-runtime-fix profiles, especially when moving
to larger batches/sequence lengths, and validate backward recomputation/masking.

## Reproducing this research

Use the original forty-five virtual environment with the built Rust extension:

```sh
cargo build --release --example forty_five_worker --features cuda

TORCHINDUCTOR_COMPILE_THREADS=4 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_training.py /path/to/native/archive \
  --forty-five-root /path/to/forty-five \
  --output target/forty_five_research/run-micro8 \
  --transitions 64 --microbatch 8 --compile --profile-torch
```

Repeat with a new output directory and `--microbatch 64`. No output directory is
overwritten. `--anchor-kl-coef .03` includes the current adapter's CPU frozen
anchor computation in native timing and the original GPU anchor in PyTorch;
report that placement difference rather than calling it a resident-kernel test.
`--fused-adam` supplies a separate optimized-PyTorch ceiling; the recorded matrix
uses the original default optimizer. For acceptance runs, increase warmups and
repeats and use an exclusive GPU window without interrupting other experiments.

To inspect native stages, `TNSR_PROFILE_PPO=1` adds a `profile` object to each
training response. `TNSR_PROFILE_READBACK=1` writes per-value copy/view timings to
stderr; this is a diagnostic run, not a throughput timing configuration:

```sh
TNSR_PROFILE_PPO=1 TNSR_PROFILE_READBACK=1 \
  target/release/examples/forty_five_worker /path/to/native/archive cuda \
  < target/forty_five_research/run-micro8/requests.jsonl \
  > target/forty_five_research/stages.jsonl \
  2> target/forty_five_research/readbacks.jsonl

nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=target/forty_five_research/native-profile \
  target/release/examples/forty_five_worker /path/to/native/archive cuda \
  < target/forty_five_research/run-micro8/requests.jsonl

nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  target/forty_five_research/native-profile.nsys-rep
```

Only this research branch adds instrumentation/benchmarks; the published seven-PR
correctness stack is unchanged. No performance optimization, model promotion or
change to ongoing forty-five training was made in this research pass.
