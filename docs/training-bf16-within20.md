# BF16 PPO within 20% of PyTorch

The final worker meets the latency target in two fresh five-mode comparisons:
**30.84 versus 30.87 ms with original Adam,
and 30.75 versus 28.17 ms with PyTorch fused Adam**.
The latter comparison has 9.2% native latency overhead.
These results apply to the width-256 shared critic, 64 real-game-derived
transitions and microbatch 64 on the RTX 4080. Mixed BF16 remains opt-in.

[Raw evidence](benchmarks/training-parity/bf16-within20.json) retains every sample,
failed intermediate comparison, numerical control, model trace summary, compiled
kernel resources, source/binary hashes, test logs and numerical driver sources.
The reference model and original updater are unchanged.

## Complete-update comparisons

Both runs start each mode from the same archive and batch. Three warmups are
excluded; 64 synchronized complete updates per mode alternate forward/reverse
mode order. Timings include feeds, loss, backward, clipping, Adam and ordinary
metrics; native timings additionally include the JSON worker request.
PyTorch uses its original updater and `torch.compile(model, dynamic=True)` for
the compiled mode. The second run enables fused Adam for every PyTorch mode;
the native settings are identical between runs.

| Complete update | Original PyTorch Adam | PyTorch fused Adam |
| --- | ---: | ---: |
| Native f32, tree/replay/TF32x3 | 42.63 ms | 42.57 ms |
| Native mixed BF16 | 30.84 ms | 30.75 ms |
| PyTorch eager strict f32 | 33.56 ms | 31.74 ms |
| PyTorch eager BF16 | 41.92 ms | 38.61 ms |
| PyTorch compiled BF16 | 30.87 ms | 28.17 ms |

The acceptance calculation is native median latency divided by the lowest
PyTorch mode median in that same run, with a limit of 1.2×. A 10,000-resample
paired circular block bootstrap (block length four, seed zero) describes timing
uncertainty:

| Comparison | Median latency ratio | 95% percentile interval |
| --- | ---: | ---: |
| Original Adam | 0.9989× | 0.9873–1.0214× |
| Fused Adam | 1.0918× | 1.0651–1.1868× |

All native retained samples use one active CUDA replay shape with zero fallback.
No unrelated CUDA compute process was observed. Desktop graphics and unrelated
host work remained running. The target describes these workload/mode medians;
it is not a claim about every matrix size, PyTorch setting or timing sample.

An earlier unchanged-worker matrix appeared to meet the target at 18.0% overhead,
but the next pointwise-only matrix measured 30.6%. Repacked GEMMs then met the
original-Adam comparison at 2.0% but missed fused Adam at 21.3%. Those results
were retained, and tuning continued through the final copy-launch change.

## Accepted changes and native controls

Independent pointwise kernels use 256-thread blocks instead of 1,024, allowing
more resident blocks when register use limits occupancy. Both fixed-binding
programs and ordinary scalar fusion regions share the launch helper. Online
reductions and embedding scatter retain their dedicated launch rules.

Long BF16 split-K GEMMs that previously mixed aligned asynchronous copies with
scalar operand loads now materialize the unaligned operand once into a
K-contiguous physical layout. The RHS uses a transposed view to preserve its
logical shape. Aligned operands then feed 64×64×32 tiles with two stages and
four accumulator fragments per computing warp, subject to the existing resource
budget. The smaller schedule remains the fallback when the wider pipeline does
not fit. Four f32 split-K partials still sum in the original order before a
single BF16 output store. The generated-code signature is version 34.

Layout materialization also uses 256-thread blocks. Float operations in the
pointwise kernels are unchanged. Temporary layout buffers participate in the
existing lifetime allocator: all 24 new allocations are reused. Retained tensor
payload remains 335,696,318 BF16 bytes plus 1,213,836,672 f32 bytes and 33,664
exact-index bytes, excluding modules, optimizer and driver overhead.

Each native ablation starts from the same archive and fixed batch, excludes
three warmups and retains 64 alternating AB/BA pairs. These are separate runs;
the absolute medians should not be combined across rows.

| Change | Before → after | Median paired throughput | Wins |
| --- | ---: | ---: | ---: |
| 256-thread pointwise blocks | 35.35 → 33.56 ms | 1.0545× | 64/64 |
| Aligned split-K operands | 32.90 → 30.40 ms | 1.0816× | 64/64 |
| 256-thread layout-copy blocks | 30.69 → 29.69 ms | 1.0281× | 56/64 |

## Model attribution

Fresh baseline/final traces verify ordered captured-node identities across 32
launches per node, excluding three warmups. The final graph has 1,438 model
kernels versus 1,414: the additional 24 are layout copies. Model span falls from
31.25 to 25.66 ms. These spans exclude optimizer,
uploads and request/metrics handling.

| Model group | Previous accepted worker | Final worker |
| --- | ---: | ---: |
| F32 pointwise | 8.51 ms | 6.90 ms |
| BF16 pointwise | 1.54 ms | 1.01 ms |
| Cast and layout copies | 1.43 ms | 1.94 ms |
| BF16 split-K GEMM | 5.33 ms | 1.35 ms |
| Other BF16 GEMM | 8.72 ms | 8.75 ms |
| Reductions | 5.51 ms | 5.51 ms |

Repacked GEMMs use 105 registers per thread, 256 threads and 35,840 static shared
bytes, with zero local-memory allocation in the recorded launches. Their copy
cost is included in both the cast/copy group and complete-update timings.

## Numerical checks and validation

Six complete backward/update passes hold both learning rates and the scheduler
floor at zero. Every saved master weight remains byte-identical to the initial
archive. Global gradient relative L2 against an independent fixed-weight pass is
1.577e-07–1.708e-07; baseline-repeat variation is
1.368e-07–1.721e-07. Full named parameter errors are retained.
These controls isolate the scheduling/layout change; they do not establish
multi-update training or learning parity. Earlier trajectory sensitivities remain
in [the reduction notes](training-bf16-packed-reductions.md).

CPU tests: 206 passed. Final CUDA tests: 419 passed, one opt-in schedule sweep
skipped. Existing ragged split-K tests cover both operand orientations, changed
feeds and replay. CPU/CUDA checks, all-target/all-feature Clippy, rustfmt and
four documentation tests pass (one documentation example ignored).

Reproduce a comparison with `integrations/forty_five/benchmark_training.py`,
archive `target/forty_five_validation/final-driver/initial`, the original
forty-five checkout and the release CUDA worker. Use `--transitions 64
--microbatch 64 --warmups 3 --repeats 64 --mixed-bf16 --tree-reductions
--tf32x3 --cuda-graphs --compile`; add `--fused-adam` for the second comparison.
The retained reports record all baseline-worker flags and exact hashes.
