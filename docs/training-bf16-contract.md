# Original forty-five bf16 training contract

Issue [94](https://github.com/tninesling/tnsr/issues/94) needs heterogeneous value
storage and explicit rounding boundaries. tnsr's existing homogeneous BF16
executor is useful infrastructure; converting the entire f32 model graph would
change indices, reductions and optimizer state incorrectly.

`integrations/forty_five/audit_precision.py` runs the original, unmodified eager
CUDA `ppo_update` with metadata-only module hooks and `TorchDispatchMode`.
Both actual shared/separate models were audited on RTX 4080 with PyTorch
2.11.0+cu130: eight transitions split into full/partial microbatches 3/3/2, one
untimed update each. Source/checkpoint hashes, every module boundary, operator
dtype flows and observed matmul shapes are in
[precision-audit.json](benchmarks/training-parity/precision-audit.json).
Full reports remain under ignored `target/forty_five_precision_audit/`.
This establishes observed dtype behavior, rather than numerical or learning parity.

| Observed boundary | Original CUDA dtype |
| --- | --- |
| Token/type/phase/action indices | int64; token IDs include 365 |
| Embedding weights and outputs | f32 |
| Linear inputs at the module boundary | f32 or bf16 |
| Linear's `addmm` operands, bias and output | bf16 |
| Attention `bmm`, including backward matmuls | bf16 inputs/output |
| LayerNorm operator inputs, outputs and statistics | f32 |
| Softmax and its backward operator | f32 |
| SiLU and its backward operator | bf16 inputs/output |
| Forward pooling/card score sums | bf16 input -> f32 output |
| Some backward/bias sums | bf16 input -> bf16 output |
| Residual add with f32 and bf16 inputs | f32 output |
| Final model output | f32 policy logits, bf16 values |
| Parameters and accumulated parameter gradients | f32: all 140 / 229 parameters |
| Adam state tensors, including step scalars | f32: 420 / 687 tensors |

The installed operator behavior agrees with PyTorch's documented
[operation-specific autocast policy](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/amp.md).
Use the recorded version and source hashes when reproducing; an operator list
from another release is insufficient to establish this model's boundaries.

## Implications for implementation

Keep exact index values in a separate integer representation through embedding,
indexed losses and replay. Checked host int64 -> u32 conversion can cover this
model's validated vocabulary without passing through a floating representation.
In the observed fixture, converting token 365 to bf16 would change it to 364.
Integer counts explicitly converted for arithmetic, such as discard-subset
normalization, are different values from lookup indices.

Introduce explicit f32/bf16 casts and per-value storage/compute metadata. Preserve
casts in fusion SSA and compilation signatures, use physically typed buffers and
dtype-specific pools, and retain them through graph replay and readback. Reusing
f32 workspaces containing rounded values can test arithmetic in isolation, but
does not establish bf16 storage or memory capacity savings.

Linear requires its input, weight and bias cast to bf16, f32 accumulation, and a
bf16 result at the final linear boundary. A separately rounded GEMM followed by
a separately rounded bias add can differ from the original `addmm`. Forward
autocast and backward casts must retain their own boundaries: gradient
contributions and shared-producer sums can round in bf16 before casting back to
an f32 master parameter. A globally straight-through cast or globally f32
backward graph does not reproduce that behavior.

The primitive graph also decomposes operations which the original executes as
single operators. SiLU's internal exp/divide should compute in f32 with rounding
at the appropriate output boundary, rather than rounding every decomposition
step. LayerNorm and softmax stay in f32. Reduction output storage depends on its
role; the trace shows both f32 forward sums and bf16 backward sums. Attention
mask limits must match the chosen dtype's finite minimum, while pooling masks
retain their original infinity behavior.

Discard-card normalization explicitly converts counts to bf16, computes a bf16
sqrt and divides bf16 subset activations. The subsequent concatenation includes
f32 embedding/query features and promotes to f32. The current f32 adapter's
precomputed reciprocal constants need auditing at this boundary. Final value
outputs cast back through the f32 PPO loss; policy logits are already f32 in
these two configurations.

Implement storage/casts and exact indices before enabling an adapter policy.
Keep f32 master weights, accumulation, clipping and Adam, along with the strict
reference. Calibrate every named gradient and multistep optimizer state against
the original bf16 trainer on both critics, full/partial batches, nonzero anchor
KL, clipping/underflow/nonfinite cases, replay and saved archive restoration.
Then measure actual tensor-core schedules, warm throughput and memory/batch
capacity. Multi-seed learning and seat-balanced evaluation remain separate gates
before quality or model-promotion claims. The audit alone does not close #94.

## Reproduction

```sh
/path/to/forty-five/.venv/bin/python integrations/forty_five/audit_precision.py \
  MODEL_SOURCE --forty-five-root /path/to/forty-five \
  --output FRESH_REPORT.json
```

Run on both the shared ONNX model and separate-critic checkpoint, separately
from timing/profiling. Only this repository's report path is written; the
original source and checkpoints are read-only inputs.
