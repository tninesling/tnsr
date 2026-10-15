# Forty-five on tnsr

The native worker reconstructs the learned-position SwiGLU forty-five model from
named f32 parameters. It includes attention padding/all-padding handling, learned
poolers, card-conditioned play/discard heads, and shared or separate legal critics.
RoPE, privileged critics, and legacy non-SwiGLU encoders are rejected explicitly.

Build from the tnsr checkout:

```sh
cargo build --release --example forty_five_worker --features cuda
```

Use the existing forty-five Python environment (PyTorch, NumPy, ONNX, ONNX Runtime,
and the built Rust game extension). Export an existing `.pt` checkpoint, or the
original `models/v8.4.onnx`:

```sh
/path/to/forty-five/.venv/bin/python integrations/forty_five/export.py \
  /path/to/forty-five/models/v8.4.onnx /tmp/tnsr-v84 \
  --forty-five-root /path/to/forty-five
```

Run repeatable inference and checkpoint-interchange checks:

```sh
/path/to/forty-five/.venv/bin/python integrations/forty_five/validate.py \
  /path/to/forty-five/models/v8.4.onnx \
  --forty-five-root /path/to/forty-five \
  --output target/forty_five_validation/inference
```

Output directories must be new. Validation uses actual playable-phase game
observations, additional phase embedding IDs and all-padding input. It compares
logits, values and masked action probabilities against PyTorch, checks recovered
ONNX weights against ONNX Runtime, and restores saved native weights into both
PyTorch and tnsr. Reports record source SHA256 and tolerances in the script.
The committed tiny fixture uses the same architecture with width 8, one layer,
16 tokens and seed 17; its CPU/PTX tests run without the sibling project.

The persistent JSON-lines worker accepts `infer` with `batch` containing
`token_ids`, `token_type_ids`, `phase_ids`, and `save` with a fresh `directory`.
Run `target/release/examples/forty_five_worker ARCHIVE cpu|cuda`. CUDA is selected
explicitly with strict-f32 matmuls; a GPU initialization failure is an error.
Up to eight batch shapes retain separate compiled executors. Graphs share the
parameter registry, so updating weights preserves parameter identity and caches.
`bridge.Worker` manages this protocol, and `bridge.Policy` supplies a callable
model adapter to the existing Python/Rust rollout collector.

The model worker uses a 64 MiB thread stack for deep symbolic graph construction.
Weights keep PyTorch's `[out, in]` layout; graph transposes make them usable by
`tnsr::nn::linear`. Saved archives contain weights/model config, not implicit
optimizer or RNG resumption state. `export.load_archive` restores them into the
original PyTorch architecture for the existing serving/export pipeline.

## PPO parity

The `train` request takes a `ppo::Batch`, `settings` and optional
`return_gradients`. Advantages must already be normalized **over the whole
rollout**. One request performs one hand minibatch: it accumulates microbatch
losses/gradients by transition count, clips their global norm, then applies Adam
with policy/value parameter groups. Bidding KL uses frozen legal-action anchor
probabilities/log-probabilities and preserves the original per-microbatch bidding
normalization. Runtime coefficients keep compiled graphs reusable. `schedule`
with `update` advances the cosine learning rates once per completed rollout.

```sh
/path/to/forty-five/.venv/bin/python integrations/forty_five/validate_ppo.py \
  /path/to/forty-five/models/v8.4.onnx \
  --forty-five-root /path/to/forty-five \
  --output target/forty_five_validation/ppo
```

The oracle is the existing `forty_five_ai.train.ppo_update` in CPU f32, including
weighted microbatches, clipping, entropy, bidding-anchor KL, two Adam updates and
scheduler steps. Every named clipped gradient and updated parameter is checked;
reports retain per-parameter errors. The fixed batch exercises both signs of
advantages, policy/value clipping, every phase and all-padding. Small committed
shared/separate-critic golden fixtures run in CPU/PTX tests. Regenerate with
`generate_ppo_fixture.py --forty-five-root ROOT` (and `--separate` into a fresh
`tiny-separate` directory).

This worker trains in f32 on either backend. Gradient accumulation, norm clipping
and Adam currently run on the host; CUDA gradients are copied back each
microbatch, and updated parameters are uploaded on the next execution. Training
is synchronous. Saved archives restore model weights for inference/export or a
fresh optimizer; exact optimizer/RNG resumption and bf16 autocast are not provided.
