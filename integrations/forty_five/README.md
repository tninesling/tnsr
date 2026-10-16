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

For larger contractions, add `--fixture-repeats 8 --microbatch 64` to the PPO
validator. The actual-model fixture then contains 64 transitions and uses the
same CPU oracle and tolerances. The default remains eight transitions in weighted
3/3/2 microbatches. With both TF32x3 and tree reductions enabled on CUDA, the
adapter partitions long rank-two gradient GEMMs into four K slices and sums their
partial outputs deterministically. This reassociates the contraction, so both
small and large fixtures are numerical gates. The response reports the number of
`split_k_contractions` in the compiled shape cache.

With `TNSR_PROFILE_PPO=1`, `cuda_compile` reports each cached module's logical
kernel steps, unique PTX entries, source bytes, PTX compilation time and module
load time. Module compilation occurs during executor warmup; it is separate from
the earlier model/autograd `compile_cache_seconds` bucket.

This worker trains in f32 on either backend. CUDA uses resident parameters,
gradient accumulation, clipping and Adam; the CPU backend uses host Adam.
Strict-order model reductions remain the default. `--tree-reductions` opts the
CUDA training/validation/benchmark scripts into fixed parallel f32 trees;
the JSON equivalent is `settings.tree_reductions=true`. The policy stays fixed
for the lifetime of a trainer. Matmul defaults to strict f32. `--tf32x3` (or
`settings.tf32x3=true`) opts CUDA training into corrected TF32 tensor-core
matmuls, with f32 storage, gradient accumulation and optimizer state. It combines
three products to retain more operand precision than ordinary TF32, while
changing accumulation order. Responses and reports identify the arithmetic
policy. See [measurements and validation](../../docs/training-tf32x3-performance.md).
The planner also avoids expanded view buffers for standalone pointwise consumers;
see [the matched follow-up measurements](../../docs/training-virtual-inputs-performance.md).
Shared dense projections also flatten leading activation rows to avoid per-example
weight-gradient matrices; see [validation and measurements](../../docs/training-dense-contractions-performance.md).
Training is synchronous.
`--cuda-graphs` (or `settings.cuda_graphs=true`) separately opts into stable-buffer
model replay. Responses report active shapes, setup and fallback reasons;
accumulation, clipping/status and Adam remain outside capture. See the
[replay contract](../../docs/resident-replay.md) and
[paired measurements](../../docs/training-replay-performance.md).
Saved archives restore model weights for inference/export or a
fresh optimizer; exact optimizer/RNG resumption and bf16 autocast are not provided.

## Real rollout, training, evaluation and serving handoff

```sh
/path/to/forty-five/.venv/bin/python integrations/forty_five/train.py \
  /path/to/forty-five/models/v8.4.onnx \
  --forty-five-root /path/to/forty-five --backend cuda \
  --output target/forty_five_validation/training \
  --updates 2 --num-envs 2 --hands-per-update 2 --microbatch 8 \
  --compare-rollout --eval-games 2 --export-onnx
```

This synchronous driver calls the original Rust `collect_trajectory` through
`collect_rollout_rust`, retaining game/reward rules, legal categorical sampling,
GAE and hand grouping. The optional rollout comparison uses the same seed and
checks observations, chosen actions, values, log-probabilities, advantages and
returns against the original model. Each update saves a fresh named archive and
checks predictions against a PyTorch restore and a fresh native worker.
Evaluation uses the existing greedy arena with both opposite seat pairs and
compares native and restored-PyTorch outcomes on the same deals. Two games are a
compatibility smoke test, not evidence of improved playing strength.

`--fixed-opponent-prob` retains the original collector's team assignment/mixing.
`--opponent-snapshot` accepts a frozen `.pt`, `.onnx` or native archive (defaults to
the initialization). The bidding anchor always remains the frozen initialization.
Every update gets a separately selectable checkpoint; restored models can enter
forty-five's existing `SelfPlayManager` snapshot pool and evaluation/selection
flow. This smoke driver writes into its new output directory and does not invoke
`maybe_promote_best`, replace serving weights, or resume/interrupt other runs.
Use the existing project's promotion gates for any candidate selected afterward.

The final weights are also written as an original-format PyTorch checkpoint.
`--export-onnx` calls forty-five's existing serving exporter, checks the ONNX graph,
and compares ONNX Runtime against restored PyTorch over the phase/padding probes.
For a separate handoff:

```sh
/path/to/forty-five/.venv/bin/python integrations/forty_five/restore.py \
  target/forty_five_validation/training/update-0002 /tmp/trained.pt \
  --forty-five-root /path/to/forty-five --onnx /tmp/trained.onnx
```

Reports separate rollout and PPO elapsed time (including graph compilation and
transfers), retain compiled-shape counts, and count JSON request/response bytes.
CUDA training payload lower bounds count the initial resident parameter upload;
normal updates do not download named gradients. These bounds exclude observation
transfers, constants, inference, checkpoint exports and explicit diagnostics.
Use Nsight traces for actual transfer counts. Stable buffers/replay and mixed
precision are the next performance work; matching the original PyTorch bf16
production throughput has not been established.

For a longer native comparison, first generate `batch.json` with
`benchmark_training.py`, then alternate two worker binaries directly:

```sh
TNSR_PROFILE_PPO=1 /path/to/forty-five/.venv/bin/python \
  integrations/forty_five/benchmark_native_pair.py /path/to/archive \
  --batch /path/to/batch.json --worker /path/to/candidate-worker \
  --baseline-worker /path/to/baseline-worker --output target/native-pairs \
  --microbatch 64 --repeats 500 --tf32x3 --tree-reductions --cuda-graphs
```

This retains the two excluded warmups, every full-request time, executable hashes
and replay state. Success requires finite metrics and active replay without
fallbacks when requested. Both workers receive identical settings and start from
the same archive; no PyTorch build runs during this native comparison.

For physical-BF16 GEMM probes with conversion costs and streaming working sets,
see [the measured BF16 headroom report](../../docs/training-bf16-headroom.md).
The separately patched whole-model arithmetic experiment in that report is not
a production AMP policy. `--bf16-compute-probe` requires its distinct candidate
precision label and retains ordinary baseline checks.

Set `TNSR_PROFILE_PPO_DUMP` to a fresh directory alongside `TNSR_PROFILE_PPO=1`
to save each compiled bucket's actual plan/PTX once. Existing files are rejected.
