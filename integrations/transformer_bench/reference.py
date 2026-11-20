"""Portable decoder fixture and synchronized PyTorch mixed-BF16 training reference.

Generate: reference.py generate --config config.json --archive fixture
Run: reference.py run --archive fixture --output run --attention matched --optimizer fused
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_config(raw):
    config = dict(raw)
    defaults = {"epsilon": 1e-5, "learning_rate": 1e-4, "max_grad_norm": 1.0}
    for name, value in defaults.items():
        config.setdefault(name, value)
    expected = {"width", "layers", "heads", "hidden", "sequence", "batch", "vocab", *defaults}
    if set(config) != expected:
        raise ValueError(f"unexpected/missing config keys: {set(config) ^ expected}")
    for name in expected - defaults.keys():
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["width"] % config["heads"]:
        raise ValueError("width must be divisible by heads")
    for name in defaults:
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"invalid {name}")
    if config["epsilon"] == 0 or config["max_grad_norm"] == 0:
        raise ValueError("epsilon and max_grad_norm must be positive")
    return config


def parameter_shapes(config):
    width, hidden = config["width"], config["hidden"]
    result = {
        "token.weight": [config["vocab"], width],
        "position.weight": [config["sequence"], width],
        "final_norm.weight": [width],
        "final_norm.bias": [width],
    }
    for layer in range(config["layers"]):
        prefix = f"layer.{layer}."
        for name in ["attn_norm", "ffn_norm"]:
            result[prefix + name + ".weight"] = [width]
            result[prefix + name + ".bias"] = [width]
        for name, out_features, in_features in [
            ("qkv", 3 * width, width), ("out", width, width),
            ("gate", hidden, width), ("up", hidden, width), ("down", width, hidden),
        ]:
            result[prefix + name + ".weight"] = [out_features, in_features]
            result[prefix + name + ".bias"] = [out_features]
    return dict(sorted(result.items()))


def generate(args):
    import numpy as np

    config = checked_config(json.loads(args.config.read_text()))
    args.archive.mkdir(parents=True, exist_ok=False)
    generator = np.random.default_rng(17)
    records, offset = [], 0
    with (args.archive / "weights.f32").open("wb") as stream:
        for name, shape in parameter_shapes(config).items():
            if name.endswith(".bias"):
                values = np.zeros(shape, dtype="<f4")
            elif "_norm.weight" in name:
                values = np.ones(shape, dtype="<f4")
            else:
                standard_deviation = 0.02 if name in {"token.weight", "position.weight"} else 1 / math.sqrt(shape[1])
                if name.endswith((".out.weight", ".down.weight")):
                    standard_deviation /= math.sqrt(2 * config["layers"])
                values = generator.normal(0.0, standard_deviation, shape).astype("<f4")
            records.append(dict(name=name, shape=shape, offset=offset, count=values.size, group="train"))
            stream.write(values.tobytes(order="C"))
            offset += values.size
    manifest = dict(version=1, dtype="f32-le", data_file="weights.f32", config=config, tensors=records)
    (args.archive / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(archive=str(args.archive), tensors=len(records), elements=offset,
                          seed=17, config=config), allow_nan=False))


def load_archive(directory):
    import numpy as np

    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != 1 or manifest["dtype"] != "f32-le":
        raise ValueError("unsupported archive version or dtype")
    if Path(manifest["data_file"]).name != manifest["data_file"]:
        raise ValueError("archive data_file must be a filename")
    config = checked_config(manifest["config"])
    # Copy-on-write mapping is writable for torch.from_numpy but never mutated.
    # No whole-model host array/copy or Python float list is constructed.
    values = np.memmap(directory / manifest["data_file"], dtype="<f4", mode="c")
    shapes = parameter_shapes(config)
    records = manifest["tensors"]
    if [record["name"] for record in records] != list(shapes):
        raise ValueError("archive parameters must match the sorted decoder contract")
    offset, arrays = 0, {}
    for record in records:
        name, shape = record["name"], record["shape"]
        count = math.prod(shape)
        if shape != shapes[name] or record["offset"] != offset or record["count"] != count or record["group"] != "train":
            raise ValueError(f"invalid tensor record {name}")
        if offset + count > values.size:
            raise ValueError(f"truncated archive tensor {name}")
        arrays[name] = values[offset:offset + count].reshape(shape)
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"nonfinite archive tensor {name}")
        offset += count
    if offset != values.size:
        raise ValueError("unreferenced archive values")
    return manifest, config, arrays


def build_model(torch, config, arrays, attention):
    import numpy as np

    functional = torch.nn.functional
    attention_scale = float(np.float32(1.0) / np.sqrt(np.float32(config["width"] // config["heads"])))

    class Decoder(torch.nn.Module):
        def __init__(self, initial_arrays):
            super().__init__()
            self.names = list(initial_arrays)
            self.masters = torch.nn.ParameterList([
                torch.nn.Parameter(torch.from_numpy(initial_arrays[name]).to("cuda")) for name in self.names
            ])
            self.named_masters = dict(zip(self.names, self.masters))
            self.register_buffer("positions", torch.arange(config["sequence"], device="cuda"))
            self.register_buffer("causal_mask", torch.triu(torch.full(
                (config["sequence"], config["sequence"]), -math.inf, dtype=torch.float32, device="cuda"), diagonal=1))
            self.register_buffer("head_bias", torch.zeros(config["vocab"], dtype=torch.bfloat16, device="cuda"))
            self.register_buffer("one_float", torch.tensor(1.0, dtype=torch.float32, device="cuda"))

        def norm(self, x, prefix):
            weight, bias = self.named_masters[prefix + ".weight"], self.named_masters[prefix + ".bias"]
            if attention == "sdpa":
                return functional.layer_norm(x, (config["width"],), weight, bias, config["epsilon"])
            centered = x - x.mean(dim=-1, keepdim=True)
            variance = (centered * centered).mean(dim=-1, keepdim=True)
            inv_std = ((variance + config["epsilon"]).log() * -0.5).exp()
            return centered * inv_std * weight + bias

        def linear(self, x, prefix):
            return functional.linear(x.to(torch.bfloat16),
                self.named_masters[prefix + ".weight"].to(torch.bfloat16),
                self.named_masters[prefix + ".bias"].to(torch.bfloat16))

        def forward(self, tokens):
            width, heads, sequence = config["width"], config["heads"], config["sequence"]
            depth = width // heads
            x = functional.embedding(tokens, self.named_masters["token.weight"])
            x = x + functional.embedding(self.positions, self.named_masters["position.weight"])
            for layer in range(config["layers"]):
                prefix = f"layer.{layer}."
                qkv = self.linear(self.norm(x, prefix + "attn_norm"), prefix + "qkv")
                query, key, value = qkv.reshape(config["batch"], sequence, 3, heads, depth).unbind(dim=2)
                query, key, value = [item.permute(0, 2, 1, 3) for item in (query, key, value)]
                if attention == "sdpa":
                    context = functional.scaled_dot_product_attention(query, key, value,
                        dropout_p=0.0, is_causal=True)
                else:
                    scores = torch.matmul(query, key.transpose(-2, -1)).float() * attention_scale
                    scores = scores + self.causal_mask
                    exponentials = (scores - scores.amax(dim=-1, keepdim=True)).exp()
                    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
                    context = torch.matmul(probabilities.to(torch.bfloat16), value)
                context = context.permute(0, 2, 1, 3).reshape(config["batch"], sequence, width)
                x = x + self.linear(context, prefix + "out").float()
                normalized = self.norm(x, prefix + "ffn_norm")
                gate = self.linear(normalized, prefix + "gate").float()
                up = self.linear(normalized, prefix + "up")
                if attention == "sdpa":
                    activated = functional.silu(gate).to(torch.bfloat16)
                else:
                    # Match nn::sigmoid's sign/magnitude/masked primitive graph,
                    # including its backward, rather than substituting sigmoid.
                    negative = ((-gate) > 0.0).to(torch.float32)
                    nonnegative = 1.0 - negative
                    magnitude = gate * (1.0 - negative * 2.0)
                    z = (-magnitude).exp()
                    denominator = z + 1.0
                    sigmoid = torch.where(nonnegative != 0.0, self.one_float / denominator, 0.0)
                    sigmoid = sigmoid + torch.where(negative != 0.0, z / denominator, 0.0)
                    activated = (gate * sigmoid).to(torch.bfloat16)
                product = (activated * up).to(torch.bfloat16)
                x = x + self.linear(product, prefix + "down").float()
            x = self.norm(x, "final_norm")
            return functional.linear(x.to(torch.bfloat16),
                self.named_masters["token.weight"].to(torch.bfloat16), self.head_bias).float()

    return Decoder(arrays)


class PrimitiveAdam:
    """Diagnostic explicit-f32 Adam grouping; not an optimized timing baseline."""
    def __init__(self, torch, parameters, learning_rate):
        self.torch = torch
        self.parameters = list(parameters)
        self.moments = [torch.zeros_like(p) for p in self.parameters]
        self.variances = [torch.zeros_like(p) for p in self.parameters]
        self.learning_rate = learning_rate
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        import numpy as np
        self.steps += 1
        b1, b2 = np.float32(0.9), np.float32(0.999)
        c1, c2 = float(np.float32(1) - np.float32(b1 ** self.steps)), float(np.float32(1) - np.float32(b2 ** self.steps))
        one1, one2 = float(np.float32(1) - b1), float(np.float32(1) - b2)
        with self.torch.no_grad():
            for parameter, moment, variance in zip(self.parameters, self.moments, self.variances):
                gradient = parameter.grad
                moment.copy_(moment * float(b1) + gradient * one1)
                variance.copy_(variance * float(b2) + (gradient * one2) * gradient)
                m_hat, v_hat = moment / c1, variance / c2
                delta = (float(np.float32(self.learning_rate)) * m_hat) / (v_hat.sqrt() + float(np.float32(1e-8)))
                parameter.copy_(parameter - delta)


def memory(torch):
    return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())


def sdpa_backend_report(torch, config, execute):
    batch, sequence, heads = config["batch"], config["sequence"], config["heads"]
    qkv = torch.zeros((batch, sequence, 3, heads, config["width"] // heads),
                      dtype=torch.bfloat16, device="cuda", requires_grad=True)
    query, key, value = [part.permute(0, 2, 1, 3) for part in qkv.unbind(dim=2)]
    cuda = torch.backends.cuda
    params = cuda.SDPAParams(query, key, value, None, 0.0, True, False)
    result = dict(used=execute, flash_built=cuda.is_flash_attention_available(),
                  flash_enabled=cuda.flash_sdp_enabled(), flash_eligible=cuda.can_use_flash_attention(params),
                  efficient_enabled=cuda.mem_efficient_sdp_enabled(), efficient_eligible=cuda.can_use_efficient_attention(params),
                  cudnn_enabled=cuda.cudnn_sdp_enabled(), cudnn_eligible=cuda.can_use_cudnn_attention(params),
                  math_enabled=cuda.math_sdp_enabled(), query_shape=list(query.shape), query_stride=list(query.stride()),
                  actual_training_backend_operators=[])
    if execute:
        # CPU operator names reveal the actually dispatched CUDA backend without
        # requiring CUPTI hardware counters; this single probe is never timed.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            torch.nn.functional.scaled_dot_product_attention(query, key, value,
                dropout_p=0.0, is_causal=True).float().sum().backward()
            torch.cuda.synchronize()
        result["actual_training_backend_operators"] = sorted({item.key for item in profile.key_averages()
            if "scaled_dot_product" in item.key})
    return result


def dump_qualification(torch, np, args, manifest, model, metrics):
    with (args.output / "qual_gradients.f32").open("wb") as gradients, (args.output / "qual_weights.f32").open("wb") as weights:
        for record in manifest["tensors"]:
            parameter = model.named_masters[record["name"]]
            if parameter.grad is None:
                raise ValueError(f"missing gradient {record['name']}")
            for tensor, stream in [(parameter.grad, gradients), (parameter, weights)]:
                values = tensor.detach().cpu().numpy().astype("<f4", copy=False)
                if not np.isfinite(values).all():
                    raise ValueError(f"nonfinite qualification tensor {record['name']}")
                stream.write(values.tobytes(order="C"))
    report = dict(metrics, optimizer_step=1, tensors=manifest["tensors"], config=manifest["config"],
                  gradients_file="qual_gradients.f32", weights_file="qual_weights.f32",
                  attention=args.attention, optimizer=args.optimizer, compiled=args.compile)
    (args.output / "qualification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run(args):
    import numpy as np
    import torch

    setup_started = time.perf_counter()
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats positive")
    if args.compile_mode != "default" and not args.compile:
        raise ValueError("a non-default compile mode requires --compile")
    if args.compile_loss and not args.compile:
        raise ValueError("--compile-loss requires --compile")
    if not torch.cuda.is_available():
        raise RuntimeError("the training reference requires CUDA")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.set_float32_matmul_precision("highest")
    manifest, config, arrays = load_archive(args.archive)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(torch, config, arrays, args.attention)
    del arrays
    compile_settings = dict(enabled=args.compile, backend="inductor", dynamic=False, fullgraph=False,
                            mode=args.compile_mode,
                            includes_cross_entropy=args.compile_loss,
                            cuda_graphs_requested=args.compile and args.compile_mode == "reduce-overhead",
                            cuda_graph_replay_directly_verified=False,
                            scope=("forward model and cross entropy with automatic backward compilation; clipping/optimizer remain outside torch.compile"
                                   if args.compile_loss else "forward model with automatic backward compilation; optimizer remains outside torch.compile"))

    def model_loss(tokens, targets):
        logits = model(tokens)
        return torch.nn.functional.cross_entropy(logits.reshape(-1, config["vocab"]), targets.reshape(-1))

    compiled_callable = model_loss if args.compile_loss else model
    forward_model = torch.compile(compiled_callable, backend="inductor", dynamic=False, fullgraph=False,
                                  mode=args.compile_mode) if args.compile else model
    parameters = list(model.masters)
    if args.optimizer == "primitive":
        optimizer = PrimitiveAdam(torch, parameters, config["learning_rate"])
    else:
        optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"], betas=(0.9, 0.999),
            eps=1e-8, weight_decay=0.0, fused=args.optimizer == "fused", foreach=False if args.optimizer == "torch" else None)
    flat = torch.arange(config["batch"] * config["sequence"], dtype=torch.int64)
    tokens_cpu = ((17 * flat + 3) % config["vocab"]).reshape(config["batch"], config["sequence"])
    targets_cpu = ((17 * (flat + 1) + 3) % config["vocab"]).reshape(config["batch"], config["sequence"])
    backend = sdpa_backend_report(torch, config, args.attention == "sdpa")
    backend["probe_scope"] = "standalone eager SDPA using model query shape/strides, outside qualification and timing"
    backend["compiled_model_backend_directly_profiled"] = False
    torch.cuda.synchronize()
    setup_memory = memory(torch)
    setup_wall_seconds = time.perf_counter() - setup_started

    def step():
        # CPU tokens/targets are fixed, but each complete update includes staging.
        tokens, targets = tokens_cpu.to("cuda"), targets_cpu.to("cuda")
        optimizer.zero_grad(set_to_none=True)
        if args.compile and args.compile_mode == "reduce-overhead":
            # Old gradients are released before starting the next graph-tree iteration.
            torch.compiler.cudagraph_mark_step_begin()
        if args.compile_loss:
            loss = forward_model(tokens, targets)
        else:
            logits = forward_model(tokens)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config["vocab"]), targets.reshape(-1))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"],
                                              error_if_nonfinite=True, foreach=True)
        loss_value, norm_value = float(loss.detach().item()), float(norm.item())
        if not math.isfinite(loss_value) or not math.isfinite(norm_value):
            raise ValueError("nonfinite loss/gradient norm")
        scale = float(np.minimum(np.float32(1), np.float32(config["max_grad_norm"]) /
                                 np.float32(np.float32(norm_value) + np.float32(1e-6))))
        optimizer.step()
        torch.cuda.synchronize()
        return dict(loss=loss_value, gradient_norm_before_clip=norm_value, gradient_scale=scale)

    qualification_started = time.perf_counter()
    qualification = step()
    qualification_wall_seconds = time.perf_counter() - qualification_started
    qualification_memory = memory(torch)
    dump_started = time.perf_counter()
    dump_qualification(torch, np, args, manifest, model, qualification)
    qualification_dump_wall_seconds = time.perf_counter() - dump_started
    warmups = []
    for _ in range(args.warmups):
        warmups.append(step())
    torch.cuda.synchronize()
    lifecycle_before_steady_reset = memory(torch)
    torch.cuda.reset_peak_memory_stats()
    steady_start_memory = memory(torch)
    samples = []
    if args.profile:
        torch.cuda.profiler.start()
    try:
        for index in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            metrics = step()
            elapsed_ms = (time.perf_counter() - started) * 1000
            samples.append(dict(metrics, wall_ms=elapsed_ms, optimizer_step=2 + args.warmups + index))
    finally:
        if args.profile:
            torch.cuda.profiler.stop()
    torch.cuda.synchronize()
    steady_memory = memory(torch)
    latencies = [sample["wall_ms"] for sample in samples]
    report = dict(
        config=config, mode=f"pytorch-{args.attention}-{args.optimizer}" + ("-compiled" if args.compile else "-eager") + ("-reduce-overhead" if args.compile_mode == "reduce-overhead" else ""), attention=args.attention,
        optimizer=args.optimizer, dtype="f32 masters; explicit BF16 linear/attention boundaries",
        compiled=args.compile, compile_settings=compile_settings, profiled=args.profile,
        setup_wall_seconds=setup_wall_seconds,
        qualification_wall_seconds=qualification_wall_seconds,
        qualification_dump_wall_seconds=qualification_dump_wall_seconds,
        startup_timing_note="setup excludes Python imports; first qualification includes lazy torch.compile/AOTAutograd work, initial optimizer state allocation and first execution; setup, qualification and binary dumps are excluded from steady samples",
        seed=17, torch_version=torch.__version__, numpy_version=np.__version__, cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(0), warmups=args.warmups, repeats=args.repeats,
        source_sha256=file_sha256(Path(__file__)),
        archive_sha256=file_sha256(args.archive / manifest["data_file"]),
        flags=dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                   math_sdpa_reduced_precision=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                   float32_matmul_precision=torch.get_float32_matmul_precision(), autocast=False),
        sdpa=backend, qualification=qualification, qualification_file="qualification.json",
        setup_memory=setup_memory, qualification_memory=qualification_memory,
        lifecycle_before_steady_reset=lifecycle_before_steady_reset,
        steady_start_memory=steady_start_memory, steady_memory=steady_memory,
        lifecycle_peak_allocated_bytes=max(lifecycle_before_steady_reset["peak_allocated_bytes"], steady_memory["peak_allocated_bytes"]),
        lifecycle_peak_reserved_bytes=max(lifecycle_before_steady_reset["peak_reserved_bytes"], steady_memory["peak_reserved_bytes"]),
        median_wall_ms=statistics.median(latencies), mean_wall_ms=statistics.mean(latencies),
        tokens_per_second=config["batch"] * config["sequence"] * 1000 / statistics.median(latencies),
        samples=samples, warmup_metrics=warmups,
        optimizer_settings=dict(learning_rate=config["learning_rate"], beta1=0.9, beta2=0.999,
                                epsilon=1e-8, weight_decay=0.0, max_grad_norm=config["max_grad_norm"],
                                clip_epsilon=1e-6, clip_foreach=True),
        protocol="one untimed qualification update with all clipped gradients/postweights saved; then excluded warmups and timed evolving-weight complete synchronized updates on the same fixed batch; timings include CPU token/target staging, forward, loss, backward, clipping, Adam and scalar loss/norm readback; binary dumps and JSON file writes excluded",
        numerical_policy=("explicit primitive f32 LayerNorm/stable sign-magnitude SiLU/softmax with materialized BF16 QK/PV boundaries" if args.attention == "matched" else
                          "optimized reference uses F.layer_norm, F.silu and causal SDPA; internal precision, reduction and attention rounding boundaries differ from matched/native materialized attention"),
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("generate")
    fixture.add_argument("--config", type=Path, required=True)
    fixture.add_argument("--archive", type=Path, required=True)
    fixture.set_defaults(function=generate)
    reference = commands.add_parser("run")
    reference.add_argument("--archive", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--warmups", type=int, default=3)
    reference.add_argument("--repeats", type=int, default=32)
    reference.add_argument("--attention", choices=["matched", "sdpa"], default="matched")
    reference.add_argument("--optimizer", choices=["fused", "torch", "primitive"], default="fused")
    reference.add_argument("--compile", action="store_true", help="compile the forward model with Inductor, dynamic=False; eager optimizer")
    reference.add_argument("--compile-mode", choices=["default", "reduce-overhead"], default="default",
                           help="Inductor mode; reduce-overhead requests CUDA Graphs, whose replay must be verified separately")
    reference.add_argument("--compile-loss", action="store_true",
                           help="include cross entropy in the compiled forward/backward scope; requires --compile")
    reference.add_argument("--profile", action="store_true",
                           help="capture only measured updates with an attached CUDA profiler; diagnostic timings")
    reference.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
