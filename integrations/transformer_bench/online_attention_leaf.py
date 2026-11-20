#!/usr/bin/env python3
"""Untimed SDPA forward/VJP diagnostics for exported paired-online leaf data.

Independent of the full-model runner/reference/validator. No fitted thresholds,
benchmarking, optimizer, loss, or numerical acceptance gate is introduced here.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def classes(np, values):
    return dict(finite=int(np.isfinite(values).sum()), nan=int(np.isnan(values).sum()),
                positive_infinity=int(np.isposinf(values).sum()),
                negative_infinity=int(np.isneginf(values).sum()))


def metrics(np, reference, actual):
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    result = dict(finite=finite, reference_classes=classes(np, reference),
                  native_classes=classes(np, actual), threshold_applied=False)
    if finite:
        ref = reference.astype(np.float64, copy=False)
        delta = actual.astype(np.float64, copy=False) - ref
        squared = float(np.square(delta).sum())
        reference_squared = float(np.square(ref).sum())
        result.update(error_squared=squared, reference_squared=reference_squared,
                      reference_l2=reference_squared ** 0.5,
                      relative_l2=squared ** 0.5 / max(reference_squared ** 0.5, 1e-12),
                      max_absolute=float(np.abs(delta).max(initial=0)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-provenance", type=Path, required=True,
                        help="matching retained native compiled-source/binary provenance JSON")
    parser.add_argument("--backend", choices=["auto", "flash", "cudnn", "efficient", "math"],
                        default="auto", help="auto keeps installed PyTorch's optimized dispatcher")
    args = parser.parse_args()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA device required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    args.output.mkdir(parents=True, exist_ok=False)
    backends = dict(flash=SDPBackend.FLASH_ATTENTION, cudnn=SDPBackend.CUDNN_ATTENTION,
                    efficient=SDPBackend.EFFICIENT_ATTENTION, math=SDPBackend.MATH)
    report = dict(version=1, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(), backend_requested=args.backend,
                  native_provenance=dict(path=str(args.native_provenance),
                                         fingerprint=fingerprint(args.native_provenance),
                                         record=json.loads(args.native_provenance.read_text())),
                  script=fingerprint(Path(__file__)), comparisons=[], performance_claim=False,
                  full_model_qualification=False, threshold_applied=False,
                  policy_difference="Native uses online unnormalized BF16 exponentials, O16 delta and div.approx.f32; SDPA uses its selected native forward/VJP policy. Mathematical attention is shared; intermediate rounding is not asserted identical.",
                  existing_counterevidence="F64 monotonic near-cancellation fixture dQ relative-L2 .279/.284; retained unchanged in native qualification-current.json",
                  flags=dict(tf32=False, bf16_reduced_precision=False,
                             fp16_reduced_precision=False, math_sdpa_reduced_precision=False))
    for manifest_path in sorted(args.exports.glob("revision-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        directory = manifest_path.parent
        shape = tuple(manifest["shape"])
        records = {record["name"]: record for record in manifest["tensors"]}
        files = {"manifest.json": fingerprint(manifest_path)}

        def load(name):
            record = records[name]
            path = directory / record["file"]
            if path.stat().st_size != record["count"] * 2:
                raise ValueError(f"incorrect BF16 payload length: {path}")
            files[record["file"]] = fingerprint(path)
            raw = np.fromfile(path, dtype="<u2").reshape(shape)
            return torch.from_numpy(raw).view(torch.bfloat16).cuda()

        q, k, v, incoming = [load(name) for name in ["query", "key", "value", "incoming"]]
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, True, False)
        eligible = dict(flash=torch.backends.cuda.can_use_flash_attention(params),
                        efficient=torch.backends.cuda.can_use_efficient_attention(params),
                        cudnn=torch.backends.cuda.can_use_cudnn_attention(params),
                        flash_built=torch.backends.cuda.is_flash_attention_available())
        selection = contextlib.nullcontext() if args.backend == "auto" else sdpa_kernel(backends[args.backend])
        with selection, torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True,
                                                    scale=0.125)
            gradients = torch.autograd.grad(output, (q, k, v), grad_outputs=incoming)
            torch.cuda.synchronize()
        operators = sorted({event.key for event in profile.key_averages()
                            if "scaled_dot_product" in event.key or "flash_attention" in event.key})
        comparison = dict(revision=manifest["revision"], shape=shape,
                          query_stride=list(q.stride()), eligible=eligible,
                          actual_forward_backward_operators=operators, tensors={}, files=files,
                          near_zero_scope="reference |gradient| <=1e-6, fixed descriptive subset; no gate")
        torch_directory = args.output / f"revision-{manifest['revision']}"
        torch_directory.mkdir(exist_ok=True)
        error_squared = reference_squared = 0.0
        concat_finite = True
        for name, tensor in zip(["output", "d_query", "d_key", "d_value"], [output, *gradients]):
            reference = tensor.detach().float().cpu().numpy()
            native = load(name).float().cpu().numpy()
            result = metrics(np, reference, native)
            result["row_zero"] = metrics(np, reference[:, :, 0, :], native[:, :, 0, :])
            if name != "output":
                near = np.isfinite(reference) & (np.abs(reference) <= 1e-6)
                result["near_zero"] = dict(count=int(near.sum()),
                                           metrics=metrics(np, reference[near], native[near]))
                concat_finite &= result["finite"]
                error_squared += result.get("error_squared", 0)
                reference_squared += result.get("reference_squared", 0)
            comparison["tensors"][name] = result
            # BF16 raw outputs retain the actual SDPA boundaries, not an F32 approximation.
            raw = tensor.detach().contiguous().view(torch.uint16).cpu().numpy().astype("<u2", copy=False)
            raw.tofile(torch_directory / f"{name}.bf16")
        comparison["concatenated_gradients"] = dict(
            finite=concat_finite, threshold_applied=False,
            relative_l2=error_squared ** 0.5 / max(reference_squared ** 0.5, 1e-12)
            if concat_finite else None)
        report["comparisons"].append(comparison)
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        del q, k, v, incoming, output, gradients
    if not report["comparisons"]:
        raise ValueError("no complete exported revision manifests found")


if __name__ == "__main__":
    main()
