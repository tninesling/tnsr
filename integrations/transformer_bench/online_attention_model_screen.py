#!/usr/bin/env python3
"""Untimed numerical screen of native attention PTX inside the reference decoder.

This is not a native-project training benchmark or runtime integration. Other
model operators and Adam are PyTorch. First validate the driver/autograd bridge
bitwise against retained native leaf exports, then use the existing fixed
first-update validator on the complete reference model.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import reference


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest)


class NativePair:
    """CUDA driver bridge for the already qualified S512/D64/H8 PTX ABI."""

    SPECS = (
        ("forward", "online-forward.ptx", ("query", "key", "value", "output_0", "statistics")),
        ("dq", "online-dquery.ptx", ("query", "key", "value", "output_0", "incoming", "statistics", "d_query")),
        ("dkv", "online-dkeyvalue.ptx", ("query", "key", "value", "incoming", "statistics", "d_key_transposed", "d_value")),
    )

    def __init__(self, torch, directory):
        if ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError("the retained kernels require 64-bit pointers")
        self.torch = torch
        torch.cuda.set_device(0)
        torch.empty(1, device="cuda")  # Establish PyTorch's primary context.
        if torch.cuda.get_device_capability() != (8, 9):
            raise RuntimeError("this retained PTX screen is qualified only on SM89")
        self.driver = ct.CDLL("libcuda.so.1")
        signatures = {
            "cuInit": [ct.c_uint],
            "cuModuleLoadData": [ct.POINTER(ct.c_void_p), ct.c_void_p],
            "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            "cuModuleUnload": [ct.c_void_p],
            "cuGetErrorString": [ct.c_int, ct.POINTER(ct.c_char_p)],
            "cuFuncGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p],
            "cuOccupancyMaxActiveBlocksPerMultiprocessor": [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t],
            "cuLaunchKernel": [ct.c_void_p, *([ct.c_uint] * 7), ct.c_void_p,
                               ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.driver, name)
            function.argtypes, function.restype = arguments, ct.c_int
        self.modules, self.functions, self.metadata = [], {}, {}
        self.calls = dict(forward=0, dq=0, dkv=0)
        self.check(self.driver.cuInit(0), "cuInit")
        try:
            for kind, filename, expected in self.SPECS:
                path = directory / filename
                source = path.read_text()
                entries = re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)", source, re.S)
                if len(entries) != 1:
                    raise ValueError(f"expected one entry in {path}")
                name, header = entries[0]
                if tuple(re.findall(r"\.param\s+\.u64\s+(\w+)", header)) != expected:
                    raise ValueError(f"unexpected native pointer ABI: {path}")
                if ".target sm_89" not in source:
                    raise ValueError(f"unexpected native target: {path}")
                buffer = ct.create_string_buffer(source.encode())
                module, function = ct.c_void_p(), ct.c_void_p()
                self.check(self.driver.cuModuleLoadData(ct.byref(module), ct.cast(buffer, ct.c_void_p)), "cuModuleLoadData")
                self.modules.append(module)
                self.check(self.driver.cuModuleGetFunction(ct.byref(function), module, name.encode()), "cuModuleGetFunction")
                attributes = {}
                # Constants verified against installed CUDA cuda.h.
                for label, attribute in [("max_threads", 0), ("shared_bytes", 1), ("local_bytes", 3), ("registers", 4)]:
                    value = ct.c_int()
                    self.check(self.driver.cuFuncGetAttribute(ct.byref(value), attribute, function), "cuFuncGetAttribute")
                    attributes[label] = value.value
                capacity = ct.c_int()
                self.check(self.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(ct.byref(capacity), function, 256, 0), "cuOccupancyMaxActiveBlocksPerMultiprocessor")
                attributes["api_active_blocks"] = capacity.value
                if not (attributes["registers"] <= 128 and attributes["local_bytes"] == 0
                        and attributes["shared_bytes"] == 49152 and attributes["max_threads"] >= 256
                        and capacity.value >= 2):
                    raise RuntimeError(f"native pair resource admission failed: {kind}: {attributes}")
                self.functions[kind] = function
                self.metadata[kind] = dict(source=fingerprint(path), entry=name, parameters=expected,
                                           resources=attributes, capacity_is_not_measured_occupancy=True)
        except BaseException:
            self.close()
            raise
        self.apply = self.make_autograd()

    def check(self, status, operation):
        if status:
            message = ct.c_char_p()
            self.driver.cuGetErrorString(status, ct.byref(message))
            detail = message.value.decode(errors="replace") if message.value else str(status)
            raise RuntimeError(f"{operation}: {detail}")

    def close(self):
        for module in reversed(self.modules):
            self.check(self.driver.cuModuleUnload(module), "cuModuleUnload")
        self.modules.clear()

    def launch(self, kind, batch_heads, tensors):
        stream = self.torch.cuda.current_stream()
        pointers = []
        for tensor in tensors:
            if tensor.device.index != 0 or not tensor.is_contiguous():
                raise ValueError("driver arguments must be contiguous tensors on CUDA0")
            tensor.record_stream(stream)
            pointers.append(ct.c_void_p(tensor.data_ptr()))
        arguments = (ct.c_void_p * len(pointers))(*(ct.addressof(p) for p in pointers))
        self.check(self.driver.cuLaunchKernel(self.functions[kind], 8, batch_heads, 1,
                   32, 8, 1, 0, ct.c_void_p(stream.cuda_stream), arguments, None), "cuLaunchKernel")
        self.calls[kind] += 1

    def make_autograd(self):
        torch, native = self.torch, self

        class Attention(torch.autograd.Function):
            @staticmethod
            def forward(ctx, query, key, value):
                shape = tuple(query.shape)
                if (len(shape) != 4 or shape[1:] != (8, 512, 64) or not 8 <= shape[0] <= 8191
                        or tuple(key.shape) != shape or tuple(value.shape) != shape):
                    raise ValueError("native screen requires B>=8,H8,S512,D64")
                if any(t.dtype != torch.bfloat16 or not t.is_cuda or t.device.index != 0
                       for t in (query, key, value)):
                    raise ValueError("native screen Q/K/V must be CUDA0 BF16")
                # These explicit packs preserve values; their cost is not benchmarked.
                q, k, v = [t.transpose(1, 2).contiguous() for t in (query, key, value)]
                output = torch.empty(shape, dtype=torch.bfloat16, device=query.device)
                statistics = torch.empty((shape[0] * 8, 512, 3), dtype=torch.float32, device=query.device)
                native.launch("forward", shape[0] * 8, (q, k, v, output, statistics))
                ctx.save_for_backward(q, k, v, output, statistics)
                return output

            @staticmethod
            def backward(ctx, incoming):
                q, k, v, output, statistics = ctx.saved_tensors
                if incoming.dtype != torch.bfloat16:
                    raise ValueError("native screen incoming gradient must be BF16")
                incoming = incoming.transpose(1, 2).contiguous()
                batch, heads, sequence, depth = output.shape
                dq, dv = torch.empty_like(output), torch.empty_like(output)
                dkt = torch.empty((batch, heads, depth, sequence), dtype=output.dtype, device=output.device)
                native.launch("dq", batch * heads, (q, k, v, output, incoming, statistics, dq))
                native.launch("dkv", batch * heads, (q, k, v, incoming, statistics, dkt, dv))
                gradients = (dq, dkt.transpose(-2, -1), dv)
                return tuple(g if needed else None for g, needed in zip(gradients, ctx.needs_input_grad))

        return Attention.apply

    def attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                  scale=None, enable_gqa=False):
        if attn_mask is not None or dropout_p != 0 or not is_causal or enable_gqa or scale not in (None, 0.125):
            raise ValueError("unsupported SDPA policy in native screen")
        return self.apply(query, key, value)


def configure(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA is required")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.set_float32_matmul_precision("highest")


def leaf_check(torch, np, args, native):
    comparisons = []
    for manifest_path in sorted(args.exports.glob("revision-*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        shape = tuple(manifest["shape"])
        records = {r["name"]: r for r in manifest["tensors"]}

        def load(name):
            record = records[name]
            raw = np.fromfile(manifest_path.parent / record["file"], dtype="<u2")
            if raw.size != record["count"]:
                raise ValueError("leaf extent mismatch")
            return torch.from_numpy(raw.reshape(shape)).view(torch.bfloat16).cuda()

        query, key, value, incoming = [load(n) for n in ("query", "key", "value", "incoming")]
        for tensor in (query, key, value):
            tensor.requires_grad_(True)
        output = native.apply(query, key, value)
        gradients = torch.autograd.grad(output, (query, key, value), grad_outputs=incoming)
        torch.cuda.synchronize()
        checks = {}
        for name, actual in zip(("output", "d_query", "d_key", "d_value"), (output, *gradients)):
            expected = load(name)
            checks[name] = bool(torch.equal(actual.detach().contiguous().view(torch.int16),
                                            expected.contiguous().view(torch.int16)))
        comparisons.append(dict(revision=manifest["revision"], full_bf16_bits_equal=checks,
                                input_manifest=fingerprint(manifest_path)))
        if not all(checks.values()):
            raise ValueError(f"driver/autograd bridge differs from native leaf export: {checks}")
    if len(comparisons) != 5:
        raise ValueError("expected all five retained leaf fixtures")
    return dict(scope="driver bridge correctness only; no model qualification/performance", comparisons=comparisons)


def update(torch, np, args, native):
    manifest, config, arrays = reference.load_archive(args.archive)
    if (config["heads"], config["sequence"], config["width"], config["batch"]) not in {
            (8, 512, 512, 8), (8, 512, 512, 16), (8, 512, 512, 32)}:
        raise ValueError("screen is restricted to the established three high-BH model cases")
    model = reference.build_model(torch, config, arrays, "sdpa")
    del arrays
    parameters = list(model.masters)
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0, fused=True)
    flat = torch.arange(config["batch"] * config["sequence"], dtype=torch.int64)
    tokens = ((17 * flat + 3) % config["vocab"]).reshape(config["batch"], config["sequence"]).cuda()
    targets = ((17 * (flat + 1) + 3) % config["vocab"]).reshape(config["batch"], config["sequence"]).cuda()
    original = torch.nn.functional.scaled_dot_product_attention
    try:
        if native is not None:
            torch.nn.functional.scaled_dot_product_attention = native.attention
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config["vocab"]), targets.reshape(-1))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"],
                                              error_if_nonfinite=True, foreach=True)
        metrics = dict(loss=float(loss.item()), gradient_norm_before_clip=float(norm.item()))
        metrics["gradient_scale"] = float(np.minimum(np.float32(1), np.float32(config["max_grad_norm"]) /
                                np.float32(np.float32(metrics["gradient_norm_before_clip"]) + np.float32(1e-6))))
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("nonfinite screen metrics")
        optimizer.step()
        torch.cuda.synchronize()
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
    if native is not None and native.calls != dict(forward=config["layers"], dq=config["layers"], dkv=config["layers"]):
        raise ValueError(f"incomplete model pair activation: {native.calls}")
    dump_args = SimpleNamespace(output=args.output, attention="sdpa", optimizer="fused", compile=False)
    reference.dump_qualification(torch, np, dump_args, manifest, model, metrics)
    metadata_path = args.output / "qualification.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(screen_only=True, attention="native-PTX-pair-in-reference-model" if native else "reference-eager-SDPA")
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return dict(scope="first-update numerical screen only; other operators and Adam are PyTorch, not tnsr",
                native_project_qualification=False, performance_claim=False, config=config,
                archive=fingerprint(args.archive / "manifest.json"), fixture=fingerprint(args.archive / manifest["data_file"]),
                metrics=metrics, memory=reference.memory(torch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("leaf-check", "update"), required=True)
    parser.add_argument("--mode", choices=("native", "torch"), default="native")
    parser.add_argument("--ptx-dir", type=Path)
    parser.add_argument("--exports", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "native" and args.ptx_dir is None:
        parser.error("native mode requires --ptx-dir")
    if args.action == "leaf-check" and (args.exports is None or args.mode != "native"):
        parser.error("leaf-check requires --exports and native mode")
    if args.action == "update" and args.archive is None:
        parser.error("update requires --archive")
    import numpy as np
    import torch
    configure(torch)
    args.output.mkdir(parents=True, exist_ok=False)
    native = NativePair(torch, args.ptx_dir) if args.mode == "native" else None
    try:
        result = leaf_check(torch, np, args, native) if args.action == "leaf-check" else update(torch, np, args, native)
        result.update(script=fingerprint(Path(__file__)), reference_source=fingerprint(Path(reference.__file__)),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      device=torch.cuda.get_device_name(), performance_claim=False, native_project_qualification=False,
                      precision_flags=dict(tf32=False, bf16_reduced_precision=False, math_sdpa_reduced_precision=False), native_kernels=native.metadata if native else None,
                      native_calls=native.calls if native else None)
        (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False))
    finally:
        torch.cuda.synchronize()
        if native is not None:
            native.close()


if __name__ == "__main__":
    main()
