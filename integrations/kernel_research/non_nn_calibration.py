#!/usr/bin/env python3
"""Separate Torch controls for actual NT fused-linear/TN F32-partial native leaves.

No code here is used by native execution. First qualify without --timing; a
later explicit parent reservation is required for timing. Mathematical F64
sample errors/near-zero bands are diagnostics, never fitted fullmodel gates.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

CHUNK = 262144

def load_json(path):
    return json.loads(Path(path).read_text())

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def mapped(path, dtype, shape):
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if Path(path).stat().st_size != expected:
        raise ValueError(f"wrong file extent: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)

def widened(data, bf16):
    if bf16:
        return (np.asarray(data, dtype=np.uint32) << 16).view(np.float32)
    return np.asarray(data, dtype=np.float32)

def metrics(candidate, reference, bf16=True):
    if candidate.shape != reference.shape:
        raise ValueError("shape mismatch")
    a, b = candidate.reshape(-1), reference.reshape(-1)
    err2 = ref2 = maxabs = 0.0
    nonfinite = bit_equal = 0
    near = {str(v): {"reference_elements": 0, "opposite_nonzero_signs": 0} for v in [1e-3, 1e-4, 1e-5]}
    for lo in range(0, a.size, CHUNK):
        aa = widened(a[lo:lo + CHUNK], bf16).astype(np.float64)
        bb = widened(b[lo:lo + CHUNK], bf16).astype(np.float64)
        valid = np.isfinite(aa) & np.isfinite(bb)
        nonfinite += int((~valid).sum())
        delta = aa[valid] - bb[valid]
        err2 += float(np.dot(delta, delta))
        ref2 += float(np.dot(bb[valid], bb[valid]))
        if delta.size:
            maxabs = max(maxabs, float(np.abs(delta).max()))
        # Raw BF16 or F32 bit equality; NaN payloads are not a numerical gate.
        abits = np.asarray(a[lo:lo + CHUNK]).view(np.uint16 if bf16 else np.uint32)
        bbits = np.asarray(b[lo:lo + CHUNK]).view(np.uint16 if bf16 else np.uint32)
        bit_equal += int((abits == bbits).sum())
        for threshold, counts in near.items():
            mask = valid & (np.abs(bb) <= float(threshold))
            counts["reference_elements"] += int(mask.sum())
            counts["opposite_nonzero_signs"] += int((mask & (aa != 0) & (bb != 0) & (np.signbit(aa) != np.signbit(bb))).sum())
    return {"elements": int(a.size), "relative_l2": math.sqrt(err2) / max(math.sqrt(ref2), 1e-12), "maxabs": maxabs,
            "nonfinite_pairs": nonfinite, "raw_bit_equal_elements": bit_equal, "near_zero_diagnostic_bands": near}

def cuda_input(path, shape):
    raw = mapped(path, "<u2", shape)
    # Explicit bounded leaf copy; no Python model-sized lists, no float widening.
    return torch.from_numpy(np.array(raw, copy=True)).view(torch.bfloat16).to("cuda")

def dump16(path, tensor):
    tensor.detach().view(torch.uint16).cpu().numpy().astype("<u2", copy=False).tofile(path)

def dump32(path, tensor):
    tensor.detach().cpu().numpy().astype("<f4", copy=False).tofile(path)

def scalar_errors(case, output, partials, bias, pure=False):
    checks = []
    for row in case["sampled_oracle"]["output"]:
        reference = row["reference_f64"]
        if pure:
            reference -= float(widened(bias[row["column"]:row["column"] + 1], True)[0])
        actual = float(widened(output[row["row"], row["column"]:row["column"] + 1], True)[0])
        checks.append({**row, "reference_f64": reference, "actual": actual, "abs_error": abs(actual - reference)})
    pieces = []
    if partials is not None:
        for row in case["sampled_oracle"]["parts"]:
            actual = float(partials[row["part"], row["row"], row["column"]])
            pieces.append({**row, "actual": actual, "abs_error": abs(actual - row["reference_f64"])})
    return {"output": checks, "F32parts": pieces, "scope": "prespecified independent widened-BF16 F64 sample diagnostics; no fitted acceptance threshold"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--profile", action="store_true", help="One separate untimed actual CUDA backend profile per supported mode")
    args = parser.parse_args()
    if args.profile and args.timing:
        raise ValueError("backend profile must be a separate untimed process")
    if args.output.exists():
        raise ValueError("preserve existing output directory")
    args.output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_num_threads(1)
    manifest = load_json(args.archive / "manifest.json")
    if manifest["dtype"] != "bf16-le":
        raise ValueError("BF16 archive required")
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "settings": {"allow_tf32": False, "allow_bf16_reduced_precision_reduction": False},
              "scope": "separate component control; no native-framework calls or fullmodel gate claim", "cases": []}
    cases = list(manifest["cases"])
    if args.reverse:
        cases.reverse()
    for case in cases:
        name = case["name"]
        out = args.output / name
        out.mkdir()
        m, n, k = case["m"], case["n"], case["k"]
        a = cuda_input(args.archive / name / "a.bf16", case["a_shape"])
        b = cuda_input(args.archive / name / "b.bf16", case["b_shape"])
        output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        bias = None
        parts = None
        functions = []
        unsupported = []
        if case["kind"] == "nt_fused_linear":
            bias = cuda_input(args.archive / name / "bias.bf16", [n])
            def addmm():
                torch.addmm(bias, a, b.t(), out=output)
            def puremm():
                torch.mm(a, b.t(), out=output)
            functions = [("addmm_bias_before_BF16", addmm, False), ("pure_NT_mm_diagnostic", puremm, True)]
        elif case["kind"] == "tn_split4":
            def single():
                torch.mm(a.t(), b, out=output)
            functions = [("single_TN_BF16_strong_control", single, False)]
            part_k = k // 4
            parts = torch.empty((4, m, n), dtype=torch.float32, device="cuda")
            t1 = torch.empty((m, n), dtype=torch.float32, device="cuda")
            t2 = torch.empty_like(t1)
            t3 = torch.empty_like(t1)
            def matched():
                for part in range(4):
                    torch.mm(a[part * part_k:(part + 1) * part_k].t(), b[part * part_k:(part + 1) * part_k], out_dtype=torch.float32, out=parts[part])
                torch.add(parts[0], parts[1], out=t1)
                torch.add(t1, parts[2], out=t2)
                torch.add(t2, parts[3], out=t3)
                output.copy_(t3)  # One final BF16 conversion, no initial +0.
            try:
                matched()
                torch.cuda.synchronize()
            except (TypeError, RuntimeError) as error:
                text = str(error).lower()
                if isinstance(error, RuntimeError) and not ("out_dtype" in text or ("not supported" in text and "dtype" in text)):
                    raise
                unsupported.append({"mode": "matched4_F32parts_left4_BF16", "status": "unsupported_local_torch_mm_out_dtype", "reason": str(error), "substitution_with_BF16partials": False})
                parts = None
            else:
                functions.append(("matched4_F32parts_left4_BF16", matched, False))
        else:
            raise ValueError("unknown contract")
        if args.reverse:
            functions.reverse()
        results = []
        for mode, function, pure in functions:
            function()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(output).all()):
                raise ValueError("finite input produced nonfinite Torch output")
            dump16(out / f"{mode}.bf16", output)
            ref = mapped(out / f"{mode}.bf16", "<u2", [m, n])
            native = mapped(args.native / name / "output.bf16", "<u2", [m, n])
            partial_map = None
            if mode == "matched4_F32parts_left4_BF16":
                dump32(out / "matched-partials.f32", parts)
                partial_map = mapped(out / "matched-partials.f32", "<f4", [4, m, n])
            bias_raw = mapped(args.archive / name / "bias.bf16", "<u2", [n]) if bias is not None else None
            native_parts = mapped(args.native / name / "partials.f32", "<f4", [4, m, n]) if case["kind"] == "tn_split4" else None
            entry = {"mode": mode, "native_comparison": None if pure else metrics(native, ref),
                     "native_F64_samples": scalar_errors(case, native, native_parts, bias_raw, pure=False),
                     "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                     "F64_samples": scalar_errors(case, ref, partial_map, bias_raw, pure=pure),
                     "partial_native_comparison": metrics(mapped(args.native / name / "partials.f32", "<f4", [4, m, n]), partial_map, False) if partial_map is not None else None}
            # Capture only after support/contract/output checks. Timing is an
            # explicitly separate later invocation, never run automatically.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(32):
                    function()
            graph.replay()
            torch.cuda.synchronize()
            dump16(out / f"{mode}-capture.bf16", output)
            replay = mapped(out / f"{mode}-capture.bf16", "<u2", [m, n])
            if not np.array_equal(ref, replay):
                raise ValueError("Torch capture/replay changed output bits")
            samples = []
            warmups = 0
            warmup_ms = 0.0
            if args.timing:
                began = time.perf_counter()
                while time.perf_counter() - began < 0.5:
                    graph.replay()
                    torch.cuda.synchronize()
                    warmups += 1
                warmup_ms = (time.perf_counter() - began) * 1000
                for _ in range(32):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / 32)
            dump16(out / f"{mode}-post.bf16", output)
            if not np.array_equal(ref, mapped(out / f"{mode}-post.bf16", "<u2", [m, n])):
                raise ValueError("posttiming output changed bits")
            partial_replay_bits_equal = None
            if partial_map is not None:
                dump32(out / "matched-partials-post.f32", parts)
                post_parts = mapped(out / "matched-partials-post.f32", "<f4", [4, m, n])
                if not np.array_equal(partial_map.view(np.uint32), post_parts.view(np.uint32)):
                    raise ValueError("F32partial replay changed bits")
                partial_replay_bits_equal = True
            entry.update({"capture_repeats": 32, "samples_us": samples, "warmups": warmups, "warmup_ms": warmup_ms, "pre_post_bits_equal": True, "partial_pre_post_bits_equal": partial_replay_bits_equal})
            if args.profile:
                # Separate profile after all timed samples, never included in
                # event intervals. Actual CUDA names retained without guessing.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                    function()
                    torch.cuda.synchronize()
                entry["profile_names"] = sorted({event.name for event in profile.events()})
                profile.export_chrome_trace(str(out / f"{mode}-profile.json"))
            results.append(entry)
            save_json(args.output / "report.json", {**report, "partial_current_case": name, "current_modes": results})
        report["cases"].append({"case": name, "m": m, "n": n, "k": k, "results": results, "unsupported": unsupported,
                                "working_set": {"input_a_bytes": a.numel() * 2, "input_b_bytes": b.numel() * 2, "output_bytes": output.numel() * 2,
                                                "parts_F32_bytes": 0 if parts is None else parts.numel() * 4},
                                "allocation_scope": "preallocated BF16 input/output; matched partial mode additionally owns4F32partials+3F32sum buffers; all setup/readback outside timings"})
        save_json(args.output / "report.json", report)
        del functions, a, b, bias, output, parts

if __name__ == "__main__":
    main()
