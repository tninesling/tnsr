#!/usr/bin/env python3
"""Independent generic pointer-only PTX bridge and shared-input leaf comparison.

No attention geometry/autograd class is reused. Exact loaded native exports,
actual ABI/launch/resources, complete Rust-output bit proofs, then optional
same-process AB/BA captured component timings. No production integration.
"""
from __future__ import annotations
import argparse
import ctypes as ct
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np
import torch
import non_nn_calibration as calibration


def fingerprint(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def parse_pointer_entry(source, expected):
    entries = re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)", source, re.S)
    if len(entries) != 1:
        raise ValueError("expected exactly one visible entry")
    entry, header = entries[0]
    names = tuple(re.findall(r"\.param\s+\.u64\s+(\w+)", header))
    remainder = re.sub(r"\.param\s+\.u64\s+\w+", "", header)
    if names != tuple(expected) or re.sub(r"[,\s]", "", remainder):
        raise ValueError("unexpected or non-pointer-only PTX ABI")
    target = re.search(r"\.target\s+sm_(\d+)(?:\s|,)", source)
    if target is None:
        raise ValueError("unsupported PTX target declaration")
    return entry, int(target.group(1))


@dataclass(frozen=True)
class Argument:
    name: str
    dtype: str
    shape: tuple[int, ...]
    write: bool = False


def interval(pointer, elements, element_size):
    if pointer <= 0 or pointer % 16 or elements <= 0 or element_size not in (2, 4):
        raise ValueError("invalid pointer alignment/extent/dtype")
    end = pointer + elements * element_size
    if end > (1 << 64) or end <= pointer:
        raise ValueError("pointer range overflow")
    return pointer, end


def disjoint_writes(ranges, writes):
    for i, ((lo, hi), write) in enumerate(zip(ranges, writes)):
        if write:
            for j, (other_lo, other_hi) in enumerate(ranges):
                if i != j and lo < other_hi and other_lo < hi:
                    raise ValueError("native writable range aliases another argument")


class Driver:
    """Pointer-only driver functions; owns modules until all graphs are reset."""
    def __init__(self):
        if ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError("64-bit host ABI required")
        torch.cuda.set_device(0)
        torch.empty(1, device="cuda")  # Establish the Torch primary context.
        self.library = ct.CDLL("libcuda.so.1")
        signatures = {
            "cuInit": [ct.c_uint],
            "cuModuleLoadData": [ct.POINTER(ct.c_void_p), ct.c_void_p],
            "cuModuleGetFunction": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
            "cuModuleUnload": [ct.c_void_p],
            "cuGetErrorString": [ct.c_int, ct.POINTER(ct.c_char_p)],
            "cuFuncGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p],
            "cuOccupancyMaxActiveBlocksPerMultiprocessor": [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t],
            "cuLaunchKernel": [ct.c_void_p, *([ct.c_uint] * 7), ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, ct.c_int
        self.modules = []
        self.check(self.library.cuInit(0), "cuInit")

    def check(self, status, operation):
        if status:
            message = ct.c_char_p()
            self.library.cuGetErrorString(status, ct.byref(message))
            detail = message.value.decode(errors="replace") if message.value else str(status)
            raise RuntimeError(f"{operation}: {detail}")

    def close(self):
        torch.cuda.synchronize()
        for module in reversed(self.modules):
            self.check(self.library.cuModuleUnload(module), "cuModuleUnload")
        self.modules.clear()


class PointerKernel:
    def __init__(self, driver, source_path, expected_resources, arguments, tensors):
        self.driver = driver
        self.arguments, self.tensors = tuple(arguments), tuple(tensors)
        source = Path(source_path).read_text()
        name, sm = parse_pointer_entry(source, [a.name for a in arguments])
        major, minor = torch.cuda.get_device_capability(0)
        if sm != major * 10 + minor:
            raise RuntimeError("exported target differs from actual device")
        if len(tensors) != len(arguments):
            raise ValueError("argument count mismatch")
        ranges = []
        for descriptor, tensor in zip(arguments, tensors):
            dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[descriptor.dtype]
            if (tensor.dtype != dtype or tuple(tensor.shape) != descriptor.shape or not tensor.is_contiguous()
                    or tensor.storage_offset() != 0 or tensor.device.type != "cuda" or tensor.device.index != 0):
                raise ValueError(f"native tensor descriptor mismatch: {descriptor}")
            ranges.append(interval(tensor.data_ptr(), math.prod(descriptor.shape), tensor.element_size()))
        disjoint_writes(ranges, [a.write for a in arguments])
        self.grid = tuple(expected_resources["grid"])
        self.block = tuple(expected_resources["block"])
        if len(self.grid) != 3 or len(self.block) != 3 or any(not 0 < x < (1 << 32) for x in (*self.grid, *self.block)):
            raise ValueError("invalid launch geometry")
        encoded = ct.create_string_buffer(source.encode())
        module, function = ct.c_void_p(), ct.c_void_p()
        driver.check(driver.library.cuModuleLoadData(ct.byref(module), ct.cast(encoded, ct.c_void_p)), "cuModuleLoadData")
        driver.modules.append(module)
        driver.check(driver.library.cuModuleGetFunction(ct.byref(function), module, name.encode()), "cuModuleGetFunction")
        self.function = function
        actual = {}
        for label, attribute in [("max_threads", 0), ("shared", 1), ("local", 3), ("registers", 4)]:
            value = ct.c_int()
            driver.check(driver.library.cuFuncGetAttribute(ct.byref(value), attribute, function), "cuFuncGetAttribute")
            actual[label] = value.value
        capacity = ct.c_int()
        driver.check(driver.library.cuOccupancyMaxActiveBlocksPerMultiprocessor(ct.byref(capacity), function, math.prod(self.block), 0), "cuOccupancyMaxActiveBlocksPerMultiprocessor")
        actual["api_capacity"] = capacity.value
        for label, value in actual.items():
            if value != expected_resources[label]:
                raise RuntimeError(f"driver resource differs from Rust capture: {label} {value}/{expected_resources[label]}")
        if actual["max_threads"] < math.prod(self.block) or actual["local"] != 0:
            raise RuntimeError("unqualified thread/spill configuration")
        self.metadata = {"source": fingerprint(source_path), "entry": name, "arguments": [a.__dict__ for a in arguments],
                         "grid": self.grid, "block": self.block, "resources": actual,
                         "capacity_is_not_measured_occupancy": True, "pointers": [t.data_ptr() for t in tensors]}

    def launch(self):
        stream = torch.cuda.current_stream()
        pointers = []
        for tensor in self.tensors:
            tensor.record_stream(stream)
            pointers.append(ct.c_void_p(tensor.data_ptr()))
        arguments = (ct.c_void_p * len(pointers))(*(ct.addressof(p) for p in pointers))
        self.driver.check(self.driver.library.cuLaunchKernel(self.function, *self.grid, *self.block, 0,
                          ct.c_void_p(stream.cuda_stream), arguments, None), "cuLaunchKernel")


def exact_output(actual, expected_path, output_path, bf16):
    shape = tuple(actual.shape)
    if bf16:
        calibration.dump16(output_path, actual)
        dtype = "<u2"
    else:
        calibration.dump32(output_path, actual)
        dtype = "<f4"
    expected = calibration.mapped(expected_path, dtype, shape)
    got = calibration.mapped(output_path, dtype, shape)
    raw_dtype = np.uint16 if bf16 else np.uint32
    if not np.array_equal(got.view(raw_dtype), expected.view(raw_dtype)):
        raise ValueError(f"full native bridge bits differ from Rust capture: {expected_path}")
    return {"elements": int(actual.numel()), "all_bits_equal": True, "expected": fingerprint(expected_path), "actual": fingerprint(output_path)}


def one_file(directory, pattern):
    paths = sorted(Path(directory).glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one exact loaded PTX artifact: {directory}/{pattern}")
    return paths[0]


def capture(function, repeats, stream):
    # Warm every library/driver path on the capture stream, outside capture.
    with torch.cuda.stream(stream):
        for _ in range(3):
            function()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            function()
    return graph


def compare_case(case, args, driver, stream, qualified_case=None):
    m, n, k, name = case["m"], case["n"], case["k"], case["name"]
    out = args.output / name
    out.mkdir()
    native_dir = args.native / name
    resources = calibration.load_json(native_dir / "actual-resources-and-abi.json")["kernels"]
    if any(type(x) is not int or x <= 0 for x in (m, n, k)):
        raise ValueError("positive integer matrix dimensions required")
    split = case["kind"] == "tn_split4"
    expected_a, expected_b = ([k, m], [k, n]) if split else ([m, k], [n, k])
    if case["a_shape"] != expected_a or case["b_shape"] != expected_b or case["output_shape"] != [m, n]:
        raise ValueError("archive physical shapes disagree with mathematical dimensions")
    # Check complete retained output/partial extents before any native launch.
    # Inputs are separately checked by cuda_input against the physical shapes.
    calibration.mapped(native_dir / "output.bf16", "<u2", (m, n))
    if split:
        if k % 4:
            raise ValueError("four contiguous K partitions require K multiple4")
        calibration.mapped(native_dir / "partials.f32", "<f4", (4, m, n))
        if f"partial[{m},{k}]x[{k},{n}]square64=true" not in resources[0]["label"].replace(" ", ""):
            raise ValueError("actual partial descriptor does not match archive dimensions")
    else:
        expected_shapes = f"inputs=[[{m},{k}],[{n},{k}],[{n}]]output=[{m},{n}]"
        if expected_shapes not in resources[0]["label"].replace(" ", ""):
            raise ValueError("actual fused-linear descriptor does not match archive dimensions")
    a = calibration.cuda_input(args.archive / name / "a.bf16", case["a_shape"])
    b = calibration.cuda_input(args.archive / name / "b.bf16", case["b_shape"])
    input_tensors = [("a", a), ("b", b)]
    input_preproofs = {label: exact_output(tensor, args.archive / name / f"{label}.bf16",
                                         out / f"input-{label}-pre.bf16", True)
                       for label, tensor in input_tensors}
    native_output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    bias = None
    native_parts = None
    controls = []
    owners = [a, b, native_output]
    kernels = []
    if case["kind"] == "nt_fused_linear":
        if len(resources) != 1:
            raise ValueError("NT native contract must contain exactly one fused linear kernel")
        bias = calibration.cuda_input(args.archive / name / "bias.bf16", [n])
        owners.append(bias)
        input_tensors.append(("bias", bias))
        input_preproofs["bias"] = exact_output(bias, args.archive / name / "bias.bf16",
                                               out / "input-bias-pre.bf16", True)
        specs = [Argument("input_0", "bf16", (m, k)), Argument("input_1", "bf16", (n, k)),
                 Argument("input_2", "bf16", (n,)), Argument("output_0", "bf16", (m, n), True)]
        kernels.append(PointerKernel(driver, one_file(native_dir, "program-*.ptx"), resources[0], specs, [a, b, bias, native_output]))
        torch_output = torch.empty_like(native_output)
        owners.append(torch_output)
        def addmm():
            torch.addmm(bias, a, b.t(), out=torch_output)
        controls.append(("Torch_addmm_matched_bias_before_BF16", addmm, torch_output, None))
    else:
        if case["kind"] != "tn_split4" or len(resources) != 2 or k % 4:
            raise ValueError("TN native contract must be four F32partials plus left4 reducer")
        native_parts = torch.empty((4, m, n), dtype=torch.float32, device="cuda")
        owners.append(native_parts)
        specs = [Argument("input_0", "bf16", (k, m)), Argument("input_1", "bf16", (k, n)), Argument("output_0", "f32", (4, m, n), True)]
        partial = one_file(args.native / "actual-partial-ptx", f"*-{m}-{n}-{k // 4}-square64.ptx")
        kernels.append(PointerKernel(driver, partial, resources[0], specs, [a, b, native_parts]))
        specs = [Argument("partial", "f32", (4, m, n)), Argument("output", "bf16", (m, n), True)]
        reducer = one_file(args.native / "actual-partial-ptx", f"reducer-{m}-{n}.ptx")
        kernels.append(PointerKernel(driver, reducer, resources[1], specs, [native_parts, native_output]))
        strong = torch.empty_like(native_output)
        matched_output = torch.empty_like(native_output)
        parts = torch.empty_like(native_parts)
        sums = [torch.empty((m, n), dtype=torch.float32, device="cuda") for _ in range(3)]
        owners.extend([strong, matched_output, parts, *sums])
        def single():
            torch.mm(a.t(), b, out=strong)
        controls.append(("Torch_single_TN_BF16_strong", single, strong, None))
        def matched():
            for part in range(4):
                lo, hi = part * k // 4, (part + 1) * k // 4
                torch.mm(a[lo:hi].t(), b[lo:hi], out_dtype=torch.float32, out=parts[part])
            torch.add(parts[0], parts[1], out=sums[0])
            torch.add(sums[0], parts[2], out=sums[1])
            torch.add(sums[1], parts[3], out=sums[2])
            matched_output.copy_(sums[2])
        # This path must already be supported by the untimed qualification.
        matched()
        torch.cuda.synchronize()
        controls.append(("Torch_matched4_F32parts_left4_BF16", matched, matched_output, parts))
    if qualified_case is not None:
        before = [kernel["source"]["sha256"] for kernel in qualified_case["native_kernels"]]
        current = [kernel.metadata["source"]["sha256"] for kernel in kernels]
        if before != current:
            raise ValueError("qualified exact loaded PTX changed before execution/timing")
    def native():
        for kernel in kernels:
            kernel.launch()
    with torch.cuda.stream(stream):
        native()
    stream.synchronize()
    proofs = {"read_only_GPU_inputs": input_preproofs, "native_output": exact_output(native_output, native_dir / "output.bf16", out / "native-pre.bf16", True)}
    if native_parts is not None:
        proofs["native_F32partials"] = exact_output(native_parts, native_dir / "partials.f32", out / "native-partials-pre.f32", False)
    graphs = []
    try:
        capture_order = [("native", native), *[(label, function) for label, function, _, _ in controls]]
        if args.reverse:
            capture_order.reverse()
        captured = {}
        for label, function in capture_order:
            graph = capture(function, 32, stream)
            graphs.append(graph)
            captured[label] = graph
        native_graph = captured["native"]
        functions = list(controls)
        if args.reverse:
            functions.reverse()
        comparisons = []
        for label, function, torch_output, torch_parts in functions:
            graph = captured[label]
            with torch.cuda.stream(stream):
                native_graph.replay()
                graph.replay()
            stream.synchronize()
            exact_output(native_output, native_dir / "output.bf16", out / f"{label}-native-capture.bf16", True)
            if native_parts is not None:
                exact_output(native_parts, native_dir / "partials.f32", out / f"{label}-native-parts-capture.f32", False)
            calibration.dump16(out / f"{label}-Torch-pre.bf16", torch_output)
            reference = calibration.mapped(out / f"{label}-Torch-pre.bf16", "<u2", (m, n))
            native_raw = calibration.mapped(native_dir / "output.bf16", "<u2", (m, n))
            errors = calibration.metrics(native_raw, reference)
            if case["kind"] == "nt_fused_linear" and errors["raw_bit_equal_elements"] != m * n:
                raise ValueError("NT matched addmm must retain qualified full bit identity")
            partial_errors = None
            if torch_parts is not None:
                calibration.dump32(out / f"{label}-Torch-parts-pre.f32", torch_parts)
                partial_errors = calibration.metrics(calibration.mapped(native_dir / "partials.f32", "<f4", (4, m, n)),
                                                     calibration.mapped(out / f"{label}-Torch-parts-pre.f32", "<f4", (4, m, n)), False)
            raw_pairs = []
            warmups = 0
            warmup_ms = 0.0
            if args.timing:
                start = time.perf_counter()
                while time.perf_counter() - start < 0.5:
                    order = [0, 1] if (warmups % 2 == 0) != args.reverse else [1, 0]
                    with torch.cuda.stream(stream):
                        for index in order:
                            [native_graph, graph][index].replay()
                    stream.synchronize()
                    warmups += 1
                warmup_ms = (time.perf_counter() - start) * 1000
                for pair in range(32):
                    order = [0, 1] if (pair % 2 == 0) != args.reverse else [1, 0]
                    us = [0.0, 0.0]
                    with torch.cuda.stream(stream):
                        for index in order:
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record(stream)
                            [native_graph, graph][index].replay()
                            end.record(stream)
                            end.synchronize()
                            us[index] = start.elapsed_time(end) * 1000 / 32
                            if not us[index] > 0 or not math.isfinite(us[index]):
                                raise ValueError("invalid event interval")
                    raw_pairs.append({"pair": pair, "order": ["native", label] if order == [0, 1] else [label, "native"],
                                      "native_us": us[0], "Torch_us": us[1], "Torch_over_native": us[1] / us[0]})
            stream.synchronize()
            post = exact_output(native_output, native_dir / "output.bf16", out / f"{label}-native-post.bf16", True)
            if native_parts is not None:
                exact_output(native_parts, native_dir / "partials.f32", out / f"{label}-native-parts-post.f32", False)
            calibration.dump16(out / f"{label}-Torch-post.bf16", torch_output)
            if not np.array_equal(reference, calibration.mapped(out / f"{label}-Torch-post.bf16", "<u2", (m, n))):
                raise ValueError("Torch postcapture output changed")
            if torch_parts is not None:
                calibration.dump32(out / f"{label}-Torch-parts-post.f32", torch_parts)
                old = calibration.mapped(out / f"{label}-Torch-parts-pre.f32", "<f4", (4, m, n))
                new = calibration.mapped(out / f"{label}-Torch-parts-post.f32", "<f4", (4, m, n))
                if not np.array_equal(old.view(np.uint32), new.view(np.uint32)):
                    raise ValueError("Torch matched F32partials changed")
            backend = None
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    with torch.cuda.stream(stream):
                        function()
                    stream.synchronize()
                profiler.export_chrome_trace(str(out / f"{label}-backend.json"))
                backend = sorted({event.name for event in profiler.events()})
            input_postproofs = {label: exact_output(tensor, args.archive / name / f"{label}.bf16",
                                                    out / f"{label}-{len(comparisons)}-input-post.bf16", True)
                                for label, tensor in input_tensors}
            comparisons.append({"control": label, "native_full_Rust_bits": post, "native_vs_Torch_output_metrics": errors,
                                "native_vs_Torch_F32partial_metrics": partial_errors, "backend_names": backend,
                                "read_only_GPU_inputs_post": input_postproofs,
                                "capture_repeats": 32, "warmup_pairs": warmups, "warmup_ms": warmup_ms, "pairs": raw_pairs,
                                "Torch_over_native_ratio_of_summed_paired_times": None if not raw_pairs else sum(p["Torch_us"] for p in raw_pairs) / sum(p["native_us"] for p in raw_pairs)})
        return {"case": name, "m": m, "n": n, "k": k, "native_kernels": [k.metadata for k in kernels], "preproofs": proofs,
                "shared_input_pointers": {"a": a.data_ptr(), "b": b.data_ptr(), "bias": None if bias is None else bias.data_ptr()},
                "allocation_scope": {"shared_identical_native_Torch_input_addresses": True, "total_live_tensor_bytes": sum(t.numel() * t.element_size() for t in owners),
                                     "excludes_driver_modules_allocator_reserved_and_library_graph_workspaces": True,
                                     "note": "All native/Torch outputs and matched4partial/sum workspaces stay live throughout comparisons; no master/cast/upload/setup/readback inside events."},
                "capture_construction_order": [label for label, _ in capture_order], "comparisons": comparisons}
    finally:
        torch.cuda.synchronize()
        for graph in reversed(graphs):
            graph.reset()
        # Driver holds native tensors/functions while every graph is reset;
        # caller retains the Driver and unloads modules only after this returns.


def cpu_self_test():
    source = ".target sm_89\n.visible .entry test(.param .u64 a,.param .u64 out) {}"
    assert parse_pointer_entry(source, ["a", "out"]) == ("test", 89)
    for bad in [source.replace(".u64 a", ".u32 a"), source + source, source.replace("sm_89", "sm_89a")]:
        try:
            parse_pointer_entry(bad, ["a", "out"])
        except ValueError:
            pass
        else:
            raise AssertionError("malformed ABI accepted")
    assert interval(256, 16, 2) == (256, 288)
    for pointer, count, size in [(0, 1, 2), (257, 1, 2), (256, 0, 2), ((1 << 64) - 16, 16, 4)]:
        try:
            interval(pointer, count, size)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid range accepted")
    disjoint_writes([(256, 288), (512, 544)], [False, True])
    try:
        disjoint_writes([(256, 288), (272, 304)], [False, True])
    except ValueError:
        pass
    else:
        raise AssertionError("aliased output accepted")
    print("Generic pointer ABI/range/write-overlap CPU checks PASS; no CUDA initialization")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--qualified", type=Path, help="Prior complete untimed bridge report required for timing")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cpu-self-test", action="store_true")
    args = parser.parse_args()
    if args.cpu_self_test:
        cpu_self_test()
        return
    if None in (args.archive, args.native, args.output):
        parser.error("archive/native/output are required")
    if args.profile and args.timing:
        parser.error("backend profiles must be a separate untimed invocation")
    if args.timing and args.qualified is None:
        parser.error("timing requires a prior qualified complete bridge report and parent reservation")
    if args.output.exists():
        raise ValueError("preserve existing output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    manifest = calibration.load_json(args.archive / "manifest.json")
    if manifest["dtype"] != "bf16-le":
        raise ValueError("BF16 archive required")
    cases = list(manifest["cases"])
    if args.reverse:
        cases.reverse()
    input_files = {f"{case['name']}/{name}": fingerprint(args.archive / case["name"] / name)
                   for case in cases for name in ("a.bf16", "b.bf16", "bias.bf16")}
    report = {"status": "in_progress", "timing": args.timing, "reverse": args.reverse,
              "torch": torch.__version__, "cuda": torch.version.cuda, "source": fingerprint(Path(__file__)),
              "calibration_helpers": fingerprint(Path(calibration.__file__)), "input_manifest": fingerprint(args.archive / "manifest.json"),
              "settings": {"allow_tf32": False, "allow_bf16_reduced_precision_reduction": False}, "input_files": input_files, "cases": []}
    if args.timing:
        qualified = calibration.load_json(args.qualified)
        if qualified.get("status") != "qualified_full_Rust_bits_and_contracts" or qualified.get("timing"):
            raise ValueError("prior bridge report is not a complete untimed qualification")
        if (qualified["source"]["sha256"] != report["source"]["sha256"]
                or qualified["calibration_helpers"]["sha256"] != report["calibration_helpers"]["sha256"]
                or qualified["input_manifest"]["sha256"] != report["input_manifest"]["sha256"]
                or qualified["torch"] != report["torch"] or qualified["cuda"] != report["cuda"]
                or {name: f["sha256"] for name, f in qualified["input_files"].items()}
                != {name: f["sha256"] for name, f in input_files.items()}):
            raise ValueError("qualification source/input mismatch")
    driver = Driver()
    report["device"] = {"name": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}
    if args.timing and qualified["device"] != json.loads(json.dumps(report["device"])):
        driver.close()
        raise ValueError("qualified device changed")
    stream = torch.cuda.Stream()
    try:
        for case in cases:
            previous = next(c for c in qualified["cases"] if c["case"] == case["name"]) if args.timing else None
            result = compare_case(case, args, driver, stream, previous)
            report["cases"].append(result)
            calibration.save_json(args.output / "report.json", report)
        report["status"] = "qualified_full_Rust_bits_and_contracts" if not args.timing else "timing_complete_full_pre_post_proofs"
        report["scope"] = "Same-process shared-input native PTX versus Torch components; no fullmodel or alternative native kernel adoption claim."
        calibration.save_json(args.output / "report.json", report)
    finally:
        torch.cuda.synchronize()
        driver.close()


if __name__ == "__main__":
    main()
