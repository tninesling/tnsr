"""Matched GPU training research; no writes to the original forty-five project.

Uses one fixed real-game-derived batch and explicit CUDA synchronization. Native
f32 is compared with original eager bf16 and a copied original updater with its
single autocast decision disabled for strict-f32 comparison. Every timed native
call includes JSON, transfers, accumulated gradients, clipping and Adam. PyTorch
updater includes H2D staging/metrics/clipping/Adam. Optional native bidding anchors
are recomputed on the CPU within each timed step, matching the current adapter.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from bridge import Worker
from export import load_archive
from validate_ppo import anchor_outputs


def synchronize():
    import torch
    torch.cuda.synchronize()


def source_updater(use_bf16):
    """Use original function body; strict baseline only disables its autocast flag."""
    from forty_five_ai.train import ppo_update
    source = inspect.getsource(ppo_update)
    decision = "use_cuda_autocast = use_cuda_prefetch and torch.cuda.is_bf16_supported()"
    if source.count(decision) != 1:
        raise RuntimeError("original autocast decision changed; audit the benchmark adaptation")
    source = source.replace(decision, f"use_cuda_autocast = {use_bf16!r}")
    namespace = dict(ppo_update.__globals__)
    exec(compile(source, inspect.getfile(ppo_update), "exec"), namespace)
    return namespace["ppo_update"], hashlib.sha256(source.encode()).hexdigest()


def make_batch(model, root, size):
    import torch
    from forty_five_ai.env import FortyFiveSelfPlayRunnerAdapter
    from forty_five_ai.train import collect_rollout_rust
    torch.manual_seed(17)
    runner = FortyFiveSelfPlayRunnerAdapter(num_envs=2, seed=17)
    original, _ = collect_rollout_rust(runner, model, 2, torch.device("cpu"), 17)
    n = len(original["actions"])
    indices = torch.arange(size) % n
    original = {key: value[indices].clone() for key, value in original.items()}
    original["hand_ids"].zero_()
    from forty_five_ai.train import normalize_advantages
    native = dict(
        observations=dict(token_ids=original["token_ids"].tolist(),
                          token_type_ids=original["token_type_ids"].tolist(),
                          phase_ids=original["phases"].tolist()),
        action_masks=original["action_masks"].tolist(), actions=original["actions"].tolist(),
        old_log_probs=original["log_probs"].tolist(), old_values=original["values"].tolist(),
        returns=original["returns"].tolist(), advantages=normalize_advantages(original["advantages"]).tolist(),
    )
    return original, native


def gpu_status():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm",
        "--format=csv,noheader,nounits"], text=True).strip()


def process_gpu_memory(pid):
    """Whole worker process MiB, sampled outside the timed interval."""
    try:
        rows = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits"], text=True).splitlines()
        for row in rows:
            process, memory = row.split(",", 1)
            if process.strip() == str(pid):
                return int(memory.strip())
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    sys.path.insert(0, str(args.forty_five_root / "python"))
    args.output.mkdir(parents=True, exist_ok=False)
    model = load_archive(args.archive, args.forty_five_root)
    original, native = make_batch(model, args.forty_five_root, args.transitions)
    report = dict(archive=str(args.archive.resolve()),
                  archive_weights_sha256=hashlib.sha256((args.archive / "weights.f32").read_bytes()).hexdigest(),
                  model_source_sha256=hashlib.sha256((args.forty_five_root / "python/forty_five_ai/model.py").read_bytes()).hexdigest(),
                  train_source_sha256=hashlib.sha256((args.forty_five_root / "python/forty_five_ai/train.py").read_bytes()).hexdigest(),
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  batch_sha256=hashlib.sha256(json.dumps(native,sort_keys=True).encode()).hexdigest(),
                  transitions=args.transitions, microbatch=args.microbatch, warmups=args.warmups,
                  repeats=args.repeats, anchor_kl_coef=args.anchor_kl_coef,
                  native_reduction_mode="deterministic_tree_f32" if args.tree_reductions else "strict_f32",
                  native_graph_replay=args.cuda_graphs,
                  native_matmul_precision="tf32x3" if args.tf32x3 else "strict_f32",
                  baseline_tree_reductions=args.baseline_tree_reductions,
                  baseline_graph_replay=args.baseline_cuda_graphs,
                  baseline_matmul_precision="tf32x3" if args.baseline_tf32x3 else "strict_f32",
                  native_memory_accounting="whole worker process MiB, including modules/workspace/graph; sampled outside timing",
                  external_workloads="left running; monitor GPU samples for contention",
                  cases={})
    (args.output / "batch.json").write_text(json.dumps(native))
    settings = dict(microbatch=args.microbatch, anchor_kl_coef=args.anchor_kl_coef,
                    scheduler_updates=1000,learning_rate=5e-5,value_learning_rate=2e-4,entropy_coef=.005)
    baseline_settings = settings.copy()
    if args.baseline_tree_reductions:
        baseline_settings["tree_reductions"] = True
    if args.baseline_cuda_graphs:
        baseline_settings["cuda_graphs"] = True
    if args.baseline_tf32x3:
        baseline_settings["tf32x3"] = True
    if args.tree_reductions:
        settings["tree_reductions"] = True
    if args.cuda_graphs:
        settings["cuda_graphs"] = True
    if args.tf32x3:
        settings["tf32x3"] = True
    if args.anchor_kl_coef:
        native["anchor"] = anchor_outputs(model, native)
    native_request = dict(op="train", batch=native, settings=settings)
    (args.output / "requests.jsonl").write_text((json.dumps(native_request)+"\n") * args.native_profile_steps)
    (args.output / "baseline-requests.jsonl").write_text(
        (json.dumps(dict(op="train", batch=native, settings=baseline_settings))+"\n") * args.native_profile_steps)
    worker = Worker(args.worker, args.archive, "cuda")
    baseline_worker = Worker(args.baseline_worker, args.archive, "cuda") if args.baseline_worker else None
    def native_step(target=worker, options=settings):
        if args.anchor_kl_coef:
            native["anchor"] = anchor_outputs(model, native)
        return target.request("train", batch=native, settings=options)
    cases = {}
    baseline_label = "tnsr_baseline_tree_f32" if args.baseline_tree_reductions else "tnsr_baseline_strict_f32"
    if args.baseline_cuda_graphs:
        baseline_label = baseline_label.replace("tnsr_baseline_", "tnsr_baseline_graph_")
    if args.baseline_tf32x3:
        baseline_label += "_tf32x3"
    if baseline_worker:
        cases[baseline_label] = lambda: native_step(baseline_worker, baseline_settings)
        report["cases"][baseline_label] = dict(samples=[])
        report["baseline_worker_sha256"] = hashlib.sha256(args.baseline_worker.read_bytes()).hexdigest()
    native_label = "tnsr_tree_f32" if args.tree_reductions else "tnsr_strict_f32"
    if args.cuda_graphs:
        native_label = native_label.replace("tnsr_", "tnsr_graph_")
    if args.tf32x3:
        native_label += "_tf32x3"
    cases[native_label] = native_step
    native_pids = {native_label: worker.process.pid}
    if baseline_worker:
        native_pids[baseline_label] = baseline_worker.process.pid
    report["worker_sha256"] = hashlib.sha256(args.worker.read_bytes()).hexdigest()
    modes = [("pytorch_eager_strict_f32",False,False),("pytorch_eager_bf16",True,False)]
    if args.compile:
        modes.append(("pytorch_compiled_bf16",True,True))
    for label, bf16, compiled in modes:
        policy = copy.deepcopy(model).cuda()
        anchor = copy.deepcopy(model).cuda()
        optimizer = torch.optim.Adam([
            dict(params=policy.policy_parameters(),lr=5e-5),
            dict(params=policy.value_parameters(),lr=2e-4)],fused=args.fused_adam)
        updater, digest = source_updater(bf16)
        if compiled:
            policy = torch.compile(policy, dynamic=True)
        def step(policy=policy,anchor=anchor,optimizer=optimizer,updater=updater):
            return updater(policy,optimizer,original,torch.device("cuda"),entropy_coef=.005,
                           epochs=1,hands_per_minibatch=100000,bid_anchor_policy=anchor if args.anchor_kl_coef else None,bid_anchor_kl_coef=args.anchor_kl_coef,
                           transitions_per_microbatch=args.microbatch)
        cases[label] = step
        report["cases"][label] = dict(updater_sha256=digest,samples=[],fused_adam=args.fused_adam)
    report["cases"][native_label] = dict(samples=[])
    try:
        # Interleave modes to reduce drift from the pre-existing GPU workload.
        for iteration in range(args.warmups + args.repeats):
            for label, step in cases.items():
                synchronize()
                torch.manual_seed(10000 + iteration)
                began = time.perf_counter()
                result = step()
                synchronize()
                seconds = time.perf_counter()-began
                metrics = result.get("metrics", result)
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise ValueError(f"nonfinite metrics in {label}")
                replay_requested = ((label == native_label and args.cuda_graphs)
                                    or (label == baseline_label and args.baseline_cuda_graphs))
                if replay_requested:
                    state = result.get("cuda_graphs", {})
                    if not state.get("active_shapes") or state.get("fallbacks"):
                        raise RuntimeError(f"requested graph benchmark fell back: {state}")
                sample = dict(seconds=seconds,gpu_status=gpu_status())
                if label in native_pids:
                    sample["worker_gpu_memory_mib"] = process_gpu_memory(native_pids[label])
                if "cuda_graphs" in result:
                    sample["graph_replay"] = result["cuda_graphs"]
                if label.startswith("tnsr") and "profile" in result:
                    sample["profile"] = result["profile"]
                if iteration >= args.warmups:
                    report["cases"][label]["samples"].append(sample)
                else:
                    report["cases"][label].setdefault("warmup_samples",[]).append(sample)
                print(iteration,label,round(seconds,6),sample["gpu_status"],flush=True)
                (args.output/"report.json").write_text(json.dumps(report,indent=2))
        for label,result in report["cases"].items():
            samples=[sample["seconds"] for sample in result["samples"]]
            result.update(median_seconds=statistics.median(samples),min_seconds=min(samples),max_seconds=max(samples),
                          transitions_per_second=args.transitions/statistics.median(samples))
        if args.profile_torch:
            from torch.profiler import profile, ProfilerActivity
            for label,step in cases.items():
                if label.startswith("pytorch"):
                    synchronize()
                    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as profiler:
                        step();synchronize()
                    profiler.export_chrome_trace(str(args.output/f"{label}-trace.json"))
                    (args.output/f"{label}-operators.txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total",row_limit=30))
    finally:
        worker.close()
        if baseline_worker:
            baseline_worker.close()
    (args.output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--forty-five-root",type=Path,required=True)
    parser.add_argument("--worker",type=Path,default=Path("target/release/examples/forty_five_worker"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline-worker",type=Path,help="optional second native binary for interleaved before/after measurements")
    parser.add_argument("--baseline-tree-reductions",action="store_true",help="select tree ordering in a compatible baseline worker too")
    parser.add_argument("--baseline-cuda-graphs",action="store_true",help="enable graph replay in a compatible baseline worker too")
    parser.add_argument("--baseline-tf32x3",action="store_true",help="select corrected TF32 matmuls in a compatible baseline worker too")
    parser.add_argument("--transitions",type=int,default=64)
    parser.add_argument("--microbatch",type=int,default=8)
    parser.add_argument("--warmups",type=int,default=2)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--native-profile-steps",type=int,default=3)
    parser.add_argument("--fused-adam",action="store_true",default=None)
    parser.add_argument("--profile-torch",action="store_true")
    parser.add_argument("--compile",action="store_true")
    parser.add_argument("--tf32x3",action="store_true",help="use corrected TF32 tensor-core matmuls with f32 storage")
    parser.add_argument("--cuda-graphs",action="store_true",help="enable private-stream native model replay")
    parser.add_argument("--tree-reductions",action="store_true",help="opt native training into fixed f32 reduction trees")
    parser.add_argument("--anchor-kl-coef",type=float,default=0.)
    args=parser.parse_args()
    if min(args.transitions,args.microbatch,args.repeats,args.native_profile_steps)<1 or args.warmups<0:
        parser.error("batch sizes/repeats must be positive and warmups nonnegative")
    run(args)
if __name__=="__main__":main()
