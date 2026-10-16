"""Profile the explicit mixed graph and validate it separately against Torch.

Native timing finishes before Torch is imported. This is a two-projection
training subgraph using forty-five widths, not a complete PPO update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


def request(worker, payload):
    worker.stdin.write(json.dumps(payload) + '\n')
    worker.stdin.flush()
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError(f'probe exited: {worker.poll()}')
    result = json.loads(line)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def gpu_state():
    return subprocess.check_output(['nvidia-smi'], text=True)


def errors(actual, reference):
    a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    delta = a - b
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite result')
    return dict(max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta * delta))),
                normalized_max=float(np.abs(delta).max() / max(float(np.abs(b).max()), 1e-30)),
                exact_fraction=float(np.mean(a == b)))


def native(args):
    args.output.mkdir(parents=True, exist_ok=False)
    measured_worker = args.output / 'worker'
    shutil.copyfile(args.worker, measured_worker)
    measured_worker.chmod(0o755)
    report = dict(worker=str(args.worker.resolve()), worker_sha256=hashlib.sha256(args.worker.read_bytes()).hexdigest(),
                  gpu_before=gpu_state(), platform=platform.platform(), timestamp=time.time(), cases=[])
    report['measured_worker'] = str(measured_worker.resolve())
    report['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    report['activation'] = 'relu' if args.relu else 'silu'
    source_files = ['src/tensor.rs', 'src/graph/mod.rs', 'src/graph/mixed.rs', 'src/graph/split_k.rs',
                    'src/lib.rs', 'src/ptx/mixed.rs', 'src/ptx/mixed/planner.rs',
                    'src/ptx/linear/program.rs', 'src/tile/graph.rs', 'examples/mixed_graph_probe.rs']
    report['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files}
    report['driver_source'] = Path(__file__).read_text()
    cases = [(4096, 683, 256, 683), (512, 683, 256, 683), (33, 65, 35, 17)]
    for index, (rows, inputs, hidden, outputs) in enumerate(cases):
        if args.case is not None and index != args.case:
            continue
        folder = args.output / str(index)
        folder.mkdir()
        spec = dict(rows=rows, inputs=inputs, hidden=hidden, outputs=outputs, relu=args.relu)
        (folder / 'spec.json').write_text(json.dumps(spec))
        rng = np.random.default_rng(20261016 + index)
        arrays = {name: rng.normal(0, .1, shape).astype(np.float32) for name, shape in
                  [('x', (rows, inputs)), ('w1', (hidden, inputs)), ('w2', (outputs, hidden)), ('upstream', (rows, outputs))]}
        for name, data in arrays.items():
            np.save(folder / f'{name}.npy', data)
        worker = subprocess.Popen([str(measured_worker.resolve()), str(folder / 'spec.json')],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            result = request(worker, dict(op='run', **{name: data.ravel().tolist() for name, data in arrays.items()}))
            (folder / 'native.json').write_text(json.dumps(result))
            bench = request(worker, dict(op='bench', samples=args.samples, repeats=args.repeats))
            a, b, c, d = bench['samples_us']
            bench['summary'] = dict(median_us=[statistics.median(t) for t in [a, b, c, d]],
                                    mixed_vs_fused_f32_paired_ratio=statistics.median(x / y for x, y in zip(d, c)),
                                    mixed_wins_vs_fused=sum(y < x for x, y in zip(d, c)),
                                    f32_region_fusion_paired_ratio=statistics.median(x / y for x, y in zip(b, c)))
            report['cases'].append(dict(case=index, spec=spec, bench=bench))
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(index, bench['summary'], flush=True)
        finally:
            worker.stdin.close()
            worker.wait(timeout=60)
    report['gpu_after'] = gpu_state()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


def validate(args):
    import torch
    import torch.nn.functional as F
    report = json.loads((args.output / 'report.json').read_text())
    report['torch'] = dict(version=torch.__version__, cuda=torch.version.cuda,
                           bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    failures = []
    for index, case in enumerate(report['cases']):
        folder = args.output / str(case.get('case', index))
        arrays = {name: np.load(folder / f'{name}.npy') for name in ['x', 'w1', 'w2', 'upstream']}
        native_result = json.loads((folder / 'native.json').read_text())
        comparisons = {}
        for result_index, mixed in [(0, False), (1, True), (2, True)]:
            values = [torch.tensor(arrays[name], device='cuda', requires_grad=True) for name in ['x', 'w1', 'w2']]
            g = torch.tensor(arrays['upstream'], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
                h = F.linear(values[0], values[1])
                h = h.relu() if report.get('activation', 'silu') == 'relu' else F.silu(h)
                y = F.linear(h, values[2])
            loss = (y.float() * g).sum()
            loss.backward()
            reference = [y.float().detach().cpu().numpy().ravel()] + [v.grad.cpu().numpy().ravel() for v in values]
            result = native_result['results'][result_index]
            actual = [result['output']] + result['gradients']
            comparison = {name: errors(a, b) for name, a, b in zip(['output', 'dx', 'dw1', 'dw2'], actual, reference)}
            comparisons[(['f32_torch', 'bf16_unfused_default_torch', 'bf16_default_torch'])[result_index]] = comparison
            # Primitive sanity only. This does not define PPO acceptance.
            limit = 1 / 128 if mixed else 2e-5
            if any(row['normalized_max'] > limit for row in comparison.values()):
                failures.append(dict(case=index, result_index=result_index, mixed=mixed, comparison=comparison))
        comparisons['fused_vs_per_op_f32_gradients'] = [errors(a, b) for a, b in
            zip(native_result['fused_f32_gradients'], native_result['results'][0]['gradients'])]
        if any(row['normalized_max'] > 2e-5 for row in comparisons['fused_vs_per_op_f32_gradients']):
            failures.append(dict(case=index, fused_comparator=True))
        case['correctness'] = comparisons
        print(index, 'validated', flush=True)
    report['limits'] = ['Contended GPU; conditional measurements only.',
                        'Two bias-free projections and the recorded activation; graph-generated backward plus f32 scalar objective.',
                        'Includes all casts and reductions; excludes upload/readback and optimizer.',
                        'F32 comparator uses whole-graph fusion and current split-K eligibility.',
                        'Not the forty-five PPO model or its autocast policy; no end-to-end speedup claim.']
    report['validation_failures'] = failures
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if failures:
        raise ValueError(f'validation failures retained in report: {failures}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=240)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--case', type=int, choices=[0, 1, 2])
    parser.add_argument('--relu', action='store_true', help='reproduce the retained rejected activation experiment')
    args = parser.parse_args()
    if not args.validate_only:
        native(args)
    if not args.native_only:
        validate(args)


if __name__ == '__main__':
    main()
