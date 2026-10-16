"""Alternate complete native training updates on one fixed batch and archive.

Start both workers from the same weights. Exclude warmups, reverse AB/BA order
on successive pairs, and retain every request time, replay state and profile.
No original-model project imports or PyTorch compilation run during measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from bridge import Worker


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    batch = json.loads(args.batch.read_text())
    settings = dict(microbatch=args.microbatch, scheduler_updates=1000,
                    learning_rate=5e-5, value_learning_rate=2e-4, entropy_coef=.005,
                    tree_reductions=args.tree_reductions, cuda_graphs=args.cuda_graphs,
                    tf32x3=args.tf32x3)
    paths = [args.baseline_worker.resolve(), args.worker.resolve()]
    compute_probe = args.bf16_compute_probe
    if compute_probe and not args.tf32x3:
        raise ValueError('the compute-only BF16 experiment substitutes TF32x3 matmuls')
    report = dict(microbatch=args.microbatch, settings=settings, warmup_count=args.warmups,
                  repeats=args.repeats, archive=str(args.archive.resolve()),
                  archive_weights_sha256=hashlib.sha256((args.archive/'weights.f32').read_bytes()).hexdigest(),
                  batch_sha256=hashlib.sha256(args.batch.read_bytes()).hexdigest(),
                  worker_paths=[str(path) for path in paths],
                  worker_hashes=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                  protocol='alternating AB/BA full train requests; same settings/batch/initial weights',
                  samples=[[], []], warmups=[[], []])
    if compute_probe:
        report['precision_probe_limits'] = [
            'Requires the separately patched experimental candidate; a production candidate fails the label check.',
            'BF16 shared operands with f32 accumulation/global storage/output; not native AMP or training parity.',
            'Initial weights match, but repeated-update numerical trajectories differ with precision.',
        ]
    workers = []
    previous_probe = os.environ.get('TNSR_BF16_COMPUTE_PROBE')
    try:
        if compute_probe:
            os.environ['TNSR_BF16_COMPUTE_PROBE'] = '1'
        for path in paths:
            workers.append(Worker(path, args.archive.resolve(), 'cuda'))
        for iteration in range(args.warmups + args.repeats):
            for index in ([0, 1] if iteration % 2 == 0 else [1, 0]):
                started = time.perf_counter()
                result = workers[index].request('train', batch=batch, settings=settings)
                seconds = time.perf_counter() - started
                if not all(math.isfinite(float(value)) for value in result['metrics'].values()):
                    raise ValueError(f'nonfinite training metrics in worker {index}')
                expected_precision = 'tf32x3' if args.tf32x3 else 'strict_f32'
                if compute_probe and index == 1:
                    expected_precision = 'bf16_compute_probe'
                if result.get('matmul_precision') != expected_precision:
                    raise RuntimeError(f'worker {index} did not use {expected_precision}')
                state = result.get('cuda_graphs', {})
                if args.cuda_graphs and (not state.get('active_shapes') or state.get('fallbacks')):
                    raise RuntimeError(f'worker {index} replay fallback: {state}')
                sample = dict(seconds=seconds, profile=result.get('profile'), graph=state,
                              metrics=result['metrics'], matmul_precision=result['matmul_precision'],
                              cuda_compile=result.get('cuda_compile'),
                              split_k_contractions=result.get('split_k_contractions'))
                report['warmups' if iteration < args.warmups else 'samples'][index].append(sample)
                print(iteration, index, round(seconds * 1000, 3), flush=True)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        a, b = report['samples']
        report['summary'] = dict(
            median_seconds=[statistics.median(row['seconds'] for row in rows) for rows in [a, b]],
            candidate_wins=sum(y['seconds'] < x['seconds'] for x, y in zip(a, b)),
            median_paired_throughput_ratio=statistics.median(x['seconds']/y['seconds'] for x, y in zip(a, b)))
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report['summary']), flush=True)
    finally:
        if compute_probe:
            if previous_probe is None:
                os.environ.pop('TNSR_BF16_COMPUTE_PROBE', None)
            else:
                os.environ['TNSR_BF16_COMPUTE_PROBE'] = previous_probe
        for worker in workers:
            worker.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--batch', type=Path, required=True, help='fixed native PPO batch JSON from benchmark_training.py')
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--baseline-worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--tf32x3', action='store_true')
    parser.add_argument('--cuda-graphs', action='store_true')
    parser.add_argument('--tree-reductions', action='store_true')
    parser.add_argument('--bf16-compute-probe', action='store_true',
                        help='research only: require the separately patched BF16-arithmetic candidate; not AMP parity')
    args = parser.parse_args()
    if min(args.microbatch, args.repeats) < 1 or args.warmups < 0:
        parser.error('microbatch/repeats must be positive and warmups nonnegative')
    run(args)


if __name__ == '__main__':
    main()
