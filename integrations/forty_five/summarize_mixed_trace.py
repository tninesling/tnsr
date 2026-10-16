"""Match a serialized mixed CUDA graph trace to its ordered kernel labels."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def category(label):
    if 'split-K partial' in label:
        return 'bf16_split_k_gemm'
    if 'split-K sum' in label:
        return 'f32_split_k_sum'
    if 'MatMulRegion' in label or ' MatMul ' in label:
        return 'bf16_gemm' if 'BF16' in label else 'f32_gemm'
    if label.startswith(('materialize', 'convert')):
        return 'cast_copy'
    if 'ReductionRegion' in label or 'Reduce' in label:
        return 'reduction'
    if 'Embedding' in label or 'Indexed' in label:
        return 'indexed'
    return 'other_bf16' if 'BF16' in label else 'other_f32'


def run(args):
    labels = json.loads(args.labels.read_text())
    if not labels or args.warmups < 0:
        raise ValueError('nonempty labels and nonnegative warmups required')
    with sqlite3.connect(args.sqlite) as db:
        rows = db.execute('select start,end,graphNodeId,streamId from CUPTI_ACTIVITY_KIND_KERNEL '
                          'where graphNodeId != 0 order by start').fetchall()
    n = len(labels)
    streams = collections.Counter(row[3] for row in rows)
    if len(streams) != 1 or not rows or len(rows) % n:
        raise ValueError(f'expected one serialized captured graph: {n} labels, {len(rows)} kernels, streams={streams}')
    repeats = len(rows) // n
    if repeats <= args.warmups:
        raise ValueError('no post-warmup graph launches')
    identities = [row[2] for row in rows[:n]]
    if len(set(identities)) != n or any([row[2] for row in rows[j*n:(j+1)*n]] != identities for j in range(repeats)):
        raise ValueError('captured node ordering/count differs between launches')
    updates = []
    for j in range(repeats):
        chunk = rows[j*n:(j+1)*n]
        groups = collections.defaultdict(float)
        for label, (start,end,_,_) in zip(labels,chunk):
            groups[category(label)] += (end-start)/1e6
        updates.append(dict(update=j+1,span_ms=(chunk[-1][1]-chunk[0][0])/1e6,
                            kernel_ms=sum((end-start)/1e6 for start,end,*_ in chunk),category_ms=dict(groups)))
    kernels = []
    for i,label in enumerate(labels):
        times = [(rows[j*n+i][1]-rows[j*n+i][0])/1e6 for j in range(args.warmups,repeats)]
        kernels.append(dict(label=label,category=category(label),median_ms=statistics.median(times),
                            min_ms=min(times),max_ms=max(times)))
    report = dict(sqlite=str(args.sqlite),labels_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                  model_kernel_count=n,launches_per_node=repeats,warmup_updates=args.warmups,
                  verified_ordered_node_ids=True,updates=updates,kernels=kernels,
                  median_span_ms=statistics.median(u['span_ms'] for u in updates[args.warmups:]),
                  median_kernel_ms=statistics.median(u['kernel_ms'] for u in updates[args.warmups:]),
                  category_median_ms={key:statistics.median(u['category_ms'][key] for u in updates[args.warmups:])
                                      for key in updates[0]['category_ms']})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['model_kernel_count','launches_per_node','median_span_ms','category_median_ms']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite',type=Path)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmups',type=int,default=3)
    run(parser.parse_args())
