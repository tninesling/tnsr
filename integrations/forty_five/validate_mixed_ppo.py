"""Compare real mixed PPO updates with the unmodified original CUDA trainer.

Keep every named gradient, moment and weight error. This is calibration data;
the existing strict-f32 tolerances and the original BF16 policy are unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from bridge import Worker
from export import export_model, load_archive, load_model
from validate_ppo import fixture, anchor_outputs


def errors(actual, expected):
    a, b = np.asarray(actual, np.float64).ravel(), np.asarray(expected, np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite or mismatched validation arrays')
    d = a - b
    return dict(reference_l2=float(np.linalg.norm(b)), difference_l2=float(np.linalg.norm(d)), max_abs=float(np.abs(d).max()), normalized_max=float(np.abs(d).max()/max(float(np.abs(b).max()), 1e-30)),
                rmse=float(np.sqrt(np.mean(d*d))), normalized_l2=float(np.linalg.norm(d)/max(np.linalg.norm(b), 1e-30)),
                scale_fit=float(np.dot(a,b)/max(np.dot(b,b),1e-30)), cosine=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)),
                sign_flips_gt_1e6=int(((a*b < 0) & (np.abs(b) > 1e-6)).sum()))


def run(args):
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.forty_five_root/'python'))
    from forty_five_ai.train import ppo_update, masked_categorical
    args.output.mkdir(parents=True, exist_ok=False)
    model = (load_archive if args.source.is_dir() else load_model)(args.source, args.forty_five_root)
    original, native = fixture(model, args.forty_five_root, tiny=args.tiny, repeats=args.fixture_repeats)
    archive = args.output/'initial'; export_model(model, archive)
    native_weights = {name: p.detach().cpu().numpy().copy() for name, p in model.named_parameters()}
    m = {name: np.zeros_like(p) for name, p in native_weights.items()}
    v = {name: np.zeros_like(p) for name, p in native_weights.items()}
    policy_names = {id(p) for p in model.policy_parameters()}
    rates = {name: 5e-5 if id(p) in policy_names else 2e-4 for name, p in model.named_parameters()}
    oracle = model.cuda(); anchor = copy.deepcopy(oracle)
    with torch.no_grad(): anchor.bid_head.bias.add_(torch.linspace(-.2, .2, 5, device='cuda'))
    # Match the original bidding-row selection and each microbatch's autocast.
    native['anchor'] = anchor_outputs(anchor, native, mixed_bf16=True, microbatch=args.microbatch)
    (args.output/'batch.json').write_text(json.dumps(native))
    optimizer = torch.optim.Adam([dict(params=oracle.policy_parameters(), lr=5e-5),
                                  dict(params=oracle.value_parameters(), lr=2e-4)])
    settings = dict(learning_rate=5e-5, value_learning_rate=2e-4, entropy_coef=.005,
                    anchor_kl_coef=.03, microbatch=args.microbatch, scheduler_updates=1000,
                    mixed_bf16=True, tree_reductions=True, tf32x3=True, cuda_graphs=not args.plain)
    report = dict(source=str(args.source.resolve()), separate_critic=oracle.use_separate_critic,
                  model_source_sha256=hashlib.sha256((args.forty_five_root/'python/forty_five_ai/model.py').read_bytes()).hexdigest(),
                  train_source_sha256=hashlib.sha256((args.forty_five_root/'python/forty_five_ai/train.py').read_bytes()).hexdigest(),
                  initial_weights_sha256=hashlib.sha256((archive/'weights.f32').read_bytes()).hexdigest(),
                  settings=settings, transitions=len(native['actions']), steps=[], torch_version=torch.__version__,
                  original_updater_sha256=hashlib.sha256(inspect.getsource(ppo_update).encode()).hexdigest(),
                  worker_sha256=hashlib.sha256(args.worker.read_bytes()).hexdigest(),
                  gpu_before=__import__('subprocess').check_output(['nvidia-smi'], text=True),
                  original_bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                  limits=['Independent native and original CUDA trajectories; all gradient/moment/weight errors retained.',
                          'No existing f32 tolerance changes. Numerical calibration is separate from execution/performance success.'])
    with Worker(args.worker, archive, 'cuda') as worker:
        for step in range(1, args.steps + 1):
            gradients = {}
            def take_step():
                gradients.update({name: (torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu().numpy().copy()
                                  for name, p in oracle.named_parameters()})
                optimizer.step()
            metrics = ppo_update(oracle, optimizer, original, torch.device('cuda'), entropy_coef=.005,
                epochs=1, hands_per_minibatch=1, bid_anchor_policy=anchor, bid_anchor_kl_coef=.03,
                optimizer_step=take_step, transitions_per_microbatch=args.microbatch)
            actual = worker.request('train', batch=native, settings=settings, return_gradients=True)
            state = worker.request('optimizer_state')
            if state['step'] != step or actual['optimizer_steps'] != step:
                raise ValueError('Adam timestep mismatch')
            saved = args.output/f'native-{step}'; worker.request('save', directory=str(saved.resolve()))
            restored = load_archive(saved, args.forty_five_root)
            updated = {name: p.detach().cpu().numpy() for name, p in restored.named_parameters()}
            rows = {}
            gradient_reference_squared = gradient_error_squared = 0.
            for name, p in oracle.named_parameters():
                g = np.asarray(actual['gradients'][name], np.float32).reshape(p.shape)
                # An independent f32 recurrence validates native Adam using its
                # own clipped gradients, separately from BF16 oracle drift.
                m[name] = np.float32(.9)*m[name] + np.float32(1.-np.float32(.9))*g
                v[name] = np.float32(.999)*v[name] + np.float32(1.-np.float32(.999))*g*g
                expected = native_weights[name] - np.float32(rates[name]) * (m[name]/np.float32(1.-np.float32(.9)**step)) / (
                    np.sqrt(v[name]/np.float32(1.-np.float32(.999)**step)) + np.float32(1e-8))
                rows[name] = dict(gradient=errors(g, gradients[name]), weight=errors(updated[name], p.detach().cpu().numpy()),
                    first_moment=errors(state['moments'][name]['first'], optimizer.state[p]['exp_avg'].cpu().numpy()),
                    second_moment=errors(state['moments'][name]['second'], optimizer.state[p]['exp_avg_sq'].cpu().numpy()),
                    native_adam_recurrence=errors(updated[name], expected))
                gradient_reference_squared += rows[name]['gradient']['reference_l2']**2
                gradient_error_squared += rows[name]['gradient']['difference_l2']**2
                if rows[name]['native_adam_recurrence']['max_abs'] > 2e-6:
                    raise ValueError(f'native Adam recurrence mismatch: {name}')
                native_weights[name] = updated[name].copy()
            entry = dict(step=step, global_gradient_normalized_l2=float(np.sqrt(gradient_error_squared/max(gradient_reference_squared,1e-30))), native_gradient_norm=actual["gradient_norm_before_clip"], native_gradient_scale=actual["gradient_scale"], original_gradient_norm=metrics["gradient_norm_mean"], native_metrics=actual['metrics'], original_metrics={k: metrics[k] for k in actual['metrics']},
                         metric_abs_errors={k: abs(actual['metrics'][k]-metrics[k]) for k in actual['metrics']}, named=rows,
                         mixed_graph=actual['mixed_graph'], graph=actual.get('cuda_graphs'),
                         max_gradient_normalized_l2=max(r['gradient']['normalized_l2'] for r in rows.values()),
                         max_weight_abs=max(r['weight']['max_abs'] for r in rows.values()))
            report['steps'].append(entry)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(step, 'metric max', max(entry['metric_abs_errors'].values()), 'grad L2 max', entry['max_gradient_normalized_l2'],
                  'weight max', entry['max_weight_abs'], flush=True)
    report['gpu_after'] = __import__('subprocess').check_output(['nvidia-smi'], text=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--forty-five-root', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=3)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--fixture-repeats', type=int, default=1)
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--plain', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
