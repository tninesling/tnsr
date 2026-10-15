"""Audit actual eager CUDA autocast boundaries without changing the source project."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--forty-five-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; use a fresh report path')
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    from export import load_model, load_archive
    from validate_ppo import fixture
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.forty_five_root / 'python'))
    from forty_five_ai.train import ppo_update
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('the original CUDA bf16 training policy requires bf16 support')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = (load_archive if args.source.is_dir() else load_model)(args.source, args.forty_five_root)
    original, _ = fixture(model, args.forty_five_root)
    model = model.cuda()
    optimizer = torch.optim.Adam([
        dict(params=model.policy_parameters(), lr=5e-5),
        dict(params=model.value_parameters(), lr=2e-4)])
    updater_hash = hashlib.sha256(inspect.getsource(ppo_update).encode()).hexdigest()
    phase = 'model_and_backward'
    calls = Counter()
    boundaries = Counter()
    scope = []

    def descriptors(value):
        return tuple((str(x.dtype), tuple(x.shape)) for x in tree_leaves(value)
                     if isinstance(x, torch.Tensor))

    class Audit(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            key = (phase, scope[-1] if scope else '', str(func),
                   descriptors((args, kwargs)), descriptors(result))
            calls[key] += 1
            return result

    handles = []
    for name, module in model.named_modules():
        def before(module, inputs, kwargs, name=name):
            scope.append(name)
        def after(module, inputs, kwargs, output, name=name):
            boundaries[(name, type(module).__name__, descriptors((inputs, kwargs)), descriptors(output))] += 1
            assert scope.pop() == name
        handles.extend([module.register_forward_pre_hook(before, with_kwargs=True),
                        module.register_forward_hook(after, with_kwargs=True)])
    gradient_dtypes = []
    def step():
        nonlocal phase
        gradient_dtypes.append(dict(Counter(str(p.grad.dtype) for p in model.parameters() if p.grad is not None)))
        phase = 'optimizer'
        optimizer.step()
        phase = 'model_and_backward'
    try:
        with Audit():
            metrics = ppo_update(model, optimizer, original, torch.device('cuda'), entropy_coef=.005,
                epochs=1, hands_per_minibatch=1, transitions_per_microbatch=3, optimizer_step=step)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    report = dict(source=str(args.source.resolve()), torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(), separate_critic=model.use_separate_critic,
        source_sha256=hashlib.sha256((args.source / 'weights.f32' if args.source.is_dir()
                                     else args.source).read_bytes()).hexdigest(),
        transitions=len(original['actions']), microbatch=3,
        notes='One untimed original PPO update, full/partial microbatches; module/operator metadata only. Phase model_and_backward includes forward, loss, clipping and backward. Empty module scope is outside forward module hooks. No source rewrites or optimizer changes; this is a dtype audit, not a learning or numerical parity gate.',
        model_source_sha256=hashlib.sha256((args.forty_five_root / 'python/forty_five_ai/model.py').read_bytes()).hexdigest(),
        train_source_sha256=hashlib.sha256((args.forty_five_root / 'python/forty_five_ai/train.py').read_bytes()).hexdigest(),
        updater_sha256=updater_hash, metrics=metrics,
        parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),
        gradient_dtypes=gradient_dtypes,
        optimizer_state_dtypes=dict(Counter(str(v.dtype) for state in optimizer.state.values()
                                           for v in state.values() if isinstance(v, torch.Tensor))),
        index_inputs={name: dict(dtype=str(value.dtype), minimum=int(value.min()), maximum=int(value.max()))
                      for name, value in original.items() if name in ('token_ids','token_type_ids','phases','actions')},
        modules=[dict(name=name, module=kind, inputs=inputs, outputs=outputs, count=count)
                 for (name, kind, inputs, outputs), count in sorted(boundaries.items())],
        operators=[dict(phase=phase, scope=scope, operator=op, inputs=inputs, outputs=outputs, count=count)
                   for (phase, scope, op, inputs, outputs), count in sorted(calls.items())])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key not in ('modules','operators')}))


if __name__ == '__main__':
    main()
