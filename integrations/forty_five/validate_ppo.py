"""Compare native PPO losses, every clipped gradient and Adam weight with forty-five."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from bridge import Worker
from export import export_model, load_archive, load_model
from validate import observations, infer_reference


def fixture(model, root, tiny=False):
    import torch
    from forty_five_ai.train import masked_categorical, normalize_advantages
    if tiny:
        n, s = 7, model.token_max_tokens
        generator = torch.Generator().manual_seed(213)
        ids = torch.randint(1, model.token_encoder.token_embedding.num_embeddings, (n,s), generator=generator)
        types = torch.randint(0, 6, (n,s), generator=generator)
        ids[-1] = 0
        batch = dict(token_ids=ids.tolist(), token_type_ids=types.tolist(), phase_ids=[1,1,2,3,4,0,5])
        masks = [[j < (5 if i < 2 else 46) for j in range(46)] for i in range(n)]
    else:
        batch, masks = observations(root)
        # Two bidding rows make the per-microbatch anchor normalization observable.
        for key in batch:
            batch[key].insert(1, copy.deepcopy(batch[key][0]))
        masks.insert(1, masks[0].copy())
    logits, values = infer_reference(model, batch)
    masks = torch.tensor(masks, dtype=torch.bool)
    dist = masked_categorical(logits, masks)
    actions = dist.probs.argmax(-1)
    n = len(actions)
    # Deliberately exercise both signs and both sides of PPO/value clipping.
    ratios = torch.tensor([.6,.9,1.,1.1,1.4,.7,1.3,1.05][:n])
    raw_advantages = torch.tensor([1.,-2.,3.,-4.,5.,-6.,7.,-8.][:n])
    old_values = values + torch.tensor([.4,-.3,.1,-.1,.5,-.5,.05,-.05][:n])
    returns = values + torch.tensor([.8,-.1,-.4,.6,-.7,.3,.2,-.2][:n])
    old_logs = dist.log_prob(actions) - ratios.log()
    original = {k: torch.tensor(v, dtype=torch.long) for k,v in batch.items() if k != 'phase_ids'}
    original.update(phases=torch.tensor(batch['phase_ids']), action_masks=masks, actions=actions,
                    log_probs=old_logs, values=old_values, returns=returns, advantages=raw_advantages,
                    hand_ids=torch.zeros(n,dtype=torch.long))
    native = dict(observations=batch, action_masks=masks.tolist(), actions=actions.tolist(),
                  old_log_probs=old_logs.tolist(), old_values=old_values.tolist(), returns=returns.tolist(),
                  advantages=normalize_advantages(raw_advantages).tolist())
    return original, native


def anchor_outputs(model, native):
    import torch
    from forty_five_ai.train import masked_categorical
    logits, _ = infer_reference(model, native['observations'])
    mask = torch.tensor(native['action_masks'])
    dist = masked_categorical(logits, mask)
    return dict(probabilities=dist.probs.tolist(), log_probabilities=torch.where(mask,dist.logits,0.).tolist())


def reference_step(model, anchor, batch, optimizer, microbatch, entropy, anchor_coef):
    import torch
    from forty_five_ai.train import ppo_update
    gradients = {}
    def step():
        for name, parameter in model.named_parameters():
            gradients[name] = (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).detach().clone()
        optimizer.step()
    metrics = ppo_update(model,optimizer,batch,torch.device('cpu'),entropy_coef=entropy,
                         epochs=1,hands_per_minibatch=1,bid_anchor_policy=anchor,
                         bid_anchor_kl_coef=anchor_coef,optimizer_step=step,transitions_per_microbatch=microbatch)
    return metrics, gradients


def validate(source, root, executable, output, backends, tiny=False, steps=2, tree_reductions=False, cuda_graphs=False):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(root/'python'))
    output.mkdir(parents=True, exist_ok=False)
    model = load_archive(source,root) if source.is_dir() else load_model(source,root)
    archive = output/'initial'; export_model(model,archive)
    original, native = fixture(model,root,tiny)
    anchor = copy.deepcopy(model)
    # A different frozen anchor gives a nonzero KL and a nontrivial KL gradient.
    with torch.no_grad(): anchor.bid_head.bias.add_(torch.linspace(-.2,.2,5))
    native['anchor'] = anchor_outputs(anchor,native)
    settings = dict(learning_rate=5e-5,value_learning_rate=2e-4,entropy_coef=.005,
                    anchor_kl_coef=.03,microbatch=3,scheduler_updates=4)
    report = dict(source=str(source.resolve()), torch_version=torch.__version__,
                  separate_critic=model.use_separate_critic, backends={})
    for backend in backends:
        native_settings = settings.copy()
        if tree_reductions and backend == 'cuda':
            native_settings['tree_reductions'] = True
        if cuda_graphs and backend == 'cuda':
            native_settings['cuda_graphs'] = True
        oracle = copy.deepcopy(model)
        optimizer = torch.optim.Adam([dict(params=oracle.policy_parameters(),lr=5e-5),
                                      dict(params=oracle.value_parameters(),lr=2e-4)])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=4,eta_min=1e-6)
        results=[]; start=time.monotonic()
        with Worker(executable,archive,backend) as worker:
            for step in range(steps):
                metrics, gradients = reference_step(oracle,anchor,original,optimizer,3,.005,.03)
                actual=worker.request('train',batch=native,settings=native_settings,return_gradients=True)
                torch.testing.assert_close(torch.tensor(list(actual['metrics'].values())),
                    torch.tensor([metrics[k] for k in actual['metrics']]),atol=3e-5,rtol=5e-4)
                errors={}
                for name, expected in gradients.items():
                    observed=torch.tensor(actual['gradients'][name]).reshape(expected.shape)
                    torch.testing.assert_close(observed,expected,atol=3e-6,rtol=3e-3,msg=lambda msg:f'{backend} step {step} {name}: {msg}')
                    errors[name]=(observed-expected).abs().max().item()
                saved=output/f'{backend}-step-{step+1}'
                worker.request('save',directory=str(saved))
                restored=load_archive(saved,root)
                weight_errors={}
                for name, expected in oracle.named_parameters():
                    observed=dict(restored.named_parameters())[name]
                    # Adam amplifies near-zero sign changes; use an absolute bound below a learning-rate step.
                    torch.testing.assert_close(observed,expected,atol=2e-5,rtol=2e-4,msg=lambda msg:f'{backend} weight {name}: {msg}')
                    weight_errors[name]=(observed-expected).abs().max().item()
                scheduler.step()
                rates=worker.request('schedule',update=step+1)
                np.testing.assert_allclose([rates['policy_learning_rate'],rates['value_learning_rate']],scheduler.get_last_lr(),rtol=2e-6)
                result=dict(step=step+1,metrics=actual['metrics'],gradient_max_absolute_error=max(errors.values()),
                            weight_max_absolute_error=max(weight_errors.values()),parameter_count=len(errors),
                            gradient_errors=errors,weight_errors=weight_errors,
                            gradient_norm_before_clip=actual['gradient_norm_before_clip'],compiled_shapes=actual['compiled_shapes'])
                if 'cuda_graphs' in actual:
                    result['cuda_graphs'] = actual['cuda_graphs']
                results.append(result)
                print(backend,step+1,json.dumps({k:v for k,v in result.items() if not k.endswith('_errors')}),flush=True)
        report['backends'][backend]=dict(steps=results,elapsed_seconds=time.monotonic()-start,
            reduction_mode='deterministic_tree_f32' if native_settings.get('tree_reductions') else 'strict_f32',
            graph_replay=native_settings.get('cuda_graphs',False))
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--forty-five-root',type=Path,required=True)
    parser.add_argument('--worker',type=Path,default=Path('target/release/examples/forty_five_worker'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backends',nargs='+',choices=['cpu','cuda'],default=['cpu','cuda'])
    parser.add_argument('--tiny',action='store_true')
    parser.add_argument('--steps',type=int,default=2)
    parser.add_argument('--cuda-graphs',action='store_true',help='enable private-stream model replay on CUDA')
    parser.add_argument('--tree-reductions',action='store_true',help='use fixed f32 reduction trees on CUDA')
    args=parser.parse_args()
    validate(args.source,args.forty_five_root,args.worker,args.output,args.backends,args.tiny,args.steps,args.tree_reductions,args.cuda_graphs)
if __name__=='__main__':main()
