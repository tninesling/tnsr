"""Run forty-five's Rust self-play collector with native tnsr inference and PPO."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from bridge import Worker, Policy
from export import export_model, load_model, load_archive
from validate import observations, infer_reference, compare
from validate_ppo import anchor_outputs
from restore import restore, validate_serving


def train_update(worker, batch, settings, *, epochs, hands_per_minibatch, anchor):
    """Match original hand grouping/shuffle; normalize once for the full rollout."""
    import torch
    from forty_five_ai.train import normalize_advantages
    advantages=normalize_advantages(batch['advantages'])
    hands={}
    for index,hand in enumerate(batch['hand_ids'].tolist()):
        hands.setdefault(hand,[]).append(index)
    keys=list(hands)
    metrics=[]
    for _ in range(epochs):
        shuffled=[keys[index] for index in torch.randperm(len(keys)).tolist()]
        for start in range(0,len(keys),hands_per_minibatch):
            rows=torch.tensor([index for hand in shuffled[start:start+hands_per_minibatch] for index in hands[hand]])
            native=dict(observations=dict(token_ids=batch['token_ids'][rows].tolist(),
                       token_type_ids=batch['token_type_ids'][rows].tolist(),phase_ids=batch['phases'][rows].tolist()),
                       action_masks=batch['action_masks'][rows].tolist(),actions=batch['actions'][rows].tolist(),
                       old_log_probs=batch['log_probs'][rows].tolist(),old_values=batch['values'][rows].tolist(),
                       returns=batch['returns'][rows].tolist(),advantages=advantages[rows].tolist())
            if settings['anchor_kl_coef']>0:
                native['anchor']=anchor_outputs(anchor,native)
            result=worker.request('train',batch=native,settings=settings)
            result['microbatches']=math.ceil(len(rows)/settings['microbatch'])
            metrics.append(result)
    return metrics


def run(args):
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(args.forty_five_root/'python'))
    from forty_five_ai.env import FortyFiveSelfPlayRunnerAdapter
    from forty_five_ai.train import collect_rollout_rust
    from forty_five_ai.eval import run_batched_policy_evaluation
    args.output.mkdir(parents=True,exist_ok=False)
    initial=load_archive(args.source,args.forty_five_root) if args.source.is_dir() else load_model(args.source,args.forty_five_root)
    if initial.use_privileged_critic or initial.use_rope or not initial.use_swiglu:
        raise ValueError('requires learned-position SwiGLU with a legal-information critic')
    initial.eval()
    archive=args.output/'initial';manifest=export_model(initial,archive)
    parameter_bytes=4*sum(record['count'] for record in manifest['tensors'])
    opponent=initial
    if args.opponent_snapshot:
        opponent=(load_archive(args.opponent_snapshot,args.forty_five_root) if args.opponent_snapshot.is_dir()
                  else load_model(args.opponent_snapshot,args.forty_five_root))
        opponent.eval()
    device=torch.device('cpu')
    runner=FortyFiveSelfPlayRunnerAdapter(num_envs=args.num_envs,seed=args.seed)
    report=dict(source=str(args.source.resolve()),backend=args.backend,seed=args.seed,
                separate_critic=initial.use_separate_critic,torch_version=torch.__version__,parameter_bytes=parameter_bytes,updates=[],
                mode='synchronous f32; resident CUDA Adam, host CPU Adam, JSON/host observations',
                reduction_mode='deterministic_tree_f32' if args.tree_reductions else 'strict_f32',
                graph_replay=args.cuda_graphs,
                matmul_precision="tf32x3" if args.tf32x3 else "strict_f32",
                transfer_accounting='training payload lower bounds; excludes observations, inference, archive saves and diagnostics',
                arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()})
    if args.source.is_file(): report['source_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest()
    settings=dict(learning_rate=args.learning_rate,value_learning_rate=args.value_learning_rate,
                  entropy_coef=args.entropy_coef,anchor_kl_coef=args.anchor_kl_coef,
                  microbatch=args.microbatch,scheduler_updates=args.updates)
    if args.tree_reductions:
        settings['tree_reductions'] = True
    if args.cuda_graphs:
        settings['cuda_graphs'] = True
    if args.tf32x3:
        settings['tf32x3'] = True
    with Worker(args.worker,archive,args.backend) as worker:
        policy=Policy(worker)
        for update in range(args.updates):
            torch.manual_seed(args.seed+update)
            began=time.monotonic()
            batch,stats=collect_rollout_rust(runner,policy,args.hands_per_update,device,args.seed+update,
                fixed_opponent=opponent if args.fixed_opponent_prob>0 else None,
                fixed_opponent_prob=args.fixed_opponent_prob,rollout_index=update)
            rollout_seconds=time.monotonic()-began
            assert stats.completed_hands>=args.hands_per_update
            assert all(mask[action] for mask,action in zip(batch['action_masks'],batch['actions'],strict=True))
            assert all(torch.isfinite(batch[key]).all() for key in ['values','log_probs','advantages','returns'])
            baseline_seconds=None;baseline_error=None
            if args.compare_rollout and update==0:
                torch.manual_seed(args.seed)
                oracle_runner=FortyFiveSelfPlayRunnerAdapter(num_envs=args.num_envs,seed=args.seed)
                began=time.monotonic()
                oracle,oracle_stats=collect_rollout_rust(oracle_runner,initial,args.hands_per_update,device,args.seed,
                    fixed_opponent=opponent if args.fixed_opponent_prob>0 else None,
                    fixed_opponent_prob=args.fixed_opponent_prob,rollout_index=0)
                baseline_seconds=time.monotonic()-began
                if stats!=oracle_stats: raise AssertionError(f'rollout stats disagree: {stats} / {oracle_stats}')
                for key in ['token_ids','token_type_ids','hand_ids','phases','action_masks','actions']:
                    torch.testing.assert_close(batch[key],oracle[key],atol=0,rtol=0)
                baseline_error={}
                for key in ['values','log_probs','advantages','returns']:
                    torch.testing.assert_close(batch[key],oracle[key],atol=5e-5,rtol=5e-4)
                    baseline_error[key]=(batch[key]-oracle[key]).abs().max().item()
            # Keep coefficients dynamic so an entropy schedule does not rebuild graphs.
            fraction=update/max(args.updates-1,1)
            settings['entropy_coef']=args.entropy_coef+(args.final_entropy_coef-args.entropy_coef)*fraction
            torch.manual_seed(args.seed+10000+update)
            began=time.monotonic()
            metrics=train_update(worker,batch,settings,epochs=args.epochs,
                                 hands_per_minibatch=args.hands_per_minibatch,anchor=initial)
            ppo_seconds=time.monotonic()-began
            rates=worker.request('schedule',update=update+1)
            saved=args.output/f'update-{update+1:04d}';worker.request('save',directory=str(saved))
            restored=load_archive(saved,args.forty_five_root)
            probe,masks=observations(args.forty_five_root)
            logits,values=infer_reference(restored,probe)
            interchange=compare(worker.request('infer',batch=probe),logits,values,masks)
            with Worker(args.worker,saved,args.backend) as fresh:
                compare(fresh.request('infer',batch=probe),logits,values,masks)
            record=dict(update=update+1,transitions=len(batch['actions']),phases=sorted(batch['phases'].unique().tolist()),
                        rollout=asdict(stats),rollout_seconds=rollout_seconds,ppo_seconds=ppo_seconds,
                        torch_rollout_seconds=baseline_seconds,rollout_max_absolute_errors=baseline_error,
                        optimizer_updates=metrics,learning_rates=rates,
                        cuda_parameter_upload_bytes_lower_bound=parameter_bytes if args.backend=='cuda' and update==0 else 0,
                        cuda_gradient_download_bytes_lower_bound=0,checkpoint=str(saved.resolve()),checkpoint_parity=interchange)
            report['updates'].append(record)
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({key:value for key,value in record.items() if key!='optimizer_updates'}),flush=True)
        began=time.monotonic()
        report['evaluation']=run_batched_policy_evaluation(policy,opponent,device=device,seed=args.seed+20000,
            games=args.eval_games,max_steps_per_episode=args.max_eval_steps)
        report['evaluation_seconds']=time.monotonic()-began
        if report['evaluation']['games']!=args.eval_games: raise AssertionError('evaluation did not complete')
        if report['evaluation']['early_terminations']!=0: raise AssertionError('evaluation truncated games')
        # Greedy native and restored PyTorch arena outcomes must agree on the same deals/seats.
        began=time.monotonic()
        report['pytorch_evaluation']=run_batched_policy_evaluation(restored,opponent,device=device,seed=args.seed+20000,
            games=args.eval_games,max_steps_per_episode=args.max_eval_steps)
        report['pytorch_evaluation_seconds']=time.monotonic()-began
        if report['evaluation']!=report['pytorch_evaluation']: raise AssertionError('native and restored arena results disagree')
        checkpoint=args.output/'final.pt'
        onnx=args.output/'final.onnx' if args.export_onnx else None
        serving=restore(saved,args.forty_five_root,checkpoint,onnx)
        report['pytorch_checkpoint']=str(checkpoint.resolve())
        if onnx:
            report['onnx_parity']=validate_serving(serving,args.forty_five_root,onnx)
            report['onnx']=str(onnx.resolve())
        report['worker_io']=worker.timing
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print('evaluation',json.dumps(report['evaluation']),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--forty-five-root',type=Path,required=True)
    parser.add_argument('--worker',type=Path,default=Path('target/release/examples/forty_five_worker'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--updates',type=int,default=1)
    parser.add_argument('--num-envs',type=int,default=2)
    parser.add_argument('--hands-per-update',type=int,default=2)
    parser.add_argument('--hands-per-minibatch',type=int,default=2)
    parser.add_argument('--microbatch',type=int,default=8)
    parser.add_argument('--tf32x3',action='store_true',help='opt into corrected TF32 matmuls with f32 storage and optimizer')
    parser.add_argument('--cuda-graphs',action='store_true',help='enable private-stream model replay on CUDA')
    parser.add_argument('--tree-reductions',action='store_true',help='opt into fixed f32 reduction trees on CUDA')
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--learning-rate',type=float,default=5e-5)
    parser.add_argument('--value-learning-rate',type=float,default=2e-4)
    parser.add_argument('--entropy-coef',type=float,default=.005)
    parser.add_argument('--final-entropy-coef',type=float,default=.005)
    parser.add_argument('--anchor-kl-coef',type=float,default=.03)
    parser.add_argument('--fixed-opponent-prob',type=float,default=0.)
    parser.add_argument('--opponent-snapshot',type=Path,help='frozen .pt/.onnx checkpoint or native archive; defaults to initialization')
    parser.add_argument('--eval-games',type=int,default=2)
    parser.add_argument('--max-eval-steps',type=int,default=10000)
    parser.add_argument('--compare-rollout',action='store_true')
    parser.add_argument('--export-onnx',action='store_true')
    args=parser.parse_args()
    if (args.tree_reductions or args.cuda_graphs or args.tf32x3) and args.backend != 'cuda':
        parser.error('tree reductions, CUDA Graphs and TF32x3 require --backend cuda')
    for field in ['updates','num_envs','hands_per_update','hands_per_minibatch','microbatch','epochs','max_eval_steps']:
        if getattr(args,field)<1:parser.error(f'{field} must be positive')
    for field in ['entropy_coef','final_entropy_coef','anchor_kl_coef','learning_rate','value_learning_rate']:
        if not math.isfinite(getattr(args,field)) or getattr(args,field)<0:parser.error(f'{field} must be finite and nonnegative')
    if args.eval_games<2 or args.eval_games%2:parser.error('eval-games must be positive and even to balance opposite seat pairs')
    if not 0<=args.fixed_opponent_prob<=1:parser.error('fixed-opponent-prob must be in [0,1]')
    run(args)
if __name__=='__main__':main()
