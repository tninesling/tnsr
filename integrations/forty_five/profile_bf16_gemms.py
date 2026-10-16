"""Measure precision headroom separately from model-policy implementation.

All timed work runs in one native probe process per case, with preallocated
storage, captured repeated operations, CUDA events and balanced mode order.
PyTorch differential checks run only after every timed probe has completed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


def status():
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,clocks.sm,power.draw','--format=csv,noheader,nounits'],text=True).strip()
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
    return dict(gpu=gpu,compute_processes=processes)


def cases(include_unsplit=False):
    # Large shapes are taken from the post-#127 shared-model plan. Attention's
    # QKV view maps and fused pointwise epilogues are deliberately not modeled.
    rows = [
        ('ffn-down-wgrad-partial',4,683,1024,256,True,False,False),
        ('ffn-down-forward',1,4096,683,256,False,True,True),
        ('ffn-gate-up-wgrad-partial',4,256,1024,683,True,False,False),
        ('ffn-gate-up-forward',1,4096,256,683,False,True,True),
        ('ffn-gate-up-input-grad',1,4096,683,256,False,False,False),
        ('qkv-forward',1,4096,256,768,False,True,True),
        ('qkv-wgrad-partial',4,256,1024,768,True,False,False),
        ('qkv-input-grad',1,4096,768,256,False,False,False),
        ('micro8-ffn-wgrad',1,683,512,256,True,False,False),
        ('micro8-ffn-down',1,512,683,256,False,True,True),
        ('ffn-down-wgrad-unsplit',1,683,4096,256,True,False,False),
    ]
    if not include_unsplit:rows=rows[:-1]
    return [dict(label=label,batch=b,m=m,k=k,n=n,transpose_lhs=tl,transpose_rhs=tr,shared_rhs=False,bias=bias) for label,b,m,k,n,tl,tr,bias in rows]


def differential(shape, directory):
    import numpy as np
    import torch
    b,m,k,n = (shape[key] for key in ['batch','m','k','n'])
    rb = 1 if shape['shared_rhs'] else b
    def values(count,period):
        return ((torch.arange(count,dtype=torch.int64)%period).float()/period-.5)
    a=values(b*m*k,29).reshape((b,k,m) if shape['transpose_lhs'] else (b,m,k)).cuda()
    w=values(rb*k*n,31).reshape((rb,n,k) if shape['transpose_rhs'] else (rb,k,n)).cuda()
    if shape['transpose_lhs']:a=a.transpose(-2,-1)
    if shape['transpose_rhs']:w=w.transpose(-2,-1)
    bias=torch.full((n,),.1,device='cuda') if shape['bias'] else None
    def compute(a,w,bias):
        if bias is not None:
            return torch.stack([torch.addmm(bias,a[index],w[0 if rb==1 else index]) for index in range(b)])
        return torch.matmul(a,w)
    def statistics_for(actual,reference):
        reference=reference.float().cpu().reshape(-1)
        delta=(actual-reference).abs()
        bits=lambda tensor:tensor.bfloat16().view(torch.int16).to(torch.int32)
        def ordered(tensor):
            raw=bits(tensor);return torch.where(raw<0,-32768-raw,raw)
        ulps=(ordered(actual)-ordered(reference)).abs()
        return dict(max_abs_error=float(delta.max()),normalized_max_error=float(delta.max())/max(1.,float(reference.abs().max())),
                    rmse=float(delta.square().mean().sqrt()),exact_fraction=float((actual==reference).float().mean()),
                    max_ulp=int(ulps.max()),more_than_one_ulp=int((ulps>1).sum()))
    actual=torch.from_numpy(np.fromfile(directory/'tf32x3.f32',dtype='<f4'))
    observed=torch.from_numpy(np.fromfile(directory/'bf16.f32',dtype='<f4'))
    reference=compute(a,w,bias)
    f32=statistics_for(actual,reference)
    f32_sanity=f32['normalized_max_error']<=2e-5
    ab,wb=a.bfloat16(),w.bfloat16()
    bb=None if bias is None else bias.bfloat16()
    default_flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    original=statistics_for(observed,compute(ab,wb,bb))
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        no_reduced=statistics_for(observed,compute(ab,wb,bb))
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=default_flag
    full=statistics_for(observed,compute(ab.float(),wb.float(),None if bb is None else bb.float()).bfloat16())
    # Declared probe semantics: BF16 operands, f32 accumulation, one BF16 output
    # rounding. Original default cuBLAS can truncate intermediate reductions;
    # retain that discrepancy instead of silently disabling its policy.
    if full['normalized_max_error']>1/128:raise AssertionError(('BF16 f32-accumulation differential',shape,full))
    return dict(tf32x3=f32,tf32x3_within_probe_sanity=f32_sanity,bf16_original_default=original,bf16_no_reduced_precision=no_reduced,
                bf16_f32_accumulation=full,original_default_within_probe_sanity=original['normalized_max_error']<=1/128,
                original_training_policy_calibrated=False,elements=actual.numel())


def run(args):
    if args.validate_only:
        report=json.loads((args.output/'report.json').read_text())
    else:
        args.output.mkdir(parents=True,exist_ok=False)
    report=report if args.validate_only else dict(probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),gpu_before=status(),cases=[],
                notes=['Timed probes precede all PyTorch imports/CUDA oracle operations.',
                       'Dense GEMM dimensions/layouts match selected current-plan families, with synthetic deterministic values.',
                       'Bias is included where indicated, but other model epilogues, attention QKV views, cross-operation reuse, backward casts, and final split-K sums are not represented.',
                       'Hot mode reuses one allocation set; streaming rotates at least twice L2 worth of BF16 plan allocations.',
                       'These measurements are precision headroom, not training throughput, AMP parity or learning-quality results.'])
    for index,shape in enumerate([] if args.validate_only else cases(args.include_unsplit_diagnostic)):
        regimes=['hot','streaming']
        l2=64*1024*1024
        for regime in regimes:
            elements=shape['batch']*shape['m']*shape['k']+shape['batch']*shape['k']*shape['n']+shape['batch']*shape['m']*shape['n']+(shape['n'] if shape['bias'] else 0)
            replicas=1 if regime=='hot' else 1<<(max(1,math.ceil(2*l2/(2*elements)))-1).bit_length()
            repeats=replicas*math.ceil(args.graph_repeats/replicas)
            spec=dict(shape=shape,samples=args.samples,graph_repeats=repeats,replicas=replicas)
            name=shape['label']+'-'+regime;path=args.output/(name+'.json');path.write_text(json.dumps(spec,indent=2)+'\n')
            before=status();began=time.monotonic()
            with (args.output/(name+'.log')).open('w') as log:
                subprocess.run([str(args.probe.resolve()),str(path.resolve()),str((args.output/name).resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((args.output/name/'report.json').read_text());l2=result['device_l2_bytes']
            if regime=='streaming' and result['physical_plan_bytes'][1]<2*l2:raise AssertionError('streaming BF16 working set is smaller than twice L2')
            if result['kernels_per_operation'][0]!=1 or result['kernels_per_operation'][1]!=1:raise AssertionError('probe GEMM/bias did not fuse into one kernel')
            row=dict(label=shape['label'],regime=regime,spec=spec,seconds=time.monotonic()-began,before=before,after=status(),probe=result)
            report['cases'].append(row);(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,'us',result['median_us'],'replicas',replicas,flush=True)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    report['oracle_driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['oracle']=dict(torch_version=torch.__version__,allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                          allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    for row in report['cases']:
        name=row['label']+'-'+row['regime'];row['differential']=differential(row['spec']['shape'],args.output/name)
        print('ORACLE',name,row['differential'],flush=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['failed_tf32x3_sanity_cases']=[row['label']+'-'+row['regime'] for row in report['cases'] if not row['differential']['tf32x3_within_probe_sanity']]
    report['original_default_disagreement_cases']=[row['label']+'-'+row['regime'] for row in report['cases'] if row['differential']['bf16_original_default']['exact_fraction']<1.]
    report['gpu_after']=status();(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,default=Path('target/release/examples/bf16_gemm_probe'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-unsplit-diagnostic',action='store_true',help='also profile the unsplit K4096 contraction, which is absent from the current large-model path')
    parser.add_argument('--validate-only',action='store_true',help='check existing outputs without repeating or replacing timed samples')
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--graph-repeats',type=int,default=32)
    args=parser.parse_args()
    if args.samples<20 or args.graph_repeats<1:parser.error('need >=20 samples and positive graph repeats')
    run(args)


if __name__=='__main__':main()
