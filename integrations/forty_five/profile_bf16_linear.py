"""Validate and time complete physical-BF16 linear islands, before Torch imports.

The native probe alternates TF32x3/BF16 complete calls, including casts, all
three matmuls, bias reduction, promotion and f32 gradient accumulation. The
TF32x3 comparator retains four-part weight-gradient contractions when eligible.
This is an isolated operation gate, not a full-model autocast implementation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import numpy as np


def status():
    return dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,clocks.sm,power.draw','--format=csv,noheader,nounits'],text=True).strip(),
                compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip())


def native(args,shape,branches,regime):
    m,k,n=(shape[x] for x in ['rows','inputs','outputs'])
    # Explicit boundary buffers, plus the f32 comparator's eligible partials.
    base_bytes=4*(2*m*k+3*n*k+3*n+2*m*n+(2*m*n if branches==2 else 0))
    replicas=1 if regime=='hot' else 1<<(math.ceil(2*64*1024**2/base_bytes)-1).bit_length()
    label=f'{m}-{k}-{n}-fanout{branches}-{regime}';out=args.output/label;out.mkdir()
    spec=dict(shape=shape,branches=branches,replay=True,replicas=replicas)
    (out/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    rng=np.random.default_rng(20261016)
    values=dict(x=(rng.standard_normal(m*k)*.1).astype(np.float32),weight=(rng.standard_normal(n*k)*.1).astype(np.float32),bias=(rng.standard_normal(n)*.01).astype(np.float32),upstream=[(rng.standard_normal(m*n)*.1).astype(np.float32) for _ in range(branches)])
    arrays={key:value.copy() for key,value in values.items() if key in ['weight','bias']}
    moments={key:[np.zeros_like(v),np.zeros_like(v)] for key,v in arrays.items()}
    steps=5 if branches==2 else 1
    report=dict(label=label,spec=spec,before=status(),validation_steps=[],notes=['Timing excludes upload/readback and CPU optimizer work.','CPU f32 Adam below is an isolated multistep boundary diagnostic, not the resident GPU optimizer.'])
    stderr=(out/'worker.log').open('w')
    process=subprocess.Popen([str(args.worker.resolve()),str((out/'spec.json').resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True)
    def request(payload):
        process.stdin.write(json.dumps(payload,allow_nan=False)+'\n');process.stdin.flush()
        line=process.stdout.readline()
        if not line:raise RuntimeError(f'linear probe exited ({process.poll()}), see {out}/worker.log')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    try:
        info=request(dict(op='info'));(out/'modules.json').write_text(json.dumps(info,indent=2)+'\n')
        report['info']=info['info']
        # Preserve binary boundaries/outputs for a separate, untimed Torch pass.
        for step in range(steps):
            inputs=dict(x=values['x'].tolist(),weight=arrays['weight'].tolist(),bias=arrays['bias'].tolist(),upstream=[x.tolist() for x in values['upstream']])
            result=request(dict(op='run',inputs=inputs,clear=True))['results']
            files={}
            for key in ['x','weight','bias']:
                path=out/f'step{step}-{key}.npy';np.save(path,np.asarray(inputs[key],dtype=np.float32));files[key]=path.name
            for index,v in enumerate(values['upstream']):np.save(out/f'step{step}-upstream{index}.npy',v)
            for mode,row in zip(['tf32x3','bf16'],result):
                for key,v in row.items():
                    v=np.asarray(v,dtype=np.float32)
                    if not np.isfinite(v).all():raise AssertionError((label,mode,key,'nonfinite'))
                    np.save(out/f'step{step}-{mode}-{key}.npy',v)
                for name in ['weight','bias']:
                    if row[name+'_gradient']!=row['accumulated_'+name+'_gradient']:raise AssertionError('fresh f32 accumulator differs')
            # Native BF16 gradients drive five genuine f32 master/state updates.
            for key in arrays:
                grad=np.asarray(result[1][key+'_gradient'],dtype=np.float32);moment,variance=moments[key]
                moment*=np.float32(.9);moment+=grad*np.float32(.1)
                variance*=np.float32(.999);variance+=grad*grad*np.float32(.001)
                arrays[key]-=np.float32(.001)*(moment/np.float32(1-.9**(step+1)))/(np.sqrt(variance/np.float32(1-.999**(step+1)))+np.float32(1e-8))
                np.save(out/f'step{step}-next-{key}.npy',arrays[key]);np.save(out/f'step{step}-moment-{key}.npy',moment);np.save(out/f'step{step}-variance-{key}.npy',variance)
            report['validation_steps'].append(dict(step=step,inputs=files))
        report['timing']=request(dict(op='bench',samples=args.samples,repeats=32 if replicas==1 else 1))
        if regime=='streaming' and min(report['timing']['physical_buffer_bytes'])<2*64*1024**2:raise AssertionError('rotating working set smaller than twice L2')
        report['median_us']=[statistics.median(times) for times in report['timing']['samples_us']]
        a,b=report['timing']['samples_us'];report['median_paired_ratio']=statistics.median(x/y for x,y in zip(a,b));report['bf16_wins']=sum(y<x for x,y in zip(a,b))
        report['after']=status();(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(label,report['median_us'],report['median_paired_ratio'],flush=True)
    finally:
        process.stdin.close();code=process.wait(timeout=30);stderr.close()
        if code:raise RuntimeError(f'probe exit {code}')
    return report


def validate(args,rows):
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    def errors(actual,reference):
        actual=actual.detach().float().cpu().flatten();reference=reference.detach().float().cpu().flatten();d=(actual-reference).abs()
        return dict(max_abs=float(d.max()),normalized_max=float(d.max())/max(1.,float(reference.abs().max())),rmse=float(d.square().mean().sqrt()),exact_fraction=float((actual==reference).float().mean()))
    for row in rows:
        out=args.output/row['label'];s=row['spec']['shape'];m,k,n=(s[x] for x in ['rows','inputs','outputs']);branches=row['spec']['branches'];checks=[]
        load=lambda step,name:torch.from_numpy(np.load(out/f'step{step}-{name}.npy')).cuda()
        independent={name:load(0,name).reshape((n,k) if name=='weight' else (n,)).clone().requires_grad_() for name in ['weight','bias']}
        optimizer=torch.optim.Adam(independent.values(),lr=.001)
        for step in range(len(row['validation_steps'])):
            x=load(step,'x').reshape(m,k).requires_grad_();w=load(step,'weight').reshape(n,k).requires_grad_();b=load(step,'bias').requires_grad_()
            seeds=[load(step,f'upstream{i}').reshape(m,n) for i in range(branches)]
            with torch.autocast('cuda',dtype=torch.bfloat16):y=torch.nn.functional.linear(x,w,b)
            torch.autograd.backward([y.float() for _ in seeds],seeds)
            original=dict(output=y,input_gradient=x.grad,weight_gradient=w.grad,bias_gradient=b.grad)
            xq,wq,bq=x.detach().bfloat16().float(),w.detach().bfloat16().float(),b.detach().bfloat16().float()
            g=seeds[0].bfloat16()
            for contribution in seeds[1:]:g=(g+contribution.bfloat16()).bfloat16()
            gf=g.float()
            declared=dict(output=torch.addmm(bq,xq,wq.T).bfloat16(),input_gradient=(gf@wq).bfloat16(),weight_gradient=(gf.T@xq).bfloat16(),bias_gradient=gf.sum(0).bfloat16())
            f32=dict(output=torch.addmm(b.detach(),x.detach(),w.detach().T))
            seed_sum=sum(seeds);f32.update(input_gradient=seed_sum@w.detach(),weight_gradient=seed_sum.T@x.detach(),bias_gradient=seed_sum.sum(0))
            comparisons={mode:{} for mode in ['original_default_bf16','quantized_f32_accumulation','tf32x3_vs_f32']}
            for key in declared:
                actual=load(step,'bf16-'+key);comparisons['original_default_bf16'][key]=errors(actual,original[key]);comparisons['quantized_f32_accumulation'][key]=errors(actual,declared[key])
                comparisons['tf32x3_vs_f32'][key]=errors(load(step,'tf32x3-'+key),f32[key])
            # A sanity flag is not a calibrated original-model tolerance.
            comparisons['declared_primitive_sanity']=all(v['normalized_max']<=1/128 for v in comparisons['quantized_f32_accumulation'].values())
            if not comparisons['declared_primitive_sanity']:raise AssertionError((row['label'],step,comparisons))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):independent_y=torch.nn.functional.linear(x.detach(),independent['weight'],independent['bias'])
            torch.autograd.backward([independent_y.float() for _ in seeds],seeds);optimizer.step()
            comparisons['independent_original_adam']={}
            for name,param in independent.items():
                comparisons['independent_original_adam'][name]=dict(weight=errors(load(step,'next-'+name),param),moment=errors(load(step,'moment-'+name),optimizer.state[param]['exp_avg']),variance=errors(load(step,'variance-'+name),optimizer.state[param]['exp_avg_sq']))
            checks.append(comparisons)
        row['oracle']=checks;row['original_training_policy_calibrated']=False
        print('ORACLE',row['label'],'original max',max(v['max_abs'] for step in checks for v in step['original_default_bf16'].values()),'Adam max',max(v['weight']['max_abs'] for step in checks for v in step['independent_original_adam'].values()),flush=True)
    return dict(torch_version=torch.__version__,allow_tf32=torch.backends.cuda.matmul.allow_tf32,allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


def reduction_diagnostic(args,row):
    import torch
    p=args.output/row["label"]
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    x=torch.from_numpy(np.load(p/'step0-x.npy')).cuda().reshape(row['spec']['shape']['rows'],row['spec']['shape']['inputs']).bfloat16();g=torch.from_numpy(np.load(p/'step0-upstream0.npy')).cuda().reshape(row['spec']['shape']['rows'],row['spec']['shape']['outputs']).bfloat16()
    a=torch.from_numpy(np.load(p/'step0-bf16-weight_gradient.npy')).cuda().reshape(row['spec']['shape']['outputs'],row['spec']['shape']['inputs'])
    flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    refs={}
    try:
     for name,reduced in [('original_default',True),('no_reduced_diagnostic',False)]:
      torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced;refs[name]=(g.T@x).float()
     refs['quantized_f32']=(g.float().T@x.float()).bfloat16().float()
     refs['quantized_f64']=(g.double().T@x.double()).bfloat16().float()
    finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
    r={}
    for name,ref in refs.items():
     d=(a-ref).abs();flip=(a*ref<0);r[name]=dict(max_abs=float(d.max()),rmse=float(d.square().mean().sqrt()),exact_fraction=float((a==ref).float().mean()),sign_flips=int(flip.sum()),native_max_abs_at_flips=float(a[flip].abs().max()) if flip.any() else 0,reference_max_abs_at_flips=float(ref[flip].abs().max()) if flip.any() else 0)
     if flip.any():
      coords=flip.nonzero()[:8];r[name]['examples']=[dict(index=c.tolist(),native=float(a[tuple(c)]),reference=float(ref[tuple(c)]),f64=float(refs['quantized_f64'][tuple(c)])) for c in coords]
    a0=refs['original_default'];b=refs['no_reduced_diagnostic'];r['original_default_vs_original_no_reduced']=dict(sign_flips=int((a0*b<0).sum()),max_abs=float((a0-b).abs().max()))
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path,default=Path('target/release/examples/bf16_linear_probe'));p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=200);p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    if args.samples<20:p.error('need at least 20 samples')
    if args.validate_only:report=json.loads((args.output/'report.json').read_text())
    else:
        args.output.mkdir(exist_ok=False);report=dict(worker_sha256=hashlib.sha256(args.worker.read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),before=status(),cases=[],limits=['Isolated complete linear calls, not model throughput or quality parity.','All timed native processes finish before Torch import/oracles; agent-owned GPU experiments are serialized and no builds overlap timing. External compute processes are recorded and left running.','Physical memory includes retained f32 masters/boundaries plus BF16 buffers; this isolation does not claim whole-model memory savings.'])
        for m,k,n,branches in [(4096,683,256,1),(4096,256,683,1),(512,683,256,1),(33,65,35,2)]:
            for regime in ['hot','streaming'] if m>33 else ['hot']:
                report['cases'].append(native(args,dict(rows=m,inputs=k,outputs=n),branches,regime));(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['oracle']=validate(args,report['cases']);report['reduction_diagnostic']=reduction_diagnostic(args,report['cases'][0]);report['validation_driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();report['after']=status();(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
