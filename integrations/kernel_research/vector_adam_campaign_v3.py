#!/usr/bin/env python3
"""Guarded fresh-process vector Adam component controller; no optimizer API.

Default mode is untimed selfqualification. Timing requires explicit root-reserved
execution and the source-bound untimed qualification. run.execute owns monitoring;
this controller owns the one campaign lock and frozen desktop descriptors.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integrations/transformer_bench'))
import run as campaign
import desktop_contexts

PYTHON=Path('/home/taylor/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12')
ENV={'PYTHONPATH':'/home/taylor/code/forty-five/.venv/lib/python3.12/site-packages','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MAX_JOBS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'}
WORKER=ROOT/'integrations/kernel_research/vector_adam_bridge_v3.py'

def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def source_hashes():
    return {str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__).resolve(),WORKER,Path(campaign.__file__).resolve(),Path(desktop_contexts.__file__).resolve(),ROOT/'integrations/kernel_research/non_nn_bridge.py',ROOT/'integrations/kernel_research/non_nn_calibration.py']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timing',action='store_true')
    parser.add_argument('--qualified',type=Path)
    parser.add_argument('--root-performance-granted',action='store_true')
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=PYTHON.resolve() or any(os.environ.get(k)!=v for k,v in ENV.items()):
        raise ValueError('use exact preflight Python/PYTHONPATH/thread2 environment')
    manifest=args.archive/'manifest.json';snapshot=source_hashes()
    if args.timing:
        if not args.root_performance_granted or args.qualified is None:
            raise ValueError('timing requires source-bound untimed proof and explicit root reservation')
        q=json.loads(args.qualified.read_text())
        if not q['qualified']or q['timing']or q['source_sha256']!=snapshot or q['archive_manifest_sha256']!=sha(manifest):
            raise ValueError('untimed qualification differs from timing controller/worker/helpers/archive')
    elif args.root_performance_granted:
        raise ValueError('performance grant flag belongs only to timing run')
    if args.output.exists():raise ValueError('fresh output required, preserve failures')
    args.output.mkdir(parents=True)
    check=subprocess.run([str(PYTHON),'-c','import torch,numpy,json,sys;print(json.dumps(dict(python=sys.executable,torch=torch.__version__,numpy=numpy.__version__)))'],capture_output=True,text=True,check=True,env=os.environ.copy())
    if args.timing and json.loads(check.stdout)!=q['python_preflight']:
        raise ValueError('Python/Torch/NumPy version differs from untimed proof')
    report={'scope':__doc__,'timing':args.timing,'performance_claim':False,'source_sha256':snapshot,'archive_manifest_sha256':sha(manifest),'python_preflight':json.loads(check.stdout),'environment':ENV,'workers':[],'qualified':False,'archive':str(args.archive.resolve()),'desktop_policy':desktop_contexts.POLICY}
    report_file=args.output/'report.json'
    save=lambda:report_file.write_text(json.dumps(report,indent=2)+'\n')
    save()
    with open('/tmp/tnsr-bench.cuda0.lock','a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Exactly once per campaign, never silently rebaseline a worker.
        expected=desktop_contexts.freeze(args.output)
        report['frozen_desktop_contexts']=list(expected.values());save()
        for index,reverse in enumerate([False,True]):
            if source_hashes()!=snapshot:raise ValueError('source changed after campaign freeze')
            directory=args.output/f'worker{index}'
            command=[str(PYTHON),str(WORKER),'run','--archive',str(args.archive.resolve()),'--output',str((directory/'artifacts').resolve())]
            if reverse:command.append('--reverse')
            if args.timing:command.append('--timing')
            print(json.dumps({'worker_start':index,'reverse':reverse,'timing':args.timing,'command':command}),flush=True)
            result=campaign.execute(command,directory,300,allow_idle_desktop_contexts=True,expected_desktop_contexts=expected)
            record={'index':index,'reverse':reverse,'process':result};report['workers'].append(record);save()
            if result['status']!='passed' or result['exit_code']!=0:
                raise RuntimeError('worker/desktop guard failed; preserve complete attempt')
            payload=json.loads((directory/'artifacts/report.json').read_text());record['component_result']=payload
            if not payload['qualified']or payload['timing']!=args.timing or payload['reverse']!=reverse:
                raise RuntimeError('worker qualification/mode mismatch')
            if args.timing:
                for case in payload['cases']:
                    if len(case['balanced_blocks'])!=36 or case['warmup_seconds']<.5:
                        raise RuntimeError('timing protocol incomplete')
            if source_hashes()!=snapshot:raise ValueError('source changed during worker')
            save();print(json.dumps({'worker_finish':index,'status':'passed'}),flush=True)
        report['qualified']=True;report['all_workers_terminal']=True;report['read_queues_closed']=True;save()
    print(json.dumps({'qualified':True,'fresh_workers':2,'timing':args.timing}),flush=True)

if __name__=='__main__':main()
