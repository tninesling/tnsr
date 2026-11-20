#!/usr/bin/env python3
"""Optional-timing v3: balanced output-slot rotation for native vector Adam.

Loads the exact fresh Rust leaf monolithic PTX without extracting/recompiling
entries. No optimizer-owner API, Torch Adam, or model claims. Timing is explicit opt-in and requires a separate root reservation.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

LENGTHS = (2818048, 3145728)
MODES = ('scalar_bf16_1024', 'scalar_bf16_256', 'vector_bf16_256')
ABI = ('weight', 'gradient', 'moment', 'variance', 'settings',
       'next_weight', 'next_moment', 'next_variance', 'health')
INPUTS = ('weight', 'gradient', 'moment', 'variance', 'settings')
OUTPUTS = ('next_weight', 'next_moment', 'next_variance', 'health')
SENTINEL_BITS = 0x4355AA55
CAPTURE_REPEATS = 32


def fingerprint(path):
    path = Path(path)
    before = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with path.open('rb') as stream:
        assert stamp(os.fstat(stream.fileno())) == stamp(before)
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert stamp(os.fstat(stream.fileno())) == stamp(before)
    assert stamp(path.stat()) == stamp(before)
    return {'path': str(path.resolve()), 'bytes': before.st_size, 'sha256': digest}


def entries(source):
    result = {}
    for name, header in re.findall(r'\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)', source, re.S):
        names = tuple(re.findall(r'\.param\s+\.u64\s+(\w+)', header))
        remainder = re.sub(r'\.param\s+\.u64\s+\w+', '', header)
        if re.sub(r'[,\s]', '', remainder) or name in result:
            raise ValueError('non-pointer-only or duplicate PTX entry')
        result[name] = names
    required = {'resident_accumulate_0', 'resident_adam_1',
                'resident_adam_bf16_first_2', 'resident_test_vector4_bf16_adam_3'}
    if set(result) != required:
        raise ValueError('not the exact four-entry Rust leaf module')
    for name in ('resident_adam_bf16_first_2', 'resident_test_vector4_bf16_adam_3'):
        if result[name] != ABI:
            raise ValueError(f'unexpected nine-pointer ABI: {name}')
    return result


def prepare(args):
    if args.archive.exists():
        raise ValueError('fresh v2 archive required; preserve v1')
    args.archive.mkdir(parents=True)
    original=json.loads((args.from_sealed/'manifest.json').read_text())
    manifest=dict(original)
    manifest['scope']=__doc__
    manifest['origin_sealed_manifest']=fingerprint(args.from_sealed/'manifest.json')
    manifest['controller']=fingerprint(Path(__file__))
    manifest['helpers']={name:fingerprint(Path(__file__).with_name(name)) for name in ['non_nn_bridge.py','non_nn_calibration.py']}
    manifest['cases']=[]
    for old in original['cases']:
        directory=args.archive/old['name'];directory.mkdir()
        case=dict(old);case['files']={}
        for key,record in old['files'].items():
            source=Path(record['path']);assert fingerprint(source)==record
            dest=directory/source.name;shutil.copyfile(source,dest)
            new=fingerprint(dest);assert new['sha256']==record['sha256']
            case['files'][key]=new
        for key,name in [('module','loaded-module.ptx'),('rust_resources','rust-resources.json')]:
            record=old[key];source=Path(record['path']);assert fingerprint(source)==record
            dest=directory/name;shutil.copyfile(source,dest)
            case[key]=fingerprint(dest);assert case[key]['sha256']==record['sha256']
        manifest['cases'].append(case)
    manifest['timing_protocol']={'blocks':36,'triplets_per_balanced_block':2,'samples_per_mode':72,'slots':3,'blocks_per_mode_slot':12,'samples_per_mode_slot':24,'captured_repeats':32,'minimum_warmup_seconds':.5,'fresh_workers':2,'reverse_cases_construction_and_first_order':True}
    (args.archive/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'prepared_v3':True,'data_identical_to_v1':True,'GPU_runs':0,'manifest':fingerprint(args.archive/'manifest.json')}))


def block_orders(index, reverse):
    first=tuple(reversed(MODES)) if bool(index%2)!=bool(reverse) else MODES
    return first,tuple(reversed(first))


def slot_for(mode,index,reverse):
    rotation=2-index%3 if reverse else index%3
    return (MODES.index(mode)+rotation)%3


def construction_pairs(reverse):
    forward=[(mode,slot)for mode in MODES for slot in range(3)]
    return list(reversed(forward))if reverse else forward


class AdamKernel:
    """Selected entry in an unchanged multi-entry module owned by generic Driver."""
    def __init__(self, bridge, torch, driver, module, source, name, length, block, resources, tensors):
        self.driver, self.tensors, self.torch = driver, tuple(tensors), torch
        if entries(source)[name] != ABI or len(self.tensors) != 9:
            raise ValueError('exact Adam pointer ABI required')
        ranges = []
        logical = (length, length, length, length, 10, length, length, length, length // 256)
        for i, (tensor, count) in enumerate(zip(self.tensors, logical)):
            dtype = torch.bfloat16 if i == 1 else torch.float32
            physical_count = count + (16 if i >= 5 else 0)
            if (tensor.dtype != dtype or tuple(tensor.shape) != (physical_count,)
                    or not tensor.is_contiguous() or tensor.storage_offset() != 0
                    or tensor.device.type != 'cuda' or tensor.device.index != 0):
                raise ValueError(f'typed argument mismatch {ABI[i]}')
            ranges.append(bridge.interval(tensor.data_ptr(), physical_count, tensor.element_size()))
        bridge.disjoint_writes(ranges, [i >= 5 for i in range(9)])
        function = ct.c_void_p()
        driver.check(driver.library.cuModuleGetFunction(ct.byref(function), module, name.encode()), 'cuModuleGetFunction')
        self.function = function
        actual = {}
        for label, attr in [('max_threads', 0), ('shared_bytes', 1), ('local_bytes', 3), ('registers', 4)]:
            value = ct.c_int()
            driver.check(driver.library.cuFuncGetAttribute(ct.byref(value), attr, function), 'cuFuncGetAttribute')
            actual[label] = value.value
            if value.value != resources[label]:
                raise ValueError(f'fresh Rust/bridge resource mismatch {label}')
        capacity = ct.c_int()
        driver.check(driver.library.cuOccupancyMaxActiveBlocksPerMultiprocessor(ct.byref(capacity), function, block, 0), 'cuOccupancyMaxActiveBlocksPerMultiprocessor')
        actual['api_capacity_at_chosen_block'] = capacity.value
        if block == 256 and capacity.value != resources['capacity256']:
            raise ValueError('256-thread capacity differs from fresh Rust capture')
        if actual['local_bytes'] or actual['max_threads'] < block or capacity.value < 1:
            raise ValueError('unqualified launch resources')
        if name.startswith('resident_test_vector') and (actual['registers'] > 128 or capacity.value < 2 or actual['shared_bytes'] != 32):
            raise ValueError('vector resource gate failed')
        self.block = (block, 1, 1)
        self.grid = (length // (1024 if name.startswith('resident_test_vector') else block), 1, 1)
        self.metadata = {'entry': name, 'ABI': list(ABI), 'typed_ABI': ['f32','bf16','f32','f32','f32','f32','f32','f32','f32'],
            'logical_elements': logical, 'physical_output_guard_elements': 16,
            'grid': self.grid, 'block': self.block, 'resources': actual,
            'pointers': [t.data_ptr() for t in tensors], 'resource_capacity_is_not_achieved_occupancy': True}

    def launch(self):
        stream = self.torch.cuda.current_stream()
        pointers = []
        for tensor in self.tensors:
            tensor.record_stream(stream)
            pointers.append(ct.c_void_p(tensor.data_ptr()))
        arguments = (ct.c_void_p * len(pointers))(*(ct.addressof(x) for x in pointers))
        self.driver.check(self.driver.library.cuLaunchKernel(self.function, *self.grid, *self.block, 0,
                          ct.c_void_p(stream.cuda_stream), arguments, None), 'cuLaunchKernel')


def inventory():
    result = {}
    for key, option in [('device', '--query-gpu=name,uuid,driver_version,memory.total,memory.free'),
                        ('compute_processes', '--query-compute-apps=pid,process_name,used_gpu_memory')]:
        p = subprocess.run(['nvidia-smi', option, '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        result[key] = p.stdout.strip()
    return result


def compare_files(np, actual, reference, health_length=None):
    actual, reference = Path(actual), Path(reference)
    if actual.stat().st_size != reference.stat().st_size:
        raise ValueError('output length mismatch')
    x, y = np.memmap(actual, dtype='<f4', mode='r'), np.memmap(reference, dtype='<f4', mode='r')
    matching = 0
    for start in range(0, x.size, 1 << 20):
        a, b = x[start:start+(1 << 20)], y[start:start+(1 << 20)]
        same = (a.view('<u4') == b.view('<u4')) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            raise ValueError(f'full output bits/NaN classes differ at {start+int(np.flatnonzero(~same)[0])}')
        matching += int(same.sum())
    sentinel = np.uint32(SENTINEL_BITS)
    if not np.all(x[-16:].view('<u4') == sentinel):
        raise ValueError('output suffix canary overwritten')
    if health_length is not None and not np.all(x[:health_length].view('<u4') == 0):
        raise ValueError('finite fixture must emit exact +0 health')
    return {'elements': int(x.size), 'nonNaN_bits_and_NaN_classes_equal': True,
            'matching_elements': matching, 'canary_elements_intact': 16,
            'actual': fingerprint(actual), 'reference': fingerprint(reference)}


def run_case(case, args, bridge, calibration, torch, np, driver, stream):
    length,name=case['length'],case['name']
    if length not in LENGTHS or length%1024:raise ValueError('outside approved lengths')
    directory=args.output/name;directory.mkdir()
    native=args.archive/name
    for record in [case['module'],case['rust_resources'],*case['files'].values()]:
        if fingerprint(record['path'])!=record:raise ValueError('sealed input/PTX changed')
    source=(native/'loaded-module.ptx').read_text();entries(source)
    sm=re.search(r'\.target\s+sm_(\d+)(?:\s|,)',source)
    if sm is None or int(sm.group(1))!=int(''.join(map(str,torch.cuda.get_device_capability(0)))):raise ValueError('target mismatch')
    encoded=ct.create_string_buffer(source.encode());module=ct.c_void_p()
    driver.check(driver.library.cuModuleLoadData(ct.byref(module),ct.cast(encoded,ct.c_void_p)),'cuModuleLoadData');driver.modules.append(module)
    inputs={}
    for label in INPUTS:
        path=Path(case['files'][label]['path'])
        if label=='gradient':inputs[label]=calibration.cuda_input(path,[length])
        else:
            count=10 if label=='settings'else length
            inputs[label]=torch.from_numpy(np.array(calibration.mapped(path,'<f4',(count,)),copy=True)).to('cuda')
    sentinel=np.array([SENTINEL_BITS],dtype='<u4').view('<f4')[0].item()
    allocation_order=[2,1,0]if args.reverse else[0,1,2]
    slots={}
    for slot in allocation_order:
        slots[slot]={label:torch.full((length//256+16 if label=='health'else length+16,),sentinel,dtype=torch.float32,device='cuda')for label in OUTPUTS}
    torch.cuda.synchronize()
    resources=json.loads((native/'rust-resources.json').read_text())
    if resources[2]['function']!='resident_adam_bf16_first_2'or resources[3]['function']!='resident_test_vector4_bf16_adam_3':raise ValueError('wrong Rust resources')
    construction=construction_pairs(args.reverse);kernels={}
    for mode,slot in construction:
        vector=mode=='vector_bf16_256';block=1024 if mode.endswith('1024')else 256
        tensors=[inputs[x]for x in INPUTS]+[slots[slot][x]for x in OUTPUTS]
        kernel=AdamKernel(bridge,torch,driver,module,source,'resident_test_vector4_bf16_adam_3'if vector else'resident_adam_bf16_first_2',length,block,resources[3 if vector else 2],tensors)
        kernel.metadata.update(mode=mode,slot=slot);kernels[(mode,slot)]=kernel
    # Check each physical slot once. Different modes intentionally alias a slot
    # only across sequential calls, never in-flight/concurrently.
    tensors=list(inputs.values())+[t for slot in range(3)for t in slots[slot].values()]
    bridge.disjoint_writes([bridge.interval(t.data_ptr(),t.numel(),t.element_size())for t in tensors],[False]*5+[True]*12)
    owners=[*tensors,encoded,module,*kernels.values()]
    def key(pair):return f'{pair[0]}/slot{pair[1]}'
    def poison(slot):
        with torch.cuda.stream(stream):
            for tensor in slots[slot].values():tensor.fill_(sentinel)
        stream.synchronize()
    def input_proof(phase):
        return {label:bridge.exact_output(tensor,case['files'][label]['path'],directory/f'input-{label}-{phase}.{ "bf16"if label=="gradient"else"f32"}',label=='gradient')for label,tensor in inputs.items()}
    def allocator_snapshot():
        return {'torch_allocated_current':torch.cuda.memory_allocated(),'torch_reserved_current':torch.cuda.memory_reserved(),'torch_peak_allocated':torch.cuda.max_memory_allocated(),'torch_peak_reserved':torch.cuda.max_memory_reserved()}
    proofs={'inputs_pre':input_proof('pre')}
    # CPU-file oracle comes from an existing slot; no extra GPU oracle tuple.
    poison(0)
    with torch.cuda.stream(stream):kernels[(MODES[0],0)].launch()
    stream.synchronize();oracle={}
    for label in OUTPUTS:
        path=directory/f'oracle-{label}.f32';calibration.dump32(path,slots[0][label]);oracle[label]=fingerprint(path)
        compare_files(np,path,path,length//256 if label=='health'else None)
    graphs={}
    def output_proof(phase,captured):
        result={}
        for pair in construction:
            mode,slot=pair
            # Reset logical output AND suffix before every binding check. No
            # previous mode's correct values can hide an omitted/misbound launch.
            poison(slot)
            with torch.cuda.stream(stream):
                if captured:graphs[pair].replay()
                else:kernels[pair].launch()
            stream.synchronize();arrays={}
            for label in OUTPUTS:
                path=directory/f'{mode}-slot{slot}-{label}-{phase}.f32';calibration.dump32(path,slots[slot][label])
                arrays[label]=compare_files(np,path,oracle[label]['path'],length//256 if label=='health'else None)
            result[key(pair)]={'mode':mode,'slot':slot,'poisoned_before_launch':True,'captured':captured,'arrays':arrays}
        return result
    proofs['outputs_pre']=output_proof('pre',False)
    memory_before_capture=allocator_snapshot()
    try:
        for pair in construction:graphs[pair]=bridge.capture(kernels[pair].launch,CAPTURE_REPEATS,stream)
        memory_after_capture=allocator_snapshot()
        proofs['outputs_captured']=output_proof('captured',True)
        proofs['inputs_captured']=input_proof('captured')
        raw_blocks=[];warmup_cycles=0;warmup_seconds=0.
        if args.timing:
            started=time.perf_counter()
            while time.perf_counter()-started<.5:
                # Complete rotation cycles warm every mode on every slot equally.
                for rotation in range(3):
                    bi=warmup_cycles*3+rotation
                    with torch.cuda.stream(stream):
                        for triplet in block_orders(bi,args.reverse):
                            for mode in triplet:graphs[(mode,slot_for(mode,bi,args.reverse))].replay()
                stream.synchronize();warmup_cycles+=1
            warmup_seconds=time.perf_counter()-started
            events={mode:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))for mode in MODES}
            for bi in range(36):
                samples=[]
                for ti,triplet in enumerate(block_orders(bi,args.reverse)):
                    for position,mode in enumerate(triplet):
                        slot=slot_for(mode,bi,args.reverse);pair=(mode,slot);start_event,end_event=events[mode]
                        with torch.cuda.stream(stream):
                            start_event.record(stream);graphs[pair].replay();end_event.record(stream)
                        end_event.synchronize();us=start_event.elapsed_time(end_event)*1000/CAPTURE_REPEATS
                        if not math.isfinite(us)or us<=0:raise ValueError('invalid event interval')
                        samples.append({'triplet':ti,'position':position,'mode':mode,'slot':slot,'binding':key(pair),'input_pointers':[inputs[x].data_ptr()for x in INPUTS],'output_pointers':[slots[slot][x].data_ptr()for x in OUTPUTS],'microseconds_per_invocation':us})
                raw_blocks.append({'block':bi,'orders':[list(x)for x in block_orders(bi,args.reverse)],'mode_to_slot':{m:slot_for(m,bi,args.reverse)for m in MODES},'samples':samples})
            stream.synchronize()
        # Post proofs poison/replay ALL nine bindings again, outside timing.
        proofs['outputs_post']=output_proof('post',True);proofs['inputs_post']=input_proof('post')
        for record in [case['module'],case['rust_resources'],*case['files'].values()]:
            if fingerprint(record['path'])!=record:raise ValueError('sealed archive changed after use')
        return {'case':name,'length':length,'qualified':True,'timing':args.timing,'performance_claim':False,'bindings':{key(pair):kernel.metadata for pair,kernel in kernels.items()},'slots':{str(slot):{'pointers':[slots[slot][label].data_ptr()for label in OUTPUTS],'physical_elements':[slots[slot][label].numel()for label in OUTPUTS]}for slot in range(3)},'shared_input_pointers':[inputs[x].data_ptr()for x in INPUTS],'slot_allocation_creation_order':allocation_order,'kernel_construction_order':[key(x)for x in construction],'capture_construction_order':[key(x)for x in construction],'proofs':proofs,'oracle_files':oracle,'extra_GPU_oracle_allocations':0,'poison_only_outside_timing':True,'capture_repeats':CAPTURE_REPEATS,'warmup_cycles':warmup_cycles,'warmup_seconds':warmup_seconds,'balanced_blocks':raw_blocks,'owned_tensor_bytes_including_guards':sum(t.numel()*t.element_size()for t in tensors),'allocator_before_capture':memory_before_capture,'allocator_after_capture':memory_after_capture,'memory_scope':'17owned tensors:5inputs+3slots×4outputs; includes16F32guard elements/output. Ninegraph runtimes/allocator reserves/context/modules separately recorded, not storage or occupancy savings.','same_readonly_inputs_all_nine_bindings':True,'sequential_output_slot_aliases_across_modes':True}
    finally:
        stream.synchronize()
        for graph in reversed(list(graphs.values())):graph.reset()
        assert len(owners)==28


def run(args):
    import numpy as np
    import torch
    import non_nn_bridge as bridge
    import non_nn_calibration as calibration
    if args.output.exists():
        raise ValueError('fresh output required; preserve all failures')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.archive/'manifest.json').read_text())
    if fingerprint(Path(__file__))!=manifest['controller']:
        raise ValueError('controller changed after archive freeze')
    for name,record in manifest['helpers'].items():
        if fingerprint(Path(__file__).with_name(name))!=record:
            raise ValueError('generic helper changed after archive freeze')
    if manifest['seed']!=17 or len(manifest['cases'])!=2:
        raise ValueError('approved seed17 archive required')
    report={'scope':__doc__,'timing':args.timing,'performance_claim':False,'reverse':args.reverse,
        'archive_manifest':fingerprint(args.archive/'manifest.json'),
        'source':{str(p):fingerprint(p) for p in [Path(__file__),Path(bridge.__file__),Path(calibration.__file__)]},
        'torch':torch.__version__,'numpy':np.__version__,'before':inventory(),'cases':[]}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    driver=bridge.Driver()
    stream=torch.cuda.Stream()
    try:
        for case in (list(reversed(manifest['cases']))if args.reverse else manifest['cases']):
            report['cases'].append(run_case(case,args,bridge,calibration,torch,np,driver,stream))
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        stream.synchronize()
        driver.close()
        report['after']=inventory()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['qualified']=len(report['cases'])==2 and all(c['qualified'] for c in report['cases'])
    report['all_GPU_jobs_terminal']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'qualified':report['qualified'],'cases':len(report['cases']),'timing':args.timing}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--from-sealed',type=Path,required=True);a.add_argument('--archive',type=Path,required=True)
    b=sub.add_parser('run');b.add_argument('--archive',type=Path,required=True);b.add_argument('--output',type=Path,required=True);b.add_argument('--timing',action='store_true');b.add_argument('--reverse',action='store_true')
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    else:run(args)
if __name__=='__main__':main()
