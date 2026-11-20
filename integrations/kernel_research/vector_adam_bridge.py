#!/usr/bin/env python3
"""Untimed, three-native-mode shared-input vector Adam component proof.

Loads the exact fresh Rust leaf monolithic PTX without extracting/recompiling
entries. No optimizer-owner API, Torch Adam, timing events, or model claims.
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
    import numpy as np
    if args.archive.exists():
        raise ValueError('preserve existing archive; fresh path required')
    args.archive.mkdir(parents=True)
    qualification = json.loads((args.leaf / 'qualification.json').read_text())
    if not qualification['qualified_leaf'] or qualification['performance_claim']:
        raise ValueError('fresh semantic leaf qualification required')
    manifest = {'scope': __doc__, 'seed': 17, 'fixture_kind': 'finite synthetic flat optimizer state at actual FFN/QKV element counts',
                'model_fixture_or_update_claim': False, 'numpy': np.__version__,
                'leaf_qualification': fingerprint(args.leaf / 'qualification.json'),
                'controller': fingerprint(Path(__file__)),
                'helpers': {name:fingerprint(Path(__file__).with_name(name)) for name in ['non_nn_bridge.py','non_nn_calibration.py']}, 'cases': []}
    for length in LENGTHS:
        directory = args.archive / f'len{length}'
        directory.mkdir()
        source = args.leaf / 'device' / f'loaded-len{length}.ptx'
        proof = next(x for x in qualification['GPU']['lengths'] if x['length'] == length)
        if fingerprint(source)['sha256'] != proof['exact_loaded_PTX']['sha256']:
            raise ValueError('loaded module differs from fresh qualified source')
        entries(source.read_text())
        resources = args.leaf / 'device' / f'admission-len{length}.json'
        shutil.copyfile(source, directory / 'loaded-module.ptx')
        shutil.copyfile(resources, directory / 'rust-resources.json')
        rng = np.random.default_rng(17)
        arrays = {
            'weight': rng.normal(0., .02, length).astype('<f4'),
            'moment': rng.normal(0., .001, length).astype('<f4'),
            'variance': rng.uniform(.001, .01, length).astype('<f4'),
        }
        gradient = rng.normal(0., .125, length).astype('<f4')
        # Explicit finite F32->BF16 RN-even checkpoint; upload raw stored bits.
        bits = gradient.view('<u4')
        raw = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype('<u2')
        arrays['gradient'] = raw
        b1, b2 = np.float32(.9), np.float32(.999)
        arrays['settings'] = np.array([np.float32(1e-4), b1, b2,
            np.float32(1)-b1, np.float32(1)-b2, np.float32(1)-b1,
            np.float32(1)-b2, np.float32(1e-8), np.float32(0), np.float32(1)], dtype='<f4')
        files = {}
        for name, array in arrays.items():
            if name != 'gradient' and not np.isfinite(array).all():
                raise ValueError('finite fixtures required')
            path = directory / (name + ('.bf16' if name == 'gradient' else '.f32'))
            array.tofile(path)
            files[name] = fingerprint(path)
        if np.any((raw & 0x7F80) == 0x7F80):
            raise ValueError('nonfinite rounded BF16 fixture')
        manifest['cases'].append({'name': f'len{length}', 'length': length,
            'semantic_role': '55M FFN weight2752x1024' if length == 2818048 else '55M QKV weight3072x1024',
            'settings_bits_u32': arrays['settings'].view('<u4').tolist(),
            'settings_scope': 'Recorded default beta/lr/epsilon/decay0/clip1 and first-step correction constants, with seeded finite nonzero moments; synthetic leaf only.',
            'files': files, 'module': fingerprint(directory / 'loaded-module.ptx'),
            'rust_resources': fingerprint(directory / 'rust-resources.json')})
    (args.archive / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'cases': 2, 'GPU_runs': 0,
                      'manifest': fingerprint(args.archive / 'manifest.json')}))


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
    length, name = case['length'], case['name']
    if length not in LENGTHS or length % 1024:
        raise ValueError('outside two approved actual parameter lengths')
    directory = args.output / name
    directory.mkdir()
    native = args.archive / name
    for record in [case['module'], case['rust_resources'], *case['files'].values()]:
        if fingerprint(record['path']) != record:
            raise ValueError('archive fingerprint changed before GPU use')
    source_path = native / 'loaded-module.ptx'
    source = source_path.read_text()
    sm = re.search(r'\.target\s+sm_(\d+)(?:\s|,)', source)
    if sm is None or int(sm.group(1)) != int(''.join(map(str, torch.cuda.get_device_capability(0)))):
        raise ValueError('PTX/actual target mismatch')
    entries(source)
    encoded = ct.create_string_buffer(source.encode())
    module = ct.c_void_p()
    driver.check(driver.library.cuModuleLoadData(ct.byref(module), ct.cast(encoded, ct.c_void_p)), 'cuModuleLoadData')
    driver.modules.append(module)
    inputs = {}
    for label in INPUTS:
        file = Path(case['files'][label]['path'])
        if label == 'gradient':
            inputs[label] = calibration.cuda_input(file, [length])
        else:
            count = 10 if label == 'settings' else length
            raw = calibration.mapped(file, '<f4', (count,))
            inputs[label] = torch.from_numpy(np.array(raw, copy=True)).to('cuda')
    sentinel = np.array([SENTINEL_BITS], dtype='<u4').view('<f4')[0].item()
    outputs = {mode: {label: torch.full((length//256+16 if label == 'health' else length+16,), sentinel,
              dtype=torch.float32, device='cuda') for label in OUTPUTS} for mode in MODES}
    torch.cuda.synchronize()
    resources = json.loads((native / 'rust-resources.json').read_text())
    if resources[2]['function'] != 'resident_adam_bf16_first_2' or resources[3]['function'] != 'resident_test_vector4_bf16_adam_3':
        raise ValueError('wrong Rust resource entries')
    kernels = {}
    for mode in MODES:
        vector = mode == 'vector_bf16_256'
        block = 1024 if mode.endswith('1024') else 256
        tensors = [inputs[x] for x in INPUTS] + [outputs[mode][x] for x in OUTPUTS]
        kernels[mode] = AdamKernel(bridge, torch, driver, module, source,
            'resident_test_vector4_bf16_adam_3' if vector else 'resident_adam_bf16_first_2',
            length, block, resources[3 if vector else 2], tensors)
    # Allocation-wide disjoint output proof across modes, not only per call.
    tensors = list(inputs.values()) + [t for mode in MODES for t in outputs[mode].values()]
    bridge.disjoint_writes([bridge.interval(t.data_ptr(),t.numel(),t.element_size()) for t in tensors],
                           [False]*5+[True]*12)
    owners = [*tensors, encoded, module, *kernels.values()]
    def input_proof(phase):
        return {label: bridge.exact_output(tensor, case['files'][label]['path'],
            directory / f'input-{label}-{phase}.{ "bf16" if label == "gradient" else "f32"}', label == 'gradient')
            for label,tensor in inputs.items()}
    def output_proof(phase):
        result = {}
        for mode in MODES:
            result[mode] = {}
            for label in OUTPUTS:
                path = directory / f'{mode}-{label}-{phase}.f32'
                calibration.dump32(path, outputs[mode][label])
                expected = directory / f'scalar_bf16_1024-{label}-pre.f32'
                result[mode][label] = compare_files(np,path,expected,length//256 if label=='health' else None)
        return result
    proofs = {'inputs_pre': input_proof('pre')}
    with torch.cuda.stream(stream):
        for mode in MODES:
            kernels[mode].launch()
    stream.synchronize()
    proofs['outputs_pre'] = output_proof('pre')
    graphs = {}
    try:
        for mode in MODES:
            graphs[mode] = bridge.capture(kernels[mode].launch, CAPTURE_REPEATS, stream)
        with torch.cuda.stream(stream):
            for mode in MODES:
                graphs[mode].replay()
        stream.synchronize()
        proofs['outputs_captured'] = output_proof('captured')
        proofs['inputs_captured'] = input_proof('captured')
        with torch.cuda.stream(stream):
            for mode in reversed(MODES):
                kernels[mode].launch()
        stream.synchronize()
        proofs['outputs_post'] = output_proof('post')
        proofs['inputs_post'] = input_proof('post')
        for record in [case['module'], case['rust_resources'], *case['files'].values()]:
            if fingerprint(record['path']) != record:
                raise ValueError('archive fingerprint changed after GPU use')
        return {'case':name,'length':length,'qualified':True,'modes':{mode:k.metadata for mode,k in kernels.items()},
            'proofs':proofs,'same_readonly_input_addresses':True,'three_disjoint_preallocated_output_tuples':True,
            'capture_repeats':CAPTURE_REPEATS,'capture_construction_order':list(MODES),
            'logical_live_tensor_bytes':sum(t.numel()*t.element_size() for t in tensors),
            'scope':'Includes16canaryF32elements per owned output; no input duplication or fake owner/prefix reuse. Tensor allocation bytes exclude allocator reserves, context, modules, graph workspaces.',
            'timing':False,'performance_claim':False}
    finally:
        stream.synchronize()
        for graph in reversed(list(graphs.values())):
            graph.reset()
        # Keep explicit owner collection live through graph retirement.
        assert len(owners) == 22


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
    report={'scope':__doc__,'timing':False,'performance_claim':False,
        'archive_manifest':fingerprint(args.archive/'manifest.json'),
        'source':{str(p):fingerprint(p) for p in [Path(__file__),Path(bridge.__file__),Path(calibration.__file__)]},
        'torch':torch.__version__,'numpy':np.__version__,'before':inventory(),'cases':[]}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    driver=bridge.Driver()
    stream=torch.cuda.Stream()
    try:
        for case in manifest['cases']:
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
    print(json.dumps({'qualified':report['qualified'],'cases':len(report['cases']),'timing':False}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--leaf',type=Path,required=True);a.add_argument('--archive',type=Path,required=True)
    b=sub.add_parser('run');b.add_argument('--archive',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    else:run(args)
if __name__=='__main__':main()
