#!/usr/bin/env python3
"""Test-only three-mode NT fused-bias calibration; no production selector hook."""
from __future__ import annotations
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import time

LABELS = ('native64', 'native128', 'Torch_addmm')


def order(round_index, reverse):
    return list(reversed(LABELS)) if bool(round_index % 2) != reverse else list(LABELS)


def case_resources(candidate, m, n):
    result = dict(candidate)
    if candidate.get('ABI') != ['BF16Arow', 'BF16Wrow_with_logical_transpose', 'BF16bias_col', 'BF16output']:
        raise ValueError('candidate typed ABI mismatch')
    schedule = candidate.get('schedule', '')
    for proof in (f'logical_shape: MatMulTile {{ m: {m}, n: {n}, k: 512 }}',
                  'block_tile: MatMulTile { m: 128, n: 128, k: 32 }',
                  'block_threads: (32, 8, 1)', 'warp_topology: (4, 2)',
                  'warp_fragments: (2, 4)', 'pipeline_stages: 2',
                  'copy_bytes: 16', 'async_operands: (true, true)',
                  'fragment_load: StreamB', 'reuse_shared_memory: true'):
        if proof not in schedule:
            raise ValueError('retained candidate schedule/launch proof mismatch')
    result.update(grid=[n // 128, m // 128, 1], block=[32, 8, 1])
    if (m % 128 or n % 128 or result['registers'] > 128 or result['registers'] < 1
            or result['shared'] != 40960 or result['local'] != 0 or result['api_capacity'] < 2
            or result['max_threads'] < 256 or not result['strict_admitted']):
        raise ValueError('candidate lacks strict retained admission/full tiles')
    return result


def compare(case, args, bridge, calibration, torch, driver, stream):
    name, m, n, k = (case[x] for x in ('name', 'm', 'n', 'k'))
    if (case['kind'] != 'nt_fused_linear' or [m, k] != case['a_shape']
            or [n, k] != case['b_shape'] or case['output_shape'] != [m, n] or k != 512
            or (m, n) not in ((16384, 1408), (16384, 1536))):
        raise ValueError('outside the two qualified actual NT contracts')
    directory = args.output / name
    directory.mkdir()
    native = args.native / name
    retained = calibration.load_json(native / 'actual-resources-and-abi.json')['kernels']
    if len(retained) != 1:
        raise ValueError('strong native64 control must be exactly one fused kernel')
    expected_shapes = f'inputs=[[{m},{k}],[{n},{k}],[{n}]]output=[{m},{n}]'
    if expected_shapes not in retained[0]['label'].replace(' ', ''):
        raise ValueError('native64 semantic contract mismatch')
    calibration.mapped(native / 'output.bf16', '<u2', (m, n))
    a = calibration.cuda_input(args.archive / name / 'a.bf16', [m, k])
    b = calibration.cuda_input(args.archive / name / 'b.bf16', [n, k])
    bias = calibration.cuda_input(args.archive / name / 'bias.bf16', [n])
    inputs = [('a', a), ('b', b), ('bias', bias)]
    outputs = {label: torch.empty((m, n), dtype=torch.bfloat16, device='cuda') for label in LABELS}
    # All three outputs and their input owners remain live until all graphs reset.
    owners = [a, b, bias, *outputs.values()]
    specs = [bridge.Argument('input_0', 'bf16', (m, k)), bridge.Argument('input_1', 'bf16', (n, k)),
             bridge.Argument('input_2', 'bf16', (n,)), bridge.Argument('output_0', 'bf16', (m, n), True)]
    candidate_resources = case_resources(calibration.load_json(args.candidate / name / 'resources.json'), m, n)
    kernels = {
        'native64': bridge.PointerKernel(driver, bridge.one_file(native, 'program-*.ptx'), retained[0], specs, [a, b, bias, outputs['native64']]),
        'native128': bridge.PointerKernel(driver, args.candidate / name / 'candidate.ptx', candidate_resources, specs, [a, b, bias, outputs['native128']]),
    }
    if kernels['native128'].metadata['entry'] != candidate_resources['entry']:
        raise ValueError('candidate entry mismatch')
    functions = {label: kernel.launch for label, kernel in kernels.items()}
    functions['Torch_addmm'] = lambda: torch.addmm(bias, a, b.t(), out=outputs['Torch_addmm'])
    def input_proof(phase):
        return {label: bridge.exact_output(tensor, args.archive / name / f'{label}.bf16',
                                          directory / f'input-{label}-{phase}.bf16', True)
                for label, tensor in inputs}
    def output_proof(phase):
        return {label: bridge.exact_output(output, native / 'output.bf16', directory / f'{label}-{phase}.bf16', True)
                for label, output in outputs.items()}
    inputs_pre = input_proof('pre')
    with torch.cuda.stream(stream):
        for label in order(0, args.reverse):
            functions[label]()
    stream.synchronize()
    pre = output_proof('pre')
    construction = order(0, args.reverse)
    graphs = {}
    raw = []
    warmups, warmup_ms = 0, 0.0
    try:
        for label in construction:
            graphs[label] = bridge.capture(functions[label], 32, stream)
        with torch.cuda.stream(stream):
            for label in construction:
                graphs[label].replay()
        stream.synchronize()
        captured = output_proof('captured')
        if args.timing:
            start = time.perf_counter()
            while time.perf_counter() - start < 0.5:
                with torch.cuda.stream(stream):
                    for label in order(warmups, args.reverse):
                        graphs[label].replay()
                stream.synchronize()
                warmups += 1
            warmup_ms = 1000 * (time.perf_counter() - start)
            # Events are constructed before the measured interval; setup/readbacks are excluded.
            events = {label: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for label in LABELS}
            for index in range(32):
                times = {}
                sequence = order(index, args.reverse)
                with torch.cuda.stream(stream):
                    for label in sequence:
                        start_event, end_event = events[label]
                        start_event.record(stream)
                        graphs[label].replay()
                        end_event.record(stream)
                        end_event.synchronize()
                        us = start_event.elapsed_time(end_event) * 1000 / 32
                        if not math.isfinite(us) or us <= 0:
                            raise ValueError('invalid event interval')
                        times[label] = us
                raw.append({'round': index, 'order': sequence, 'microseconds_per_invocation': times})
        stream.synchronize()
        post = output_proof('post')
        inputs_post = input_proof('post')
        sums = {label: sum(r['microseconds_per_invocation'][label] for r in raw) for label in LABELS}
        return {'case': name, 'm': m, 'n': n, 'k': k, 'native_kernels': {label: kernel.metadata for label, kernel in kernels.items()},
                'shared_input_pointers': {label: tensor.data_ptr() for label, tensor in inputs},
                'output_pointers': {label: tensor.data_ptr() for label, tensor in outputs.items()},
                'proofs': {'inputs_pre': inputs_pre, 'outputs_pre': pre, 'outputs_captured': captured,
                           'outputs_post': post, 'inputs_post': inputs_post},
                'allocation_scope': {'identical_three_mode_input_addresses': True,
                                     'live_tensor_bytes': sum(t.numel() * t.element_size() for t in owners),
                                     'three_separate_preallocated_outputs': True,
                                     'excludes_modules_reserved_allocator_library_graph_workspaces': True},
                'capture_construction_order': construction, 'capture_repeats': 32,
                'warmup_rounds': warmups, 'warmup_ms': warmup_ms, 'rounds': raw,
                'ratios_of_summed_paired_times': None if not raw else {
                    'native64_over_native128': sums['native64'] / sums['native128'],
                    'Torch_over_native128': sums['Torch_addmm'] / sums['native128'],
                    'Torch_over_native64': sums['Torch_addmm'] / sums['native64']}}
    finally:
        torch.cuda.synchronize()
        for graph in reversed(list(graphs.values())):
            graph.reset()


def main():
    parser = argparse.ArgumentParser()
    for flag in ('helpers', 'archive', 'native', 'candidate', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--qualified', type=Path)
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--cpu-self-test', action='store_true')
    args = parser.parse_args()
    if args.cpu_self_test:
        assert order(0, False) == list(LABELS) and order(1, False) == list(reversed(LABELS))
        assert order(0, True) == order(1, False) and order(1, True) == order(0, False)
        assert all(sum(order(i, reverse) == list(LABELS) for i in range(32)) == 16 for reverse in (False, True))
        try:
            case_resources({'ABI': []}, 16384, 1408)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed ABI accepted')
        print('ABC/CBA balanced, genuinely reversed and malformed ABI rejected; CPU-only PASS')
        return
    if args.timing and args.qualified is None:
        parser.error('timing requires prior complete untimed report and root CLEANACK')
    if args.output.exists():
        raise ValueError('preserve previous output directory')
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.helpers.resolve()))
    bridge = importlib.import_module('non_nn_bridge')
    calibration = bridge.calibration
    torch = bridge.torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cases = [c for c in calibration.load_json(args.archive / 'manifest.json')['cases'] if c['kind'] == 'nt_fused_linear']
    if len(cases) != 2:
        raise ValueError('exactly two qualified NT cases required')
    if args.reverse:
        cases.reverse()
    paths = [Path(__file__), Path(bridge.__file__), Path(calibration.__file__), args.archive / 'manifest.json']
    for c in cases:
        name = c['name']
        paths += [args.archive / name / f'{label}.bf16' for label in ('a', 'b', 'bias')]
        paths += [args.native / name / 'output.bf16', args.native / name / 'actual-resources-and-abi.json',
                  bridge.one_file(args.native / name, 'program-*.ptx'),
                  args.candidate / name / 'candidate.ptx', args.candidate / name / 'resources.json']
    provenance = {str(p.resolve()): bridge.fingerprint(p) for p in paths}
    # Only immutable content hashes are compared; source/output path relocations are allowed.
    hashes = sorted((v['bytes'], v['sha256']) for v in provenance.values())
    report = {'status': 'in_progress', 'timing': args.timing, 'reverse': args.reverse, 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'provenance': provenance, 'content_identities': hashes, 'cases': [],
              'settings': {'allow_tf32': False, 'allow_bf16_reduced_precision_reduction': False},
              'scope': 'component only; no profiling; no production selector or whole-model qualification'}
    qualified = calibration.load_json(args.qualified) if args.timing else None
    if qualified is not None:
        if (qualified['status'] != 'qualified_three_mode_full_bits' or qualified['timing']
                or qualified['content_identities'] != json.loads(json.dumps(hashes))
                or qualified['torch'] != report['torch'] or qualified['cuda'] != report['cuda']):
            raise ValueError('prior untimed qualification content/runtime mismatch')
    driver = bridge.Driver()
    report['device'] = {'name': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0))}
    try:
        if qualified is not None and report['device'] != qualified['device']:
            raise ValueError('device changed')
        stream = torch.cuda.Stream()
        for case in cases:
            report['cases'].append(compare(case, args, bridge, calibration, torch, driver, stream))
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        report['status'] = 'timing_complete_full_pre_post_bits' if args.timing else 'qualified_three_mode_full_bits'
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        driver.close()
    print(report['status'])


if __name__ == '__main__':
    main()
