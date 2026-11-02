"""Validate external ST execution and measure a complete simulated tensor loop."""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
import torch
from .case import (SOURCE, INPUTS, OUTPUTS, COracle, Controller, ConveyorSimulation,
                   encode_neural, decode_neural, direct_tensor_step, neural_model)


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def cpu_name():
    if platform.system() == 'Darwin':
        try:
            return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or 'unavailable'


def all_transitions(oracle, device):
    states = np.array(list(itertools.product(range(8), [0, 1], [0, 1], [0, 1])), dtype=np.int32)
    inputs = np.array(list(itertools.product([0, 1], repeat=9)), dtype=np.int32)
    before, values = np.repeat(states, 512, axis=0), np.tile(inputs, (64, 1))
    expected = oracle.step(before, values)
    model = neural_model(device)
    neural_bad = direct_bad = 0
    raw_error = 0.0
    for start in range(0, len(before), 1024):
        sl = slice(start, start + 1024)
        st = torch.tensor(before[sl], device=device)
        ins = torch.tensor(values[sl], device=device)
        bits = encode_neural(st)
        bits[:, :9] = ins
        result = model(bits)
        observed = decode_neural(result).cpu().numpy()
        neural_bad += int(np.any(observed != expected[sl], axis=1).sum())
        direct = direct_tensor_step(st, ins).cpu().numpy()
        direct_bad += int(np.any(direct != expected[sl], axis=1).sum())
        target = encode_neural(torch.tensor(expected[sl], device=device))
        idx = [16, 17, 18] + list(range(32, 40))
        raw_error = max(raw_error, float((result[:, idx] - target[:, idx]).abs().max().cpu()))
    assert neural_bad == direct_bad == 0 and raw_error < 1e-4
    table = expected.reshape(64, 512, 4)
    key = lambda s: int(np.dot(s, [8, 4, 2, 1]))
    parents, queue, unsafe, auto_reverse = {0: None}, deque([0]), 0, None
    while queue:
        current = queue.popleft()
        for j, after in enumerate(table[current]):
            unsafe += bool(after[1] and after[2])
            if auto_reverse is None and inputs[j, 0] and after[2]:
                path, node = [j], current
                while parents[node] is not None:
                    node, edge = parents[node]
                    path.append(edge)
                auto_reverse = [dict(zip(INPUTS, map(int, inputs[x]))) for x in reversed(path)]
            nxt = key(after)
            if nxt not in parents:
                parents[nxt] = (current, j)
                queue.append(nxt)
    assert unsafe == 0
    return {'device': str(device), 'transitions': len(before),
            'neural_mismatches': neural_bad, 'direct_tensor_mismatches': direct_bad,
            'raw_semantic_max_abs_error': raw_error, 'reachable_reference_states': len(parents),
            'reachable_reference_transitions': len(parents) * 512,
            'reachable_mutual_exclusion_violations': unsafe,
            'upstream_auto_reverse_counterexample': auto_reverse}


def supplied_traces(oracle):
    record = json.loads((SOURCE / 'source_record.json').read_text())
    results = []
    for index, case in enumerate(record['test_cases'], 1):
        for completion in ([False, True] if index == 3 else [False]):
            initial = np.array([[case['initial_state'].get(n, 0) for n in OUTPUTS]], dtype=np.int32)
            if completion:
                initial[0, 1] = 1  # Explicitly reported completion of missing fixture state.
            neural = Controller('plcformer', torch.tensor(initial))
            state, expected_bad, oracle_bad = initial.copy(), 0, 0
            expected_trace = dict(case['expected_output_trace'])
            expected_trace.update(case.get('expected_internal_state') or {})
            observed = []
            for t in range(case['simulation_rounds']):
                inputs = np.array([[case['input_trace'][n][t] for n in INPUTS]], dtype=np.int32)
                state = oracle.step(state, inputs)
                neural.step(torch.tensor(inputs))
                got = neural.semantic_state().numpy()
                oracle_bad += int(np.any(got != state))
                target = np.array([[expected_trace[n][t] for n in OUTPUTS]])
                expected_bad += int(np.any(state != target))
                observed.append(state[0].tolist())
            assert oracle_bad == 0
            results.append({'case': index, 'explicit_missing_forward_initialization': completion,
                            'initial': initial[0].tolist(), 'scans': case['simulation_rounds'],
                            'neural_vs_matiec_mismatched_scans': oracle_bad,
                            'upstream_expected_mismatched_scans': expected_bad,
                            'observed_states': observed})
    return results


def closed_loop(oracle, device, batch=32, scans=256):
    initial = torch.zeros((batch, 4), dtype=torch.int32, device=device)
    controllers = [Controller(k, initial, oracle) for k in ['matiec_host', 'direct_tensor', 'plcformer']]
    plant = ConveyorSimulation(batch, device)
    max_error = 0.0
    for _ in range(scans):
        inputs = plant.inputs()
        commands = [c.step(inputs) for c in controllers]
        states = [c.semantic_state() for c in controllers]
        for state, command in zip(states[1:], commands[1:]):
            assert torch.equal(state, states[0]) and torch.equal(command, commands[0])
        target = encode_neural(states[0])
        idx = [16, 17, 18] + list(range(32, 40))
        max_error = max(max_error, float((controllers[2].state[:, idx] - target[:, idx]).abs().max().cpu()))
        assert not bool((commands[0][:, 0] & commands[0][:, 1]).any().cpu())
        plant.advance(commands[0])
        assert bool(((plant.position >= -1) & (plant.position <= 20)).all().cpu())
    completed = int(plant.completed.sum().cpu())
    assert completed > 0
    return {'device': str(device), 'batch': batch, 'scans_per_line': scans,
            'controller_scans': batch*scans, 'state_mismatches': 0,
            'completed_parts': completed, 'raw_semantic_max_abs_error': max_error}


def measure(oracle, device, batches=(1, 64, 1024), repeats=3, scans=64):
    rows, rng = [], random.Random(249)
    for batch in batches:
        initial = torch.zeros((batch, 4), dtype=torch.int32, device=device)
        controllers = {k: Controller(k, initial, oracle) for k in ['matiec_host', 'direct_tensor', 'plcformer']}
        plants = {k: ConveyorSimulation(batch, device) for k in controllers}
        times = {k: [] for k in controllers}
        digests = {}
        for k, controller in controllers.items():
            for _ in range(8):
                plants[k].advance(controller.step(plants[k].inputs()))
        for repeat in range(repeats):
            order = list(controllers)
            rng.shuffle(order)
            for k in order:
                controller, plant = controllers[k], plants[k]
                controller.reset()
                plant.reset()
                sync(device)
                start = time.perf_counter()
                for _ in range(scans):
                    plant.advance(controller.step(plant.inputs()))
                sync(device)
                times[k].append(time.perf_counter() - start)
                # Result consumption and equality checks outside the timing boundary.
                values = np.concatenate([controller.semantic_state().cpu().numpy().ravel()] +
                                        [getattr(plant, name).cpu().numpy().astype(np.int32).ravel()
                                         for name in ['position', 'dwell', 'removal', 'wait',
                                                      'completed', 'previous']])
                digest = hashlib.sha256(values.tobytes()).hexdigest()
                if digests:
                    assert digest == next(iter(digests.values()))
                digests[k] = digest
        for k in controllers:
            median = float(np.median(times[k]))
            rows.append({'device': str(device), 'batch': batch, 'method': k,
                         'scans_per_repeat': scans, 'repeat_seconds': times[k],
                         'median_batch_scan_ms': median/scans*1000,
                         'controller_scans_per_second': batch*scans/median,
                         'completed_parts_per_repeat': int(plants[k].completed.sum().cpu()),
                         'final_state_and_plant_sha256': digests[k]})
        print(f'timing complete: {device}, batch={batch}', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', required=True)
    parser.add_argument('--devices', default='cpu')
    parser.add_argument('--output', default='benchmarks/production_line_results.json')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    devices = [torch.device(x) for x in args.devices.split(',')]
    if any(x.type == 'mps' for x in devices) and not torch.backends.mps.is_available():
        raise RuntimeError('Requested MPS is unavailable; do not silently substitute CPU.')
    oracle = COracle(args.oracle)
    result = {'experiment': 'External ST production-line controller and synthetic tensor plant',
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'provenance': json.loads((SOURCE / 'PROVENANCE.json').read_text()),
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'torch': torch.__version__, 'numpy': np.__version__, 'torch_threads': 4,
                              'cpu': cpu_name()},
              'oracle_build': json.loads(Path(args.oracle).with_name('build.json').read_text()),
              'timing_protocol': {'warmup_scans': 8, 'repeats': 3, 'scans_per_repeat': 64,
                                  'method_order_seed': 249,
                                  'boundary': 'plant input construction, controller, command decoding and plant advance; MPS synchronized; reset/compilation excluded',
                                  'host_boundary': 'matiec method includes device-host input/state transfers, C batch call and return transfer',
                                  'plant': 'synthetic discrete-event conveyor, not physical plant data'},
              'exhaustive': [], 'closed_loop': [], 'timing': []}
    with torch.inference_mode():
        result['supplied_traces'] = supplied_traces(oracle)
        print('supplied traces checked', flush=True)
        for device in devices:
            result['exhaustive'].append(all_transitions(oracle, device))
            print(f'exhaustive transitions checked: {device}', flush=True)
            result['closed_loop'].append(closed_loop(oracle, device))
            print(f'closed loop checked: {device}', flush=True)
            if not args.validation_only:
                result['timing'].extend(measure(oracle, device))
    root = Path(__file__).resolve().parents[2]
    sources = list((root / 'benchmarks/production_line_case').glob('*.py'))
    sources += [p for p in SOURCE.iterdir() if p.is_file()]
    sources += list((root / 'compiler').glob('*.py'))
    result['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print('Saved '+args.output, flush=True)


if __name__ == '__main__':
    main()
