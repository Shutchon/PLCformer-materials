"""Measured throughput on identical synthetic workloads; no WCET guarantee."""
import json
import time
import numpy as np
import torch
from benchmarks.verify_equivalence import BENCHMARKS, PROGRAMS_DIR
from benchmarks.direct_baseline import run_direct
from runtime.fast_engine import FastNumbaEngine
from runtime.host import PLCformerHost
from runtime.plant_sim import ReferenceSoftPLC


def benchmark_single_instance_modes():
    results = []
    for name, filename, generator, seed in BENCHMARKS:
        host = PLCformerHost()
        instrs = host.load_program_file(PROGRAMS_DIR / filename)
        program = np.array([i.to_vector() for i in instrs], dtype=np.int64)
        schedule = generator(50_000, seed=seed)
        engine = FastNumbaEngine(host)
        initial = engine.state.copy()
        run_direct(program, schedule[:10])  # JIT compilation excluded
        direct_times, neural_times = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            direct_y = run_direct(program, schedule)
            direct_times.append(time.perf_counter()-t0)
            engine.state[:] = initial
            neural_y, _, elapsed = engine.run_batch_cycles(schedule)
            neural_times.append(elapsed)
        if not np.array_equal(neural_y, direct_y):
            raise AssertionError('Direct baseline and compiled weights disagree')
        eager_times, ref_times = [], []
        for _ in range(3):
            eager = PLCformerHost()
            eager.load_program_file(PROGRAMS_DIR / filename)
            reference = ReferenceSoftPLC.from_file(PROGRAMS_DIR / filename)
            for row in schedule[:10]:
                eager.run_scan_cycle({f'X{i}': float(row[i]) for i in range(16)})
            eager.gdn_memory.reset_state()
            t0 = time.perf_counter()
            for row in schedule[:200]:
                eager.run_scan_cycle({f'X{i}': float(row[i]) for i in range(16)})
            eager_times.append(time.perf_counter()-t0)
            t0 = time.perf_counter()
            for row in schedule[:200]:
                reference.step_scan_cycle(row)
            ref_times.append(time.perf_counter()-t0)
        neural_rate = 50_000 / float(np.median(neural_times))
        eager_rate = 200 / float(np.median(eager_times))
        results.append({
            'program': name, 'program_file': filename, 'num_instructions': len(instrs),
            'input_seed': seed, 'jit_cycles_per_repeat': 50_000, 'python_cycles_per_repeat': 200,
            'pytorch_mode_a_cycles_per_sec': eager_rate,
            'reference_python_cycles_per_sec': 200/float(np.median(ref_times)),
            'numba_jit_cycles_per_sec': neural_rate,
            'numba_jit_tokens_per_sec': neural_rate*len(instrs),
            'numba_jit_amortized_us': 1e6/neural_rate,
            'direct_numba_cycles_per_sec': 50_000/float(np.median(direct_times)),
            'speedup_jit_over_eager': neural_rate/eager_rate,
            'direct_numba_output_match': True,
            'neural_jit_repeat_seconds': neural_times,
            'direct_numba_repeat_seconds': direct_times,
            'pytorch_repeat_seconds': eager_times, 'reference_repeat_seconds': ref_times,
            'timing_scope': 'batch JIT includes input latching and output arrays; Python includes per-scan API overhead',
        })
    return results


def benchmark_batched_digital_twins():
    """Legacy function name: measures Boolean controller batches, without a plant."""
    host = PLCformerHost()
    instrs = host.load_program_file(PROGRAMS_DIR / 'self_hold.il')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = host.unrolled_model.to(device)
    result = []
    for batch in [1, 64, 256, 1024, 10_000]:
        initial = torch.zeros((batch, 128), device=device)
        initial[:, 0] = 1
        seconds = []
        with torch.no_grad():
            bits = initial.clone()
            for _ in range(10):
                bits = model(bits)
            if device.type == 'mps':
                torch.mps.synchronize()
            for _ in range(5):
                bits = initial.clone()
                if device.type == 'mps':
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                for _ in range(200):
                    bits = model(bits)
                if device.type == 'mps':
                    torch.mps.synchronize()
                seconds.append(time.perf_counter()-t0)
        if not torch.equal(bits[:,16], torch.ones(batch, device=device)):
            raise AssertionError('Batched self-hold output mismatch')
        elapsed = float(np.median(seconds))
        rate = batch*200/elapsed
        result.append({
            'batch_size_plcs': batch, 'device': str(device), 'workload': 'self_hold.il; constant start input; no plant',
            'wall_time_ms_per_step': elapsed*1000/200,
            'amortized_us_per_plc_cycle': 1e6/rate,
            'total_scan_cycles_per_sec': rate,
            'effective_instructions_per_sec': rate*len(instrs), 'repeat_seconds': seconds,
        })
    return result


def run_all_speed_benchmarks():
    return {'experiment': 'Execution throughput (median of repeats)',
            'single_instance': benchmark_single_instance_modes(),
            'batched_controllers': benchmark_batched_digital_twins()}

if __name__ == '__main__':
    print(json.dumps(run_all_speed_benchmarks(), indent=2))
