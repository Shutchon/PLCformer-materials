"""Reproducible output and process-image checks against a shared-IR reference.

State is inspected after instruction-level integer normalization, before output
thresholding. This is trace-based differential testing, not a compiler proof.
"""
import json
from pathlib import Path
import time
import numpy as np
from runtime.fast_engine import FastNumbaEngine
from runtime.host import PLCformerHost
from runtime.plant_sim import IndustrialPlantInputGenerator, ReferenceSoftPLC

PROGRAMS_DIR = Path(__file__).resolve().parent / 'programs'
BENCHMARKS = [
    ('Motor Start/Stop Interlock', 'self_hold.il', IndustrialPlantInputGenerator.generate_motor_start_stop_trace, 42),
    ('Star-Delta Motor Starter', 'star_delta.il', IndustrialPlantInputGenerator.generate_star_delta_trace, 43),
    ('4-Way Traffic Light Controller', 'traffic_light.il', IndustrialPlantInputGenerator.generate_traffic_light_trace, 44),
    ('Automated Conveyor Sorting', 'sorting.il', IndustrialPlantInputGenerator.generate_conveyor_sorting_trace, 45),
]

def process_error(state, reference):
    """All 128 image bits, 64 D words, 32 timer/counter triples, and branch stack."""
    return float(max(
        np.max(np.abs(state[0, :, 0] - reference.bits)),
        np.max(np.abs(state[1, :96, :3] - reference.words[:96, :3])),
        np.max(np.abs(state[2, 2:10, 0] - reference.stack)),
    ))

def run_equivalence_verification(n_cycles=100_000):
    results = []
    for name, filename, generator, seed in BENCHMARKS:
        trace = generator(n_cycles, seed=seed)
        reference = ReferenceSoftPLC.from_file(PROGRAMS_DIR / filename)
        host = PLCformerHost()
        instrs = host.load_program_file(PROGRAMS_DIR / filename)
        engine = FastNumbaEngine(host)
        output_bad = state_bad = 0
        max_state_error = 0.0
        tokens = 0
        t0 = time.perf_counter()
        for i in range(n_cycles):
            ref_y = reference.step_scan_cycle(trace[i])
            # Full input-image update outside the measured execution kernel.
            engine.state[0, :16, 0] = trace[i]
            tokens += engine.run_scan_cycle()
            error = process_error(engine.state, reference)
            max_state_error = max(max_state_error, error)
            state_bad += error != 0.0
            output_bad += not np.array_equal(engine.state[0, 16:32, 0], ref_y)
        elapsed = time.perf_counter() - t0
        eager = PLCformerHost()
        eager.load_program_file(PROGRAMS_DIR / filename)
        eager_ref = ReferenceSoftPLC.from_file(PROGRAMS_DIR / filename)
        eager_n = min(500, n_cycles)
        eager_error = 0.0
        for i in range(eager_n):
            eager.run_scan_cycle({f'X{j}': float(trace[i, j]) for j in range(16)})
            eager_ref.step_scan_cycle(trace[i])
            eager_error = max(eager_error, process_error(eager.gdn_memory.state[0].detach().numpy(), eager_ref))
        results.append({
            'benchmark': name, 'program_file': filename, 'input_seed': seed,
            'num_il_instructions': len(instrs), 'scan_cycles_tested': n_cycles,
            'total_tokens_executed': tokens, 'plcformer_accuracy_pct': 100*(1-output_bad/n_cycles),
            'mismatched_cycles': int(output_bad), 'full_state_mismatched_cycles': int(state_bad),
            'max_linf_error': max_state_error,
            'state_measurement': 'post-instruction normalization; before output threshold',
            'state_fields': 'X,Y,M,T.DN,C.DN,D,T.ACC/PRE/previous-enable,C.ACC/PRE/PREV,stack',
            'pytorch_eager_cycles': eager_n, 'pytorch_eager_500_exact': eager_error == 0,
            'pytorch_max_state_error': eager_error, 'verification_wall_sec': elapsed,
        })
    return {'experiment': 'Trace-based process-image equivalence', 'benchmarks': results}

if __name__ == '__main__':
    print(json.dumps(run_equivalence_verification(), indent=2))
