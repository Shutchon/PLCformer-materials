"""Reproduce the manuscript experiments and record source/environment metadata."""

import json
from pathlib import Path
import time
import hashlib
import platform
import subprocess
import importlib.metadata
import torch

from benchmarks.benchmark_speed import run_all_speed_benchmarks
from benchmarks.experiment_safety_and_diff import (
    run_differentiable_plc_tuning_experiment,
    run_safety_guardrail_experiment,
)
from benchmarks.verify_equivalence import run_equivalence_verification
from benchmarks.run_ablations import run_ablation_swiglu_gate, run_latency_measurements, run_diff_plc_trajectory


def main() -> None:
    t0 = time.perf_counter()
    print('Checking process-image equivalence (400,000 scans)...', flush=True)
    exp1 = run_equivalence_verification(n_cycles=100_000)
    print('Measuring matched-workload throughput and controller batches...', flush=True)
    exp2 = run_all_speed_benchmarks()
    print('Checking interlocks, surrogate fitting, and latency...', flush=True)
    exp3 = {
        "experiment": "Synthetic interlock filtering and separate sigmoid surrogate fitting",
        "part_a_safety_guardrail": run_safety_guardrail_experiment(n_trials=10_000),
        "part_b_differentiable_tuning": run_differentiable_plc_tuning_experiment(),
    }
    ablations = {
        'swiglu_ablation': run_ablation_swiglu_gate(),
        'latency_stats': run_latency_measurements(),
        'diff_trajectory_every_5': run_diff_plc_trajectory()[::5],
    }
    root = Path(__file__).resolve().parent.parent
    def sysctl(key):
        try:
            return subprocess.check_output(['sysctl', '-n', key], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unavailable'
    metadata = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'machine': platform.machine(), 'cpu': sysctl('machdep.cpu.brand_string'),
        'logical_cpus': sysctl('hw.logicalcpu'), 'memory_bytes': sysctl('hw.memsize'),
        'torch_threads': torch.get_num_threads(), 'mps_available': torch.backends.mps.is_available(),
        'packages': {name: importlib.metadata.version(name) for name in ['torch','numpy','numba','safetensors','pytest']},
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for directory in ['compiler','runtime','benchmarks']
                          for p in sorted((root/directory).rglob('*.py'))},
        'timing_note': 'wall-clock samples on a general-purpose OS; no certified WCET',
    }
    (root/'benchmarks'/'ablation_results.json').write_text(json.dumps(ablations, indent=2))
    total_elapsed = round(time.perf_counter() - t0, 3)

    summary = {
        "project": "PLCformer: Constructive PLC-IR Execution with Gated-Delta Memory and SwiGLU Networks",
        "total_evaluation_time_sec": total_elapsed,
        "environment": metadata,
        "experiment_1_equivalence": exp1,
        "experiment_2_throughput": exp2,
        "experiment_3_safety_and_diff": exp3,
    }

    out_path = Path(__file__).resolve().parent / "results_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved complete experimental results to {out_path} in {total_elapsed}s.")


if __name__ == "__main__":
    main()

