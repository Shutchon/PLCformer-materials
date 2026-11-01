"""Primitive float32 error, empirical scan-kernel latency, and scalar fitting.

The 50-addition stress test repeats a scalar gate output, not a PLC program.
Latency percentiles and observed maxima are measurements, not WCET bounds.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

from compiler.mlp_alu import SwiGLUScalingCoeffs
from runtime.host import PLCformerHost
from runtime.fast_engine import FastNumbaEngine, _jit_execute_scan_cycle
from runtime.plant_sim import DifferentiablePLCTuner


def run_ablation_swiglu_gate() -> List[Dict[str, Any]]:
    """Compare Single-Neuron Asymptotic Gate vs. Symmetric Dual-Neuron Gate (Lemma 2)."""
    scales = [2.0, 5.0, 10.0, 15.0, 20.0, 32.0]
    results = []

    triples = [
        (float(s), float(a), float(b))
        for s in (0, 1)
        for a in (0, 1)
        for b in (0, 1)
    ]

    for c in scales:
        c_t = torch.tensor(c, dtype=torch.float32)
        alpha_p = F.silu(c_t)
        coeffs = SwiGLUScalingCoeffs.compute(c=c)
        w_p = torch.tensor(coeffs.w_pos, dtype=torch.float32)
        w_m = torch.tensor(coeffs.w_neg, dtype=torch.float32)

        max_err_single = 0.0
        max_err_dual = 0.0

        for s, a, b in triples:
            target = s * a * b
            z = torch.tensor(s + a - 1.0, dtype=torch.float32)
            b_t = torch.tensor(b, dtype=torch.float32)

            # Single-neuron asymptotic approximation: (SiLU(c * z) / SiLU(c)) * b
            y_single = (F.silu(c_t * z) / alpha_p) * b_t
            err_single = abs(float(y_single.item()) - target)
            max_err_single = max(max_err_single, err_single)

            # Symmetric Dual-Neuron Gate (Lemma 2)
            y_dual = (w_p * F.silu(c_t * z) + w_m * F.silu(-c_t * z)) * b_t
            err_dual = abs(float(y_dual.item()) - target)
            max_err_dual = max(max_err_dual, err_dual)

        # Repeat the inactive gate output in 50 float32 additions (s=0, a=0, b=1)
        z_zero = torch.tensor(-1.0, dtype=torch.float32)
        single_leak = float((F.silu(c_t * z_zero) / alpha_p).item())
        dual_leak = float((w_p * F.silu(c_t * z_zero) + w_m * F.silu(-c_t * z_zero)).item())

        single_acc = torch.tensor(0., dtype=torch.float32)
        dual_acc = torch.tensor(0., dtype=torch.float32)
        for _ in range(50):
            single_acc = single_acc + single_leak
            dual_acc = dual_acc + dual_leak
        drift_50_single = abs(float(single_acc))
        drift_50_dual = abs(float(dual_acc))

        results.append({
            "scale_c": c,
            "single_neuron_max_err": max_err_single,
            "single_neuron_50step_drift": drift_50_single,
            "single_abs_error_exceeds_half": bool(drift_50_single >= 0.5),
            "dual_neuron_max_err": max_err_dual,
            "dual_neuron_50step_drift": drift_50_dual,
            "dual_abs_error_exceeds_half": bool(drift_50_dual >= 0.5),
        })

    return results


def run_latency_measurements() -> Dict[str, Any]:
    """Measure per-cycle latency distribution (Mean, Std, P99, P99.9) over 10,000 cycles."""
    prog_dir = Path(__file__).resolve().parent / "programs"
    benchmarks = ["self_hold.il", "star_delta.il", "traffic_light.il", "sorting.il"]
    latency_stats: Dict[str, Any] = {}

    for fname in benchmarks:
        host = PLCformerHost()
        host.load_program_file(prog_dir / fname)
        engine = FastNumbaEngine(host)

        # Warmup
        for _ in range(500):
            engine.run_scan_cycle({"X0": 1, "X1": 0})

        # Measure 10,000 individual scan cycles for empirical latency statistics
        n_measure = 10_000
        latencies_us = np.empty(n_measure, dtype=np.float64)
        for i in range(n_measure):
            engine.state[0, 0, 0] = float(i % 2)
            t0 = time.perf_counter_ns()
            _jit_execute_scan_cycle(
                engine.state,
                engine.host.dt,
                256,
                engine.gate_row_ptr,
                engine.gate_col_idx,
                engine.gate_vals,
                engine.up_row_ptr,
                engine.up_col_idx,
                engine.up_vals,
                engine.down_T_row_ptr,
                engine.down_T_col_idx,
                engine.down_T_vals,
                engine.active_neurons,
            )
            t1 = time.perf_counter_ns()
            latencies_us[i] = (t1 - t0) / 1000.0

        latency_stats[fname] = {
            "mean_us": round(float(np.mean(latencies_us)), 3),
            "std_us": round(float(np.std(latencies_us)), 3),
            "median_us": round(float(np.median(latencies_us)), 3),
            "p99_us": round(float(np.percentile(latencies_us, 99.0)), 3),
            "p999_us": round(float(np.percentile(latencies_us, 99.9)), 3),
            "max_us": round(float(np.max(latencies_us)), 3),
            "sample_count": n_measure,
        }

    return latency_stats


def run_diff_plc_trajectory() -> List[Dict[str, float]]:
    """Record loss and learned timer preset K at every epoch (0..80) for Figure 3."""
    target_k = 5.0
    initial_k = 1.5
    tuner = DifferentiablePLCTuner(initial_k=initial_k, tau=2.5)
    optimizer = torch.optim.Adam(tuner.parameters(), lr=0.15)

    ticks = torch.arange(1, 11, dtype=torch.float32)
    target_star = (ticks < target_k).to(torch.float32)
    target_delta = (ticks >= target_k).to(torch.float32)

    trajectory = []
    for step in range(81):
        optimizer.zero_grad()
        pred_star, pred_delta = tuner.forward_star_delta_trace(ticks)
        loss = F.mse_loss(pred_star, target_star) + F.mse_loss(pred_delta, target_delta)
        trajectory.append({
            "epoch": step,
            "loss": round(float(loss.item()), 5),
            "K": round(float(tuner.k_preset.item()), 4),
        })
        if step < 80:
            loss.backward()
            optimizer.step()

    return trajectory


if __name__ == "__main__":
    swiglu_ablation = run_ablation_swiglu_gate()
    latency_stats = run_latency_measurements()
    diff_traj = run_diff_plc_trajectory()

    summary = {
        "swiglu_ablation": swiglu_ablation,
        "latency_stats": latency_stats,
        "diff_trajectory_every_5": diff_traj[::5],
    }

    out_path = Path(__file__).resolve().parent / "ablation_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
