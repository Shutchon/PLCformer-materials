"""Synthetic Boolean interlock checks and separate scalar sigmoid fitting.

No language model is called and no gradient passes through a compiled PLC.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict
import torch
import torch.nn.functional as F

from runtime.plant_sim import DifferentiablePLCTuner, HybridSafetyGuardrail


def run_safety_guardrail_experiment(n_trials: int = 10_000) -> Dict[str, Any]:
    """Synthetic binary proposals plus all 16 possible command/sensor patterns."""
    import itertools
    import numpy as np
    torch.manual_seed(2026)
    guardrail = HybridSafetyGuardrail()
    raw_states = torch.zeros((n_trials, 128), dtype=torch.float32)
    raw_states[:, :4] = torch.randint(0, 2, (n_trials, 4), dtype=torch.float32)

    def hazards(states, outputs):
        bits = outputs[:, 16:19] >= .5
        short = bits[:, 0] & bits[:, 1]
        estop = (states[:, 3] >= .5) & bits.any(dim=1)
        return short, estop

    raw_outputs = torch.zeros_like(raw_states)
    raw_outputs[:, 16:19] = raw_states[:, :3]
    raw_short, raw_stop = hazards(raw_states, raw_outputs)
    raw_hazard = raw_short | raw_stop
    timings = []
    with torch.no_grad():
        guardrail.forward_guardrail_batch(raw_states)  # warm-up excluded
        for _ in range(5):
            t0 = time.perf_counter()
            filtered = guardrail.forward_guardrail_batch(raw_states)
            timings.append((time.perf_counter() - t0)*1000)
        combinations = torch.zeros((16,128))
        combinations[:, :4] = torch.tensor(list(itertools.product([0.,1.], repeat=4)))
        exhaustive = guardrail.forward_guardrail_batch(combinations)
    post_short, post_stop = hazards(raw_states, filtered)
    ex_short, ex_stop = hazards(combinations, exhaustive)
    expected = torch.stack([
        combinations[:,0]*(1-combinations[:,1])*(1-combinations[:,3]),
        combinations[:,1]*(1-combinations[:,0])*(1-combinations[:,3]),
        combinations[:,2]*(1-combinations[:,3]),
        torch.maximum(combinations[:,0]*combinations[:,1], combinations[:,3]),
    ], dim=1)
    preserved = torch.all(filtered[:,16:19] == raw_outputs[:,16:19], dim=1)
    n_hazards = int(raw_hazard.sum())
    return {
        'input_source': 'synthetic independent Bernoulli(0.5) bits; not an LLM',
        'seed': 2026, 'total_agent_actions_tested': n_trials,
        'raw_short_circuit_commands': int(raw_short.sum()),
        'raw_estop_violation_commands': int(raw_stop.sum()),
        'total_raw_hazardous_commands': n_hazards,
        'post_guardrail_short_circuits': int(post_short.sum()),
        'post_guardrail_estop_violations': int(post_stop.sum()),
        'hazard_interception_rate_pct': 100*int((raw_hazard & ~(post_short | post_stop)).sum())/n_hazards if n_hazards else None,
        'safe_proposals_tested': int((~raw_hazard).sum()),
        'safe_proposals_changed': int((~raw_hazard & ~preserved).sum()),
        'interlock_alarms_raised': int((filtered[:,19] >= .5).sum()),
        'exhaustive_patterns': 16,
        'exhaustive_hazards_after_filter': int((ex_short | ex_stop).sum()),
        'exhaustive_raw_max_error': float(torch.max(torch.abs(exhaustive[:,16:20]-expected))),
        'batch_10k_inference_time_ms': float(np.median(timings)),
        'batch_repeat_ms': timings,
    }


def run_differentiable_plc_tuning_experiment() -> Dict[str, Any]:
    """
    Demonstrate Differentiable PLC tuning: recover target Timer Preset K* = 5.0
    starting from misconfigured K_init = 1.5 via gradient descent through a standalone sigmoid surrogate.
    """
    target_k = 5.0
    initial_k = 1.5
    tuner = DifferentiablePLCTuner(initial_k=initial_k, tau=2.5)
    optimizer = torch.optim.Adam(tuner.parameters(), lr=0.15)

    ticks = torch.arange(1, 11, dtype=torch.float32)
    # Target discrete Star-Delta trajectory for K* = 5.0:
    # Star (Y1) is 1 for t=1..4 and 0 for t=5..10; Delta (Y2) is 0 for t=1..4 and 1 for t=5..10
    target_star = (ticks < target_k).to(torch.float32)
    target_delta = (ticks >= target_k).to(torch.float32)

    loss_history = []
    k_history = [round(initial_k, 4)]

    for step in range(80):
        optimizer.zero_grad()
        pred_star, pred_delta = tuner.forward_star_delta_trace(ticks)
        loss = F.mse_loss(pred_star, target_star) + F.mse_loss(pred_delta, target_delta)
        loss.backward()
        optimizer.step()
        if step in {0, 19, 39, 59, 79}:
            loss_history.append(round(float(loss.item()), 6))
            k_history.append(round(float(tuner.k_preset.item()), 4))

    final_k = float(tuner.k_preset.item())
    return {
        "initial_k": initial_k,
        "target_k": target_k,
        "recovered_k": round(final_k, 4),
        "absolute_error": round(abs(final_k - target_k), 4),
        "k_trajectory": k_history,
        "loss_trajectory": loss_history,
    }


if __name__ == "__main__":
    res = {
        "experiment": "Synthetic interlock and separate sigmoid surrogate",
        "part_a_safety_guardrail": run_safety_guardrail_experiment(),
        "part_b_differentiable_tuning": run_differentiable_plc_tuning_experiment(),
    }
    print(json.dumps(res, indent=2))

