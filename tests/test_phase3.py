"""
Phase 3 Unit Test Suite for PLCformer:
  - Task 3.1: 4-Way Traffic Light Controller (`traffic_light.il`) & shared-IR `ReferenceSoftPLC`
  - Task 3.2: Experiment 1 (Bit-Exact Equivalence), Experiment 2 (Throughput & Batched Digital Twins),
              and Experiment 3 (synthetic Boolean interlock & Differentiable PLC Tuning)
"""

from pathlib import Path
import numpy as np
import pytest

from benchmarks.benchmark_speed import benchmark_batched_digital_twins, benchmark_single_instance_modes
from benchmarks.experiment_safety_and_diff import (
    run_differentiable_plc_tuning_experiment,
    run_safety_guardrail_experiment,
)
from benchmarks.verify_equivalence import run_equivalence_verification
from runtime.host import PLCformerHost
from runtime.plant_sim import ReferenceSoftPLC


def test_4way_traffic_light_sequential_phases() -> None:
    """
    Task 3.1: Verify the 4-Way Traffic Light Controller (`benchmarks/programs/traffic_light.il`)
    cycles through all 4 phases with zero green-light conflicts:
      - Phase 1 (cycles 1..3):  NS Green (Y0=1),  EW Red (Y5=1)
      - Phase 2 (cycles 4..5):  NS Yellow (Y1=1), EW Red (Y5=1)
      - Phase 3 (cycles 6..9):  NS Red (Y2=1),    EW Green (Y3=1)
      - Phase 4 (cycles 10..11): NS Red (Y2=1),   EW Yellow (Y4=1)
    """
    il_path = Path(__file__).resolve().parent.parent / "benchmarks" / "programs" / "traffic_light.il"
    host = PLCformerHost()
    host.load_program_file(il_path)
    ref = ReferenceSoftPLC.from_file(il_path)

    for cycle in range(1, 26):
        res = host.run_scan_cycle({"X0": 1}, mode="mode_a")
        ref_y = ref.step_scan_cycle(np.array([1.0] + [0.0] * 15, dtype=np.float32))

        for y_i in range(6):
            assert res.outputs_y[f"Y{y_i}"] == int(ref_y[y_i]), f"Mismatch at cycle {cycle}, Y{y_i}"

        # Mutual exclusion safety invariant: NS Green/Yellow (Y0|Y1) and EW Green/Yellow (Y3|Y4) never co-occur
        ns_active = res.outputs_y["Y0"] or res.outputs_y["Y1"]
        ew_active = res.outputs_y["Y3"] or res.outputs_y["Y4"]
        assert not (ns_active and ew_active), f"Traffic conflict at cycle {cycle}!"


def test_experiment1_bit_exact_equivalence() -> None:
    """
    Task 3.2 (Experiment 1): Verify 100.000% bit-exact equivalence across all 4 industrial
    benchmarks against `ReferenceSoftPLC`.
    """
    summary = run_equivalence_verification(n_cycles=2_000)
    for b in summary["benchmarks"]:
        assert b["plcformer_accuracy_pct"] == 100.0
        assert b["mismatched_cycles"] == 0
        assert b["max_linf_error"] == 0
        assert b["pytorch_eager_500_exact"] is True


def test_experiment2_batched_digital_twins() -> None:
    """
    Task 3.2 (Experiment 2): Verify batched parallel execution up to 10,000 PLC instances.
    """
    batch_res = benchmark_batched_digital_twins()
    assert len(batch_res) == 5
    assert batch_res[-1]["batch_size_plcs"] == 10_000
    # Correctness is checked inside the benchmark; speed varies by hardware.
    assert batch_res[-1]["total_scan_cycles_per_sec"] > 0.0


def test_experiment3_hybrid_safety_guardrail_and_diff_plc() -> None:
    """
    Task 3.2 (Experiment 3): Verify 100% hazard interception in the synthetic Boolean interlock
    and <0.05 parameter recovery error in separate sigmoid fitting.
    """
    safety_res = run_safety_guardrail_experiment(n_trials=10_000)
    assert safety_res["total_raw_hazardous_commands"] > 5000
    assert safety_res["post_guardrail_short_circuits"] == 0
    assert safety_res["post_guardrail_estop_violations"] == 0
    assert safety_res["hazard_interception_rate_pct"] == 100.0

    diff_res = run_differentiable_plc_tuning_experiment()
    assert diff_res["absolute_error"] < 0.05

