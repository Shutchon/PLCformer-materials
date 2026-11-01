"""
Phase 2 Unit Test Suite for PLCformer:
  - Task 2.1: Full ISA Expansion (SET, RST, MPS/MRD/MPP Stack, TON Timer, CTU Counter, MOV/ADD/CMP Word ALU)
  - Task 2.2: Structured Text (ST) Transpiler & .safetensors Checkpoint Builder
  - Task 2.3: Fast Numba JIT Sparse Engine (`runtime/fast_engine.py`) Bit-Exactness & >5,000 tok/s Speed Verification
"""

from pathlib import Path
import numpy as np
import pytest

from compiler.build_checkpoint import export_plcformer_safetensors, load_plcformer_safetensors
from runtime.fast_engine import FastNumbaEngine
from runtime.host import PLCformerHost


def test_set_and_rst_latching() -> None:
    """Verify SET and RST latching instructions preserve state across scan cycles."""
    il_code = """
    LD  X0
    SET Y0
    LD  X1
    RST Y0
    END
    """
    host = PLCformerHost(program_source=il_code)

    # Idle: Y0 = 0
    host.run_scan_cycle({"X0": 0, "X1": 0})
    assert host.read_bit("Y0") == 0

    # Pulse X0=1 -> SET Y0 latches Y0 = 1
    host.run_scan_cycle({"X0": 1, "X1": 0})
    assert host.read_bit("Y0") == 1

    # Release X0=0 -> Y0 remains 1 without needing an OR seal-in contact
    for _ in range(5):
        host.run_scan_cycle({"X0": 0, "X1": 0})
        assert host.read_bit("Y0") == 1

    # Pulse X1=1 -> RST Y0 clears Y0 = 0
    host.run_scan_cycle({"X0": 0, "X1": 1})
    assert host.read_bit("Y0") == 0

    # Release X1=0 -> Y0 remains 0
    host.run_scan_cycle({"X0": 0, "X1": 0})
    assert host.read_bit("Y0") == 0


def test_stack_branching_mps_mrd_mpp() -> None:
    """
    Verify MPS (Push), MRD (Read), and MPP (Pop) instructions on a 3-branch rung:
      Common condition: X0
        +- Branch 1: AND X1 -> OUT Y0
        +- Branch 2: AND X2 -> OUT Y1
        +- Branch 3: ANI X3 -> OUT Y2
    """
    il_code = """
    LD  X0
    MPS
    AND X1
    OUT Y0
    MRD
    AND X2
    OUT Y1
    MPP
    ANI X3
    OUT Y2
    END
    """
    host = PLCformerHost(program_source=il_code)

    # When master condition X0 = 0 -> all branches Y0, Y1, Y2 must be 0
    res0 = host.run_scan_cycle({"X0": 0, "X1": 1, "X2": 1, "X3": 0})
    assert (res0.outputs_y["Y0"], res0.outputs_y["Y1"], res0.outputs_y["Y2"]) == (0, 0, 0)

    # When master condition X0 = 1, X1 = 1, X2 = 0, X3 = 0 -> Y0=1, Y1=0, Y2=1
    res1 = host.run_scan_cycle({"X0": 1, "X1": 1, "X2": 0, "X3": 0})
    assert (res1.outputs_y["Y0"], res1.outputs_y["Y1"], res1.outputs_y["Y2"]) == (1, 0, 1)

    # When master condition X0 = 1, X1 = 0, X2 = 1, X3 = 1 -> Y0=0, Y1=1, Y2=0
    res2 = host.run_scan_cycle({"X0": 1, "X1": 0, "X2": 1, "X3": 1})
    assert (res2.outputs_y["Y0"], res2.outputs_y["Y1"], res2.outputs_y["Y2"]) == (0, 1, 0)


def test_ton_timer_and_star_delta_starter() -> None:
    """
    Verify TON On-Delay Timer and Star-Delta Motor Starter (`benchmarks/programs/star_delta.il`):
      - Pressing X0 starts Main Contactor (Y0=1) and Star Contactor (Y1=1, Y2=0).
      - During cycles 1..4 (T0.ACC < 5), Star remains ON (Y1=1, Y2=0).
      - On cycle 5 (T0.ACC == 5 >= K5), T0.DN becomes 1 -> Star turns OFF (Y1=0) and Delta turns ON (Y2=1).
      - Pressing Stop X1=1 resets M0, T0.ACC, T0.DN, and turns OFF Y0, Y1, Y2.
    """
    il_path = Path(__file__).resolve().parent.parent / "benchmarks" / "programs" / "star_delta.il"
    host = PLCformerHost()
    host.load_program_file(il_path)

    # Cycle 1: Press Start (X0=1, X1=0) -> T0.ACC = 1, Y0=1 (Main), Y1=1 (Star), Y2=0 (Delta)
    host.run_scan_cycle({"X0": 1, "X1": 0})
    assert host.read_word("T0.ACC") == 1.0
    assert host.read_bit("T0.DN") == 0
    assert (host.read_bit("Y0"), host.read_bit("Y1"), host.read_bit("Y2")) == (1, 1, 0)

    # Cycles 2..4: Release Start (X0=0, M0 holds) -> T0.ACC = 2, 3, 4; Star still ON
    for expected_acc in (2.0, 3.0, 4.0):
        host.run_scan_cycle({"X0": 0})
        assert host.read_word("T0.ACC") == expected_acc
        assert host.read_bit("T0.DN") == 0
        assert (host.read_bit("Y0"), host.read_bit("Y1"), host.read_bit("Y2")) == (1, 1, 0)

    # Cycle 5: T0.ACC reaches 5 -> T0.DN = 1 -> transitions from Star (Y1=0) to Delta (Y2=1)!
    host.run_scan_cycle({"X0": 0})
    assert host.read_word("T0.ACC") == 5.0
    assert host.read_bit("T0.DN") == 1
    assert (host.read_bit("Y0"), host.read_bit("Y1"), host.read_bit("Y2")) == (1, 0, 1)

    # Cycle 6: Stop pressed (X1=1) -> M0=0, T0 resets to 0, all contactors OFF
    host.run_scan_cycle({"X1": 1})
    assert host.read_word("T0.ACC") == 0.0
    assert host.read_bit("T0.DN") == 0
    assert (host.read_bit("Y0"), host.read_bit("Y1"), host.read_bit("Y2")) == (0, 0, 0)


def test_ctu_counter_and_conveyor_sorting() -> None:
    """
    Verify CTU Count-Up Counter rising-edge semantics and RST on `benchmarks/programs/sorting.il`.
    """
    il_path = Path(__file__).resolve().parent.parent / "benchmarks" / "programs" / "sorting.il"
    host = PLCformerHost()
    host.load_program_file(il_path)

    # Start conveyor (X2=1, X0=0, X1=0) -> Y0=1 (Conveyor running), Y1=0 (Box not full), C0.ACC=0
    host.run_scan_cycle({"X0": 0, "X1": 0, "X2": 1})
    assert host.read_word("C0.ACC") == 0.0
    assert (host.read_bit("Y0"), host.read_bit("Y1")) == (1, 0)

    # Part 1 arrives (X0: 0 -> 1 rising edge) -> C0.ACC = 1
    host.run_scan_cycle({"X0": 1})
    assert host.read_word("C0.ACC") == 1.0
    # Holding X0=1 for another cycle must NOT increment C0.ACC (edge-triggered!)
    host.run_scan_cycle({"X0": 1})
    assert host.read_word("C0.ACC") == 1.0
    # Sensor clears (X0: 1 -> 0)
    host.run_scan_cycle({"X0": 0})

    # Part 2 arrives (X0: 0 -> 1 -> 0) -> C0.ACC = 2
    host.run_scan_cycle({"X0": 1})
    assert host.read_word("C0.ACC") == 2.0
    host.run_scan_cycle({"X0": 0})
    assert (host.read_bit("Y0"), host.read_bit("Y1")) == (1, 0)

    # Part 3 arrives (X0: 0 -> 1) -> C0.ACC = 3 >= K3 -> C0.DN = 1!
    # Conveyor Y0 stops automatically (Y0=0) and Box Full Diverter Y1 turns ON (Y1=1)
    host.run_scan_cycle({"X0": 1})
    assert host.read_word("C0.ACC") == 3.0
    assert host.read_bit("C0.DN") == 1
    assert (host.read_bit("Y0"), host.read_bit("Y1")) == (0, 1)

    # Operator presses Batch Reset X1=1 (with X0=0) -> C0 resets to 0, Conveyor Y0 resumes!
    host.run_scan_cycle({"X0": 0, "X1": 1})
    assert host.read_word("C0.ACC") == 0.0
    assert host.read_bit("C0.DN") == 0
    assert (host.read_bit("Y0"), host.read_bit("Y1")) == (1, 0)


def test_word_alu_mov_add_and_comparisons_via_st() -> None:
    """
    Verify MOV, ADD, CMP_EQ, CMP_GT, and CMP_LT compiled from Structured Text (ST).
    """
    st_program = """
    IF X0 THEN D0 := 10; END_IF;
    IF X1 THEN D0 := D0 + 5; END_IF;
    Y0 := (D0 == 10);
    Y1 := (D0 > 12);
    Y2 := (D0 < 12);
    """
    host = PLCformerHost(program_source=st_program, is_structured_text=True)

    # Cycle 1: Load D0 := 10 (X0=1, X1=0) -> D0=10 -> Y0(==10)=1, Y1(>12)=0, Y2(<12)=1
    host.run_scan_cycle({"X0": 1, "X1": 0})
    assert host.read_word("D0") == 10.0
    assert (host.read_bit("Y0"), host.read_bit("Y1"), host.read_bit("Y2")) == (1, 0, 1)

    # Cycle 2: Add 5 to D0 (X0=0, X1=1) -> D0=15 -> Y0(==10)=0, Y1(>12)=1, Y2(<12)=0
    host.run_scan_cycle({"X0": 0, "X1": 1})
    assert host.read_word("D0") == 15.0
    assert (host.read_bit("Y0"), host.read_bit("Y1"), host.read_bit("Y2")) == (0, 1, 0)


def test_safetensors_checkpoint_export_and_load(tmp_path: Path) -> None:
    """Verify exporting and reloading a compiled PLCformer checkpoint in .safetensors format."""
    il_path = Path(__file__).resolve().parent.parent / "benchmarks" / "programs" / "star_delta.il"
    host = PLCformerHost()
    host.load_program_file(il_path)

    ckpt_file = tmp_path / "plcformer_star_delta.safetensors"
    export_plcformer_safetensors(host, ckpt_file)
    assert ckpt_file.exists()

    loaded = load_plcformer_safetensors(ckpt_file)
    assert "model.layers.1.gdn.state_init" in loaded
    assert "model.layers.2.mlp.gate_proj.weight" in loaded
    assert loaded["model.layers.1.gdn.state_init"].shape == (1, 5, 128, 128)


def test_fast_numba_engine_equivalence_and_throughput() -> None:
    """
    Task 2.3: Verify that `FastNumbaEngine` produces bit-exact identical outputs to `PLCformerHost`
    on Star-Delta starter and achieves >5,000 tokens/sec on CPU.
    """
    il_path = Path(__file__).resolve().parent.parent / "benchmarks" / "programs" / "star_delta.il"
    host_ref = PLCformerHost()
    host_ref.load_program_file(il_path)

    host_jit = PLCformerHost()
    host_jit.load_program_file(il_path)
    fast_engine = FastNumbaEngine(host_jit)

    # Build a 50-cycle test schedule exercising Start, Star-to-Delta timer transition, and Stop
    n_test = 50
    schedule = np.zeros((n_test, 16), dtype=np.float32)
    schedule[0, 0] = 1.0   # Press Start X0=1 at cycle 0
    schedule[15, 1] = 1.0  # Press Stop X1=1 at cycle 15
    schedule[20, 0] = 1.0  # Press Start X0=1 again at cycle 20

    # Run on Reference PyTorch Host
    ref_outputs = np.zeros((n_test, 16), dtype=np.int32)
    for c in range(n_test):
        res = host_ref.run_scan_cycle({"X0": schedule[c, 0], "X1": schedule[c, 1]}, mode="mode_a")
        for y_i in range(16):
            ref_outputs[c, y_i] = res.outputs_y[f"Y{y_i}"]

    # Run on FastNumbaEngine
    jit_outputs, _, _ = fast_engine.run_batch_cycles(schedule)
    assert np.array_equal(ref_outputs, jit_outputs)

    # Benchmark throughput over 2,000 scan cycles (30,000 instruction tokens)
    bench_schedule = np.zeros((2000, 16), dtype=np.float32)
    bench_schedule[0, 0] = 1.0
    _, total_tokens, elapsed_sec = fast_engine.run_batch_cycles(bench_schedule)
    tokens_per_sec = total_tokens / max(elapsed_sec, 1e-9)

    # Blueprint target: > 5,000 tok/s
    assert tokens_per_sec > 5000.0, f"Expected >5,000 tok/s, got {tokens_per_sec:.1f} tok/s"

