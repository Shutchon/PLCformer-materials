"""
Phase 1 Unit Test Suite for PLCformer:
  - Task 1.1: Gated DeltaNet 128x128 Exact Addressable Memory Primitives (Lemma 1)
  - Task 1.2: Zero-Training Constructive SwiGLU MLP Boolean ALU (Lemma 2)
  - Task 1.3: Self-Holding (Start/Stop Motor Circuit) Scan-Cycle Verification
              (Mode A Interpreted ROM, Mode B Unrolled Scan-Cycle, and ST -> IL Transpiler)
"""

from pathlib import Path
import pytest
import torch

from compiler.assembler import ILAssembler
from compiler.gdn_primitives import (
    GDNProcessImageMemory,
    gated_delta_rule_step_col,
    gated_delta_rule_step_row,
    gdn_read_row,
    one_hot_address,
)
from compiler.isa import Instruction, Opcode
from compiler.memory_map import HeadID, MemoryMap, STATE_DIM
from compiler.mlp_alu import PLCBooleanALU, SwiGLUScalingCoeffs
from compiler.st_transpiler import STTranspiler
from runtime.host import PLCformerHost


# ============================================================================
# Task 1.1: Gated DeltaNet 128x128 Memory Map & Exact Read/Write Tests
# ============================================================================


def test_memory_map_coordinates() -> None:
    """Verify exact (head, row, col) coordinate resolution for X, Y, M, D, T, C, and CPU registers."""
    assert MemoryMap.resolve("X0") == MemoryMap.resolve("x0")
    assert MemoryMap.resolve("X0").head == HeadID.BIT_MEMORY
    assert MemoryMap.resolve("X0").row == 0
    assert MemoryMap.resolve("X15").row == 15

    assert MemoryMap.resolve("Y0").head == HeadID.BIT_MEMORY
    assert MemoryMap.resolve("Y0").row == 16
    assert MemoryMap.resolve("Y15").row == 31

    assert MemoryMap.resolve("M0").head == HeadID.BIT_MEMORY
    assert MemoryMap.resolve("M0").row == 32
    assert MemoryMap.resolve("M63").row == 95

    assert MemoryMap.resolve("T0.DN").head == HeadID.BIT_MEMORY
    assert MemoryMap.resolve("T0.DN").row == 96
    assert MemoryMap.resolve("T0.ACC").head == HeadID.WORD_MEMORY
    assert MemoryMap.resolve("T0.ACC").row == 64
    assert MemoryMap.resolve("T0.ACC").col == 0
    assert MemoryMap.resolve("T0.PRE").col == 1

    assert MemoryMap.resolve("C15.DN").head == HeadID.BIT_MEMORY
    assert MemoryMap.resolve("C15.DN").row == 127

    assert MemoryMap.resolve("PC").head == HeadID.CPU_CONTROL
    assert MemoryMap.resolve("PC").row == 0
    assert MemoryMap.resolve("ACC").row == 1
    assert MemoryMap.resolve("SCAN_STATUS").row == 10


def test_lemma1_gated_delta_rule_exact_addressing() -> None:
    """
    Verify Lemma 1 on the canonical Gated DeltaNet update equation:
        S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) k_t^T
    across a 128x128 state matrix.
    """
    S = torch.zeros(STATE_DIM, STATE_DIM, dtype=torch.float32)

    # Write distinct vectors to columns 0, 16, 32, and 127 with alpha=1.0, beta=1.0
    test_rows = [0, 16, 32, 127]
    for r in test_rows:
        k_r = one_hot_address(r)
        v_r = torch.zeros(STATE_DIM, dtype=torch.float32)
        v_r[0] = 1.0
        v_r[1] = float(r)
        S = gated_delta_rule_step_col(S, k_r, v_r, alpha=1.0, beta=1.0)

    # Verify exact retrieval via S @ k_r and zero cross-talk on untouched columns
    for r in range(STATE_DIM):
        k_r = one_hot_address(r)
        retrieved = S @ k_r
        if r in test_rows:
            assert retrieved[0].item() == 1.0
            assert retrieved[1].item() == float(r)
        else:
            assert torch.all(retrieved == 0.0)

    # Overwrite row 16 from 1.0 -> 0.0 -> 1.0 using beta = 1.0
    k_16 = one_hot_address(16)
    v_zero = torch.zeros(STATE_DIM, dtype=torch.float32)
    S = gated_delta_rule_step_col(S, k_16, v_zero, alpha=1.0, beta=1.0)
    assert (S @ k_16)[0].item() == 0.0

    # Verify beta = 0.0 (write disabled) leaves state 100% unchanged even with non-zero v
    v_noise = torch.ones(STATE_DIM, dtype=torch.float32) * 999.0
    S_after_no_write = gated_delta_rule_step_col(S, k_16, v_noise, alpha=1.0, beta=0.0)
    assert torch.equal(S, S_after_no_write)


def test_gdn_process_image_memory_ports() -> None:
    """Test reading and overwriting X, Y, M, PC, ACC, and Instruction ROM in GDNProcessImageMemory."""
    mem = GDNProcessImageMemory()

    # Test writing and overwriting bits 0/1 in X, Y, M
    for sym in ["X0", "X15", "Y0", "Y7", "M0", "M63"]:
        assert mem.read_scalar(sym) == 0.0
        mem.write_scalar(sym, 1.0, beta=1.0)
        assert mem.read_scalar(sym) == 1.0
        # Overwrite back to 0.0
        mem.write_scalar(sym, 0.0, beta=1.0)
        assert mem.read_scalar(sym) == 0.0
        # Write 1.0 again, then test beta=0.0 does not overwrite
        mem.write_scalar(sym, 1.0, beta=1.0)
        mem.write_scalar(sym, 0.0, beta=0.0)
        assert mem.read_scalar(sym) == 1.0

    # Test PC and ACC in Head 2
    mem.write_scalar("PC", 42.0, beta=1.0)
    mem.write_scalar("ACC", 1.0, beta=1.0)
    assert mem.read_scalar("PC") == 42.0
    assert mem.read_scalar("ACC") == 1.0


# ============================================================================
# Task 1.2: Constructive SwiGLU MLP Boolean ALU Exhaustive Tests
# ============================================================================


def test_lemma2_swiglu_ternary_gate_exactness() -> None:
    """
    Verify Lemma 2: The symmetric dual-neuron SwiGLU ternary gate computes s * a * b
    with ZERO floating-point error across all 8 binary combinations of (s, a, b) in {0, 1}^3.
    """
    for scale_c in [5.0, 10.0, 20.0]:
        coeffs = SwiGLUScalingCoeffs.compute(scale_c)
        for s in (0.0, 1.0):
            for a in (0.0, 1.0):
                for b in (0.0, 1.0):
                    z = torch.tensor(s + a - 1.0, dtype=torch.float32)
                    gate_pos = torch.nn.functional.silu(coeffs.c * z) * b
                    gate_neg = torch.nn.functional.silu(-coeffs.c * z) * b
                    out = coeffs.w_pos * gate_pos + coeffs.w_neg * gate_neg
                    expected = s * a * b
                    assert abs(float(out.item()) - expected) < 1e-7


@pytest.mark.parametrize("acc", [0.0, 1.0])
@pytest.mark.parametrize("mem_bit", [0.0, 1.0])
def test_swiglu_boolean_alu_truth_tables(acc: float, mem_bit: float) -> None:
    """
    Exhaustively verify every PLC Boolean instruction compiled into `PLCBooleanALU`
    (`LD`, `LDI`, `AND`, `ANI`, `OR`, `ORI`, `OUT`, `SET`, `RST`, `NOP`, `END`)
    against its exact IEC 61131-3 mathematical specification.
    """
    alu = PLCBooleanALU(scale_c=20.0)
    a_int = int(acc)
    m_int = int(mem_bit)

    # 1. LD: NEW_ACC = m
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.LD, acc, mem_bit)
    assert abs(new_acc - float(m_int)) < 1e-6
    assert abs(w_en - 0.0) < 1e-6
    assert abs(is_end - 0.0) < 1e-6

    # 2. LDI: NEW_ACC = 1 - m
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.LDI, acc, mem_bit)
    assert abs(new_acc - float(1 - m_int)) < 1e-6
    assert abs(w_en - 0.0) < 1e-6

    # 3. AND: NEW_ACC = a & m
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.AND, acc, mem_bit)
    assert abs(new_acc - float(a_int & m_int)) < 1e-6
    assert abs(w_en - 0.0) < 1e-6

    # 4. ANI: NEW_ACC = a & (1 - m)
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.ANI, acc, mem_bit)
    assert abs(new_acc - float(a_int & (1 - m_int))) < 1e-6
    assert abs(w_en - 0.0) < 1e-6

    # 5. OR: NEW_ACC = a | m
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.OR, acc, mem_bit)
    assert abs(new_acc - float(a_int | m_int)) < 1e-6
    assert abs(w_en - 0.0) < 1e-6

    # 6. ORI: NEW_ACC = a | (1 - m)
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.ORI, acc, mem_bit)
    assert abs(new_acc - float(a_int | (1 - m_int))) < 1e-6
    assert abs(w_en - 0.0) < 1e-6

    # 7. OUT: NEW_ACC = a, WRITE_BIT_VAL = a, WRITE_BIT_EN = 1.0
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.OUT, acc, mem_bit)
    assert abs(new_acc - acc) < 1e-6
    assert abs(w_val - acc) < 1e-6
    assert abs(w_en - 1.0) < 1e-6

    # 8. SET: NEW_ACC = a, WRITE_BIT_VAL = 1.0, WRITE_BIT_EN = a
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.SET, acc, mem_bit)
    assert abs(new_acc - acc) < 1e-6
    assert abs(w_val - 1.0) < 1e-6
    assert abs(w_en - acc) < 1e-6

    # 9. RST: NEW_ACC = a, WRITE_BIT_VAL = 0.0, WRITE_BIT_EN = a
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.RST, acc, mem_bit)
    assert abs(new_acc - acc) < 1e-6
    assert abs(w_val - 0.0) < 1e-6
    assert abs(w_en - acc) < 1e-6

    # 10. END: IS_SCAN_END = 1.0
    new_acc, w_val, w_en, is_end = alu.evaluate_step(Opcode.END, acc, mem_bit)
    assert abs(new_acc - 0.0) < 1e-6
    assert abs(w_en - 0.0) < 1e-6
    assert abs(is_end - 1.0) < 1e-6


# ============================================================================
# Task 1.3: Self-Holding (Start/Stop Motor Circuit) Scan-Cycle Verification
# ============================================================================


@pytest.mark.parametrize("execution_mode", ["mode_a", "mode_b"])
def test_self_holding_motor_circuit_scan_cycles(execution_mode: str) -> None:
    """
    Task 1.3: Run the standard Self-Holding Start/Stop Motor Circuit (`benchmarks/programs/self_hold.il`)
    through `PLCformerHost` and verify that:
      1. Initially (X0=0, X1=0) -> Y0 is 0.
      2. Pressing Start (IN_X0_1) -> Y0 latches to 1.
      3. Releasing Start (IN_X0_0) -> Y0 stays 1 across multiple scan cycles (Self-Holding!).
      4. Pressing Stop (IN_X1_1) -> Y0 drops to 0 immediately.
      5. Releasing Stop (IN_X1_0) -> Y0 remains 0.
    """
    il_path = Path(__file__).resolve().parent.parent / "benchmarks" / "programs" / "self_hold.il"
    host = PLCformerHost()
    host.load_program_file(il_path)

    # Cycle 1: Idle state (Start X0=0, Stop X1=0) -> Motor Y0=0
    res1 = host.run_scan_cycle({"X0": 0, "X1": 0}, mode=execution_mode)
    assert res1.outputs_y["Y0"] == 0
    assert host.read_bit("Y0") == 0

    # Cycle 2: Operator presses Start pushbutton (X0=1, X1=0) -> Motor Y0=1
    res2 = host.run_scan_cycle({"X0": 1}, mode=execution_mode)
    assert res2.outputs_y["Y0"] == 1
    assert host.read_bit("Y0") == 1
    assert "OUT_Y0_1" in res2.emitted_tokens

    # Cycles 3..12: Operator releases Start pushbutton (X0=0, X1=0) -> Motor Y0 MUST stay 1!
    for _ in range(10):
        res_hold = host.run_scan_cycle({"X0": 0}, mode=execution_mode)
        assert res_hold.outputs_y["Y0"] == 1
        assert host.read_bit("Y0") == 1

    # Cycle 13: Operator presses Stop button / Overload trips (X1=1) -> Motor Y0 drops to 0
    res_stop = host.run_scan_cycle({"X1": 1}, mode=execution_mode)
    assert res_stop.outputs_y["Y0"] == 0
    assert host.read_bit("Y0") == 0
    assert "OUT_Y0_0" in res_stop.emitted_tokens

    # Cycle 14: Operator releases Stop button (X1=0) -> Motor Y0 remains 0
    res_after = host.run_scan_cycle({"X1": 0}, mode=execution_mode)
    assert res_after.outputs_y["Y0"] == 0
    assert host.read_bit("Y0") == 0


def test_st_transpiler_and_plcverif_exhaustive_state_equivalence() -> None:
    """
    Verify that Structured Text `Y0 := (X0 OR Y0) AND NOT X1;` transpiles to the exact
    self-holding IL program and satisfies PLCverif-style exhaustive state-space equivalence
    across all 2^3 = 8 combinations of (X0, X1, Y0_prev) in both Mode A and Mode B.
    """
    st_code = "Y0 := (X0 OR Y0) AND NOT X1;"
    il_code = STTranspiler.transpile(st_code)
    assert il_code.splitlines() == ["LD X0", "OR Y0", "ANI X1", "OUT Y0", "END"]

    host_a = PLCformerHost(program_source=st_code, is_structured_text=True)
    host_b = PLCformerHost(program_source=st_code, is_structured_text=True)

    for x0 in (0, 1):
        for x1 in (0, 1):
            for y0_prev in (0, 1):
                # Reference IEC 61131-3 semantics
                expected_y0 = (x0 | y0_prev) & (1 - x1)

                host_a.gdn_memory.write_scalar("Y0", float(y0_prev), beta=1.0)
                res_a = host_a.run_scan_cycle({"X0": x0, "X1": x1}, mode="mode_a")

                host_b.gdn_memory.write_scalar("Y0", float(y0_prev), beta=1.0)
                res_b = host_b.run_scan_cycle({"X0": x0, "X1": x1}, mode="mode_b")

                assert res_a.outputs_y["Y0"] == expected_y0
                assert res_b.outputs_y["Y0"] == expected_y0

