"""Regression and boundary checks for the revised execution contract."""

import itertools
import numpy as np
import pytest
import torch

from compiler.assembler import ILAssembler
from compiler.gdn_primitives import gated_delta_rule_step_col
from compiler.isa import Opcode
from compiler.mlp_alu import PLCBooleanALU
from runtime.fast_engine import FastNumbaEngine
from runtime.host import PLCformerHost
from runtime.plant_sim import ReferenceSoftPLC


def test_word_comparators_and_inactive_gates_at_boundaries():
    alu = PLCBooleanALU()
    values = [-32768, -32767, -6000, -1, 0, 1, 4999, 5000, 6000, 32766, 32767]
    rows, expected = [], []
    for a, u, k in itertools.product([0, 1], values, values):
        for op, answer in [(Opcode.CMP_LT, u < k), (Opcode.CMP_GT, u > k),
                           (Opcode.CMP_EQ, u == k)]:
            rows.append(alu.build_input_vector(op, a, 0, mem_word=u, imm_val=k))
            target = torch.zeros(8)
            target[0] = a * answer
            expected.append(target)
    with torch.no_grad():
        actual = alu(torch.stack(rows))
    # Inspect all raw ALU outputs, including inactive write-enable channels.
    assert torch.max(torch.abs(actual - torch.stack(expected))) < 1e-4


@pytest.mark.parametrize("op,expected", [("CMP_LT", 0), ("CMP_GT", 1), ("CMP_EQ", 0)])
def test_6000_word_regression_in_both_engines(op, expected):
    program = f"LD X0\nMOV D0, K6000\n{op} D0, K0\nOUT Y0\nEND"
    host = PLCformerHost(program)
    engine = FastNumbaEngine(PLCformerHost(program))
    assert host.run_scan_cycle({"X0": 1}).outputs_y["Y0"] == expected
    engine.run_scan_cycle({"X0": 1})
    assert engine.state[0, 16, 0] == expected
    assert host.gdn_memory.read_scalar("Y0") == expected


@pytest.mark.parametrize("program", [
    "MOV D0, K32768", "MOV D0, K-32769", "MOV D0, K1.5",
    "TON T0, K-1", "TON C0, K5", "CTU T0, K5", "MOV Y0, K1",
    "LD D0", "MRD", "MPS\nEND", "END\nLD X0", "NOP\n" * 256,
])
def test_invalid_program_is_rejected(program):
    with pytest.raises(ValueError):
        ILAssembler.assemble(program)


@pytest.mark.parametrize("source", [
    "LD X0\nMOV D0, K32767\nADD D0, K1\nEND",
    "LD X0\nMOV D0, K-32768\nADD D0, K-1\nEND",
    "LD X0\nTON T0, K5\nEND",
    "LD X0\nCTU C0, K5\nEND",
])
def test_overflow_is_reported_by_all_runtimes(source):
    host = PLCformerHost(source)
    reference = ReferenceSoftPLC.from_source(source)
    if "TON" in source:
        host.gdn_memory.write_scalar("T0.ACC", 32767)
        reference.words[64, 0] = 32767
    if "CTU" in source:
        host.gdn_memory.write_scalar("C0.ACC", 32767)
        reference.words[80, 0] = 32767
    engine = FastNumbaEngine(PLCformerHost(source))
    engine.state[:] = host.gdn_memory.state[0].detach().numpy()
    with pytest.raises(OverflowError):
        host.run_scan_cycle({"X0": 1})
    with pytest.raises(OverflowError):
        engine.run_scan_cycle({"X0": 1})
    with pytest.raises(OverflowError):
        reference.step_scan_cycle(np.array([1]))


def test_gdn_overwrite_does_not_lose_small_new_value():
    state = torch.tensor([[1e8, 2.0], [4.0, 5.0]])
    updated = gated_delta_rule_step_col(state, torch.tensor([1., 0.]), torch.tensor([1., 3.]))
    assert torch.equal(updated, torch.tensor([[1., 2.], [3., 5.]]))


def test_mode_b_fallback_reports_actual_work():
    host = PLCformerHost("LD X0\nTON T0, K5\nOUT Y0\nEND")
    result = host.run_scan_cycle({"X0": 1}, mode="mode_b")
    assert result.steps_executed == len(result.traces) == 4


def test_scan_budget_exhaustion_is_not_success():
    source = "LD X0\nOUT Y0\nEND"
    with pytest.raises(RuntimeError):
        PLCformerHost(source).run_scan_cycle({"X0": 1}, max_steps=1)
    with pytest.raises(RuntimeError):
        FastNumbaEngine(PLCformerHost(source)).run_scan_cycle({"X0": 1}, max_steps=1)


def test_st_comparison_does_not_depend_on_relay_m63():
    host = PLCformerHost("Y0 := (D0 == 0);", is_structured_text=True)
    host.gdn_memory.write_scalar("M63", 1)
    assert host.run_scan_cycle().outputs_y["Y0"] == 1
    with pytest.raises(ValueError):
        PLCformerHost("Y0 := (D0 < 1.5);", is_structured_text=True)


def test_direct_numba_baseline_against_reference_on_all_programs():
    from benchmarks.direct_baseline import run_direct
    from benchmarks.verify_equivalence import BENCHMARKS, PROGRAMS_DIR
    for _, filename, generator, seed in BENCHMARKS:
        instructions = ILAssembler.assemble_file(PROGRAMS_DIR / filename)
        program = np.array([i.to_vector() for i in instructions], dtype=np.int64)
        inputs = generator(1000, seed)
        reference = ReferenceSoftPLC(instructions)
        expected = np.stack([reference.step_scan_cycle(row) for row in inputs])
        assert np.array_equal(run_direct(program, inputs), expected)
