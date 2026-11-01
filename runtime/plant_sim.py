"""
Clean-Room Reference Soft-PLC, Industrial Plant Simulators, In-Model Hybrid Safety Guardrail,
and Differentiable PLC Parameter Tuner (`runtime/plant_sim.py`).

Supports the 4 standard industrial automation benchmarks:
  1. Motor Start/Stop with Thermal Overload Interlock (`self_hold.il`)
  2. Star-Delta Motor Starter (`star_delta.il`)
  3. 4-Way Sequential Traffic Light Controller (`traffic_light.il`)
  4. Automated Conveyor Sorting & Batching System (`sorting.il`)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compiler.assembler import ILAssembler
from compiler.isa import Instruction, Opcode
from compiler.memory_map import HeadID, MemoryMap, STATE_DIM
from compiler.mlp_alu import ConstructiveSwiGLUMLP, SwiGLUScalingCoeffs
from compiler.numeric import WORD_MIN, WORD_MAX, require_integer


class ReferenceSoftPLC:
    """
    Clean-room Reference IEC 61131-3 Soft-PLC Interpreter written in pure Python/NumPy
    (Zero neural network / matrix-multiplication primitives).

    Used as the Ground-Truth Formal Oracle (following `PLCverif` semantics, Adiego et al., 2015)
    to verify Bit-Exact Equivalence against PLCformer across 100,000+ scan cycles.
    """

    def __init__(self, instructions: List[Instruction], dt: float = 1.0) -> None:
        self.instructions = instructions
        self.dt = require_integer(dt, 1)
        self.bits = np.zeros(STATE_DIM, dtype=np.int32)
        self.words = np.zeros((STATE_DIM, 3), dtype=np.float64)
        self.stack = np.zeros(MemoryMap.STACK_DEPTH, dtype=np.int32)

    @classmethod
    def from_file(cls, path: str | Path, dt: float = 1.0) -> "ReferenceSoftPLC":
        return cls(ILAssembler.assemble_file(path), dt=dt)

    @classmethod
    def from_source(cls, source: str, is_st: bool = False, dt: float = 1.0) -> "ReferenceSoftPLC":
        instrs = ILAssembler.assemble_st(source) if is_st else ILAssembler.assemble(source)
        return cls(instrs, dt=dt)

    def reset(self) -> None:
        self.bits.fill(0)
        self.words.fill(0.0)
        self.stack.fill(0)

    def set_inputs(self, inputs: Dict[str, int | float]) -> None:
        for sym, val in inputs.items():
            spec = MemoryMap.resolve(sym, bit_context=True)
            self.bits[spec.row] = 1 if float(val) >= 0.5 else 0

    def step_scan_cycle(self, input_row: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute 1 complete IEC 61131-3 Scan Cycle and return Y0..Y15 outputs."""
        if input_row is not None:
            for i in range(min(16, len(input_row))):
                self.bits[i] = 1 if float(input_row[i]) >= 0.5 else 0

        acc = 0
        for instr in self.instructions:
            op = instr.opcode
            addr = instr.operand_addr
            imm = instr.immediate

            if op == Opcode.NOP:
                continue
            elif op == Opcode.END:
                break
            elif op == Opcode.LD:
                acc = int(self.bits[addr])
            elif op == Opcode.LDI:
                acc = 1 - int(self.bits[addr])
            elif op == Opcode.AND:
                acc = acc & int(self.bits[addr])
            elif op == Opcode.ANI:
                acc = acc & (1 - int(self.bits[addr]))
            elif op == Opcode.OR:
                acc = acc | int(self.bits[addr])
            elif op == Opcode.ORI:
                acc = acc | (1 - int(self.bits[addr]))
            elif op == Opcode.OUT:
                self.bits[addr] = acc
            elif op == Opcode.SET:
                if acc == 1:
                    self.bits[addr] = 1
            elif op == Opcode.RST:
                if acc == 1:
                    self.bits[addr] = 0
                    if addr >= MemoryMap.T_DN_START:
                        w_row = addr - 32
                        self.words[w_row, :] = 0.0
            elif op == Opcode.MPS:
                self.stack[1:] = self.stack[:-1]
                self.stack[0] = acc
            elif op == Opcode.MRD:
                acc = int(self.stack[0])
            elif op == Opcode.MPP:
                acc = int(self.stack[0])
                self.stack[:-1] = self.stack[1:]
                self.stack[-1] = 0
            elif op == Opcode.TON:
                w_row = addr - 32
                if acc == 1:
                    new_t = self.words[w_row, 0] + self.dt
                    if new_t > WORD_MAX:
                        raise OverflowError("PLC-IR timer overflow")
                    self.words[w_row, 0] = new_t
                    self.words[w_row, 1] = imm
                    self.words[w_row, 2] = 1.0
                    dn = 1 if new_t >= imm else 0
                    self.bits[addr] = dn
                    acc = dn
                else:
                    self.words[w_row, 0] = 0.0
                    self.words[w_row, 1] = imm
                    self.words[w_row, 2] = 0.0
                    self.bits[addr] = 0
                    acc = 0
            elif op == Opcode.CTU:
                w_row = addr - 32
                prev_clk = int(round(self.words[w_row, 2]))
                rising = 1 if (acc == 1 and prev_clk == 0) else 0
                new_c = self.words[w_row, 0] + rising
                if new_c > WORD_MAX:
                    raise OverflowError("PLC-IR counter overflow")
                self.words[w_row, 0] = new_c
                self.words[w_row, 1] = imm
                self.words[w_row, 2] = float(acc)
                dn = 1 if new_c >= imm else 0
                self.bits[addr] = dn
                acc = dn
            elif op == Opcode.MOV:
                if acc == 1:
                    self.words[addr, 0] = imm
            elif op == Opcode.ADD:
                if acc == 1:
                    new_word = self.words[addr, 0] + imm
                    if not WORD_MIN <= new_word <= WORD_MAX:
                        raise OverflowError("PLC-IR word overflow")
                    self.words[addr, 0] = new_word
            elif op == Opcode.CMP_EQ:
                acc = acc & (1 if int(round(self.words[addr, 0])) == int(round(imm)) else 0)
            elif op == Opcode.CMP_GT:
                acc = acc & (1 if int(round(self.words[addr, 0])) > int(round(imm)) else 0)
            elif op == Opcode.CMP_LT:
                acc = acc & (1 if int(round(self.words[addr, 0])) < int(round(imm)) else 0)

        return self.bits[MemoryMap.Y_START : MemoryMap.Y_START + MemoryMap.Y_COUNT].copy()


class IndustrialPlantInputGenerator:
    """
    Generates synthetic input traces of shape `(n_cycles, 16)` for four
    bundled examples. There is no plant dynamics model or closed-loop feedback.
    """

    @staticmethod
    def generate_motor_start_stop_trace(n_cycles: int, seed: int = 42) -> np.ndarray:
        """
        Task 1: Momentary Start pulses (X0) and periodic Thermal Overload / Stop trips (X1).
        """
        rng = np.random.default_rng(seed)
        inputs = np.zeros((n_cycles, 16), dtype=np.float32)
        for c in range(n_cycles):
            r = rng.random()
            if r < 0.08:
                inputs[c, 0] = 1.0  # Start button pulse
            elif r < 0.12:
                inputs[c, 1] = 1.0  # Stop / Thermal Overload trip
        return inputs

    @staticmethod
    def generate_star_delta_trace(n_cycles: int, seed: int = 43) -> np.ndarray:
        """
        Task 2: Start pulse (X0) held long enough for Star-to-Delta transition (K=5),
        followed by periodic Stop/Reset (X1).
        """
        rng = np.random.default_rng(seed)
        inputs = np.zeros((n_cycles, 16), dtype=np.float32)
        c = 0
        while c < n_cycles:
            inputs[c, 0] = 1.0  # Start pulse
            run_len = int(rng.integers(3, 15))
            c += run_len
            if c < n_cycles:
                inputs[c, 1] = 1.0  # Stop pulse
            c += int(rng.integers(1, 4))
        return inputs

    @staticmethod
    def generate_traffic_light_trace(n_cycles: int, seed: int = 44) -> np.ndarray:
        """
        Task 3: Master enable X0=1 for multi-phase cascades, with occasional maintenance resets (X0=0).
        """
        rng = np.random.default_rng(seed)
        inputs = np.zeros((n_cycles, 16), dtype=np.float32)
        inputs[:, 0] = 1.0
        # Every ~40 cycles, briefly toggle X0=0 to test mid-sequence reset
        for c in range(0, n_cycles, 40):
            if rng.random() < 0.5:
                inputs[c, 0] = 0.0
        return inputs

    @staticmethod
    def generate_conveyor_sorting_trace(n_cycles: int, seed: int = 45) -> np.ndarray:
        """
        Task 4: Conveyor run switch X2=1, part sensor pulses on X0, and batch reset pulses on X1.
        """
        rng = np.random.default_rng(seed)
        inputs = np.zeros((n_cycles, 16), dtype=np.float32)
        inputs[:, 2] = 1.0  # Conveyor run switch ON
        for c in range(n_cycles):
            r = rng.random()
            if r < 0.25:
                inputs[c, 0] = 1.0  # Part sensor pulse (may span 1 or more cycles)
            elif r < 0.32:
                inputs[c, 1] = 1.0  # Batch reset pulse
        return inputs


class HybridSafetyGuardrail(nn.Module):
    """Compiled Boolean interlock over three proposed commands and one stop bit.

    The benchmark uses synthetic binary proposals; no LLM is executed.
    Outputs implement mutual exclusion, stop inhibition, and an alarm.
    """

    SAFETY_ST_PROGRAM = """
    Y0 := X0 AND NOT X1 AND NOT X3;
    Y1 := X1 AND NOT X0 AND NOT X3;
    Y2 := X2 AND NOT X3;
    M0 := X0 AND X1;
    Y3 := M0 OR X3;
    """

    def __init__(self) -> None:
        super().__init__()
        from runtime.host import PLCformerHost

        self.host = PLCformerHost(
            program_source=self.SAFETY_ST_PROGRAM,
            is_structured_text=True,
        )

    def forward_guardrail_batch(self, raw_inputs: torch.Tensor) -> torch.Tensor:
        """
        Filter `(B, 128)` process-image vectors using the Boolean unrolled model.
        """
        assert self.host.unrolled_model is not None
        return self.host.unrolled_model(raw_inputs)


class DifferentiablePLCTuner(nn.Module):
    """Standalone scalar sigmoid surrogate for fitting a switching threshold.

    This model does not differentiate through compiled SwiGLU weights, gated
    memory, or plant dynamics. The historical class name is retained for callers.
    """

    def __init__(self, initial_k: float = 1.5, tau: float = 3.0) -> None:
        super().__init__()
        self.k_preset = nn.Parameter(torch.tensor(float(initial_k), dtype=torch.float32))
        self.tau = float(tau)

    def forward_star_delta_trace(self, time_ticks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable Star-Delta transition over scan cycle ticks `t = 1..T`:
          - T0.DN(t) = sigmoid(tau * (t - K + 0.5))
          - Star Contactor Y1(t)  = 1.0 - T0.DN(t)
          - Delta Contactor Y2(t) = T0.DN(t)
        """
        t_dn = torch.sigmoid(self.tau * (time_ticks - self.k_preset + 0.5))
        y_star = 1.0 - t_dn
        y_delta = t_dn
        return y_star, y_delta

