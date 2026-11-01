"""Constructive SwiGLU Boolean and bounded-integer PLC-IR ALU.

The two-neuron Boolean selection is an identity over real arithmetic.
Comparators approximate rectified ramps with scaled SiLU and use bounded
opcode inhibition. Float32 execution also requires the runtime normalization
and overflow policy; no infinite-horizon numerical theorem is claimed.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .isa import Opcode
from .numeric import COMPARATOR_INHIBITION, DEFAULT_SCALE


# Number of opcodes in ISA (0..20)
NUM_OPCODES: int = len(Opcode)

# Residual stream layout entering the Layer 2 SwiGLU ALU:
#   idx 0: CONST_1    (always 1.0)
#   idx 1: ACC        (current Boolean accumulator in {0.0, 1.0})
#   idx 2: MEM_BIT    (fetched operand bit Mem[addr] in {0.0, 1.0})
#   idx 3: MEM_WORD   (fetched operand word/accumulator value in R)
#   idx 4: IMM_VAL    (instruction immediate value K_val in R)
#   idx 5: DT_VAL     (scan cycle delta-time dt for TON timer, default 1.0)
#   idx 6: PREV_CLK   (previous rung input state C[i].PREV for CTU rising edge)
#   idx 7: STACK_TOP  (top of CPU boolean stack STACK[0] for MRD/MPP)
#   idx 8 .. 8 + NUM_OPCODES - 1: One-hot Opcode selector vector e_{opcode}
ALU_INPUT_DIM: int = 8 + NUM_OPCODES

# Output channels produced by the Layer 2 SwiGLU ALU:
#   out 0: NEW_ACC         (updated Boolean accumulator value in {0.0, 1.0})
#   out 1: WRITE_BIT_VAL   (new bit value to write to Head 0 when WRITE_BIT_EN == 1)
#   out 2: WRITE_BIT_EN    (write gate beta_bit in {0.0, 1.0} for Head 0)
#   out 3: IS_SCAN_END     (1.0 if Opcode == END else 0.0)
#   out 4: WRITE_WORD_VAL  (new word/register value to write to Head 1)
#   out 5: WRITE_WORD_EN   (write gate beta_word in {0.0, 1.0} for Head 1)
#   out 6: STACK_PUSH      (1.0 if Opcode == MPS else 0.0)
#   out 7: STACK_POP       (1.0 if Opcode == MPP else 0.0)
ALU_OUTPUT_DIM: int = 8


@dataclass(frozen=True)
class SwiGLUScalingCoeffs:
    """Closed-form coefficients for Exact Bilinear and Ternary SwiGLU Gates."""
    c: float
    alpha_pos: float
    alpha_neg: float
    gamma_bilinear: float
    w_pos: float
    w_neg: float

    @classmethod
    def compute(cls, c: float = DEFAULT_SCALE) -> "SwiGLUScalingCoeffs":
        t_pos = torch.tensor(float(c), dtype=torch.float32)
        t_neg = torch.tensor(-float(c), dtype=torch.float32)
        alpha_pos = float(F.silu(t_pos).item())
        alpha_neg = float(F.silu(t_neg).item())
        denom = float(
            (F.silu(t_pos) * F.silu(t_pos) - F.silu(t_neg) * F.silu(t_neg)).item()
        )
        w_pos = float((F.silu(t_pos) / denom).item())
        w_neg = float((-F.silu(t_neg) / denom).item())
        gamma_bilinear = float((1.0 / F.silu(t_pos)).item())
        return cls(
            c=float(c),
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            gamma_bilinear=gamma_bilinear,
            w_pos=w_pos,
            w_neg=w_neg,
        )


class ConstructiveSwiGLUMLP(nn.Module):
    """
    Custom bias-free SwiGLU module:
        y = (SiLU(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T
    No bias terms are used. This class does not instantiate a transformer model.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False, dtype=dtype, device=device)

        with torch.no_grad():
            self.gate_proj.weight.zero_()
            self.up_proj.weight.zero_()
            self.down_proj.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PLCBooleanALU(nn.Module):
    """
    Zero-training Constructive SwiGLU Boolean & Word ALU for IEC 61131-3 IL (PLC-IR).

    Compiles the full PLCformer ISA (`LD`, `LDI`, `AND`, `ANI`, `OR`, `ORI`,
    `OUT`, `SET`, `RST`, `MPS`, `MRD`, `MPP`, `TON`, `CTU`, `MOV`, `ADD`,
    `CMP_EQ`, `CMP_GT`, `CMP_LT`, `NOP`, `END`) into the weights of a single
    custom bias-free SwiGLU MLP layer.
    """

    IDX_CONST = 0
    IDX_ACC = 1
    IDX_MEM_BIT = 2
    IDX_MEM_WORD = 3
    IDX_IMM = 4
    IDX_DT = 5
    IDX_PREV_CLK = 6
    IDX_STACK_TOP = 7
    IDX_OPCODE_BASE = 8

    OUT_NEW_ACC = 0
    OUT_WRITE_BIT_VAL = 1
    OUT_WRITE_BIT_EN = 2
    OUT_IS_SCAN_END = 3
    OUT_WRITE_WORD_VAL = 4
    OUT_WRITE_WORD_EN = 5
    OUT_STACK_PUSH = 6
    OUT_STACK_POP = 7

    # Larger than any signed-16-bit operand difference plus a timer increment.
    # The default power-of-two scale avoids reciprocal scaling error on integers.
    # Arithmetic overflow is rejected by the runtime, never silently wrapped.
    BIG_M: float = COMPARATOR_INHIBITION

    def __init__(
        self,
        scale_c: float = DEFAULT_SCALE,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.coeffs = SwiGLUScalingCoeffs.compute(scale_c)
        self.hidden_dim = 128
        self.mlp = ConstructiveSwiGLUMLP(
            in_features=ALU_INPUT_DIM,
            hidden_features=self.hidden_dim,
            out_features=ALU_OUTPUT_DIM,
            dtype=dtype,
            device=device,
        )
        self._cursor = 0
        self._compile_weights()

    def _op_idx(self, opcode: Opcode) -> int:
        return self.IDX_OPCODE_BASE + int(opcode)

    def _add_bilinear_gate(
        self,
        gate_terms: Dict[int, float],
        up_terms: Dict[int, float],
        out_channel: int,
        scale_out: float = 1.0,
    ) -> None:
        """
        Compile a 2-way product:
            out[out_channel] += scale_out * gamma_c * SiLU(c * sum(gate_terms)) * sum(up_terms)
        Valid when sum(gate_terms) in {0, 1}.
        """
        h = self._cursor
        if h >= self.hidden_dim:
            raise RuntimeError("Exceeded allocated SwiGLU hidden neurons")
        self._cursor += 1

        with torch.no_grad():
            for idx, weight in gate_terms.items():
                self.mlp.gate_proj.weight[h, idx] = self.coeffs.c * weight
            for idx, weight in up_terms.items():
                self.mlp.up_proj.weight[h, idx] = weight
            self.mlp.down_proj.weight[out_channel, h] = scale_out * self.coeffs.gamma_bilinear

    def _add_ternary_gate(
        self,
        sel_terms: Dict[int, float],
        a_terms: Dict[int, float],
        up_terms: Dict[int, float],
        out_channel: int,
        scale_out: float = 1.0,
    ) -> None:
        """
        Compile an exact 3-way product (Lemma 2):
            out[out_channel] += scale_out * (sel * a * up)
        where `sel = sum(sel_terms) in {0, 1}` and `a = sum(a_terms) in {0, 1}`.
        """
        h_pos = self._cursor
        h_neg = self._cursor + 1
        if h_neg >= self.hidden_dim:
            raise RuntimeError("Exceeded allocated SwiGLU hidden neurons")
        self._cursor += 2

        z_terms: Dict[int, float] = {self.IDX_CONST: -1.0}
        for idx, w in sel_terms.items():
            z_terms[idx] = z_terms.get(idx, 0.0) + w
        for idx, w in a_terms.items():
            z_terms[idx] = z_terms.get(idx, 0.0) + w

        with torch.no_grad():
            for idx, w in z_terms.items():
                self.mlp.gate_proj.weight[h_pos, idx] = self.coeffs.c * w
                self.mlp.gate_proj.weight[h_neg, idx] = -self.coeffs.c * w
            for idx, w in up_terms.items():
                self.mlp.up_proj.weight[h_pos, idx] = w
                self.mlp.up_proj.weight[h_neg, idx] = w

            self.mlp.down_proj.weight[out_channel, h_pos] = scale_out * self.coeffs.w_pos
            self.mlp.down_proj.weight[out_channel, h_neg] = scale_out * self.coeffs.w_neg

    def _add_ramp_neuron(
        self,
        diff_terms: Dict[int, float],
        up_terms: Dict[int, float],
        out_components: List[Tuple[int, float]],
    ) -> None:
        """
        Add a single ramp neuron computing:
            r = (1 / c) * SiLU(c * sum(diff_terms)) * sum(up_terms)
              == max(0, sum(diff_terms)) * sum(up_terms)  (for integer sum(diff_terms) in Z)
        and routing `scale * r` to each `(out_channel, scale)` in `out_components`.
        """
        h = self._cursor
        if h >= self.hidden_dim:
            raise RuntimeError("Exceeded allocated SwiGLU hidden neurons")
        self._cursor += 1

        inv_c = 1.0 / self.coeffs.c
        with torch.no_grad():
            for idx, w in diff_terms.items():
                self.mlp.gate_proj.weight[h, idx] = self.coeffs.c * w
            for idx, w in up_terms.items():
                self.mlp.up_proj.weight[h, idx] = w
            for out_ch, scale in out_components:
                self.mlp.down_proj.weight[out_ch, h] = scale * inv_c

    def _add_step_ge_zero(
        self,
        diff_terms: Dict[int, float],
        opcode: Opcode,
        up_terms: Dict[int, float],
        out_components: List[Tuple[int, float]],
        extra_inhibit_terms: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Compile an exact integer step comparator:
            I[sum(diff_terms) >= 0] * I[Opcode == opcode] * sum(up_terms)
        using 2 ramp neurons:
            max(0, d + 1) - max(0, d)
        where d = sum(diff_terms) - BIG_M * (1 - e_{opcode}) - BIG_M * extra_inhibit.
        """
        op_i = self._op_idx(opcode)
        c1 = self.IDX_CONST

        # Base d terms with Big-M inhibition when opcode == 0
        d_base: Dict[int, float] = {
            c1: -self.BIG_M,
            op_i: self.BIG_M,
        }
        for idx, w in diff_terms.items():
            d_base[idx] = d_base.get(idx, 0.0) + w
        if extra_inhibit_terms:
            for idx, w in extra_inhibit_terms.items():
                d_base[idx] = d_base.get(idx, 0.0) - self.BIG_M * w

        # Neuron 1: max(0, d + 1) with +1 weight
        d_plus_1 = dict(d_base)
        d_plus_1[c1] = d_plus_1.get(c1, 0.0) + 1.0
        self._add_ramp_neuron(
            diff_terms=d_plus_1,
            up_terms=up_terms,
            out_components=[(ch, +sc) for ch, sc in out_components],
        )

        # Neuron 2: -max(0, d) with -1 weight
        self._add_ramp_neuron(
            diff_terms=d_base,
            up_terms=up_terms,
            out_components=[(ch, -sc) for ch, sc in out_components],
        )

    def _add_step_eq_zero(
        self,
        diff_terms: Dict[int, float],
        opcode: Opcode,
        up_terms: Dict[int, float],
        out_components: List[Tuple[int, float]],
        extra_inhibit_terms: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Compile an exact integer equality comparator:
            I[sum(diff_terms) == 0] * I[Opcode == opcode] * sum(up_terms)
        using 3 ramp neurons:
            max(0, d + 1) - 2 * max(0, d) + max(0, d - 1).
        """
        op_i = self._op_idx(opcode)
        c1 = self.IDX_CONST

        d_base: Dict[int, float] = {
            c1: -self.BIG_M,
            op_i: self.BIG_M,
        }
        for idx, w in diff_terms.items():
            d_base[idx] = d_base.get(idx, 0.0) + w
        if extra_inhibit_terms:
            for idx, w in extra_inhibit_terms.items():
                d_base[idx] = d_base.get(idx, 0.0) - self.BIG_M * w

        d_plus_1 = dict(d_base)
        d_plus_1[c1] = d_plus_1.get(c1, 0.0) + 1.0

        d_minus_1 = dict(d_base)
        d_minus_1[c1] = d_minus_1.get(c1, 0.0) - 1.0

        self._add_ramp_neuron(d_plus_1, up_terms, [(ch, +1.0 * sc) for ch, sc in out_components])
        self._add_ramp_neuron(d_base, up_terms, [(ch, -2.0 * sc) for ch, sc in out_components])
        self._add_ramp_neuron(d_minus_1, up_terms, [(ch, +1.0 * sc) for ch, sc in out_components])

    def _compile_weights(self) -> None:
        """
        Constructively compile all Phase 1 & Phase 2 IEC 61131-3 IL instructions into SwiGLU weights.
        """
        c1 = self.IDX_CONST
        acc = self.IDX_ACC
        m = self.IDX_MEM_BIT
        w_mem = self.IDX_MEM_WORD
        imm = self.IDX_IMM
        dt = self.IDX_DT
        prev_clk = self.IDX_PREV_CLK
        stack_top = self.IDX_STACK_TOP

        # --------------------------------------------------------------------
        # Phase 1: Core Boolean Instructions
        # --------------------------------------------------------------------

        # 1. LD addr: NEW_ACC = Mem[addr]
        self._add_bilinear_gate({self._op_idx(Opcode.LD): 1.0}, {m: 1.0}, self.OUT_NEW_ACC)

        # 2. LDI addr: NEW_ACC = 1.0 - Mem[addr]
        self._add_bilinear_gate({self._op_idx(Opcode.LDI): 1.0}, {c1: 1.0, m: -1.0}, self.OUT_NEW_ACC)

        # 3. AND addr: NEW_ACC = ACC * Mem[addr]
        self._add_ternary_gate({self._op_idx(Opcode.AND): 1.0}, {acc: 1.0}, {m: 1.0}, self.OUT_NEW_ACC)

        # 4. ANI addr: NEW_ACC = ACC * (1.0 - Mem[addr])
        self._add_ternary_gate({self._op_idx(Opcode.ANI): 1.0}, {acc: 1.0}, {c1: 1.0, m: -1.0}, self.OUT_NEW_ACC)

        # 5. OR addr: NEW_ACC = ACC + (1 - ACC) * Mem[addr]
        self._add_bilinear_gate({self._op_idx(Opcode.OR): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)
        self._add_ternary_gate({self._op_idx(Opcode.OR): 1.0}, {c1: 1.0, acc: -1.0}, {m: 1.0}, self.OUT_NEW_ACC)

        # 6. ORI addr: NEW_ACC = ACC + (1 - ACC) * (1 - Mem[addr])
        self._add_bilinear_gate({self._op_idx(Opcode.ORI): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)
        self._add_ternary_gate(
            {self._op_idx(Opcode.ORI): 1.0}, {c1: 1.0, acc: -1.0}, {c1: 1.0, m: -1.0}, self.OUT_NEW_ACC
        )

        # 7. OUT addr
        self._add_bilinear_gate({self._op_idx(Opcode.OUT): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)
        self._add_bilinear_gate({self._op_idx(Opcode.OUT): 1.0}, {acc: 1.0}, self.OUT_WRITE_BIT_VAL)
        self._add_bilinear_gate({self._op_idx(Opcode.OUT): 1.0}, {c1: 1.0}, self.OUT_WRITE_BIT_EN)

        # 8. SET addr (Latch ON when ACC == 1)
        self._add_bilinear_gate({self._op_idx(Opcode.SET): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)
        self._add_bilinear_gate({self._op_idx(Opcode.SET): 1.0}, {c1: 1.0}, self.OUT_WRITE_BIT_VAL)
        self._add_bilinear_gate({self._op_idx(Opcode.SET): 1.0}, {acc: 1.0}, self.OUT_WRITE_BIT_EN)

        # 9. RST addr (Reset OFF to 0.0 when ACC == 1; also resets Timer/Counter word accumulator)
        self._add_bilinear_gate({self._op_idx(Opcode.RST): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)
        self._add_bilinear_gate({self._op_idx(Opcode.RST): 1.0}, {acc: 1.0}, self.OUT_WRITE_BIT_EN)
        self._add_bilinear_gate({self._op_idx(Opcode.RST): 1.0}, {acc: 1.0}, self.OUT_WRITE_WORD_EN)

        # 10. NOP
        self._add_bilinear_gate({self._op_idx(Opcode.NOP): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)

        # 11. END
        self._add_bilinear_gate({self._op_idx(Opcode.END): 1.0}, {c1: 1.0}, self.OUT_IS_SCAN_END)

        # --------------------------------------------------------------------
        # Phase 2: Stack Instructions (MPS, MRD, MPP)
        # --------------------------------------------------------------------

        # 12. MPS (Push ACC onto stack): preserves ACC, emits STACK_PUSH = 1.0
        self._add_bilinear_gate({self._op_idx(Opcode.MPS): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)
        self._add_bilinear_gate({self._op_idx(Opcode.MPS): 1.0}, {c1: 1.0}, self.OUT_STACK_PUSH)

        # 13. MRD (Read top of stack into ACC): NEW_ACC = STACK_TOP
        self._add_bilinear_gate({self._op_idx(Opcode.MRD): 1.0}, {stack_top: 1.0}, self.OUT_NEW_ACC)

        # 14. MPP (Pop top of stack into ACC): NEW_ACC = STACK_TOP, emits STACK_POP = 1.0
        self._add_bilinear_gate({self._op_idx(Opcode.MPP): 1.0}, {stack_top: 1.0}, self.OUT_NEW_ACC)
        self._add_bilinear_gate({self._op_idx(Opcode.MPP): 1.0}, {c1: 1.0}, self.OUT_STACK_POP)

        # --------------------------------------------------------------------
        # Phase 2: Timer (TON) & Counter (CTU)
        # --------------------------------------------------------------------

        # 15. TON T_idx, K_val (On-Delay Timer):
        #   - WRITE_WORD_EN = 1.0 (always updates T.ACC: increments when ACC==1, resets to 0 when ACC==0)
        #   - WRITE_WORD_VAL = ACC * (MEM_WORD + DT)
        #   - WRITE_BIT_EN = 1.0
        #   - TON_DN = ACC * I[MEM_WORD + DT - IMM >= 0] -> routed to WRITE_BIT_VAL and NEW_ACC
        self._add_bilinear_gate({self._op_idx(Opcode.TON): 1.0}, {c1: 1.0}, self.OUT_WRITE_WORD_EN)
        self._add_bilinear_gate({self._op_idx(Opcode.TON): 1.0}, {c1: 1.0}, self.OUT_WRITE_BIT_EN)
        self._add_ternary_gate(
            sel_terms={self._op_idx(Opcode.TON): 1.0},
            a_terms={acc: 1.0},
            up_terms={w_mem: 1.0, dt: 1.0},
            out_channel=self.OUT_WRITE_WORD_VAL,
        )
        self._add_step_ge_zero(
            diff_terms={w_mem: 1.0, dt: 1.0, imm: -1.0},
            opcode=Opcode.TON,
            up_terms={acc: 1.0},
            out_components=[(self.OUT_WRITE_BIT_VAL, 1.0), (self.OUT_NEW_ACC, 1.0)],
        )

        # 16. CTU C_idx, K_val (Count-Up Counter on rising edge of ACC):
        #   - rising_edge = ACC * (1.0 - PREV_CLK)
        #   - WRITE_WORD_VAL = MEM_WORD + ACC * (1.0 - PREV_CLK)
        #   - WRITE_WORD_EN = 1.0, WRITE_BIT_EN = 1.0
        #   - CTU_DN = I[MEM_WORD >= IMM] + ACC * (1 - PREV_CLK) * I[MEM_WORD == IMM - 1]
        self._add_bilinear_gate({self._op_idx(Opcode.CTU): 1.0}, {c1: 1.0}, self.OUT_WRITE_WORD_EN)
        self._add_bilinear_gate({self._op_idx(Opcode.CTU): 1.0}, {c1: 1.0}, self.OUT_WRITE_BIT_EN)
        self._add_bilinear_gate({self._op_idx(Opcode.CTU): 1.0}, {w_mem: 1.0}, self.OUT_WRITE_WORD_VAL)
        self._add_ternary_gate(
            sel_terms={self._op_idx(Opcode.CTU): 1.0},
            a_terms={acc: 1.0},
            up_terms={c1: 1.0, prev_clk: -1.0},
            out_channel=self.OUT_WRITE_WORD_VAL,
        )
        # Term A of CTU_DN: I[MEM_WORD - IMM >= 0]
        self._add_step_ge_zero(
            diff_terms={w_mem: 1.0, imm: -1.0},
            opcode=Opcode.CTU,
            up_terms={c1: 1.0},
            out_components=[(self.OUT_WRITE_BIT_VAL, 1.0), (self.OUT_NEW_ACC, 1.0)],
        )
        # Term B of CTU_DN: ACC * I[PREV_CLK == 0] * I[MEM_WORD - IMM + 1 == 0]
        self._add_step_eq_zero(
            diff_terms={w_mem: 1.0, imm: -1.0, c1: 1.0},
            opcode=Opcode.CTU,
            up_terms={acc: 1.0},
            out_components=[(self.OUT_WRITE_BIT_VAL, 1.0), (self.OUT_NEW_ACC, 1.0)],
            extra_inhibit_terms={prev_clk: 1.0},
        )

        # --------------------------------------------------------------------
        # Phase 2: Word / Register Move, Addition, and Comparisons
        # --------------------------------------------------------------------

        # 17. MOV D_idx, K_val (when ACC == 1, D[idx] = IMM)
        self._add_bilinear_gate({self._op_idx(Opcode.MOV): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)
        self._add_bilinear_gate({self._op_idx(Opcode.MOV): 1.0}, {acc: 1.0}, self.OUT_WRITE_WORD_EN)
        self._add_bilinear_gate({self._op_idx(Opcode.MOV): 1.0}, {imm: 1.0}, self.OUT_WRITE_WORD_VAL)

        # 18. ADD D_idx, K_val (when ACC == 1, D[idx] = MEM_WORD + IMM)
        self._add_bilinear_gate({self._op_idx(Opcode.ADD): 1.0}, {acc: 1.0}, self.OUT_NEW_ACC)
        self._add_bilinear_gate({self._op_idx(Opcode.ADD): 1.0}, {acc: 1.0}, self.OUT_WRITE_WORD_EN)
        self._add_bilinear_gate({self._op_idx(Opcode.ADD): 1.0}, {w_mem: 1.0, imm: 1.0}, self.OUT_WRITE_WORD_VAL)

        # 19. CMP_EQ D_idx, K_val: NEW_ACC = ACC * I[MEM_WORD - IMM == 0]
        self._add_step_eq_zero(
            diff_terms={w_mem: 1.0, imm: -1.0},
            opcode=Opcode.CMP_EQ,
            up_terms={acc: 1.0},
            out_components=[(self.OUT_NEW_ACC, 1.0)],
        )

        # 20. CMP_GT D_idx, K_val: NEW_ACC = ACC * I[MEM_WORD - IMM - 1 >= 0]
        self._add_step_ge_zero(
            diff_terms={w_mem: 1.0, imm: -1.0, c1: -1.0},
            opcode=Opcode.CMP_GT,
            up_terms={acc: 1.0},
            out_components=[(self.OUT_NEW_ACC, 1.0)],
        )

        # 21. CMP_LT D_idx, K_val: NEW_ACC = ACC * I[IMM - MEM_WORD - 1 >= 0]
        self._add_step_ge_zero(
            diff_terms={imm: 1.0, w_mem: -1.0, c1: -1.0},
            opcode=Opcode.CMP_LT,
            up_terms={acc: 1.0},
            out_components=[(self.OUT_NEW_ACC, 1.0)],
        )

    @property
    def active_neurons(self) -> int:
        """Number of compiled hidden neurons actually used in the SwiGLU layer."""
        return self._cursor

    def build_input_vector(
        self,
        opcode: Opcode,
        acc: float,
        mem_bit: float,
        mem_word: float = 0.0,
        imm_val: float = 0.0,
        dt_val: float = 1.0,
        prev_clk: float = 0.0,
        stack_top: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Construct the 1D residual stream input vector x for the SwiGLU ALU."""
        x = torch.zeros(ALU_INPUT_DIM, dtype=dtype, device=device)
        x[self.IDX_CONST] = 1.0
        x[self.IDX_ACC] = float(acc)
        x[self.IDX_MEM_BIT] = float(mem_bit)
        x[self.IDX_MEM_WORD] = float(mem_word)
        x[self.IDX_IMM] = float(imm_val)
        x[self.IDX_DT] = float(dt_val)
        x[self.IDX_PREV_CLK] = float(prev_clk)
        x[self.IDX_STACK_TOP] = float(stack_top)
        x[self._op_idx(opcode)] = 1.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the compiled SwiGLU MLP forward pass."""
        return self.mlp(x)

    def evaluate_step(
        self,
        opcode: Opcode,
        acc: float,
        mem_bit: float = 0.0,
        mem_word: float = 0.0,
        imm_val: float = 0.0,
        dt_val: float = 1.0,
        prev_clk: float = 0.0,
        stack_top: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate a single instruction step and return the 4 primary boolean channels
        (new_acc, write_bit_val, write_bit_en, is_scan_end) for backward compatibility.
        """
        y = self.evaluate_full(
            opcode=opcode,
            acc=acc,
            mem_bit=mem_bit,
            mem_word=mem_word,
            imm_val=imm_val,
            dt_val=dt_val,
            prev_clk=prev_clk,
            stack_top=stack_top,
        )
        return (
            float(y[self.OUT_NEW_ACC].item()),
            float(y[self.OUT_WRITE_BIT_VAL].item()),
            float(y[self.OUT_WRITE_BIT_EN].item()),
            float(y[self.OUT_IS_SCAN_END].item()),
        )

    def evaluate_full(
        self,
        opcode: Opcode,
        acc: float,
        mem_bit: float = 0.0,
        mem_word: float = 0.0,
        imm_val: float = 0.0,
        dt_val: float = 1.0,
        prev_clk: float = 0.0,
        stack_top: float = 0.0,
    ) -> torch.Tensor:
        """Evaluate a single instruction step and return all 8 ALU output channels."""
        x = self.build_input_vector(
            opcode=opcode,
            acc=acc,
            mem_bit=mem_bit,
            mem_word=mem_word,
            imm_val=imm_val,
            dt_val=dt_val,
            prev_clk=prev_clk,
            stack_top=stack_top,
            dtype=self.mlp.gate_proj.weight.dtype,
            device=self.mlp.gate_proj.weight.device,
        )
        return self.forward(x)
