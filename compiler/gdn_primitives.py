"""
Gated DeltaNet (GDN) Constructive Weight Primitives for PLCformer.

Implements Lemma 1 (Exact Addressable Process Image via Gated Delta Rule):
    S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) k_t^T   (Column-major Key)
and its row-major transpose equivalent:
    S_t = alpha_t * S_{t-1} + beta_t * k_t (v_t^T - k_t^T * alpha_t * S_{t-1})  (Row-major Key)

With alpha_t = 1.0 (no decay), beta_t in {0.0, 1.0} (write enable gate), and
one-hot key vector k_t = e_r in R^{128}, the Gated DeltaNet recurrent state
implements an addressed read/write table of size 128 x 128 per head.
The identity is over real arithmetic; the dense implementation computes full
matrix updates and uses erase-before-write evaluation for one-hot keys.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .isa import Instruction, Opcode
from .memory_map import AddressSpec, HeadID, MemoryMap, NUM_HEADS, STATE_DIM


def one_hot_address(
    index: int,
    dim: int = STATE_DIM,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create a 1D one-hot basis vector e_{index} in R^{dim}."""
    if not (0 <= index < dim):
        raise ValueError(f"Address index {index} out of bounds [0, {dim - 1}]")
    vec = torch.zeros(dim, dtype=dtype, device=device)
    vec[index] = 1.0
    return vec


def gated_delta_rule_step_col(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    beta: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Canonical Gated DeltaNet state update equation (Yang et al., ICLR 2025):
        S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} @ k_t) @ k_t^T

    Args:
        state: S_{t-1} of shape (..., d_v, d_k), where d_v = d_k = 128.
        k: Key vector k_t of shape (..., d_k). When k_t = e_r, addresses column r.
        v: Value vector v_t of shape (..., d_v).
        alpha: Decay gate in [0, 1] (scalar or tensor broadcastable to state).
        beta: Write strength gate in [0, 1] (scalar or tensor broadcastable to state).

    Returns:
        Updated state S_t of shape (..., d_v, d_k).
    """
    decayed_state = alpha * state
    # S_{t-1} @ k_t -> shape (..., d_v)
    retrieved = torch.einsum("...vk,...k->...v", decayed_state, k)
    # Erase then write. For binary beta and a one-hot key this cancels the
    # old cell before adding v, avoiding old + (v-old) cancellation loss.
    erase = torch.einsum("...v,...k->...vk", retrieved, k)
    write = torch.einsum("...v,...k->...vk", v, k)
    return (decayed_state - beta * erase) + beta * write


def gated_delta_rule_step_row(
    state: torch.Tensor,
    k_row: torch.Tensor,
    v_row: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    beta: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Row-addressed Gated DeltaNet state update (exact transpose of `gated_delta_rule_step_col`):
        S_t = alpha_t * S_{t-1} + beta_t * k_t @ (v_t^T - k_t^T @ (alpha_t * S_{t-1}))

    In this convention, row `r` of S_t (i.e. S_t[..., r, :]) stores the value vector v_row,
    matching the human-readable table layout in Section 4.2 of the PLCformer blueprint.
    Mathematically identical to applying `gated_delta_rule_step_col` on S_t^T.
    """
    return gated_delta_rule_step_col(state.transpose(-1, -2), k_row, v_row, alpha=alpha, beta=beta).transpose(-1, -2)


def gdn_read_row(state: torch.Tensor, q_row: torch.Tensor) -> torch.Tensor:
    """
    Read a row vector from state S_t using one-hot query vector q_row = e_r:
        o_t = S_t^T @ q_row = q_row^T @ S_t
    """
    return torch.einsum("...r,...rc->...c", q_row, state)


class GDNProcessImageMemory(nn.Module):
    """
    Multi-Head Gated DeltaNet State Manager for PLCformer.

    Maintains the recurrent state tensor S in R^{B x H x 128 x 128} across 5 heads:
      - Head 0: Bit Memory (X0..15, Y0..15, M0..63, T0..15.DN, C0..15.DN)
      - Head 1: Word / Register Memory (D0..63, T0..15.ACC/PRE, C0..15.ACC/PRE/PREV)
      - Head 2: CPU Internal Control Registers (PC, ACC, STACK[0..7], SCAN_STATUS, SP)
      - Head 3: Instruction ROM Bank 0 (lines 0..127)
      - Head 4: Instruction ROM Bank 1 (lines 128..255)

    Addressed reads/writes use gated-delta primitives. The host also directly
    manages control registers, stack movement, and initialization.
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = NUM_HEADS,
        state_dim: int = STATE_DIM,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.dtype = dtype
        self.register_buffer(
            "state",
            torch.zeros(batch_size, num_heads, state_dim, state_dim, dtype=dtype, device=device),
        )
        # Precomputed identity basis matrix I_{128} for fast one-hot key lookup
        self.register_buffer(
            "basis",
            torch.eye(state_dim, dtype=dtype, device=device),
        )

    def reset_state(self, clear_rom: bool = False) -> None:
        """Reset volatile PLC memory (Heads 0, 1, 2) while optionally preserving ROM (Heads 3, 4)."""
        if clear_rom:
            self.state.zero_()
        else:
            self.state[:, :3, :, :].zero_()

    def read_row_vector(self, head: int | HeadID, row: int, batch_idx: int = 0) -> torch.Tensor:
        """
        Perform a Gated DeltaNet linear attention query read on (head, row):
            o = q_row^T @ S[batch_idx, head]
        where q_row = e_{row} in R^{128}.
        """
        q_row = self.basis[row]  # One-hot vector e_{row}
        return gdn_read_row(self.state[batch_idx, int(head)], q_row)

    def read_scalar(self, symbol_or_spec: str | AddressSpec, batch_idx: int = 0) -> float:
        """Read a single scalar value from the GDN state via one-hot query projection."""
        spec = MemoryMap.resolve(symbol_or_spec) if isinstance(symbol_or_spec, str) else symbol_or_spec
        row_vec = self.read_row_vector(spec.head, spec.row, batch_idx=batch_idx)
        q_col = self.basis[spec.col]
        val = torch.dot(row_vec, q_col)
        return float(val.item())

    def write_row_vector(
        self,
        head: int | HeadID,
        row: int,
        v_row: torch.Tensor,
        beta: float = 1.0,
        alpha: float = 1.0,
        batch_idx: Optional[int] = None,
    ) -> None:
        """
        Update row `row` in `head` using the exact Gated DeltaNet update rule:
            S_t = alpha * S_{t-1} + beta * k_row (v_row^T - k_row^T (alpha * S_{t-1}))
        """
        k_row = self.basis[row]
        h = int(head)
        if batch_idx is None:
            self.state[:, h] = gated_delta_rule_step_row(
                self.state[:, h], k_row, v_row, alpha=alpha, beta=beta
            )
        else:
            self.state[batch_idx, h] = gated_delta_rule_step_row(
                self.state[batch_idx, h], k_row, v_row, alpha=alpha, beta=beta
            )

    def write_scalar(
        self,
        symbol_or_spec: str | AddressSpec,
        value: float,
        beta: float = 1.0,
        alpha: float = 1.0,
        batch_idx: int = 0,
    ) -> None:
        """
        Write a scalar value to an exact PLC address (head, row, col) via the Gated Delta Rule.
        When col == 0 and the row stores a single scalar (e.g. X, Y, M, D, PC, ACC),
        v_row = value * e_0. When a row has multiple fields (e.g. T0.ACC at col 0, T0.PRE at col 1),
        we preserve other columns in the row vector and update column `spec.col`.
        """
        spec = MemoryMap.resolve(symbol_or_spec) if isinstance(symbol_or_spec, str) else symbol_or_spec
        current_row = self.read_row_vector(spec.head, spec.row, batch_idx=batch_idx).clone()
        current_row[spec.col] = float(value)
        self.write_row_vector(spec.head, spec.row, current_row, beta=beta, alpha=alpha, batch_idx=batch_idx)

    def load_rom_instructions(self, instructions: List[Instruction], batch_idx: int = 0) -> None:
        """
        Compile a sequence of PLC-IR (IL) instructions into GDN Instruction ROM (Heads 3 & 4)
        using Gated DeltaNet write steps with beta = 1.0.
        """
        if len(instructions) > 2 * self.state_dim:
            raise ValueError(f"Program length {len(instructions)} exceeds 256-instruction ROM capacity")

        for pc, instr in enumerate(instructions):
            rom_head, rom_row = MemoryMap.rom_address(pc)
            v_row = torch.zeros(self.state_dim, dtype=self.dtype, device=self.state.device)
            vec4 = instr.to_vector()
            v_row[0] = vec4[0]  # Opcode_ID
            v_row[1] = vec4[1]  # Operand_Head
            v_row[2] = vec4[2]  # Operand_Address
            v_row[3] = vec4[3]  # Immediate_Value
            self.write_row_vector(rom_head, rom_row, v_row, beta=1.0, alpha=1.0, batch_idx=batch_idx)

    def fetch_instruction(self, pc: Optional[int] = None, batch_idx: int = 0) -> Instruction:
        """
        Fetch the instruction at Program Counter `pc` (or read `PC` from Head 2 if None)
        via Gated DeltaNet query read.
        """
        if pc is None:
            pc = int(round(self.read_scalar("PC", batch_idx=batch_idx)))
        rom_head, rom_row = MemoryMap.rom_address(pc)
        row_vec = self.read_row_vector(rom_head, rom_row, batch_idx=batch_idx)
        vec4 = (
            float(row_vec[0].item()),
            float(row_vec[1].item()),
            float(row_vec[2].item()),
            float(row_vec[3].item()),
        )
        return Instruction.from_vector(vec4)

    def snapshot_bits(self, prefix: str = "Y", count: int = 16, batch_idx: int = 0) -> Dict[str, int]:
        """Return a dictionary of bit states (e.g. {'Y0': 1, 'Y1': 0, ...}) for online monitoring."""
        out: Dict[str, int] = {}
        for i in range(count):
            sym = f"{prefix}{i}"
            out[sym] = int(round(self.read_scalar(sym, batch_idx=batch_idx)))
        return out

