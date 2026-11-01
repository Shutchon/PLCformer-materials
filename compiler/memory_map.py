"""
Gated DeltaNet State Memory Map (128 x 128 Matrix per Head) for PLCformer.

Maps IEC 61131-3 PLC Process Image addresses (X, Y, M, D, T, C) and internal
CPU registers (PC, ACC, STACK, SCAN_STATUS) to exact (head_id, row_idx, col_idx)
coordinates inside the Gated DeltaNet recurrent state tensor S_t in R^{H x 128 x 128}.
"""

from dataclasses import dataclass
from enum import IntEnum
import re
from typing import Dict, Tuple


STATE_DIM: int = 128
NUM_HEADS: int = 5


class HeadID(IntEnum):
    """Gated DeltaNet Head allocation in PLCformer."""
    BIT_MEMORY = 0     # Process Image Bit Table: X0..15, Y0..15, M0..63, T0..15.DN, C0..15.DN
    WORD_MEMORY = 1    # Word / Register Memory: D0..63, T0..15.ACC/PRE, C0..15.ACC/PRE/PREV
    CPU_CONTROL = 2    # CPU Control Registers: PC, ACC, STACK[0..7], SCAN_STATUS, SP
    ROM_BANK_0 = 3     # Instruction ROM Bank 0 (lines 0..127)
    ROM_BANK_1 = 4     # Instruction ROM Bank 1 (lines 128..255)


@dataclass(frozen=True)
class AddressSpec:
    """Exact coordinate in the Gated DeltaNet recurrent state tensor S[head, row, col]."""
    symbol: str
    head: HeadID
    row: int
    col: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.row < STATE_DIM):
            raise ValueError(f"Row index {self.row} out of range [0, {STATE_DIM - 1}] for {self.symbol}")
        if not (0 <= self.col < STATE_DIM):
            raise ValueError(f"Col index {self.col} out of range [0, {STATE_DIM - 1}] for {self.symbol}")


class MemoryMap:
    """
    Address resolver and coordinate directory for PLCformer's 128x128 Gated DeltaNet state.

    Convention for State Matrix S[head] in R^{128 x 128} (d_v x d_k):
      - Under the Gated Delta Rule:
            S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
        where k_t = e_{row} in R^{128} is the one-hot address key vector,
        reading row `r` is performed via S_t @ e_r (extracting column `r` of S_t,
        or equivalently row `r` when S_t is indexed as [row, col_feature]).
      - To make both row-major indexing S[head, row, col] and matrix-vector product
        consistent and intuitive, we store the addressed vector in column `r` (or
        provide symmetric helper methods in `gdn_primitives.py`).
    """

    # Head 0: Bit Memory row offsets
    X_START, X_COUNT = 0, 16       # Rows 0..15:   X0 - X15
    Y_START, Y_COUNT = 16, 16      # Rows 16..31:  Y0 - Y15
    M_START, M_COUNT = 32, 64      # Rows 32..95:  M0 - M63
    T_DN_START, T_COUNT = 96, 16   # Rows 96..111: T0.DN - T15.DN
    C_DN_START, C_COUNT = 112, 16  # Rows 112..127: C0.DN - C15.DN

    # Head 1: Word / Register Memory row offsets
    D_START, D_COUNT = 0, 64       # Rows 0..63:   D0 - D63
    T_WORD_START = 64              # Rows 64..79:  T0..T15 (col 0 = ACC, col 1 = PRE)
    C_WORD_START = 80              # Rows 80..95:  C0..C15 (col 0 = ACC, col 1 = PRE, col 2 = PREV_CLK)

    # Head 2: CPU Internal Control Registers row offsets
    ROW_PC = 0
    ROW_ACC = 1
    ROW_STACK_BASE = 2             # Rows 2..9 (8 stack slots)
    STACK_DEPTH = 8
    ROW_SCAN_STATUS = 10
    ROW_SP = 11

    _SYMBOL_REGEX = re.compile(
        r"^(?P<prefix>X|Y|M|D|T|C)(?P<index>\d+)(?:\.(?P<field>DN|ACC|PRE|PREV))?$",
        re.IGNORECASE,
    )

    @classmethod
    def resolve(cls, symbol: str, bit_context: bool = True) -> AddressSpec:
        """
        Resolve a PLC symbol (e.g. 'X0', 'Y15', 'M3', 'T0', 'T0.DN', 'T0.ACC',
        'C1.PRE', 'D10', 'PC', 'ACC', 'SCAN_STATUS') into an AddressSpec.
        """
        sym = symbol.strip().upper()

        # Check CPU internal control registers first
        if sym == "PC":
            return AddressSpec("PC", HeadID.CPU_CONTROL, cls.ROW_PC, 0)
        if sym == "ACC":
            return AddressSpec("ACC", HeadID.CPU_CONTROL, cls.ROW_ACC, 0)
        if sym == "SCAN_STATUS":
            return AddressSpec("SCAN_STATUS", HeadID.CPU_CONTROL, cls.ROW_SCAN_STATUS, 0)
        if sym == "SP":
            return AddressSpec("SP", HeadID.CPU_CONTROL, cls.ROW_SP, 0)
        if sym.startswith("STACK_"):
            slot = int(sym.split("_")[1])
            if not (0 <= slot < cls.STACK_DEPTH):
                raise ValueError(f"Stack slot {slot} out of range [0, {cls.STACK_DEPTH - 1}]")
            return AddressSpec(sym, HeadID.CPU_CONTROL, cls.ROW_STACK_BASE + slot, 0)

        match = cls._SYMBOL_REGEX.match(sym)
        if not match:
            raise ValueError(f"Invalid PLC address symbol: '{symbol}'")

        prefix = match.group("prefix").upper()
        idx = int(match.group("index"))
        field = (match.group("field") or "").upper()

        if prefix == "X":
            if not (0 <= idx < cls.X_COUNT):
                raise ValueError(f"Input X{idx} out of range [X0..X{cls.X_COUNT - 1}]")
            return AddressSpec(f"X{idx}", HeadID.BIT_MEMORY, cls.X_START + idx, 0)

        if prefix == "Y":
            if not (0 <= idx < cls.Y_COUNT):
                raise ValueError(f"Output Y{idx} out of range [Y0..Y{cls.Y_COUNT - 1}]")
            return AddressSpec(f"Y{idx}", HeadID.BIT_MEMORY, cls.Y_START + idx, 0)

        if prefix == "M":
            if not (0 <= idx < cls.M_COUNT):
                raise ValueError(f"Internal relay M{idx} out of range [M0..M{cls.M_COUNT - 1}]")
            return AddressSpec(f"M{idx}", HeadID.BIT_MEMORY, cls.M_START + idx, 0)

        if prefix == "D":
            if not (0 <= idx < cls.D_COUNT):
                raise ValueError(f"Data register D{idx} out of range [D0..D{cls.D_COUNT - 1}]")
            return AddressSpec(f"D{idx}", HeadID.WORD_MEMORY, cls.D_START + idx, 0)

        if prefix == "T":
            if not (0 <= idx < cls.T_COUNT):
                raise ValueError(f"Timer T{idx} out of range [T0..T{cls.T_COUNT - 1}]")
            if field == "ACC":
                return AddressSpec(f"T{idx}.ACC", HeadID.WORD_MEMORY, cls.T_WORD_START + idx, 0)
            if field == "PRE":
                return AddressSpec(f"T{idx}.PRE", HeadID.WORD_MEMORY, cls.T_WORD_START + idx, 1)
            if field == "DN" or bit_context:
                return AddressSpec(f"T{idx}.DN", HeadID.BIT_MEMORY, cls.T_DN_START + idx, 0)
            return AddressSpec(f"T{idx}.ACC", HeadID.WORD_MEMORY, cls.T_WORD_START + idx, 0)

        if prefix == "C":
            if not (0 <= idx < cls.C_COUNT):
                raise ValueError(f"Counter C{idx} out of range [C0..C{cls.C_COUNT - 1}]")
            if field == "ACC":
                return AddressSpec(f"C{idx}.ACC", HeadID.WORD_MEMORY, cls.C_WORD_START + idx, 0)
            if field == "PRE":
                return AddressSpec(f"C{idx}.PRE", HeadID.WORD_MEMORY, cls.C_WORD_START + idx, 1)
            if field == "PREV":
                return AddressSpec(f"C{idx}.PREV", HeadID.WORD_MEMORY, cls.C_WORD_START + idx, 2)
            if field == "DN" or bit_context:
                return AddressSpec(f"C{idx}.DN", HeadID.BIT_MEMORY, cls.C_DN_START + idx, 0)
            return AddressSpec(f"C{idx}.ACC", HeadID.WORD_MEMORY, cls.C_WORD_START + idx, 0)

        raise ValueError(f"Unhandled PLC symbol: '{symbol}'")

    @classmethod
    def rom_address(cls, pc: int) -> Tuple[HeadID, int]:
        """Map a Program Counter (0..255) to (ROM HeadID, row_index 0..127)."""
        if not (0 <= pc < 2 * STATE_DIM):
            raise ValueError(f"Program counter {pc} out of range [0, {2 * STATE_DIM - 1}]")
        if pc < STATE_DIM:
            return (HeadID.ROM_BANK_0, pc)
        return (HeadID.ROM_BANK_1, pc - STATE_DIM)

    @classmethod
    def describe_bit_row(cls, row: int) -> str:
        """Return human-readable PLC symbol for a row index (0..127) in Head 0."""
        if 0 <= row < 16:
            return f"X{row}"
        if 16 <= row < 32:
            return f"Y{row - 16}"
        if 32 <= row < 96:
            return f"M{row - 32}"
        if 96 <= row < 112:
            return f"T{row - 96}.DN"
        if 112 <= row < 128:
            return f"C{row - 112}.DN"
        raise ValueError(f"Row {row} out of range [0, 127]")

