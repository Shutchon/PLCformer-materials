"""
IEC 61131-3 Instruction List (IL / PLC-IR) Instruction Set Architecture (ISA)
and Token Vocabulary definitions for PLCformer.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class Opcode(IntEnum):
    """IEC 61131-3 IL (PLC-IR) Opcodes compiled into SwiGLU & Gated DeltaNet weights."""
    NOP = 0
    # Bit Load
    LD = 1      # ACC = Mem[addr]
    LDI = 2     # ACC = 1.0 - Mem[addr]
    # Bit Logic
    AND = 3     # ACC = ACC * Mem[addr]
    ANI = 4     # ACC = ACC * (1.0 - Mem[addr])
    OR = 5      # ACC = ACC + Mem[addr] - (ACC * Mem[addr])
    ORI = 6     # ACC = ACC + (1.0 - Mem[addr]) - (ACC * (1.0 - Mem[addr]))
    # Output & Latching
    OUT = 7     # Mem[addr] = ACC
    SET = 8     # if ACC == 1: Mem[addr] = 1.0
    RST = 9     # if ACC == 1: Mem[addr] = 0.0
    # Stack Operations (Branching Rungs)
    MPS = 10    # Push ACC onto stack
    MRD = 11    # Read top of stack into ACC
    MPP = 12    # Pop top of stack into ACC
    # Timers & Counters
    TON = 13    # On-Delay Timer: if ACC == 1 -> T.ACC += dt; T.DN = (T.ACC >= K)
    CTU = 14    # Count-Up Counter: on rising edge -> C.ACC += 1; C.DN = (C.ACC >= K)
    # Word / Arithmetic & Comparison
    MOV = 15    # Move immediate or register value
    ADD = 16    # Add immediate or register value
    CMP_EQ = 17 # Compare equal
    CMP_GT = 18 # Compare greater than
    CMP_LT = 19 # Compare less than
    # Scan Control
    END = 20    # End of scan cycle: reset PC = 0, emit SCAN_END


# Set of opcodes that write back to Bit Memory (Head 0) or Word Memory (Head 1)
WRITE_OPCODES = {Opcode.OUT, Opcode.SET, Opcode.RST, Opcode.TON, Opcode.CTU, Opcode.MOV, Opcode.ADD}

# Phase 1 Core Boolean Opcodes
CORE_BOOLEAN_OPCODES = {
    Opcode.LD,
    Opcode.LDI,
    Opcode.AND,
    Opcode.ANI,
    Opcode.OR,
    Opcode.ORI,
    Opcode.OUT,
    Opcode.SET,
    Opcode.RST,
    Opcode.END,
}


@dataclass(frozen=True)
class Instruction:
    """
    Represents a single compiled PLC-IR (IL) instruction stored in Instruction ROM (Head 3/4).
    Encoded as a 4-tuple vector: [Opcode_ID, Operand_Head, Operand_Address, Immediate_Value].
    """
    opcode: Opcode
    operand_head: int = 0
    operand_addr: int = 0
    immediate: float = 0.0
    symbol: str = ""

    def to_vector(self) -> Tuple[float, float, float, float]:
        """Convert instruction to its 4-float ROM row vector."""
        return (
            float(int(self.opcode)),
            float(self.operand_head),
            float(self.operand_addr),
            float(self.immediate),
        )

    @classmethod
    def from_vector(cls, vec: Tuple[float, float, float, float], symbol: str = "") -> "Instruction":
        """Reconstruct an Instruction from a 4-float ROM row vector."""
        return cls(
            opcode=Opcode(int(round(vec[0]))),
            operand_head=int(round(vec[1])),
            operand_addr=int(round(vec[2])),
            immediate=float(vec[3]),
            symbol=symbol,
        )


class PLCVocabulary:
    """
    Token vocabulary mapping PLC I/O events and execution control signals
    to discrete Transformer token IDs.
    """

    def __init__(self, num_inputs: int = 16, num_outputs: int = 16) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_vocab()

    def _add_token(self, token: str) -> int:
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def _build_vocab(self) -> None:
        # Control & execution tokens
        for ctrl in ["PAD", "STEP", "SCAN_START", "STEP_OK", "SCAN_END", "HALT", "INTERLOCK_TRIP"]:
            self._add_token(ctrl)

        # Discrete input latch tokens: IN_X{i}_{0|1}
        for i in range(self.num_inputs):
            self._add_token(f"IN_X{i}_0")
            self._add_token(f"IN_X{i}_1")

        # Discrete output trace tokens: OUT_Y{i}_{0|1}
        for i in range(self.num_outputs):
            self._add_token(f"OUT_Y{i}_0")
            self._add_token(f"OUT_Y{i}_1")

    def encode(self, token: str) -> int:
        if token not in self.token_to_id:
            raise KeyError(f"Unknown PLC token: {token}")
        return self.token_to_id[token]

    def decode(self, token_id: int) -> str:
        if token_id not in self.id_to_token:
            raise KeyError(f"Unknown PLC token ID: {token_id}")
        return self.id_to_token[token_id]

    def parse_input_token(self, token_id: int) -> Optional[Tuple[int, float]]:
        """
        If token_id is an input latch token IN_X{i}_{val}, return (row_index_i, float_val).
        Otherwise return None.
        """
        token = self.id_to_token.get(token_id, "")
        if token.startswith("IN_X"):
            parts = token.split("_")
            x_idx = int(parts[1][1:])
            val = float(int(parts[2]))
            return (x_idx, val)
        return None

    @property
    def size(self) -> int:
        return len(self.token_to_id)

