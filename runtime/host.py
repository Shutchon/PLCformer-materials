"""Custom host-managed PLC-IR interpreter with gated-delta memory and SwiGLU ALU.

Mode A executes individual instructions. Mode B unrolls Boolean-only programs;
stateful/word programs fall back to Mode A. No stock language model is loaded.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from compiler.assembler import ILAssembler, UnrolledRungCompiler
from compiler.gdn_primitives import GDNProcessImageMemory
from compiler.isa import Instruction, Opcode, PLCVocabulary
from compiler.memory_map import HeadID, MemoryMap, STATE_DIM
from compiler.mlp_alu import PLCBooleanALU
from compiler.numeric import DEFAULT_SCALE, WORD_MIN, WORD_MAX, require_integer


@dataclass
class StepTrace:
    """Detailed execution trace of one host-managed instruction step."""
    pc_before: int
    instruction: Instruction
    acc_before: float
    operand_val: float
    acc_after: float
    write_en: float
    write_val: float
    pc_after: int
    emitted_token: str


@dataclass
class ScanCycleResult:
    """Result of executing 1 complete PLC Scan Cycle."""
    cycle_index: int
    mode: str
    steps_executed: int
    emitted_tokens: List[str]
    outputs_y: Dict[str, int]
    relays_m: Dict[str, int]
    traces: List[StepTrace] = field(default_factory=list)


class PLCformerHost(nn.Module):
    """
    Host controller for the custom PLCformer execution pipeline.
    """

    def __init__(
        self,
        program_source: Optional[str] = None,
        is_structured_text: bool = False,
        scale_c: float = DEFAULT_SCALE,
        dt: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.vocab = PLCVocabulary()
        self.gdn_memory = GDNProcessImageMemory(batch_size=1, dtype=dtype, device=device)
        self.alu = PLCBooleanALU(scale_c=scale_c, dtype=dtype, device=device)
        self.dt = require_integer(dt, 1)
        self.dtype = dtype
        self.device = device
        self.cycle_count = 0

        self.instructions: List[Instruction] = []
        self.unrolled_model: Optional[UnrolledRungCompiler] = None

        if program_source is not None:
            self.load_program(program_source, is_structured_text=is_structured_text)

    def load_program(self, source: str, is_structured_text: bool = False) -> List[Instruction]:
        """Assemble IL or ST source code and load it into both Mode A (ROM) and Mode B (Unrolled MLP)."""
        if is_structured_text:
            self.instructions = ILAssembler.assemble_st(source)
        else:
            self.instructions = ILAssembler.assemble(source)

        self.gdn_memory.load_rom_instructions(self.instructions)
        self.unrolled_model = UnrolledRungCompiler(
            self.instructions,
            scale_c=self.alu.coeffs.c,
            dtype=self.dtype,
            device=self.device,
        )
        return self.instructions

    def load_program_file(self, path: str | Path) -> List[Instruction]:
        """Load an `.il` file from disk into ROM and Unrolled MLP weights."""
        text = Path(path).read_text(encoding="utf-8")
        return self.load_program(text, is_structured_text=False)

    def latch_input_token(self, token: str | int) -> None:
        """
        Layer 1 Input Latch: Process an input token such as 'IN_X0_1' or its token ID
        and write the bit into Head 0 via the Gated Delta Rule (beta = 1.0).
        """
        tok_id = self.vocab.encode(token) if isinstance(token, str) else token
        parsed = self.vocab.parse_input_token(tok_id)
        if parsed is not None:
            x_idx, val = parsed
            self.gdn_memory.write_scalar(f"X{x_idx}", val, beta=1.0, alpha=1.0)

    def set_inputs(self, inputs: Dict[str, int | float]) -> List[str]:
        """
        Convert a dictionary of input changes (e.g. {'X0': 1, 'X1': 0}) into input tokens
        and latch each into Head 0 via Layer 1 Gated Delta Rule.
        """
        tokens: List[str] = []
        for sym, val in inputs.items():
            sym_clean = sym.strip().upper()
            bit_int = 1 if float(val) >= 0.5 else 0
            tok = f"IN_{sym_clean}_{bit_int}"
            self.latch_input_token(tok)
            tokens.append(tok)
        return tokens

    def step_instruction(self, input_token: Optional[str | int] = None) -> StepTrace:
        """
        Execute one host-managed instruction step (Mode A).
        """
        # --- Layer 1: GDN Fetch & Input Latch ---
        if input_token is not None:
            self.latch_input_token(input_token)

        pc_before = int(round(self.gdn_memory.read_scalar("PC")))
        instr = self.gdn_memory.fetch_instruction(pc=pc_before)

        # --- Layer 2a: GDN Operand Read ---
        acc_before = self.gdn_memory.read_scalar("ACC")
        stack_top = self.gdn_memory.read_scalar("STACK_0")
        mem_bit = 0.0
        mem_word = 0.0
        prev_clk = 0.0

        if instr.opcode in {Opcode.TON, Opcode.CTU}:
            bit_row_vec = self.gdn_memory.read_row_vector(HeadID.BIT_MEMORY, instr.operand_addr)
            mem_bit = float(bit_row_vec[0].item())
            word_row = instr.operand_addr - 32
            word_row_vec = self.gdn_memory.read_row_vector(HeadID.WORD_MEMORY, word_row)
            mem_word = float(word_row_vec[0].item())
            prev_clk = float(word_row_vec[2].item())
        elif instr.operand_head == int(HeadID.WORD_MEMORY):
            word_row_vec = self.gdn_memory.read_row_vector(HeadID.WORD_MEMORY, instr.operand_addr)
            mem_word = float(word_row_vec[0].item())
        elif instr.operand_head == int(HeadID.BIT_MEMORY) and instr.opcode not in {
            Opcode.NOP,
            Opcode.END,
            Opcode.MPS,
            Opcode.MRD,
            Opcode.MPP,
        }:
            bit_row_vec = self.gdn_memory.read_row_vector(HeadID.BIT_MEMORY, instr.operand_addr)
            mem_bit = float(bit_row_vec[0].item())

        # --- Layer 2b: SwiGLU MLP Boolean & Word ALU ---
        y = self.alu.evaluate_full(
            opcode=instr.opcode,
            acc=acc_before,
            mem_bit=mem_bit,
            mem_word=mem_word,
            imm_val=instr.immediate,
            dt_val=self.dt,
            prev_clk=prev_clk,
            stack_top=stack_top,
        )
        new_acc = float(y[PLCBooleanALU.OUT_NEW_ACC].item())
        write_bit_val = float(y[PLCBooleanALU.OUT_WRITE_BIT_VAL].item())
        write_bit_en = float(y[PLCBooleanALU.OUT_WRITE_BIT_EN].item())
        is_scan_end = float(y[PLCBooleanALU.OUT_IS_SCAN_END].item())
        write_word_val = float(y[PLCBooleanALU.OUT_WRITE_WORD_VAL].item())
        write_word_en = float(y[PLCBooleanALU.OUT_WRITE_WORD_EN].item())
        stack_push = float(y[PLCBooleanALU.OUT_STACK_PUSH].item())
        stack_pop = float(y[PLCBooleanALU.OUT_STACK_POP].item())

        # Clean near-zero/one float32 values
        new_acc = float(round(new_acc)) if abs(new_acc - round(new_acc)) < 1e-4 else new_acc
        write_bit_val = (
            float(round(write_bit_val)) if abs(write_bit_val - round(write_bit_val)) < 1e-4 else write_bit_val
        )
        write_bit_en = (
            float(round(write_bit_en)) if abs(write_bit_en - round(write_bit_en)) < 1e-4 else write_bit_en
        )
        write_word_val = (
            float(round(write_word_val)) if abs(write_word_val - round(write_word_val)) < 1e-4 else write_word_val
        )
        write_word_en = (
            float(round(write_word_en)) if abs(write_word_en - round(write_word_en)) < 1e-4 else write_word_en
        )

        if write_word_en >= 0.5 and not WORD_MIN <= write_word_val <= WORD_MAX:
            raise OverflowError("PLC-IR word/timer/counter overflow; scan aborted")

        # --- Layer 3: GDN State Writeback & PC Increment ---
        # 3.1 Update ACC in Head 2 via Gated Delta Rule
        self.gdn_memory.write_scalar("ACC", new_acc, beta=1.0, alpha=1.0)

        # 3.2 Stack Push (MPS) / Stack Pop (MPP) in Head 2 via Gated Delta Rule
        if stack_push >= 0.5:
            for slot in range(MemoryMap.STACK_DEPTH - 1, 0, -1):
                prev_val = self.gdn_memory.read_scalar(f"STACK_{slot - 1}")
                self.gdn_memory.write_scalar(f"STACK_{slot}", prev_val, beta=stack_push, alpha=1.0)
            self.gdn_memory.write_scalar("STACK_0", acc_before, beta=stack_push, alpha=1.0)
        elif stack_pop >= 0.5:
            for slot in range(0, MemoryMap.STACK_DEPTH - 1):
                next_val = self.gdn_memory.read_scalar(f"STACK_{slot + 1}")
                self.gdn_memory.write_scalar(f"STACK_{slot}", next_val, beta=stack_pop, alpha=1.0)
            self.gdn_memory.write_scalar(
                f"STACK_{MemoryMap.STACK_DEPTH - 1}", 0.0, beta=stack_pop, alpha=1.0
            )

        # 3.3 Bit Memory writeback in Head 0 via Gated Delta Rule (with gate beta = write_bit_en)
        if instr.operand_head == int(HeadID.BIT_MEMORY):
            v_bit = torch.zeros(STATE_DIM, dtype=self.dtype, device=self.device)
            v_bit[0] = write_bit_val
            self.gdn_memory.write_row_vector(
                HeadID.BIT_MEMORY,
                instr.operand_addr,
                v_bit,
                beta=write_bit_en,
                alpha=1.0,
            )

        # 3.4 Word Memory writeback in Head 1 via Gated Delta Rule (with gate beta = write_word_en)
        if instr.opcode in {Opcode.TON, Opcode.CTU}:
            word_row = instr.operand_addr - 32
            v_word = torch.zeros(STATE_DIM, dtype=self.dtype, device=self.device)
            v_word[0] = write_word_val
            v_word[1] = instr.immediate
            v_word[2] = acc_before
            self.gdn_memory.write_row_vector(
                HeadID.WORD_MEMORY,
                word_row,
                v_word,
                beta=write_word_en,
                alpha=1.0,
            )
        elif instr.opcode == Opcode.RST and instr.operand_addr >= MemoryMap.T_DN_START:
            word_row = instr.operand_addr - 32
            v_word = torch.zeros(STATE_DIM, dtype=self.dtype, device=self.device)
            self.gdn_memory.write_row_vector(
                HeadID.WORD_MEMORY,
                word_row,
                v_word,
                beta=write_word_en,
                alpha=1.0,
            )
        elif instr.operand_head == int(HeadID.WORD_MEMORY) and instr.opcode in {Opcode.MOV, Opcode.ADD}:
            v_word = torch.zeros(STATE_DIM, dtype=self.dtype, device=self.device)
            v_word[0] = write_word_val
            self.gdn_memory.write_row_vector(
                HeadID.WORD_MEMORY,
                instr.operand_addr,
                v_word,
                beta=write_word_en,
                alpha=1.0,
            )

        # 3.5 Update PC and SCAN_STATUS in Head 2
        pc_after = 0 if is_scan_end >= 0.5 else (pc_before + 1)
        self.gdn_memory.write_scalar("PC", float(pc_after), beta=1.0, alpha=1.0)
        self.gdn_memory.write_scalar("SCAN_STATUS", float(is_scan_end), beta=1.0, alpha=1.0)

        # --- Layer 4: LM Head Trace & I/O Emission ---
        if is_scan_end >= 0.5:
            emitted_token = "SCAN_END"
        elif write_bit_en >= 0.5 and instr.operand_head == int(HeadID.BIT_MEMORY):
            row = instr.operand_addr
            if MemoryMap.Y_START <= row < MemoryMap.Y_START + MemoryMap.Y_COUNT:
                y_idx = row - MemoryMap.Y_START
                bit_int = 1 if write_bit_val >= 0.5 else 0
                emitted_token = f"OUT_Y{y_idx}_{bit_int}"
            else:
                emitted_token = "STEP_OK"
        else:
            emitted_token = "STEP_OK"

        operand_val = mem_word if instr.operand_head == int(HeadID.WORD_MEMORY) else mem_bit
        return StepTrace(
            pc_before=pc_before,
            instruction=instr,
            acc_before=acc_before,
            operand_val=operand_val,
            acc_after=new_acc,
            write_en=max(write_bit_en, write_word_en),
            write_val=write_word_val if instr.operand_head == int(HeadID.WORD_MEMORY) else write_bit_val,
            pc_after=pc_after,
            emitted_token=emitted_token,
        )

    def run_scan_cycle(
        self,
        inputs: Optional[Dict[str, int | float]] = None,
        mode: str = "mode_a",
        max_steps: int = 256,
    ) -> ScanCycleResult:
        """
        Execute 1 complete PLC Scan Cycle (Input Scan -> Logic Solve -> Output Update).
        """
        emitted_tokens: List[str] = []
        if inputs:
            in_tokens = self.set_inputs(inputs)
            emitted_tokens.extend(in_tokens)

        traces: List[StepTrace] = []

        if mode.lower() == "mode_a" or (
            mode.lower() == "mode_b"
            and self.unrolled_model is not None
            and not self.unrolled_model.is_pure_boolean
        ):
            self.gdn_memory.write_scalar("PC", 0.0, beta=1.0, alpha=1.0)
            self.gdn_memory.write_scalar("ACC", 0.0, beta=1.0, alpha=1.0)
            self.gdn_memory.write_scalar("SCAN_STATUS", 0.0, beta=1.0, alpha=1.0)

            steps = 0
            while steps < max_steps:
                trace = self.step_instruction()
                traces.append(trace)
                emitted_tokens.append(trace.emitted_token)
                steps += 1
                if trace.emitted_token == "SCAN_END":
                    break
            if not traces or traces[-1].emitted_token != "SCAN_END":
                raise RuntimeError("Scan instruction budget exhausted before END")
        elif mode.lower() == "mode_b":
            if self.unrolled_model is None:
                raise RuntimeError("No program compiled for Mode B")
            bit_vec = self.gdn_memory.state[0, int(HeadID.BIT_MEMORY), :, 0].clone()
            updated_bits = self.unrolled_model(bit_vec)
            for r in range(STATE_DIM):
                v_row = torch.zeros(STATE_DIM, dtype=self.dtype, device=self.device)
                v_row[0] = updated_bits[r]
                self.gdn_memory.write_row_vector(HeadID.BIT_MEMORY, r, v_row, beta=1.0, alpha=1.0)
            for y_idx in range(MemoryMap.Y_COUNT):
                r = MemoryMap.Y_START + y_idx
                bit_int = 1 if float(updated_bits[r].item()) >= 0.5 else 0
                emitted_tokens.append(f"OUT_Y{y_idx}_{bit_int}")
            emitted_tokens.append("SCAN_END")
            steps = 1
        else:
            raise ValueError(f"Unknown execution mode '{mode}'. Expected 'mode_a' or 'mode_b'.")

        self.cycle_count += 1
        return ScanCycleResult(
            cycle_index=self.cycle_count,
            mode=mode,
            steps_executed=steps,
            emitted_tokens=emitted_tokens,
            outputs_y=self.gdn_memory.snapshot_bits("Y", 16),
            relays_m=self.gdn_memory.snapshot_bits("M", 64),
            traces=traces,
        )

    def read_bit(self, symbol: str) -> int:
        """Read a discrete bit (0 or 1) from the GDN Process Image."""
        return int(round(self.gdn_memory.read_scalar(symbol)))

    def read_word(self, symbol: str) -> float:
        """Read a word/register value (e.g. 'D0', 'T0.ACC', 'C0.ACC') from Head 1."""
        return float(self.gdn_memory.read_scalar(symbol))
