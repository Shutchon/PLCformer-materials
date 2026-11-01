"""Bounded PLC-IR assembler and Boolean weight compiler.

Restricted ST-like syntax is transpiled to PLC-IR. Mode A stores instructions
in memory heads 3 and 4. Mode B supports Boolean residual blocks only; programs
with stack, timer, counter, or word operations use the host interpreter.
"""

from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn

from .isa import Instruction, Opcode
from .memory_map import HeadID, MemoryMap, STATE_DIM
from .mlp_alu import PLCBooleanALU, SwiGLUScalingCoeffs
from .st_transpiler import STTranspiler
from .numeric import DEFAULT_SCALE, require_integer


class ILAssembler:
    """Assembler converting IEC 61131-3 IL (or ST) source code into PLCformer Instructions."""

    @classmethod
    def parse_line(cls, line: str) -> Optional[Instruction]:
        clean = line.split(";")[0].split("//")[0].strip()
        if not clean:
            return None

        tokens = clean.replace(",", " ").split()
        mnemonic = tokens[0].upper()
        try:
            opcode = Opcode[mnemonic]
        except KeyError as exc:
            raise ValueError(f"Unknown IL mnemonic '{mnemonic}' in line: '{line}'") from exc

        if opcode in {Opcode.NOP, Opcode.END, Opcode.MPS, Opcode.MRD, Opcode.MPP}:
            return Instruction(
                opcode=opcode,
                operand_head=int(HeadID.CPU_CONTROL),
                operand_addr=0,
                immediate=0.0,
                symbol=mnemonic,
            )

        if len(tokens) < 2:
            raise ValueError(f"Instruction '{mnemonic}' requires an operand in line: '{line}'")

        operand_str = tokens[1].upper()

        # Timer / Counter with preset K value (e.g., TON T0, K10 or CTU C0, K5)
        if opcode in {Opcode.TON, Opcode.CTU}:
            spec = MemoryMap.resolve(operand_str, bit_context=True)
            imm = 0.0
            if len(tokens) >= 3:
                k_tok = tokens[2].upper().lstrip("K#")
                imm = float(k_tok)
            return Instruction(
                opcode=opcode,
                operand_head=int(spec.head),
                operand_addr=spec.row,
                immediate=imm,
                symbol=spec.symbol,
            )

        # Word / Register operations (MOV, ADD, CMP_EQ, CMP_GT, CMP_LT)
        if opcode in {Opcode.MOV, Opcode.ADD, Opcode.CMP_EQ, Opcode.CMP_GT, Opcode.CMP_LT}:
            spec = MemoryMap.resolve(operand_str, bit_context=False)
            imm = 0.0
            if len(tokens) >= 3:
                k_tok = tokens[2].upper().lstrip("K#")
                imm = float(k_tok)
            return Instruction(
                opcode=opcode,
                operand_head=int(spec.head),
                operand_addr=spec.row,
                immediate=imm,
                symbol=spec.symbol,
            )

        # Standard bit operand (LD, LDI, AND, ANI, OR, ORI, OUT, SET, RST)
        spec = MemoryMap.resolve(operand_str, bit_context=True)
        imm = 0.0
        if len(tokens) >= 3:
            k_tok = tokens[2].upper().lstrip("K#")
            imm = float(k_tok)

        return Instruction(
            opcode=opcode,
            operand_head=int(spec.head),
            operand_addr=spec.row,
            immediate=imm,
            symbol=spec.symbol,
        )

    @classmethod
    def assemble(cls, source: str) -> List[Instruction]:
        """Assemble multi-line IL source code into a list of Instructions."""
        instructions: List[Instruction] = []
        for line in source.splitlines():
            instr = cls.parse_line(line)
            if instr is not None:
                instructions.append(instr)
        if not instructions or instructions[-1].opcode != Opcode.END:
            instructions.append(
                Instruction(opcode=Opcode.END, operand_head=int(HeadID.CPU_CONTROL), operand_addr=0)
            )
        if len(instructions) > 256:
            raise ValueError("PLC-IR programs are limited to 256 instructions including END")
        depth = 0
        word_ops = {Opcode.MOV, Opcode.ADD, Opcode.CMP_EQ, Opcode.CMP_GT, Opcode.CMP_LT}
        bit_ops = {Opcode.LD, Opcode.LDI, Opcode.AND, Opcode.ANI, Opcode.OR,
                   Opcode.ORI, Opcode.OUT, Opcode.SET, Opcode.RST}
        for i, ins in enumerate(instructions):
            require_integer(ins.immediate)
            if ins.opcode in {Opcode.TON, Opcode.CTU}:
                require_integer(ins.immediate, 0)
                start = MemoryMap.T_DN_START if ins.opcode == Opcode.TON else MemoryMap.C_DN_START
                if ins.operand_head != HeadID.BIT_MEMORY or not start <= ins.operand_addr < start + 16:
                    raise ValueError("TON requires Tn and CTU requires Cn")
            elif ins.opcode in word_ops:
                if ins.operand_head != HeadID.WORD_MEMORY or not 0 <= ins.operand_addr < 64:
                    raise ValueError("Word instructions require a D register")
            elif ins.opcode in bit_ops and ins.operand_head != HeadID.BIT_MEMORY:
                raise ValueError("Boolean instructions require a bit address")
            if ins.opcode == Opcode.END and i != len(instructions) - 1:
                raise ValueError("END must be the final instruction")
            if ins.opcode == Opcode.MPS:
                depth += 1
                if depth > MemoryMap.STACK_DEPTH:
                    raise ValueError("PLC-IR branch stack overflow")
            elif ins.opcode in {Opcode.MRD, Opcode.MPP}:
                if depth == 0:
                    raise ValueError("PLC-IR branch stack underflow")
                if ins.opcode == Opcode.MPP:
                    depth -= 1
        if depth:
            raise ValueError("Every MPS must have a matching MPP within the scan")
        return instructions

    @classmethod
    def assemble_file(cls, path: str | Path) -> List[Instruction]:
        """Assemble an `.il` file from disk."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.assemble(text)

    @classmethod
    def assemble_st(cls, st_source: str) -> List[Instruction]:
        """Transpile Structured Text (ST) to IL and assemble into Instructions."""
        il_code = STTranspiler.transpile(st_source)
        return cls.assemble(il_code)


class UnrolledRungCompiler(nn.Module):
    """
    Mode B (Unrolled Scan-Cycle Mode): Compiles a complete sequence of PLC-IR (IL)
    instructions so that 1 Forward Pass executes the entire PLC Scan Cycle.

    For pure boolean rungs (`LD`, `LDI`, `AND`, `ANI`, `OR`, `ORI`, `OUT`, `SET`, `RST`),
    each instruction is compiled into a dedicated 1-neuron residual SwiGLU block.
    For stack, timer, counter, and word programs, the host falls back to Mode A.
    Those programs are not tensor-unrolled by this module.
    """

    def __init__(
        self,
        instructions: List[Instruction],
        scale_c: float = DEFAULT_SCALE,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.instructions = [ins for ins in instructions if ins.opcode != Opcode.END]
        self.coeffs = SwiGLUScalingCoeffs.compute(scale_c)
        self.dtype = dtype
        self.device = device
        self.shared_alu = PLCBooleanALU(scale_c=scale_c, dtype=dtype, device=device)

        self.vec_dim = STATE_DIM + 2
        self.idx_acc = STATE_DIM
        self.idx_const = STATE_DIM + 1

        self.is_pure_boolean = all(
            ins.opcode
            in {
                Opcode.LD,
                Opcode.LDI,
                Opcode.AND,
                Opcode.ANI,
                Opcode.OR,
                Opcode.ORI,
                Opcode.OUT,
                Opcode.SET,
                Opcode.RST,
            }
            and ins.operand_head == int(HeadID.BIT_MEMORY)
            and ins.operand_addr < MemoryMap.T_DN_START
            for ins in self.instructions
        )

        self.step_layers = nn.ModuleList()
        if self.is_pure_boolean:
            for instr in self.instructions:
                layer = self._compile_instruction_layer(instr)
                self.step_layers.append(layer)

    def _compile_instruction_layer(self, instr: Instruction) -> nn.Module:
        """
        Compile a single boolean instruction into a Residual SwiGLU Layer:
            v_{next} = v + (SiLU(v @ W_g^T) * (v @ W_u^T)) @ W_d^T
        where v in R^{130} holds [Bit_Memory(0..127), ACC(128), CONST_1(129)].
        """
        from .mlp_alu import ConstructiveSwiGLUMLP

        mlp = ConstructiveSwiGLUMLP(
            in_features=self.vec_dim,
            hidden_features=4,
            out_features=self.vec_dim,
            dtype=self.dtype,
            device=self.device,
        )
        c = self.coeffs.c
        gamma = self.coeffs.gamma_bilinear
        acc = self.idx_acc
        c1 = self.idx_const
        addr = instr.operand_addr
        op = instr.opcode

        with torch.no_grad():
            if op == Opcode.LD:
                mlp.gate_proj.weight[0, c1] = c
                mlp.up_proj.weight[0, addr] = 1.0
                mlp.up_proj.weight[0, acc] = -1.0
                mlp.down_proj.weight[acc, 0] = gamma

            elif op == Opcode.LDI:
                mlp.gate_proj.weight[0, c1] = c
                mlp.up_proj.weight[0, c1] = 1.0
                mlp.up_proj.weight[0, addr] = -1.0
                mlp.up_proj.weight[0, acc] = -1.0
                mlp.down_proj.weight[acc, 0] = gamma

            elif op == Opcode.AND:
                mlp.gate_proj.weight[0, acc] = c
                mlp.up_proj.weight[0, addr] = 1.0
                mlp.up_proj.weight[0, c1] = -1.0
                mlp.down_proj.weight[acc, 0] = gamma

            elif op == Opcode.ANI:
                mlp.gate_proj.weight[0, acc] = c
                mlp.up_proj.weight[0, addr] = -1.0
                mlp.down_proj.weight[acc, 0] = gamma

            elif op == Opcode.OR:
                mlp.gate_proj.weight[0, c1] = c
                mlp.gate_proj.weight[0, acc] = -c
                mlp.up_proj.weight[0, addr] = 1.0
                mlp.down_proj.weight[acc, 0] = gamma

            elif op == Opcode.ORI:
                mlp.gate_proj.weight[0, c1] = c
                mlp.gate_proj.weight[0, acc] = -c
                mlp.up_proj.weight[0, c1] = 1.0
                mlp.up_proj.weight[0, addr] = -1.0
                mlp.down_proj.weight[acc, 0] = gamma

            elif op == Opcode.OUT:
                mlp.gate_proj.weight[0, c1] = c
                mlp.up_proj.weight[0, acc] = 1.0
                mlp.up_proj.weight[0, addr] = -1.0
                mlp.down_proj.weight[addr, 0] = gamma

            elif op == Opcode.SET:
                mlp.gate_proj.weight[0, acc] = c
                mlp.up_proj.weight[0, c1] = 1.0
                mlp.up_proj.weight[0, addr] = -1.0
                mlp.down_proj.weight[addr, 0] = gamma

            elif op == Opcode.RST:
                mlp.gate_proj.weight[0, acc] = c
                mlp.up_proj.weight[0, addr] = -1.0
                mlp.down_proj.weight[addr, 0] = gamma

        return mlp

    def forward(self, bit_vector: torch.Tensor) -> torch.Tensor:
        """
        Execute 1 full PLC scan cycle in a single forward pass over the 128-bit Process Image.
        """
        prefix_shape = bit_vector.shape[:-1]
        zeros_acc = torch.zeros(*prefix_shape, 1, dtype=bit_vector.dtype, device=bit_vector.device)
        ones_const = torch.ones(*prefix_shape, 1, dtype=bit_vector.dtype, device=bit_vector.device)
        v = torch.cat([bit_vector, zeros_acc, ones_const], dim=-1)

        for mlp in self.step_layers:
            v = v + mlp(v)

        return v[..., :STATE_DIM]
