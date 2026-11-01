"""PLCformer Compiler Package: Compiling IEC 61131-3 PLC Programs into Linear Transformer Weights."""

from .assembler import ILAssembler, UnrolledRungCompiler
from .build_checkpoint import export_plcformer_safetensors, load_plcformer_safetensors
from .gdn_primitives import GDNProcessImageMemory, gated_delta_rule_step_col, gated_delta_rule_step_row
from .isa import Instruction, Opcode, PLCVocabulary
from .memory_map import AddressSpec, HeadID, MemoryMap, NUM_HEADS, STATE_DIM
from .mlp_alu import ConstructiveSwiGLUMLP, PLCBooleanALU, SwiGLUScalingCoeffs
from .st_transpiler import STTranspiler

__all__ = [
    "Opcode",
    "Instruction",
    "PLCVocabulary",
    "MemoryMap",
    "HeadID",
    "AddressSpec",
    "STATE_DIM",
    "NUM_HEADS",
    "GDNProcessImageMemory",
    "gated_delta_rule_step_col",
    "gated_delta_rule_step_row",
    "SwiGLUScalingCoeffs",
    "ConstructiveSwiGLUMLP",
    "PLCBooleanALU",
    "STTranspiler",
    "ILAssembler",
    "UnrolledRungCompiler",
    "export_plcformer_safetensors",
    "load_plcformer_safetensors",
]
