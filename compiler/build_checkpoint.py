"""Export implementation-specific weights and current memory as safetensors.

Tensor names describe this custom runtime; they are not a loadable stock Qwen
checkpoint. Export a fresh host to obtain initial program memory.
"""

from pathlib import Path
from typing import Dict
from safetensors.torch import load_file, save_file
import torch

from runtime.host import PLCformerHost


def export_plcformer_safetensors(host: PLCformerHost, output_path: str | Path) -> Path:
    """
    Save the compiled PLCformer model weights (`W_gate`, `W_up`, `W_down`, GDN basis,
    and initial Gated DeltaNet 5x128x128 state with Instruction ROM) to a `.safetensors` file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tensors: Dict[str, torch.Tensor] = {
        "model.layers.1.gdn.state_init": host.gdn_memory.state.detach().cpu().contiguous(),
        "model.layers.1.gdn.address_basis": host.gdn_memory.basis.detach().cpu().contiguous(),
        "model.layers.2.mlp.gate_proj.weight": host.alu.mlp.gate_proj.weight.detach().cpu().contiguous(),
        "model.layers.2.mlp.up_proj.weight": host.alu.mlp.up_proj.weight.detach().cpu().contiguous(),
        "model.layers.2.mlp.down_proj.weight": host.alu.mlp.down_proj.weight.detach().cpu().contiguous(),
    }

    if host.unrolled_model is not None and host.unrolled_model.is_pure_boolean:
        for idx, step_mlp in enumerate(host.unrolled_model.step_layers):
            tensors[f"model.unrolled.rung_{idx}.gate_proj.weight"] = (
                step_mlp.gate_proj.weight.detach().cpu().contiguous()
            )
            tensors[f"model.unrolled.rung_{idx}.up_proj.weight"] = (
                step_mlp.up_proj.weight.detach().cpu().contiguous()
            )
            tensors[f"model.unrolled.rung_{idx}.down_proj.weight"] = (
                step_mlp.down_proj.weight.detach().cpu().contiguous()
            )

    save_file(tensors, str(out))
    return out


def load_plcformer_safetensors(checkpoint_path: str | Path) -> Dict[str, torch.Tensor]:
    """Load a compiled PLCformer `.safetensors` checkpoint."""
    return load_file(str(checkpoint_path))

