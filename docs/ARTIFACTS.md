# Artifact inventory and interpretation

This repository contains the computational materials only. The manuscript, submission documents, review notes, and development history from the main project are excluded.

| Material | Location | Interpretation |
| --- | --- | --- |
| Compiler and numeric contract | `compiler/` | Restricted PLC-IR/ST-like front end; constructive memory and ALU |
| Execution engines and reference | `runtime/` | Custom PyTorch/Numba execution; shared-IR reference |
| Four control programs | `benchmarks/programs/*.il` | Synthetic self-hold, star–delta, traffic, and sorting examples |
| Input schedules | `data/*.npz` | 100,000 scans × 16 input bits per workload, float32, seeds 42–45 |
| Tensor checkpoints | `checkpoints/*.safetensors` | Constructed ALU weights and fresh memory with program ROM |
| Reference measurements | `results/reference/*.json` | Snapshot used for the reported evaluation; do not overwrite |
| Working measurements | `benchmarks/*results*.json` | Rewritten when the complete experiment runner is used |
| Regression/differential tests | `tests/` | 47 tests in the supplied snapshot |
| Input/checkpoint metadata | `artifact_manifest.json` | Seeds, dimensions, program names, and SHA-256 hashes |
| Full package inventory | `MATERIALS_MANIFEST.json` | SHA-256 of all distributed material files except itself |
| Source provenance | `PROVENANCE.json` | Source snapshot, snapshot hashes, and creation information |

## Input files

Load with `numpy.load(path, allow_pickle=False)`. `inputs` has shape `(100000, 16)`, with columns X0 through X15. `seed` identifies the generator seed; `scan_ticks` is 1. These are synthetic binary traces, not physical-plant measurements. The exact generators are in `runtime/plant_sim.py` and their mapping is in `benchmarks/verify_equivalence.py`.

## Checkpoints

Use `safetensors.torch.load_file` or `compiler.build_checkpoint.load_plcformer_safetensors` to inspect the tensor dictionary. The checkpoints are implementation-specific exports, not Hugging Face or stock Qwen checkpoints. The exporter does not provide a complete host restoration API; execute programs by constructing a fresh `PLCformerHost` from the `.il` file. `examples/self_hold.py` demonstrates both engines.

## Scope of the evidence

State comparisons include all image bits, data registers, timer/counter records, and branch stack, after integer normalization and before output thresholding. The sparse path checks 100,000 scans per program; the eager path checks a 500-scan prefix. This is differential testing, not a full compiler proof or IEC certification.

The interlock checks all 16 binary input patterns and 10,000 seeded synthetic proposals. It tests the two specified violations and safe-command preservation. No language model is called. Scalar sigmoid fitting is separate from the compiled controller.

CPU timing includes both neural execution and a direct integer Numba baseline. The direct baseline is faster for these standalone workloads. The MPS batch experiment runs Boolean self-hold with constant start input and no plant dynamics; CPU fallback is recorded when MPS is unavailable. Timing percentiles and observed maxima are not worst-case bounds.

## Attribution

The construction is inspired by [NESFormer](https://github.com/dbreunig/nesformer). PyTorch, NumPy, Numba, pytest, and safetensors are external dependencies; they are installed separately and are not vendored here. No reuse license was specified in the source repository; this materials package does not introduce a new license grant.
