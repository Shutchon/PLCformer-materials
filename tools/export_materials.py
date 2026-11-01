"""Export synthetic inputs and implementation-specific initial tensor checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.verify_equivalence import BENCHMARKS, PROGRAMS_DIR
from compiler.build_checkpoint import export_plcformer_safetensors
from runtime.host import PLCformerHost

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/local/export')
    args = parser.parse_args()
    for folder in ['data', 'checkpoints']:
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    entries = []
    for name, filename, generator, seed in BENCHMARKS:
        stem = Path(filename).stem
        inputs = generator(100_000, seed=seed)
        data = args.output / 'data' / f'{stem}.npz'
        np.savez_compressed(data, inputs=inputs, seed=np.array(seed), scan_ticks=np.array(1))
        host = PLCformerHost()
        instructions = host.load_program_file(PROGRAMS_DIR / filename)
        checkpoint = export_plcformer_safetensors(host, args.output / 'checkpoints' / f'{stem}.safetensors')
        entries.append({'name': name, 'program': f'benchmarks/programs/{filename}',
                        'seed': seed, 'input_shape': list(inputs.shape), 'input_dtype': str(inputs.dtype),
                        'scan_ticks': 1, 'instructions': len(instructions),
                        'data': str(data.relative_to(args.output)),
                        'checkpoint': str(checkpoint.relative_to(args.output)),
                        'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
                        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    (args.output/'artifact_manifest.json').write_text(json.dumps({'format_version': 1, 'workloads': entries}, indent=2)+'\n')
    print(f'Exported {len(entries)} traces and checkpoints to {args.output}')

if __name__ == '__main__':
    main()
