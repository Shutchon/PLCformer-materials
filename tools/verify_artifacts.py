"""Verify the reference source, trace/checkpoint artifacts, and file inventory (stdlib only)."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary = json.loads((ROOT/'results/reference/results_summary.json').read_text())
    for name, expected in summary['environment']['source_sha256'].items():
        assert digest(ROOT/name) == expected, f'Reference source mismatch: {name}'
    artifacts = json.loads((ROOT/'artifact_manifest.json').read_text())
    for item in artifacts['workloads']:
        for kind in ['data','checkpoint']:
            assert digest(ROOT/item[kind]) == item[f'{kind}_sha256'], item[kind]
    inventory = json.loads((ROOT/'MATERIALS_MANIFEST.json').read_text())
    for name, expected in inventory['files_sha256'].items():
        assert digest(ROOT/name) == expected, f'Material changed: {name}'
    assert not (ROOT/'paper').exists(), 'Manuscript directory must not be packaged'
    forbidden = [name for name in inventory['files_sha256'] if Path(name).suffix.lower() in {'.pdf','.tex','.docx'}]
    assert not forbidden, f'Manuscript files in package: {forbidden}'
    print(f"Verified {len(summary['environment']['source_sha256'])} source hashes, "
          f"{len(artifacts['workloads'])} trace/checkpoint pairs, and {len(inventory['files_sha256'])} material files.")

if __name__ == '__main__':
    main()
