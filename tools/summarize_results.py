"""Write Markdown and CSV summaries without manuscript/LaTeX dependencies."""
from pathlib import Path
import argparse
import csv
import json
import sys
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/reference')
    parser.add_argument('--output', type=Path, default=ROOT/'results/local/summary')
    args = parser.parse_args()
    s = json.loads((args.input/'results_summary.json').read_text())
    a = json.loads((args.input/'ablation_results.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {
        'equivalence': s['experiment_1_equivalence']['benchmarks'],
        'cpu_throughput': s['experiment_2_throughput']['single_instance'],
        'controller_batches': s['experiment_2_throughput']['batched_controllers'],
        'latency': [dict(program_file=k,**v) for k,v in a['latency_stats'].items()],
        'primitive_error': a['swiglu_ablation'],
        'surrogate_trajectory': a['diff_trajectory_every_5'],
    }
    for name, rows in datasets.items():
        with (args.output/f'{name}.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    eq=datasets['equivalence']; safety=s['experiment_3_safety_and_diff']['part_a_safety_guardrail']
    text = '# Recorded result summary\n\n'
    text += f"Scans: {sum(r['scan_cycles_tested'] for r in eq):,}; mismatched process-image/stack scans: {sum(r['full_state_mismatched_cycles'] for r in eq)}.\n\n"
    text += 'State checks follow instruction-level normalization and precede output thresholding.\n\n'
    text += f"Synthetic proposals: {safety['total_agent_actions_tested']:,}; exhaustive binary patterns: {safety['exhaustive_patterns']}; safe proposals changed: {safety['safe_proposals_changed']}.\n\n"
    text += '| Program | Neural Numba scans/s | Direct integer scans/s |\n| --- | ---: | ---: |\n'
    for r in datasets['cpu_throughput']:
        text += f"| {r['program_file']} | {r['numba_jit_cycles_per_sec']:,.0f} | {r['direct_numba_cycles_per_sec']:,.0f} |\n"
    text += '\nTimings describe the recorded environment; batch amortized cost is not response latency, and latency percentiles are not WCET bounds.\n'
    (args.output/'SUMMARY.md').write_text(text)
    print(f'Wrote SUMMARY.md and {len(datasets)} CSV files to {args.output}')

if __name__ == '__main__':
    main()
