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
    external = json.loads((args.input/'production_line_results.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {
        'equivalence': s['experiment_1_equivalence']['benchmarks'],
        'cpu_throughput': s['experiment_2_throughput']['single_instance'],
        'controller_batches': s['experiment_2_throughput']['batched_controllers'],
        'latency': [dict(program_file=k,**v) for k,v in a['latency_stats'].items()],
        'primitive_error': a['swiglu_ablation'],
        'surrogate_trajectory': a['diff_trajectory_every_5'],
        'external_transitions': external['exhaustive'],
        'external_closed_loop': external['closed_loop'],
        'external_fixtures': external['supplied_traces'],
        'external_timing': external['timing'],
        'external_timing_repeats': [
            dict(device=r['device'], batch=r['batch'], method=r['method'],
                 repeat=i+1, scans=r['scans_per_repeat'], seconds=seconds,
                 batch_scan_ms=seconds/r['scans_per_repeat']*1000)
            for r in external['timing'] for i,seconds in enumerate(r['repeat_seconds'])],
    }
    for name, rows in datasets.items():
        with (args.output/f'{name}.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                              for k,v in row.items()} for row in rows)
    eq=datasets['equivalence']; safety=s['experiment_3_safety_and_diff']['part_a_safety_guardrail']
    text = '# Recorded result summary\n\n'
    text += f"Scans: {sum(r['scan_cycles_tested'] for r in eq):,}; mismatched process-image/stack scans: {sum(r['full_state_mismatched_cycles'] for r in eq)}.\n\n"
    text += 'State checks follow instruction-level normalization and precede output thresholding.\n\n'
    text += f"Synthetic proposals: {safety['total_agent_actions_tested']:,}; exhaustive binary patterns: {safety['exhaustive_patterns']}; safe proposals changed: {safety['safe_proposals_changed']}.\n\n"
    text += '| Program | Neural Numba scans/s | Direct integer scans/s |\n| --- | ---: | ---: |\n'
    for r in datasets['cpu_throughput']:
        text += f"| {r['program_file']} | {r['numba_jit_cycles_per_sec']:,.0f} | {r['direct_numba_cycles_per_sec']:,.0f} |\n"
    text += '\nTimings describe the recorded environment; batch amortized cost is not response latency, and latency percentiles are not WCET bounds.\n'
    text += '\n## External production-line case\n\n'
    for check in external['exhaustive']:
        loop = next(r for r in external['closed_loop'] if r['device'] == check['device'])
        text += (f"{check['device'].upper()}: {check['transitions']:,} canonical transitions; "
                 f"{check['neural_mismatches']} neural mismatches; "
                 f"{loop['controller_scans']:,} closed-loop controller scans; "
                 f"{loop['state_mismatches']} state mismatches; "
                 f"{loop['completed_parts']} completed parts.\n\n")
    text += '| Device | Lines | Native C (ms) | Direct tensor (ms) | PLCformer (ms) |\n| --- | ---: | ---: | ---: | ---: |\n'
    groups = sorted({(r['device'],r['batch']) for r in external['timing']})
    for device,batch in groups:
        values = {r['method']:r['median_batch_scan_ms'] for r in external['timing']
                  if (r['device'],r['batch']) == (device,batch)}
        text += (f"| {device.upper()} | {batch:,} | {values['matiec_host']:.3f} | "
                 f"{values['direct_tensor']:.3f} | {values['plcformer']:.3f} |\n")
    text += ('\nMedians describe a complete synthetic loop; native C includes host transfers. '
             'PLCformer is slower than both baselines at all recorded device/batch pairs. '
             'All 54 repeat durations are exported separately. Two upstream fixture scans '
             'disagree with zero-default initialization, although all 20 match between '
             'PLCformer and matiec. The completed fixture and the retained reverse '
             'command after a manual-to-automatic mode switch are reported separately '
             'in the JSON and case protocol. No physical plant validation is claimed.\n')
    (args.output/'SUMMARY.md').write_text(text)
    print(f'Wrote SUMMARY.md and {len(datasets)} CSV files to {args.output}')

if __name__ == '__main__':
    main()
