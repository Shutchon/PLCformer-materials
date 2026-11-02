# Reported results and reproduction map

Paths below are relative to this materials repository. The reference JSON files
are immutable snapshots; reruns can have different timing. No manuscript files
or LaTeX installation are needed to inspect or reproduce the computations.

| Evidence | Reference record | Code / input |
| --- | --- | --- |
| 400,000 normalized scans; eager prefixes | `results/reference/results_summary.json`, `experiment_1_equivalence` | `benchmarks/verify_equivalence.py`, `benchmarks/programs/`, `data/*.npz` |
| CPU and self-hold batch timing | Same JSON, `experiment_2_throughput` | `benchmarks/benchmark_speed.py`, `benchmarks/direct_baseline.py` |
| Interlock and safe-proposal checks | Same JSON, `experiment_3_safety_and_diff` | `benchmarks/experiment_safety_and_diff.py` |
| Gate error, latency and surrogate trajectory | `results/reference/ablation_results.json` | `benchmarks/run_ablations.py` |
| External canonical transitions and reachability | `results/reference/production_line_results.json`, `exhaustive` | `benchmarks/production_line_case/run.py`, `all_transitions` |
| External original/completed fixtures | Same JSON, `supplied_traces` | `benchmarks/external/agents4plc/source_record.json`, `supplied_traces` |
| External 32-line closed loop | Same JSON, `closed_loop` | `benchmarks/production_line_case/case.py`, `ConveyorSimulation`; runner `closed_loop` |
| External end-to-end timing and repeat durations | Same JSON, `timing` / `timing_protocol` | Runner `measure`; batch sizes 1, 64 and 1024 |
| External source and compiler identity | Same JSON, `provenance` / `oracle_build` | `benchmarks/external/agents4plc/PROVENANCE.json`, `build_oracle.py` |

## Fast inspection without experiment dependencies

```sh
python tools/verify_artifacts.py
python tools/summarize_results.py
```

Both use only the Python standard library. The first checks the package inventory,
original and external source hashes, and all four input/checkpoint pairs. The
second exports a Markdown summary and eleven CSVs to `results/local/summary/`.
Identical exports are supplied in `results/reference/summary/`.

## Execute checks

After installing the dependencies described in the root README:

```sh
python examples/self_hold.py
python -m pytest -q
```

The 51 Python tests include four external-case checks and require no matiec binary.
They do not replace the separate oracle/exhaustive/closed-loop runner. To rerun the
four synthetic workloads use `python -m benchmarks.run_all_experiments`; to build
the external oracle and run that case follow
[its protocol](../benchmarks/production_line_case/README.md).

## Expected invariants and limits

- Original workloads: zero process-image/stack mismatches after normalization.
- External case: zero mismatches on 32,768 canonical transitions per device and
  8,192 controller scans in the specified 32-line simulation; 224 completed parts.
- Original upstream fixtures: 18/20 supplied expectations match zero defaults;
  all 20 match between neural and matiec execution. The separately completed
  third fixture matches all five expected scans. Do not silently edit the source.
- A two-scan witness retains reverse after switching to automatic mode. Faithful
  execution does not establish correctness of the task's natural-language rules.
- Throughput and latency are environment-dependent measurements, not exact
  reproduction targets, certified limits or demonstrated neural speedups.

## Source adaptation

`original.st` and `source_record.json` retain the upstream material. `portable.st`
removes only the Siemens storage attribute and line comments. `production_line.il`
is a manual one-hot lowering, not an extension of the ST parser to general CASE.
The original source license and adaptation notice remain alongside these files.
