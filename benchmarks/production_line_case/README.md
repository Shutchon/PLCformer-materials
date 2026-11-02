# External production-line case

This case evaluates **Agents4PLC task 249_ProductionLineControl**, a published
three-station production-line controller. It adds an independent compiler oracle
and an end-to-end tensor simulation workload. It is **not physical plant data**.

The reference record, all three supplied test cases, original ST, source license,
and immutable commit provenance are in `../external/agents4plc/`. The nine inputs
are Mode, ForwardButton, ReverseButton, SensorA/B/C and ButtonA/B/C. Persistent
semantic state is the integer State (0–7) and three Boolean outputs. The neural
representation uses Y0–Y2 for outputs, M0–M7 for one-hot State and M8–M14 for
transition guards. **State 0 requires M0=1**, not an all-zero process image.

The 85-instruction PLC-IR adaptation includes END and has 84 Boolean blocks.
All guards are computed before updating one-hot state, preserving the source
CASE behavior. Source initialization executes before CASE in the same scan.
This manual lowering does not extend PLCformer's ST parser to general CASE.

## Build the independent oracle

Use Python dependencies in the root `requirements-repro.txt`. Build tools are a
C/C++ compiler, make, autoconf, automake, flex, and bison >=2.4. On macOS, Apple's
bundled bison may be too old; put an installed newer bison first on PATH.

```sh
git clone https://github.com/beremiz/matiec /tmp/matiec
git -C /tmp/matiec checkout 3a41303bcb4e50403417bc39c63f88981d7d961e
cd /tmp/matiec
autoreconf -i
./configure
make -j4
```

Return to the PLCformer/materials root, then run:

```sh
python -m benchmarks.production_line_case.build_oracle --matiec-root /tmp/matiec
mkdir -p results/local
python -m benchmarks.production_line_case.run \
  --oracle benchmarks/local/production_line_oracle/oracle.so \
  --devices cpu,mps --output results/local/production_line_results.json
```

Use `--devices cpu` on machines without Apple MPS. Device substitution is never
silent. The oracle builder removes no executable statement: the supplied
`portable.st` differs only by omission of the Siemens storage attribute and `//`
comments. matiec parses that ST and generates C, which clang compiles with `-O3`.
The wrapper initializes a function-block instance, restores all persistent
semantic variables, applies the nine inputs, calls the generated body, and reads
State plus the three outputs. No PLCformer parser is used in this path. Compiler
identity, commands and generated-file checksums are recorded with each run.
The matiec source is obtained separately under its upstream license; its runtime
headers and the generated native library are not vendored in this case.

## Checks and interpretation

1. Run all supplied traces against matiec and PLCformer. Report supplied expected
   output discrepancies separately from implementation disagreement. Case 3 gives
   only State=3 initially but expects MotorForward=true during two idle scans;
   its source does not derive MotorForward from State. Also report an explicitly
   completed fixture with MotorForward=true, without modifying the original.
2. Enumerate all 8 states × 8 output combinations × 512 input patterns = 32,768
   one-scan transitions. Compare PLCformer and direct tensor operations with
   matiec, including the one-hot state encoding and raw float32 error.
3. Explore reachable reference states from State=0/all outputs off. Check motor
   mutual exclusion. Record a reachable counterexample to the natural-language
   claim of no reverse command in automatic mode: the source can retain reverse
   from manual mode. Faithful compilation does not repair that source behavior.
4. Run 32 parallel lines for 256 scans with a shared synthetic discrete-event
   conveyor plant, comparing all three implementations on every scan. Parts
   visit positions 0, 10 and 20; processing takes three stopped scans. Completed
   parts are removed after three scans and replacement waits two scans. Initial
   arrival offsets vary by line index modulo eight. These are explicit scenario
   assumptions, not identified physical parameters.

The timing workload includes input construction, controller execution, output
decoding and plant advance. Three methods use the same inputs and simulator:

- `plcformer`: unrolled SwiGLU network, persistent float32 process image.
- `direct_tensor`: direct predicates/where operations, integer semantic state.
- `matiec_host`: compiled native C plus device–host transfers in both directions.

Batch sizes are 1, 64 and 1024. Eight warm-up scans precede three repetitions of
64 scans; method order is shuffled with seed 249. Setup/reset/compilation are
excluded. MPS is synchronized at timing boundaries. Every method/repetition must
produce identical final semantic state and every plant-state field (positions,
dwell/removal/wait counters, completion counts and previous commands). All
repeat durations are saved, including unfavorable results. These wall times are
batch-step latency, not individual-controller latency or a real-time bound.

The direct tensor method is essential: avoiding a CPU round trip alone would
not establish a benefit specific to neural compilation. Apple unified memory,
Python dispatch, a small program and a synthetic plant limit generalization.
No trained perception model, field I/O, hardware-in-the-loop, external safety
certification, or full IEC-language equivalence is evaluated.

## Recorded outcome

The supplied `results/reference/production_line_results.json` records the CPU and MPS run; `benchmarks/production_line_results.json` is an initial working copy.
Each backend has zero mismatches on 32,768 canonical transitions and 8,192
closed-loop controller scans, with 224 completed parts in that scenario and zero
observed raw error in the eleven semantic bits. All 20 original fixture scans
agree between PLCformer and matiec; two disagree with the upstream expectations
under zero defaults. The separately completed third fixture matches all five
expected scans. The two-scan automatic-mode reverse witness is also retained.

PLCformer is slower than both baselines for every recorded device/batch pair.
These results support execution agreement, not a measured performance advantage.
Run timing without concurrent tests or benchmark jobs, and save a reproduction
to another output path to retain the supplied reference measurements.
