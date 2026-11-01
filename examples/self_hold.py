"""Run a start/hold/stop trace with both implementations."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.host import PLCformerHost
from runtime.fast_engine import FastNumbaEngine

program = 'LD X0\nOR Y0\nANI X1\nOUT Y0\nEND'
eager = PLCformerHost(program)
fast = FastNumbaEngine(PLCformerHost(program))
expected = [0, 1, 1, 0, 0]
trace = [(0, 0), (1, 0), (0, 0), (0, 1), (0, 0)]
for scan, ((start, stop), answer) in enumerate(zip(trace, expected), 1):
    inputs = {'X0': start, 'X1': stop}
    output = eager.run_scan_cycle(inputs).outputs_y['Y0']
    fast.run_scan_cycle(inputs)
    assert output == fast.read_bit('Y0') == answer
    print(f'scan={scan} start={start} stop={stop} motor={output}')
