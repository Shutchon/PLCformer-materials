"""Compile the original external ST using a separately installed pinned matiec.

Usage: python -m benchmarks.production_line_case.build_oracle --matiec-root PATH
No network access or compiler installation is performed by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from .case import SOURCE, INPUTS, OUTPUTS


def build(root, output, cc='clang'):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [str(root / 'iec2c'), '-I', str(root / 'lib'), '-T', str(output),
               str(SOURCE / 'portable.st')]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assignments = '\n'.join(
        f'    __SET_VAR(fb.,{name.upper()},,inputs[i*9+{j}]);'
        for j, name in enumerate(INPUTS))
    restore = '\n'.join(
        f'    __SET_VAR(fb.,{name.upper()},,states[i*4+{j}]);'
        for j, name in enumerate(OUTPUTS))
    collect = '\n'.join(
        f'    states[i*4+{j}] = __GET_VAR(fb.{name.upper()});'
        for j, name in enumerate(OUTPUTS))
    wrapper = '''#include <stdint.h>
#include "iec_types_all.h"
#include "accessor.h"
#include "iec_std_lib.h"
#include "POUS.h"
#include "POUS.c"
void batch_step(int32_t *states, const int32_t *inputs, int32_t count) {
  for (int32_t i=0; i<count; ++i) {
    FB_PRODUCTIONLINECONTROL_data__ fb;
    FB_PRODUCTIONLINECONTROL_init__(&fb, 0);
''' + restore + '\n' + assignments + '''
    FB_PRODUCTIONLINECONTROL_body__(&fb);
''' + collect + '\n  }\n}\n'
    (output / 'wrapper.c').write_text(wrapper)
    native = [cc, '-O3', '-shared', '-fPIC', '-I', str(root / 'lib/C'),
              '-I', str(output), str(output / 'wrapper.c'), '-o', str(output / 'oracle.so')]
    subprocess.run(native, check=True, capture_output=True, text=True)
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in ['POUS.c', 'POUS.h', 'wrapper.c', 'oracle.so']}
    info = {'matiec_commit': json.loads((SOURCE / 'PROVENANCE.json').read_text())['matiec_commit'],
            'matiec_command': command, 'native_command': native,
            'cc_version': subprocess.check_output([cc, '--version'], text=True).splitlines()[0],
            'matiec_binary_sha256': hashlib.sha256((root / 'iec2c').read_bytes()).hexdigest(),
            'generated_hashes': hashes}
    (output / 'build.json').write_text(json.dumps(info, indent=2)+'\n')
    return output / 'oracle.so'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matiec-root', required=True)
    parser.add_argument('--output', default='benchmarks/local/production_line_oracle')
    parser.add_argument('--cc', default='clang')
    args = parser.parse_args()
    print(build(args.matiec_root, args.output, args.cc))
