"""Public trace fixtures and initialization semantics of the external case."""
import json
import re
import torch
from benchmarks.production_line_case.case import SOURCE, INPUTS, OUTPUTS, Controller


def test_portable_source_preserves_executable_statements():
    original = (SOURCE / 'original.st').read_text()
    expected = re.sub(r'//[^\n]*', '', original.replace("{ S7_Optimized_Access := 'TRUE' }\n", ''))
    assert (SOURCE / 'portable.st').read_text() == expected


def test_public_traces_and_explicit_fixture_initialization():
    record = json.loads((SOURCE / 'source_record.json').read_text())
    with torch.inference_mode():
        for number, case in enumerate(record['test_cases'], 1):
            initial = torch.tensor([[case['initial_state'].get(n, 0) for n in OUTPUTS]], dtype=torch.int32)
            # Case 3 omits an output initializer that its expected trace assumes.
            if number == 3:
                initial[0, 1] = 1
            controllers = [Controller(k, initial) for k in ['plcformer', 'direct_tensor']]
            expected = dict(case['expected_output_trace'])
            expected.update(case.get('expected_internal_state') or {})
            for tick in range(case['simulation_rounds']):
                inputs = torch.tensor([[case['input_trace'][n][tick] for n in INPUTS]], dtype=torch.int32)
                target = torch.tensor([[expected[n][tick] for n in OUTPUTS]], dtype=torch.int32)
                for controller in controllers:
                    controller.step(inputs)
                    assert torch.equal(controller.semantic_state(), target)


def test_initialization_and_automatic_branch_execute_in_same_scan():
    # Source clears pre-existing outputs in State 0, then executes State 1 CASE.
    initial = torch.tensor([[0, 1, 1, 1]], dtype=torch.int32)
    inputs = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.int32)
    with torch.inference_mode():
        for kind in ['plcformer', 'direct_tensor']:
            controller = Controller(kind, initial)
            controller.step(inputs)
            assert torch.equal(controller.semantic_state(), torch.tensor([[2, 0, 0, 0]]))


def test_upstream_reverse_retention_is_preserved():
    with torch.inference_mode():
        for kind in ['plcformer', 'direct_tensor']:
            controller = Controller(kind, torch.zeros((1, 4), dtype=torch.int32))
            controller.step(torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.int32))
            controller.step(torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32))
            assert torch.equal(controller.semantic_state(), torch.tensor([[1, 0, 1, 0]]))
