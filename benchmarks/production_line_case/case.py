"""Adapters and a direct tensor baseline for Agents4PLC task 249.

The direct baseline expresses the original sequential IF/CASE semantics without
PLC-IR parsing or neural weights. The neural adapter uses an explicitly documented
one-hot lowering; it is not a general Structured Text compiler.
"""
from pathlib import Path
import ctypes
import numpy as np
import torch
from compiler.assembler import ILAssembler, UnrolledRungCompiler

SOURCE = Path(__file__).resolve().parents[1] / 'external' / 'agents4plc'
INPUTS = ['Mode', 'ForwardButton', 'ReverseButton', 'SensorA', 'SensorB',
          'SensorC', 'ButtonA', 'ButtonB', 'ButtonC']
OUTPUTS = ['State', 'MotorForward', 'MotorReverse', 'CompletionLight']


class COracle:
    """ctypes interface to matiec-generated C, not a Python/PLC-IR interpreter."""
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        ptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')
        self.lib.batch_step.argtypes = [ptr, ptr, ctypes.c_int32]
        self.lib.batch_step.restype = None

    def step(self, states, inputs):
        result = np.ascontiguousarray(states, dtype=np.int32).copy()
        values = np.ascontiguousarray(inputs, dtype=np.int32)
        assert result.shape == (len(values), 4) and values.shape[1] == 9
        self.lib.batch_step(result, values, len(values))
        return result


def direct_tensor_step(states, inputs):
    """Source-level state machine using where/Boolean operators on the device."""
    old, f, r, light = states.unbind(1)
    mode, forward, reverse, a, b, c, ba, bb, bc = (inputs != 0).unbind(1)
    init = old == 0
    old = torch.where(init, 1, old)
    f = torch.where(init, 0, f)
    r = torch.where(init, 0, r)
    light = torch.where(init, 0, light)
    t1 = mode & (old == 1) & a
    t2 = mode & (old == 2) & ba
    t3 = mode & (old == 3) & b
    t4 = mode & (old == 4) & bb
    t5 = mode & (old == 5) & c
    t6 = mode & (old == 6) & bc
    t7 = mode & (old == 7) & ~c
    new = old + (t1 | t2 | t3 | t4 | t5 | t6).to(torch.int32)
    new = torch.where(t7, 1, new)
    f = torch.where(~mode, (forward & ~reverse).to(torch.int32), f)
    r = torch.where(~mode, (reverse & ~forward).to(torch.int32), r)
    f = torch.where(t2 | t4, 1, f)
    r = torch.where(t2 | t4, 0, r)
    f = torch.where(t3 | t5, 0, f)
    light = torch.where(t6, 1, light)
    light = torch.where(t7, 0, light)
    return torch.stack((new, f, r, light), dim=1)


def encode_neural(states):
    bits = torch.zeros((len(states), 128), device=states.device)
    bits[:, 16:19] = states[:, 1:].float()
    bits[:, 32:40] = (states[:, :1] == torch.arange(8, device=states.device)).float()
    return bits


def decode_neural(bits):
    return torch.cat((bits[:, 32:40].argmax(dim=1, keepdim=True),
                      (bits[:, 16:19] >= .5).to(torch.int64)), dim=1).to(torch.int32)


def neural_model(device):
    instructions = ILAssembler.assemble_file(SOURCE / 'production_line.il')
    model = UnrolledRungCompiler(instructions).to(device).eval()
    assert model.is_pure_boolean
    return model


class Controller:
    def __init__(self, kind, initial, oracle=None):
        self.kind, self.oracle = kind, oracle
        self.model = neural_model(initial.device) if kind == 'plcformer' else None
        self.initial = initial.clone()
        self.reset()

    def reset(self):
        self.state = encode_neural(self.initial) if self.model is not None else self.initial.clone()

    def step(self, inputs):
        if self.kind == 'plcformer':
            self.state[:, :9] = inputs
            self.state = self.model(self.state)
            # Return command bits; the persistent neural state is not normalized.
            return self.state[:, 16:19] >= .5
        if self.kind == 'direct_tensor':
            self.state = direct_tensor_step(self.state, inputs)
        else:
            # Both device-to-host transfer and result copy are inside the boundary.
            result = self.oracle.step(self.state.cpu().numpy(), inputs.cpu().numpy())
            self.state = torch.as_tensor(result, device=inputs.device)
        return self.state[:, 1:] != 0

    def semantic_state(self):
        return decode_neural(self.state) if self.model is not None else self.state


class ConveyorSimulation:
    """Explicitly synthetic discrete-event plant; not a calibrated physical model.

    A part moves one position tick per scan through stations 0, 10 and 20.
    Processing takes three stopped scans at each station. A completed part is
    removed after three scans and the next part is supplied after two scans.
    This avoids inventing dimensional speed, mass, force or sensor accuracy.
    """
    def __init__(self, batch, device):
        self.batch, self.device = batch, device
        self.reset()

    def reset(self):
        self.position = torch.full((self.batch,), -1, device=self.device, dtype=torch.int32)
        self.dwell = torch.zeros_like(self.position)
        self.removal = torch.zeros_like(self.position)
        self.wait = torch.arange(self.batch, device=self.device, dtype=torch.int32) % 8
        self.completed = torch.zeros_like(self.position)
        self.previous = torch.zeros((self.batch, 3), device=self.device, dtype=torch.bool)

    def inputs(self):
        at_a, at_b, at_c = (self.position == n for n in (0, 10, 20))
        stopped = ~(self.previous[:, 0] | self.previous[:, 1])
        self.dwell = torch.where((at_a | at_b | at_c) & stopped, self.dwell + 1, 0)
        ready = self.dwell >= 3
        zero = torch.zeros_like(at_a)
        return torch.stack((~zero, zero, zero, at_a, at_b, at_c,
                            ready & at_a, ready & at_b, ready & at_c), dim=1).to(torch.int32)

    def advance(self, commands):
        f, r, done = commands.unbind(1)
        present = self.position >= 0
        movement = f.to(torch.int32) - r.to(torch.int32)
        self.position = torch.where(present, self.position + movement, self.position)
        self.removal = torch.where(done & present, self.removal + 1, 0)
        remove = present & (self.removal >= 3)
        self.completed += remove.to(torch.int32)
        self.position = torch.where(remove, -1, self.position)
        self.wait = torch.where(remove, 2, self.wait)
        absent = self.position < 0
        self.wait = torch.where(absent & ~remove, self.wait - 1, self.wait)
        self.position = torch.where(absent & ~done & (self.wait <= 0), 0, self.position)
        self.previous = commands
