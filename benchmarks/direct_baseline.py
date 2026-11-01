"""Direct integer PLC-IR interpreter compiled by Numba, without neural weights.

Shares the instruction encoding, not the SwiGLU implementation. Includes input
latching and output collection just as the batched neural JIT measurement does.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def run_direct(program, inputs):
    bits = np.zeros(128, dtype=np.int32)
    words = np.zeros((96, 3), dtype=np.int64)
    stack = np.zeros(8, dtype=np.int32)
    outputs = np.zeros((len(inputs), 16), dtype=np.int32)
    for cycle in range(len(inputs)):
        for i in range(16):
            bits[i] = int(inputs[cycle, i] >= .5)
        acc = 0
        for j in range(len(program)):
            op, _, addr, imm = program[j]
            if op == 20:
                break
            if op == 1:
                acc = bits[addr]
            elif op == 2:
                acc = 1 - bits[addr]
            elif op == 3:
                acc &= bits[addr]
            elif op == 4:
                acc &= 1 - bits[addr]
            elif op == 5:
                acc |= bits[addr]
            elif op == 6:
                acc |= 1 - bits[addr]
            elif op == 7:
                bits[addr] = acc
            elif op == 8 and acc:
                bits[addr] = 1
            elif op == 9 and acc:
                bits[addr] = 0
                if addr >= 96:
                    words[addr-32, :] = 0
            elif op == 10:
                for i in range(7, 0, -1):
                    stack[i] = stack[i-1]
                stack[0] = acc
            elif op == 11:
                acc = stack[0]
            elif op == 12:
                acc = stack[0]
                for i in range(7):
                    stack[i] = stack[i+1]
                stack[7] = 0
            elif op == 13:
                row = addr - 32
                count = words[row, 0] + 1 if acc else 0
                if count > 32767:
                    raise OverflowError('timer overflow')
                words[row, 0] = count
                words[row, 1] = imm
                words[row, 2] = acc
                acc = int(bool(acc) and count >= imm)
                bits[addr] = acc
            elif op == 14:
                row = addr - 32
                count = words[row, 0] + int(bool(acc) and words[row, 2] == 0)
                if count > 32767:
                    raise OverflowError('counter overflow')
                words[row, 0] = count
                words[row, 1] = imm
                words[row, 2] = acc
                acc = int(count >= imm)
                bits[addr] = acc
            elif op == 15 and acc:
                words[addr, 0] = imm
            elif op == 16 and acc:
                value = words[addr, 0] + imm
                if value < -32768 or value > 32767:
                    raise OverflowError('word overflow')
                words[addr, 0] = value
            elif op == 17:
                acc &= int(words[addr, 0] == imm)
            elif op == 18:
                acc &= int(words[addr, 0] > imm)
            elif op == 19:
                acc &= int(words[addr, 0] < imm)
        outputs[cycle] = bits[16:32]
    return outputs
