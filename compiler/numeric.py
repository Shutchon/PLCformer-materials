"""Numeric contract for the executable PLC-IR subset (not IEC conformance)."""

import math

WORD_MIN = -32768
WORD_MAX = 32767
DEFAULT_SCALE = 32.0
COMPARATOR_INHIBITION = 131072.0
NORMALIZATION_TOLERANCE = 1e-4


def require_integer(value: float, lower: int = WORD_MIN, upper: int = WORD_MAX) -> float:
    value = float(value)
    if not math.isfinite(value) or value != round(value) or not lower <= value <= upper:
        raise ValueError(f"Expected an integer in [{lower}, {upper}], got {value}")
    return value
