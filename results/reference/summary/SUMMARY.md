# Recorded result summary

Scans: 400,000; mismatched process-image/stack scans: 0.

State checks follow instruction-level normalization and precede output thresholding.

Synthetic proposals: 10,000; exhaustive binary patterns: 16; safe proposals changed: 0.

| Program | Neural Numba scans/s | Direct integer scans/s |
| --- | ---: | ---: |
| self_hold.il | 1,226,193 | 114,843,526 |
| star_delta.il | 391,905 | 55,419,593 |
| traffic_light.il | 175,869 | 33,114,403 |
| sorting.il | 617,583 | 71,241,917 |

Timings describe the recorded environment; batch amortized cost is not response latency, and latency percentiles are not WCET bounds.

## External production-line case

CPU: 32,768 canonical transitions; 0 neural mismatches; 8,192 closed-loop controller scans; 0 state mismatches; 224 completed parts.

MPS: 32,768 canonical transitions; 0 neural mismatches; 8,192 closed-loop controller scans; 0 state mismatches; 224 completed parts.

| Device | Lines | Native C (ms) | Direct tensor (ms) | PLCformer (ms) |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 0.085 | 0.150 | 1.282 |
| CPU | 64 | 0.086 | 0.160 | 2.331 |
| CPU | 1,024 | 0.116 | 0.199 | 10.735 |
| MPS | 1 | 1.835 | 0.807 | 4.615 |
| MPS | 64 | 1.881 | 0.867 | 4.659 |
| MPS | 1,024 | 1.612 | 0.875 | 4.754 |

Medians describe a complete synthetic loop; native C includes host transfers. PLCformer is slower than both baselines at all recorded device/batch pairs. All 54 repeat durations are exported separately. Two upstream fixture scans disagree with zero-default initialization, although all 20 match between PLCformer and matiec. The completed fixture and the retained reverse command after a manual-to-automatic mode switch are reported separately in the JSON and case protocol. No physical plant validation is claimed.
