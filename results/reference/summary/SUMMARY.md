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
