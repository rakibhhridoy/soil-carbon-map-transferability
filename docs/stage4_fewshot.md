# Stage 4 --- Few-shot calibration

Data cost of making a model usable on an unsampled delta: median skill over held-out deltas as $k$ local cores are added to training (10 random draws each).

| $k$ local cores | median RMSE (Mg/ha) | median within-delta $r$ | deltas |
|---:|---:|---:|---:|
| 0 | 75.5 | 0.137 | 5 |
| 5 | 81.6 | 0.322 | 5 |
| 10 | 70.1 | 0.545 | 5 |
| 25 | 62.4 | 0.671 | 5 |

Adding even a handful of local cores recovers substantial skill, quantifying the local-data requirement for crediting-grade estimates in an unsampled delta.
