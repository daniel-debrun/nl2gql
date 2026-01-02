# Results

Held-out test set: set E (300 lines, {'query': 225, 'clarify': 45, 'reject': 30}), never used for training or tuning. Reference date 2026-11-18.

## Ablation on the held-out set

| training data | system | decision acc. | exact match | subgraph | metric | query type | grouping | direction | filters | time | limit | valid queries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic only | bayes-net + text evidence | 0.987 | 0.956 | 0.987 | 0.982 | 0.982 | 0.987 | 0.978 | 0.987 | 0.982 | 0.973 | 1.000 (222) |
| synthetic only | bayes-net, no text evidence | 0.983 | 0.956 | 0.987 | 0.982 | 0.982 | 0.987 | 0.978 | 0.987 | 0.982 | 0.973 | 1.000 (222) |
| synthetic only | independent per-slot | 0.957 | 0.738 | 0.987 | 0.978 | 0.831 | 0.836 | 0.911 | 0.987 | 0.982 | 0.973 | 1.000 (231) |
| sets A+B+C+D | bayes-net + text evidence | 0.983 | 0.916 | 0.973 | 0.964 | 0.982 | 0.956 | 0.973 | 0.987 | 0.982 | 0.978 | 1.000 (222) |
| sets A+B+C+D | bayes-net, no text evidence | 0.983 | 0.929 | 0.987 | 0.982 | 0.982 | 0.956 | 0.982 | 0.987 | 0.982 | 0.978 | 1.000 (222) |
| sets A+B+C+D | independent per-slot | 0.957 | 0.720 | 0.987 | 0.916 | 0.889 | 0.836 | 0.956 | 0.987 | 0.982 | 0.973 | 0.961 (231) |
| sets A+B+C+D + synthetic | bayes-net + text evidence | 0.983 | 0.956 | 0.987 | 0.982 | 0.982 | 0.987 | 0.978 | 0.987 | 0.982 | 0.973 | 1.000 (223) |
| sets A+B+C+D + synthetic | bayes-net, no text evidence | 0.983 | 0.956 | 0.987 | 0.982 | 0.982 | 0.987 | 0.978 | 0.987 | 0.982 | 0.973 | 1.000 (222) |
| sets A+B+C+D + synthetic | independent per-slot | 0.953 | 0.787 | 0.987 | 0.982 | 0.844 | 0.836 | 0.960 | 0.987 | 0.982 | 0.973 | 1.000 (231) |
| (none) | rules only | 0.940 | 0.964 | 0.987 | 0.982 | 0.982 | 0.987 | 0.982 | 0.987 | 0.982 | 0.978 | 1.000 (223) |

## Leave one set out (train on the other three + synthetic)

| held out | exact match | decision acc. |
|---|---|---|
| A | 0.956 | 1.000 |
| B | 0.987 | 0.993 |
| C | 0.991 | 1.000 |
| D | 0.982 | 0.997 |

## Final model

Trained on 7200 examples, 62 KB on disk. On the held-out set: decision accuracy 0.983, exact match 0.956, valid emitted queries 1.000, median latency 2.741 ms (p95 3.878 ms).

Decision confusion (gold -> predicted): clarify->clarify: 44, clarify->query: 1, query->clarify: 3, query->query: 222, reject->clarify: 1, reject->reject: 29

## Example data

| file | lines | kinds | failed the checker |
|---|---|---|---|
| A.jsonl | 300 | {'query': 225, 'clarify': 45, 'reject': 30} | 0 |
| B.jsonl | 300 | {'query': 225, 'clarify': 45, 'reject': 30} | 0 |
| C.jsonl | 300 | {'query': 225, 'clarify': 45, 'reject': 30} | 0 |
| D.jsonl | 300 | {'query': 225, 'clarify': 45, 'reject': 30} | 0 |
| E.jsonl | 300 | {'query': 225, 'clarify': 45, 'reject': 30} | 0 |

Tuned configuration: `{"alpha": 0.5, "nb_alpha": 0.5, "nb_weight": 0.2, "use_nb": true, "use_network": true, "tau_ood": 0.7, "tau_ambiguous": 0.15, "tau_missing": 0.3}`
