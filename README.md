# CSE-MRTA: certified selective evaluation for bottleneck task assignment

Core implementation of the method in *Certified selective evaluation of
congestion-aware pair costs for large-scale multi-robot target assignment* (under review).

Given R robots, R targets, a cheap lower bound l_ij (e.g. BFS distance) and an
expensive exact-cost oracle c_ij >= l_ij, the solver returns an assignment whose
bottleneck max_i c_{i,pi(i)} is certified optimal while evaluating only a small
fraction of the R x R pairs. Oracle calls of one repair round are independent,
so each round can run as one parallel batch.

## Contents

| module | what it provides |
|---|---|
| `cse_mrta/certify.py` | certify/repair loop (`run_certified`), witness-closure certificate, `+MS` min-sum tie-break |
| `cse_mrta/bottleneck.py` | cold bottleneck solver and the incremental (warm-started) solver |
| `cse_mrta/admission.py` | initial evaluated set: Lazy-LB, Heuristic-B, Random-B, Learned-B (all contain S_safe) |
| `cse_mrta/features.py` | the 14 pair features computed before any oracle call |
| `cse_mrta/scorer.py` | scorer architecture and inference interface (`load_scorer`, `score_pairs`) |
| `examples/toy_example.py` | end-to-end run with a placeholder oracle |
| `tests/test_core.py` | incremental = cold solver; every rule certifies the full-evaluation optimum |

## Quick start

```bash
pip install -r requirements.txt
python examples/toy_example.py
python -m pytest tests
```

```python
from cse_mrta import instance_context, heuristic_b, run_certified, min_sum_tiebreak

ctx = instance_context(lb, grid, starts, goals)      # lb: R x R lower bounds
s0 = heuristic_b(lb, ctx, budget)                   # or lazy_lb(lb), learned_b(lb, ctx, scores, budget)
res = run_certified(oracle, lb, s0)                 # oracle(i, j) -> exact cost >= lb[i, j]
assignment = min_sum_tiebreak(len(lb), res.costs, res.assignment)
```

The certified bottleneck does not depend on the admission rule or on the scorer;
these only change the number of oracle calls and rounds.

## Not included in this release

The congestion-aware oracle and its settings, instance generators and seeds,
trained scorer checkpoints, training labels and training scripts, budget
settings, baselines, experiment drivers, raw results and the scripts that
produce the paper's tables and figures. They will be released after the
paper is accepted.

## License

MIT (see `LICENSE`).
