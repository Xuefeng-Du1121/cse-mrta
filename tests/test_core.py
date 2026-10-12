import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from cse_mrta import (FEATURE_NAMES, IncrementalBottleneck, bottleneck_assignment, heuristic_b, instance_context, lazy_lb, learned_b, min_sum_tiebreak, pair_features, random_b, run_certified)
from toy_example import ToyOracle, make_instance

def test_incremental_matches_cold():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 30)); m = rng.integers(1, int(rng.choice([3, 10, 100])), size=(n, n)).astype(np.float64); inc = IncrementalBottleneck(m)
        for _ in range(int(rng.integers(1, 10))):
            theta, asg = inc.solve(); assert theta == bottleneck_assignment(m)[0]; assert sorted(asg) == list(range(n)) and m[np.arange(n), asg].max() == theta
            idx = rng.choice(n * n, size=int(rng.integers(1, n * n + 1)), replace=False); m.ravel()[idx] += rng.integers(0, 20, size=idx.size)

def test_every_rule_certifies_full_eval_optimum():
    for seed in range(5):
        grid, starts, goals, lb = make_instance(32, 40, seed); n = lb.shape[0]
        full = np.array([[ToyOracle(lb, seed)(i, j) for j in range(n)] for i in range(n)]); ref, _ = bottleneck_assignment(full)
        ctx = instance_context(lb, grid, starts, goals); feats = pair_features(lb, ctx)
        assert feats.shape == (n, n, len(FEATURE_NAMES)) and np.isfinite(feats).all()
        fake_scores = np.random.default_rng(seed).random((n, n))
        for s0 in (lazy_lb(lb), heuristic_b(lb, ctx, 2 * n), random_b(lb, ctx, 2 * n, seed), learned_b(lb, ctx, fake_scores, 2 * n)):
            for solver in ("incremental", "cold"):
                res = run_certified(ToyOracle(lb, seed), lb, s0, solver=solver); assert res.bottleneck == ref
                asg = min_sum_tiebreak(n, res.costs, res.assignment); assert max(res.costs[(i, j)] for i, j in enumerate(asg)) == ref
