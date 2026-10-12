#!/usr/bin/env python3
"""Minimal end-to-end example on a random grid with a stand-in oracle.

The oracle here is a placeholder (BFS distance times a random per-pair factor
>= 1). It is not the congestion-aware oracle used in the paper; replace
ToyOracle with any exact-cost routine whose values are >= the lower bounds.
"""
import os
import sys
from collections import deque
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cse_mrta import bottleneck_assignment, heuristic_b, instance_context, lazy_lb, min_sum_tiebreak, random_b, run_certified

def bfs_lower_bounds(grid, starts, goals):
    rows, cols = grid.shape; out = np.full((len(starts), len(goals)), np.inf)
    for i, s in enumerate(starts):
        dist = np.full(grid.shape, np.inf); dist[s] = 0; q = deque([s])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1; q.append((nr, nc))
        out[i] = [dist[g] for g in goals]
    return out

class ToyOracle:
    def __init__(self, lb, seed):
        self.lb = lb; self.factor = 1.0 + np.random.default_rng(seed).exponential(0.3, size=lb.shape); self.calls = 0
    def __call__(self, i, j):
        self.calls += 1; return float(np.ceil(self.lb[i, j] * self.factor[i, j]))

def make_instance(size, n, seed):
    rng = np.random.default_rng(seed); grid = (rng.random((size, size)) < 0.15).astype(np.int8); free = [tuple(x) for x in np.argwhere(grid == 0)]
    cells = [free[k] for k in rng.choice(len(free), size=2 * n, replace=False)]; starts, goals = cells[:n], cells[n:]; lb = bfs_lower_bounds(grid, starts, goals)
    ok = np.isfinite(lb).all(axis=1) & np.isfinite(lb).all(axis=0)
    if not ok.all(): return make_instance(size, n, seed + 1)
    return grid, starts, goals, lb

def main(size=48, n=60, seed=0):
    grid, starts, goals, lb = make_instance(size, n, seed); oracle = ToyOracle(lb, seed); full = np.array([[oracle(i, j) for j in range(n)] for i in range(n)]); ref, _ = bottleneck_assignment(full)
    print(f"R=G={n}, full evaluation: {n * n} oracle calls, bottleneck {ref:g}")
    ctx = instance_context(lb, grid, starts, goals); budget = 2 * n
    rules = {"Lazy-LB": lazy_lb(lb), "Heuristic-B": heuristic_b(lb, ctx, budget), "Random-B": random_b(lb, ctx, budget, seed)}
    for name, s0 in rules.items():
        res = run_certified(ToyOracle(lb, seed), lb, s0); asg = min_sum_tiebreak(n, res.costs, res.assignment); assert res.bottleneck == ref
        print(f"{name:12s} bottleneck {res.bottleneck:g}  oracle calls {res.num_eval:5d}  rounds {res.rounds:3d}  +MS sum {sum(res.costs[(i, j)] for i, j in enumerate(asg)):g}")

if __name__ == "__main__": main()
