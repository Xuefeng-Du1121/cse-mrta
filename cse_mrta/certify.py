"""Certified selective evaluation and the min-sum tie-break."""

import math
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from .bottleneck import IncrementalBottleneck, bottleneck_assignment

class CertifiedResult:
    __slots__ = ("assignment", "bottleneck", "rounds", "num_eval", "evaluated", "costs", "assign_time", "runtime")
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
    def __repr__(self): return f"CertifiedResult(bottleneck={self.bottleneck}, rounds={self.rounds}, num_eval={self.num_eval}, runtime={self.runtime:.3f}s)"

def run_certified(oracle, lower_bounds, initial_pairs, solver="incremental", max_rounds=None):
    lb = np.asarray(lower_bounds, dtype=np.float64); n = lb.shape[0]
    if lb.shape != (n, n): raise ValueError("square R x G matrix expected (G = R)")
    if max_rounds is None: max_rounds = max(1, n * n)
    mixed, evaluated, costs = lb.copy(), np.zeros((n, n), dtype=bool), {}
    if solver == "incremental": inc = IncrementalBottleneck(mixed); solve = inc.solve
    elif solver == "cold": inc = None; solve = lambda: bottleneck_assignment(mixed)
    else: raise ValueError(solver)
    batch = hasattr(oracle, "evaluate_many"); rounds = 0; assign_time = 0.0; started = time.perf_counter()
    def add_pairs(pairs):
        new = [(int(i), int(j)) for i, j in dict.fromkeys(pairs) if not evaluated[i, j]]
        values = oracle.evaluate_many(new) if batch else [oracle(i, j) for i, j in new]
        for (i, j), v in zip(new, values):
            v = float(v)
            if inc is not None and v < mixed[i, j]: inc.invalidate()
            mixed[i, j], evaluated[i, j], costs[(i, j)] = v, True, v
    def lowest_batch():
        ii, jj = np.nonzero(~evaluated)
        if ii.size == 0: return []
        order = np.lexsort((jj, ii, lb[ii, jj]))[:max(1, min(n, ii.size))]
        return list(zip(ii[order].tolist(), jj[order].tolist()))
    def result(assignment, value):
        return CertifiedResult(assignment=assignment, bottleneck=value, rounds=rounds, num_eval=int(evaluated.sum()), evaluated=evaluated, costs=costs, assign_time=assign_time, runtime=time.perf_counter() - started)
    add_pairs(initial_pairs)
    while rounds < max_rounds:
        t0 = time.perf_counter(); _, assignment = solve(); assign_time += time.perf_counter() - t0
        if assignment is None:
            nxt = lowest_batch()
            if not nxt: return result(None, math.inf)
            add_pairs(nxt); rounds += 1; continue
        missing = [(i, j) for i, j in enumerate(assignment) if not evaluated[i, j]]
        if not missing: return result(list(assignment), max(costs[(i, j)] for i, j in enumerate(assignment)))
        add_pairs(missing); rounds += 1
    return result(None, math.inf)

def min_sum_tiebreak(n, costs, assignment):
    theta = max(costs[(i, j)] for i, j in enumerate(assignment)); big = 1e9; m = np.full((n, n), big)
    for (i, j), c in costs.items():
        if c <= theta + 1e-9: m[i, j] = c
    r, c = linear_sum_assignment(m); assert m[r, c].max() < big
    return [int(j) for j in c]
