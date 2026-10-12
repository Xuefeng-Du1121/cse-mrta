"""Bottleneck assignment solvers."""

import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def _perfect_matching_under(matrix, threshold):
    graph = csr_matrix(matrix <= threshold)
    match = maximum_bipartite_matching(graph, perm_type="column")
    if np.any(match < 0): return None
    return match

def bottleneck_assignment(matrix):
    """Minimise max_i C[i, a(i)] over perfect matchings."""
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0: return math.inf, None
    values = np.unique(finite)
    floor = max(float(np.min(matrix, axis=1).max()), float(np.min(matrix, axis=0).max()))
    if not math.isfinite(floor): return math.inf, None
    lo, hi = int(np.searchsorted(values, floor, side="left")), len(values) - 1
    best = _perfect_matching_under(matrix, values[hi])
    if best is None: return math.inf, None
    while lo < hi:
        mid = (lo + hi) // 2
        match = _perfect_matching_under(matrix, values[mid])
        if match is None: lo = mid + 1
        else: hi, best = mid, match
    if lo == len(values) - 1 or best is None or float(matrix[np.arange(len(best)), best].max()) != float(values[lo]):
        best = _perfect_matching_under(matrix, values[lo])
    assignment = [int(j) for j in best]
    return float(matrix[np.arange(len(assignment)), assignment].max()), assignment

def _augment(adj, row_match, col_match, r):
    n = adj.shape[1]
    parent, seen = np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=bool)
    frontier = np.array([r], dtype=np.int64)
    while frontier.size:
        sub = adj[frontier]
        cols = np.flatnonzero(sub.any(axis=0) & ~seen)
        if cols.size == 0: return False
        parent[cols] = frontier[np.argmax(sub[:, cols], axis=0)]
        seen[cols] = True
        free = cols[col_match[cols] < 0]
        if free.size:
            c = int(free[0])
            while True:
                rr, prev = int(parent[c]), int(row_match[int(parent[c])])
                row_match[rr], col_match[c] = c, rr
                if rr == r: return True
                c = prev
        frontier = col_match[cols]
    return False

class IncrementalBottleneck:
    """Bottleneck assignment for a matrix whose entries only increase between calls."""
    def __init__(self, matrix):
        self.matrix = matrix; self.theta = None; self.row = None; self.col = None; self.cold_calls = 0; self.probes = 0
    def invalidate(self): self.theta = None
    def _cold(self):
        self.cold_calls += 1
        theta, assignment = bottleneck_assignment(self.matrix)
        if assignment is None: self.theta = None; return math.inf, None
        n = len(assignment); self.row = np.asarray(assignment, dtype=np.int64); self.col = np.full(n, -1, dtype=np.int64)
        self.col[self.row] = np.arange(n); self.theta = float(theta)
        return self.theta, assignment
    def _decide(self, threshold, row, col):
        self.probes += 1; adj = self.matrix <= threshold
        for r in np.flatnonzero(row < 0):
            if not _augment(adj, row, col, int(r)): return False
        return True
    def solve(self):
        if self.theta is None: return self._cold()
        m, n = self.matrix, self.matrix.shape[0]
        row, col = self.row.copy(), self.col.copy()
        bad = np.flatnonzero(m[np.arange(n), row] > self.theta); col[row[bad]] = -1; row[bad] = -1
        if bad.size == 0: self.row, self.col = row, col; return self.theta, row.tolist()
        floor = max(float(m.min(axis=1).max()), float(m.min(axis=0).max()))
        if not math.isfinite(floor): self.theta = None; return math.inf, None
        finite = m[np.isfinite(m)]; start = max(self.theta, floor); vals = np.unique(finite[finite >= start])
        if vals.size == 0: self.theta = None; return math.inf, None
        lo_row, lo_col, lo_idx = row, col, -1; hi_idx, hi_match = None, None; i = 0
        while True:
            r2, c2 = lo_row.copy(), lo_col.copy()
            if self._decide(vals[i], r2, c2): hi_idx, hi_match = i, (r2, c2); break
            lo_row, lo_col, lo_idx = r2, c2, i
            if i == vals.size - 1: self.theta = None; return math.inf, None
            i = min(2 * i + 1, vals.size - 1)
        while hi_idx - lo_idx > 1:
            mid = (lo_idx + hi_idx) // 2; r2, c2 = lo_row.copy(), lo_col.copy()
            if self._decide(vals[mid], r2, c2): hi_idx, hi_match = mid, (r2, c2)
            else: lo_row, lo_col, lo_idx = r2, c2, mid
        self.row, self.col = hi_match; self.theta = float(vals[hi_idx])
        return self.theta, self.row.tolist()
