"""Admission rules: the initial evaluated set S0 of the certify/repair loop."""
import numpy as np

def argmin_pairs(lower_bounds):
    return [(i, int(np.argmin(lower_bounds[i]))) for i in range(lower_bounds.shape[0])]

def safe_pairs(lower_bounds, ctx):
    return sorted(set(argmin_pairs(lower_bounds)) | set(ctx["safe_matching"]))

def _with_safe(base, extra_mask):
    ii, jj = np.nonzero(extra_mask); chosen = set(base); chosen.update(zip(ii.tolist(), jj.tolist())); return sorted(chosen)

def budget_admission(lower_bounds, ctx, priority, budget):
    """S_safe plus the budget non-safe pairs with highest priority."""
    base = safe_pairs(lower_bounds, ctx); n = lower_bounds.shape[0]; pri = np.asarray(priority, dtype=np.float64).ravel().copy()
    for i, j in base: pri[i * n + j] = -np.inf
    k = int(min(max(0, budget), n * n - len(base)))
    if k == 0: return base
    lbf = lower_bounds.astype(np.float64).ravel(); flat = np.arange(n * n); cand = np.argpartition(-pri, k - 1)[:k]; cut = pri[cand].min()
    above = flat[pri > cut]; tied = flat[pri == cut]; tied = tied[np.lexsort((tied, lbf[tied]))][:k - above.size]
    extra = np.zeros(n * n, dtype=bool); extra[above] = True; extra[tied] = True
    return _with_safe(base, extra.reshape(n, n))

def random_admission(lower_bounds, ctx, num_extra, seed):
    base = safe_pairs(lower_bounds, ctx); n = lower_bounds.shape[0]; mask = np.ones((n, n), dtype=bool)
    for i, j in base: mask[i, j] = False
    pool = np.flatnonzero(mask.ravel()); k = int(min(max(0, num_extra), pool.size)); pick = np.random.default_rng(seed).choice(pool, size=k, replace=False)
    extra = np.zeros(n * n, dtype=bool); extra[pick] = True; return _with_safe(base, extra.reshape(n, n))

def lazy_lb(lower_bounds, ctx=None): return argmin_pairs(lower_bounds)
def heuristic_b(lower_bounds, ctx, budget): return budget_admission(lower_bounds, ctx, -np.asarray(lower_bounds, dtype=np.float64), budget)
def random_b(lower_bounds, ctx, budget, seed): return random_admission(lower_bounds, ctx, budget, seed)
def learned_b(lower_bounds, ctx, scores, budget): return budget_admission(lower_bounds, ctx, scores, budget)
