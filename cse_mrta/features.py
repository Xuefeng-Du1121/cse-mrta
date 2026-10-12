"""Pair features for the learned admission rule.

All 14 features use only quantities available before any oracle call: the
lower-bound matrix, the grid, the start/goal cells and a static per-cell
congestion field (any non-negative array the oracle can provide at set-up;
use zeros if the oracle has none).
"""

import math

import numpy as np

from .bottleneck import bottleneck_assignment

FEATURE_NAMES = [
    "lb_over_theta", "row_gap_over_theta", "col_gap_over_theta", "row_min_over_theta", "col_min_over_theta",
    "log_robot_rank", "log_target_rank", "bbox_field_mean", "bbox_obstacle_density", "start_field", "goal_field",
    "detour_ratio", "log_n", "robot_density",
]

def _integral(a):
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s

def _box_sum(s, r0, r1, c0, c1):
    return s[r1 + 1, c1 + 1] - s[r0, c1 + 1] - s[r1 + 1, c0] + s[r0, c0]

def _ranks(matrix, axis):
    order = np.argsort(matrix, axis=axis, kind="stable")
    ranks = np.empty_like(order)
    if axis == 1:
        ranks[np.arange(matrix.shape[0])[:, None], order] = np.arange(matrix.shape[1])[None, :]
    else:
        ranks[order, np.arange(matrix.shape[1])[None, :]] = np.arange(matrix.shape[0])[:, None]
    return ranks

def instance_context(lower_bounds, grid, starts, goals, field=None):
    """Per-instance quantities shared by the admission rules and the features.

    grid: 2-D array, 0 = free, 1 = obstacle. field: optional static congestion
    field of the same shape (default zeros).
    """
    theta, safe = bottleneck_assignment(lower_bounds)
    arr = np.asarray(grid, dtype=np.float64)
    return {"theta_lb": float(theta), "safe_matching": [(i, int(j)) for i, j in enumerate(safe)],
            "obstacles": arr, "field": np.zeros_like(arr) if field is None else np.asarray(field, dtype=np.float64),
            "starts": np.asarray(starts, dtype=np.int64), "goals": np.asarray(goals, dtype=np.int64)}

def pair_features(lower_bounds, ctx):
    """n x n x F float32 feature tensor (see FEATURE_NAMES)."""
    lb = lower_bounds.astype(np.float64)
    n = lb.shape[0]
    theta = max(1.0, ctx["theta_lb"])
    row_min, col_min = lb.min(axis=1, keepdims=True), lb.min(axis=0, keepdims=True)
    log_n1 = math.log(1 + n)
    s, g = ctx["starts"], ctx["goals"]
    r0, r1 = np.minimum(s[:, None, 0], g[None, :, 0]), np.maximum(s[:, None, 0], g[None, :, 0])
    c0, c1 = np.minimum(s[:, None, 1], g[None, :, 1]), np.maximum(s[:, None, 1], g[None, :, 1])
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    field_sum = _box_sum(_integral(ctx["field"]), r0, r1, c0, c1)
    obs_sum = _box_sum(_integral(ctx["obstacles"]), r0, r1, c0, c1)
    manh = np.abs(s[:, None, 0] - g[None, :, 0]) + np.abs(s[:, None, 1] - g[None, :, 1])
    field, free_cells = ctx["field"], float((ctx["obstacles"] == 0).sum())
    feats = np.empty((n, n, len(FEATURE_NAMES)), dtype=np.float32)
    feats[..., 0] = lb / theta; feats[..., 1] = (lb - row_min) / theta; feats[..., 2] = (lb - col_min) / theta
    feats[..., 3] = np.broadcast_to(row_min / theta, (n, n)); feats[..., 4] = np.broadcast_to(col_min / theta, (n, n))
    feats[..., 5] = np.log1p(_ranks(lb, axis=1)) / log_n1; feats[..., 6] = np.log1p(_ranks(lb, axis=0)) / log_n1
    feats[..., 7] = field_sum / area; feats[..., 8] = obs_sum / area
    feats[..., 9] = np.broadcast_to(field[s[:, 0], s[:, 1]][:, None], (n, n)); feats[..., 10] = np.broadcast_to(field[g[:, 0], g[:, 1]][None, :], (n, n))
    feats[..., 11] = lb / np.maximum(1, manh); feats[..., 12] = math.log(n); feats[..., 13] = n / max(1.0, free_cells)
    return feats
