"""Pair scorer used by Learned-B: an MLP over the 14 pair features with a sigmoid output."""

import numpy as np
import torch
from torch import nn
from .features import FEATURE_NAMES

class PairScorer(nn.Module):
    def __init__(self, hidden_dim, feature_dim=len(FEATURE_NAMES)):
        super().__init__(); self.feature_dim = feature_dim; self.hidden_dim = hidden_dim
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)

def load_scorer(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if list(ckpt["feature_names"]) != FEATURE_NAMES: raise ValueError("checkpoint feature names do not match features.FEATURE_NAMES")
    model = PairScorer(hidden_dim=ckpt["hidden_dim"]); model.load_state_dict(ckpt["state_dict"]); model.eval()
    return {"model": model.to(device), "mean": np.asarray(ckpt["mean"], dtype=np.float32), "std": np.asarray(ckpt["std"], dtype=np.float32), "device": device}

def score_pairs(scorer, feats, chunk=1 << 18):
    n = feats.shape[0]; flat = feats.reshape(-1, feats.shape[-1]); out = np.empty(flat.shape[0], dtype=np.float32)
    with torch.no_grad():
        for k in range(0, flat.shape[0], chunk):
            x = (flat[k:k + chunk] - scorer["mean"]) / scorer["std"]
            out[k:k + chunk] = scorer["model"](torch.from_numpy(x).to(scorer["device"])).cpu().numpy()
    return out.reshape(n, n)
