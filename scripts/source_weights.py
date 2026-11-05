"""How much could weighting the source lines gain? Hiding one public line at a time, on the 300 final-phase
targets: score the plain source mean, then a weighted mean with the best weights found by looking at the hidden
line's answer (an upper bound: if even these gain little, similarity weights cannot gain more).
Proxies as in src/proxy.py (raw mse at the best single scale, pds rank; lower is better), target genes left out.

Usage: python scripts/source_weights.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.proxy import best_scale, scores
from map_model import DATA, EXPR, LABEL_LINES, final_targets, genes

LAB = {k: DATA[k][0] for k in DATA}
col = {x: i for i, x in enumerate(genes)}
tcols = np.array([col[t] for t in final_targets if t in col])
GRID = [0, 0.5, 1, 2, 4]


def weighted_mean(S, w):
    w = np.asarray(w, float)[:, None, None] * ~np.isnan(S)
    return np.nan_to_num((np.nan_to_num(S) * w).sum(0) / np.where(w.sum(0) > 0, w.sum(0), np.nan))


for H in LABEL_LINES:
    perts = np.array([t for t in final_targets if t in LAB[H].index])
    real = LAB[H].reindex(perts).to_numpy(np.float64)
    e = EXPR[H]
    own = np.zeros_like(real, bool); own[:, tcols] = True
    others = [k for k in LAB if k != H]
    S = np.stack([LAB[k].reindex(perts).to_numpy(np.float64) for k in others])

    def score(w):
        p = weighted_mean(S, w)
        s = best_scale(p, real, e, own)
        return scores(s * p, real, e, own)

    raw0, pds0 = score([1] * len(others))
    res = [(w, *score(w)) for w in itertools.product(GRID, repeat=len(others)) if any(w) and max(w) == GRID[-1] or w == (1,) * len(others)]
    by_pds, by_mse = min(res, key=lambda r: r[2]), min(res, key=lambda r: r[1])
    print(f"hidden {H}, sources {others}")
    print(f"  plain mean        : raw mse {raw0:.4f}  pds {pds0:.3f}")
    print(f"  best for pds {by_pds[0]}: raw mse {by_pds[1]:.4f}  pds {by_pds[2]:.3f}")
    print(f"  best for mse {by_mse[0]}: raw mse {by_mse[1]:.4f}  pds {by_mse[2]:.3f}", flush=True)
