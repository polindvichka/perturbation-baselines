"""Do the iPSC CRISPRi screens help? Hiding one public line at a time, on the 300 final-phase targets.

Feng (HipSci iPSC, log2 FCs; scripts/build_feng_table.py) and KOLF2.1J (iPSC, z-scores; scripts/build_kolf_table.py)
are compared with the plain mean of the other lines, by adding each to that mean with a few weights. KOLF z-scores
are first rescaled to the other lines' size (least squares over the knockdowns both have).
Proxies as in src/proxy.py (raw mse at the best single scale, pds rank; lower is better), target genes left out.
Coverage of the hidden line's knockdowns is printed, since a source only changes the knockdowns it measured.

Usage: VCC_PANEL=test python scripts/ipsc_check.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.proxy import best_scale, scores
from map_model import DATA, EXPR, LABEL_LINES, final_targets, genes, load

LAB = {k: DATA[k][0] for k in DATA}
EXTRA = {"feng": load("feng_lfc.npz")[0], "kolf": load("kolf_z.npz")[0]}
col = {x: i for i, x in enumerate(genes)}
tcols = np.array([col[t] for t in final_targets if t in col])
print({k: f"{len(np.intersect1d(v.index, final_targets))} of 300 targets" for k, v in EXTRA.items()}, flush=True)

for H in LABEL_LINES:
    perts = np.array([t for t in final_targets if t in LAB[H].index])
    real = LAB[H].reindex(perts).to_numpy(np.float64)
    e = EXPR[H]
    own = np.zeros_like(real, bool); own[:, tcols] = True
    with np.errstate(all="ignore"):
        S = np.stack([LAB[k].reindex(perts).to_numpy(np.float64) for k in LAB if k != H])
        base_n = (~np.isnan(S)).sum(0)
        base = np.nan_to_num(np.nanmean(S, 0))

    def score(p):
        s = best_scale(p, real, e, own)
        return scores(s * p, real, e, own)

    raw0, pds0 = score(base)
    print(f"hidden {H}: plain mean raw mse {raw0:.4f}  pds {pds0:.3f}", flush=True)
    for name, tab in EXTRA.items():
        x = tab.reindex(perts).to_numpy(np.float64)
        have = ~np.isnan(x)
        both = have & (base_n > 0)
        scale = float((x[both] * base[both]).sum() / (x[both] ** 2).sum())   # onto the other lines' size
        x = np.nan_to_num(x) * scale
        print(f"  {name}: measures {have.any(1).sum()} of {len(perts)} knockdowns, size factor {scale:.3f}")
        for w in [0.5, 1, 2]:                       # w = weight of the extra source next to each other line
            p = np.where(have, (base * base_n + w * x) / (base_n + w * have), base)
            raw, pds = score(p)
            print(f"    weight {w}: raw mse {raw:.4f}  pds {pds:.3f}", flush=True)
