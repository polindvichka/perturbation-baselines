"""Offline stand-ins for the challenge's mse and pds, computed on pseudobulk effect tables of a held-out public line.

Both follow cell-eval2's vcc2026 definitions (metrics/delta.py, configs/vcc2026.yaml):
  space      each group's expression as log1p(counts per 50,000), here log1p(5 * CP10K)
  genes      control CP10K >= 0.05 (the scorer's 5-CPM gene filter); each knockdown's own gene is left out
  raw mse    sum over knockdowns and genes of (pred - real)^2 / sum of (real - control)^2
             -> 1.0 = predicting no change; the leaderboard maps the context mean (~0.98 on A/B/C) to 0
  pds        cosine of each predicted change (pred - control) with every real change, rank of its own one
             -> 0 = always the best match, 0.5 = chance (the leaderboard's pds score is 1 - 2 * this, roughly)
Not reproduced: the scorer's sampling-noise correction, so absolute values sit closer to 1.0 (mse) and 0.5 (pds)
than on the leaderboard. Use them to compare methods, not to predict the leaderboard number.
"""
import numpy as np


def to_space(cp10k):
    return np.log1p(5.0 * np.asarray(cp10k, dtype=np.float64))


def predicted_delta(control_cp10k, lfc):
    """Change in scorer space when a log2 fold change is applied to the control level."""
    c = np.nan_to_num(np.asarray(control_cp10k, dtype=np.float64))
    return to_space(c * 2.0 ** np.nan_to_num(lfc)) - to_space(c)


def scores(pred_lfc, real_lfc, control_cp10k, own, min_expr=0.05):
    """pred_lfc, real_lfc: (knockdowns, genes) log2 FCs; own: same-shape bool mask of each knockdown's own gene.
    Returns (raw mse, pds) on the scorable genes; knockdowns or genes the real line did not measure are skipped."""
    c = np.nan_to_num(np.asarray(control_cp10k, dtype=np.float64))
    cols = c >= min_expr
    real = np.asarray(real_lfc, dtype=np.float64)[:, cols]
    keep = ~np.isnan(real) & ~np.asarray(own)[:, cols]
    d = np.where(keep, predicted_delta(c[cols], np.asarray(pred_lfc)[:, cols]), 0.0)
    t = np.where(keep, predicted_delta(c[cols], real), 0.0)
    raw = float(((d - t) ** 2).sum() / (t ** 2).sum())
    # pds only over knockdowns that have both a prediction and a measurement: an all-zero prediction has no
    # direction, and counting it would make it look like a perfect match
    rows = (np.linalg.norm(d, axis=1) > 0) & (np.linalg.norm(t, axis=1) > 0)
    dn = d[rows] / np.linalg.norm(d[rows], axis=1, keepdims=True)
    tn = t[rows] / np.linalg.norm(t[rows], axis=1, keepdims=True)
    s = dn @ tn.T
    pds = float(np.mean((s > np.diag(s)[:, None]).sum(1) / (len(s) - 1)))
    return raw, pds


def best_scale(pred_lfc, real_lfc, control_cp10k, own, min_expr=0.05):
    """The single multiplier of pred_lfc's scorer-space change that minimizes raw mse (linear approximation)."""
    c = np.nan_to_num(np.asarray(control_cp10k, dtype=np.float64))
    cols = c >= min_expr
    real = np.asarray(real_lfc, dtype=np.float64)[:, cols]
    keep = ~np.isnan(real) & ~np.asarray(own)[:, cols]
    d = np.where(keep, predicted_delta(c[cols], np.asarray(pred_lfc)[:, cols]), 0.0)
    t = np.where(keep, predicted_delta(c[cols], real), 0.0)
    return float((d * t).sum() / max((d * d).sum(), 1e-12))
