"""Trained transfer model on 7 cell contexts: a weighted ridge regression that learns how a knockdown's measured
effect in other lines carries over to an unseen line.

Compared with transfer_model.py (v14), which learned from 3 hidden lines and only our 300 knockdowns:
  - training contexts: K562, RPE1, Jurkat, HepG2, HCT116, HEK293T, H1 (each hidden in turn), and the training
    examples are ALL knockdowns the hidden line shares with at least one other line (~2,000 genes), not just ours
  - one extra input: the sources' average weighted by how similar each source's control cells are to the target's
    (correlation of control profiles after removing the shared average profile)
CD4 enters as an input value only (its table holds just our 300 knockdowns and no control expression).

One example = (knockdown p, measured gene j) in the hidden line; label = its log2 FC there; weight = how much that
gene counts in the scorer's log1p(5 * CP10K) space (as in transfer_model.py).

Modes
  cv     hide K562, HCT116, HEK293T in turn (on our 300 knockdowns) and compare: plain average, the v14-style ridge
         (3 lines, our knockdowns only) and this model (trained on the other 6 contexts, all shared knockdowns)
  final  train on all 7 contexts, predict A, B, C (each with its own control expression and similarities)
         -> PROCESSED/effects_ridge7.npz, lfc shape (3, 300, 18533); use with make_submission.py SCALE
Usage: python scripts/transfer_wide.py cv|final
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED

MODE = sys.argv[1]
EPS, MIN_CELLS, MIN_EXPR = 0.1, 20, 0.05
ROWS_PER_FOLD = 2_000_000
SEED = 0
LINES = {"k562": ["k562_delta.npz"], "rpe1": ["rpe1_delta.npz"], "jurkat": ["jurkat_delta.npz"],
         "hepg2": ["hepg2_delta.npz"], "hct116": ["xatlas_HCT116_delta.npz"], "hek293t": ["xatlas_HEK293T_delta.npz"],
         "h1": ["h1_train_delta.npz", "h1_val_delta.npz", "h1_test_delta.npz"]}
OLD_LINES = ["k562", "hct116", "hek293t"]          # the 3 hidden lines v14 learned from

genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"]
G = len(genes)
rng = np.random.default_rng(SEED)


def load(files):
    """(lfc DataFrame knockdowns x challenge genes, float32; control CP10K on the challenge axis)."""
    frames, ctrl = [], []
    for file in files:
        z = np.load(PROCESSED / file)
        keep = z["n_cells"] >= MIN_CELLS
        c = z["basal_cp10k"] if "basal_cp10k" in z.files else z["control_cp10k"]
        l = (np.log2(z["mean_cp10k"][keep] + EPS) - np.log2(c + EPS)).astype(np.float32)
        l[:, c == 0] = np.nan
        col = pd.Index(genes).get_indexer(z["genes"])
        ok = (col >= 0) & ~pd.Series(z["genes"]).duplicated().to_numpy()
        full = np.full((int(keep.sum()), G), np.nan, np.float32)
        full[:, col[ok]] = l[:, ok]
        frames.append(pd.DataFrame(full, index=z["perts"][keep].astype(str)))
        e = np.full(G, np.nan, np.float32); e[col[ok]] = c[ok]
        ctrl.append(e)
    lfc = pd.concat(frames).groupby(level=0).mean() if len(frames) > 1 else frames[0]
    return lfc[~lfc.index.duplicated()], np.nanmean(np.stack(ctrl), 0)


def context_expr(ctx):
    a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
    s = np.asarray(a.X.sum(0)).ravel()
    return pd.Series(1e4 * s / s.sum(), index=a.var_names.astype(str)).reindex(genes).to_numpy(np.float32)


print("loading tables ...", flush=True)
DATA = {k: load(v) for k, v in LINES.items()}
z = np.load(PROCESSED / "cd4_lfc.npz")
CD4 = pd.DataFrame(z["lfc"].astype(np.float32), index=z["perts"].astype(str))
CTX = {c: context_expr(c) for c in "ABC"}
print({k: v[0].shape[0] for k, v in DATA.items()}, "knockdowns per line", flush=True)

# similarity of control profiles: correlation after removing the average profile of all lines and contexts
PROFILES = {**{k: v[1] for k, v in DATA.items()}, **{f"ctx_{c}": e for c, e in CTX.items()}}
P = np.log1p(np.stack(list(PROFILES.values())))
use = np.all(np.nan_to_num(P, nan=-1) > np.log1p(0.05), 0)
Pc = P[:, use] - P[:, use].mean(0)
SIM = pd.DataFrame(np.corrcoef(Pc), index=list(PROFILES), columns=list(PROFILES))
print("control-profile similarity to contexts:\n", SIM.loc[list(LINES), ["ctx_A", "ctx_B", "ctx_C"]].round(2), flush=True)


def features(perts, sources, tgt_expr, sims):
    """Ridge inputs for every (p in perts, j); sources: names of lines used as evidence; sims: {source: similarity}."""
    S = np.stack([DATA[s][0].reindex(perts).to_numpy(np.float32) for s in sources])     # (n_src, P, G)
    have = ~np.isnan(S)
    n = have.sum(0).astype(np.float32)
    with np.errstate(all="ignore"):
        mean = np.nan_to_num(np.nanmean(S, 0))
        w = np.array([max(sims[s], 0.0) + 0.05 for s in sources], np.float32)[:, None, None] * have
        simmean = np.nan_to_num(np.nansum(S * w, 0) / w.sum(0))
    up, down = (S > 0).sum(0), (S < 0).sum(0)
    agree = (np.maximum(up, down) / np.maximum(n, 1)).astype(np.float32)
    del S, w, have
    te = np.log1p(np.nan_to_num(tgt_expr)).astype(np.float32)[None, :]
    cd4 = np.nan_to_num(CD4.reindex(perts).reindex(columns=range(G)).to_numpy(np.float32))
    h1 = np.nan_to_num(DATA["h1"][0].reindex(perts).to_numpy(np.float32)) if "h1" in sources else np.zeros_like(mean)
    f = {"mean": mean, "mean*agree": mean * agree, "mean*tgt_expr": mean * te, "mean*n": mean * n,
         "simmean": simmean, "simmean*agree": simmean * agree, "cd4": cd4, "h1": h1}
    return f, n


def own_mask(perts):
    pos = {g: i for i, g in enumerate(genes)}
    m = np.zeros((len(perts), G), bool)
    for i, p in enumerate(perts):
        if p in pos:
            m[i, pos[p]] = True
    return m


def bulk_weight(c):
    c = np.nan_to_num(c)
    return ((np.log(2) * 5 * c / (1 + 5 * c)) ** 2).astype(np.float32)


def fold(held, exclude=(), only_targets=False):
    """Training rows with `held` hidden: X (rows x inputs), y, weights, plus the full arrays for evaluation."""
    sources = [s for s in LINES if s != held and s not in exclude]
    lfc, expr = DATA[held]
    others = set().union(*(DATA[s][0].index for s in sources)) | set(CD4.index)
    perts = [p for p in lfc.index if p in others and (p in set(targets) or not only_targets)]
    f, n = features(perts, sources, expr, {s: SIM.loc[s, held] for s in sources})
    label = lfc.reindex(perts).to_numpy(np.float32)
    mask = ~np.isnan(label) & (n > 0) & (np.nan_to_num(expr) >= MIN_EXPR)[None, :] & ~own_mask(perts)
    idx = np.flatnonzero(mask.ravel())
    if len(idx) > ROWS_PER_FOLD:
        idx = np.sort(rng.choice(idx, ROWS_PER_FOLD, replace=False))
    X = np.column_stack([v.ravel()[idx] for v in f.values()])
    w = np.broadcast_to(bulk_weight(expr), mask.shape).ravel()[idx]
    return X, label.ravel()[idx], w, (perts, f, mask, label, expr)


def fit(folds, names):
    X = np.vstack([fo[0] for fo in folds]); y = np.concatenate([fo[1] for fo in folds])
    w = np.concatenate([fo[2] for fo in folds])
    model = Ridge(alpha=1.0, fit_intercept=False).fit(X, y, sample_weight=w)
    print(f"  fitted on {len(y):,} examples; weights:", dict(zip(names, np.round(model.coef_, 3))), flush=True)
    return model


def predict(model, f, perts):
    out = (np.stack(list(f.values()), -1) @ model.coef_).astype(np.float32)
    own = own_mask(perts)
    out[own] = f["mean"][own]                  # the knocked-down gene itself: keep the plain average
    return out


NAMES = ["mean", "mean*agree", "mean*tgt_expr", "mean*n", "simmean", "simmean*agree", "cd4", "h1"]

if MODE == "cv":
    for test in OLD_LINES:
        print(f"hidden line: {test}", flush=True)
        wide = fit([fold(h, exclude=(test,))[:3] for h in LINES if h != test], NAMES)
        old = fit([fold(h, exclude=(test,), only_targets=True)[:3] for h in OLD_LINES if h != test], NAMES)
        _, _, _, (perts, f, mask, label, expr) = fold(test, only_targets=True)
        c = np.broadcast_to(np.nan_to_num(expr), mask.shape)[mask]
        to_bulk = lambda l: np.log1p(5 * c * 2.0 ** l) - np.log1p(5 * c)
        yb = to_bulk(label[mask])
        T = np.where(mask, np.nan_to_num(label), 0)
        for name, pred in [("plain average", f["mean"]), ("v14-style ridge (3 lines)", predict(old, f, perts)),
                           ("new ridge (6 other contexts)", predict(wide, f, perts))]:
            for s in (1, 4, 7):
                pb = to_bulk(s * pred[mask])
                r2 = 1 - np.mean((yb - pb) ** 2) / np.mean((yb - yb.mean()) ** 2)
                print(f"  {name:29s} x{s}: scorer-space R2 {r2:+.4f}", flush=True)
            Pn = np.where(mask, pred, 0); Pn = Pn / np.maximum(np.linalg.norm(Pn, axis=1, keepdims=True), 1e-9)
            Tn = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
            Sm = Pn @ Tn.T
            print(f"  {name:29s}    : pds-like {np.mean((Sm > np.diag(Sm)[:, None]).sum(1) / (len(perts) - 1)):.3f}"
                  " (0 = best, 0.5 = chance)", flush=True)

elif MODE == "final":
    model = fit([fold(h)[:3] for h in LINES], NAMES)
    preds = []
    for ctx in "ABC":
        sources = list(LINES)
        f, n = features(list(targets), sources, CTX[ctx], {s: SIM.loc[s, f"ctx_{ctx}"] for s in sources})
        p = predict(model, f, list(targets))
        p[n == 0] = 0.0
        preds.append(p)
        print(f"context {ctx}: mean|lfc| {np.abs(p).mean():.4f}", flush=True)
    out = PROCESSED / "effects_ridge7.npz"
    np.savez_compressed(out, perts=targets.to_numpy(str), genes=genes.to_numpy(str), lfc=np.stack(preds),
                        contexts=np.array(list("ABC")))
    print("wrote", out)
