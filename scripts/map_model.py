"""Learned map between lines: predict a knockdown's whole response in an unseen line from its responses in other
lines, trained on ALL knockdowns the public lines share (~1,850), not only the 300 challenge genes.

Why: transfer_model.py learns 6 + 6 weights from the 300 challenge knockdowns and sees one (knockdown, gene) pair
at a time. Here each response is a vector over genes; the map can learn that a pattern in K562 shows up as a
somewhat different pattern elsewhere, and it removes the noise outside the main response patterns.

Steps, with one line H hidden (cv) or none (final):
  space    each line's log2 FCs on the shared genes G0, weighted by the scorer's sensitivity sqrt(bulk weight)
  basis    top K principal patterns of all responses of the source lines (H left out)
  maps     per source S: y_L = s * x_S + x_S B_S in the basis, pooled over the other label lines L != S, H;
           ridge on B_S (alpha), so it starts from a plain rescaled copy of x_S and learns corrections
  combine  mean over the sources that measured the knockdown, then back to genes
Evaluated on the 300 final-phase (D/E/F) targets of H, which are never used for training, with the offline proxies
(src/proxy.py: raw mse at the best single scale, lower is better; pds rank, lower is better).

Usage: python scripts/map_model.py cv [K=100] [ALPHA=10]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PANEL_DIRS, PROCESSED
from src.proxy import best_scale, scores

MODE = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else None   # None when imported
K = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
EPS, MIN_CELLS, MIN_EXPR = 0.1, 20, 0.05
FILES = {"k562": "k562_delta.npz", "hct116": "xatlas_HCT116_delta.npz", "hek293t": "xatlas_HEK293T_delta.npz",
         "cd4": "cd4_lfc.npz"}
LABEL_LINES = ["k562", "hct116", "hek293t"]           # lines with their own control expression (CD4: source only)

genes = pd.read_csv(PANEL_DIRS["test"] / "gene_names.csv")["gene_name"].astype(str)
final_targets = pd.read_csv(PANEL_DIRS["test"] / "pert_counts.csv")["target_gene"].astype(str).to_numpy()


def load(file):
    """(lfc DataFrame all knockdowns x challenge genes, control CP10K Series or None)."""
    z = np.load(PROCESSED / file)
    keep = z["n_cells"] >= MIN_CELLS
    perts = z["perts"][keep].astype(str)
    if "lfc" in z.files:
        d = pd.DataFrame(z["lfc"][keep], index=perts, columns=z["genes"].astype(str))
        return d.loc[:, ~d.columns.duplicated()].reindex(columns=genes).groupby(level=0).mean(), None
    c = z["basal_cp10k"] if "basal_cp10k" in z.files else z["control_cp10k"]
    lfc = np.log2(z["mean_cp10k"][keep] + EPS) - np.log2(c + EPS)
    lfc[:, c == 0] = np.nan
    d = pd.DataFrame(lfc, index=perts, columns=z["genes"].astype(str))
    d = d.loc[:, ~d.columns.duplicated()].reindex(columns=genes).groupby(level=0).mean()
    e = pd.Series(c, index=z["genes"].astype(str)).groupby(level=0).mean().reindex(genes)
    return d, e


DATA = {k: load(f) for k, f in FILES.items()}
EXPR = {k: v[1].to_numpy() for k, v in DATA.items() if v[1] is not None}
# shared genes: measured by every line and expressed in all label lines
measured = np.all([~v[0].isna().all(axis=0).to_numpy() for v in DATA.values()], 0)
expressed = np.all([np.nan_to_num(e) >= MIN_EXPR for e in EXPR.values()], 0)
G0 = np.flatnonzero(measured & expressed)
mean_c = np.nanmean(np.stack([EXPR[k][G0] for k in EXPR]), 0)
W = np.log(2) * 5 * mean_c / (1 + 5 * mean_c)          # sqrt of the scorer's bulk weight per gene
print(f"shared genes G0: {len(G0)}; knockdowns per line:", {k: len(v[0]) for k, v in DATA.items()}, flush=True)


def vectors(name):
    """Weighted responses on G0: DataFrame knockdowns x G0 (NaN -> 0)."""
    d = DATA[name][0].iloc[:, G0]
    return pd.DataFrame(np.nan_to_num(d.to_numpy()) * W, index=d.index)


VEC = {k: vectors(k) for k in DATA}


def fit(hidden):
    """Basis and per-source maps without line `hidden`; training knockdowns exclude the final targets."""
    lines = [k for k in VEC if k != hidden]
    allv = np.vstack([VEC[k].to_numpy() for k in lines])
    mu = allv.mean(0)
    _, _, Vt = np.linalg.svd(allv - mu, full_matrices=False)
    P = Vt[:K].T                                                        # (G0, K)
    maps = {}
    for S in lines:
        X, Y = [], []
        for L in [l for l in LABEL_LINES if l not in (S, hidden)]:
            common = VEC[S].index.intersection(VEC[L].index).difference(final_targets)
            X.append(VEC[S].loc[common].to_numpy() @ P); Y.append(VEC[L].loc[common].to_numpy() @ P)
        if not X:
            continue
        X, Y = np.vstack(X), np.vstack(Y)
        s = float((X * Y).sum() / (X * X).sum())
        R = Y - s * X
        B = np.linalg.solve(X.T @ X + ALPHA * len(X) * np.eye(K), X.T @ R)
        maps[S] = (s, B, len(X))
    return P, maps


def predict(P, maps, sources, perts, use_map=True, keep=0.0):
    """Mean over the sources that measured each knockdown, back on G0 (weighted space). keep: share of each
    response's part outside the basis that is kept (the knockdown-specific detail pds needs)."""
    out, cnt = np.zeros((len(perts), len(G0))), np.zeros(len(perts))
    for S in sources:
        have = np.isin(perts, VEC[S].index)
        if not have.any() or S not in maps:
            continue
        x = VEC[S].reindex(perts[have]).to_numpy()
        if use_map == "raw":
            y = x
        else:
            s, B, _ = maps[S]
            a = x @ P
            y = (s * a + a @ B if use_map else s * a) @ P.T + keep * s * (x - a @ P.T)
        out[have] += y; cnt[have] += 1
    return out / np.maximum(cnt, 1)[:, None], cnt


if MODE == "cv":
    for H in LABEL_LINES:
        P, maps = fit(H)
        print(f"hidden {H}: maps", {S: (round(m[0], 3), m[2]) for S, m in maps.items()}, flush=True)
        perts = np.array([t for t in final_targets if t in VEC[H].index])
        sources = [k for k in VEC if k != H]
        real = DATA[H][0].reindex(perts).iloc[:, G0].to_numpy()
        e = EXPR[H][G0]
        own = np.array([[g == p for g in genes.to_numpy()[G0]] for p in perts])
        for name, how, keep in [("source mean (no basis)", "raw", 0), ("basis only", False, 0),
                                ("basis + learned map", True, 0), ("map + 25% detail", True, 0.25),
                                ("map + 50% detail", True, 0.5), ("map + all detail", True, 1.0)]:
            pw, cnt = predict(P, maps, sources, perts, how, keep)
            lfc = pw / W                                               # back to log2 FCs
            ok = cnt > 0
            s = best_scale(lfc[ok], real[ok], e, own[ok])
            raw, pds = scores(s * lfc[ok], real[ok], e, own[ok])
            print(f"  {name:24s}: raw mse {raw:.4f}  pds {pds:.3f}  (best scale {s:.2f}, {ok.sum()} knockdowns)",
                  flush=True)
