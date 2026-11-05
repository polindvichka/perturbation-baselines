"""Remove the main axes of the context's own control-cell variation from every predicted change (a pds idea that
other teams report helps: Sidechain's ablation, rpinho's LESSONS file; see notes/research_2026-10-25.md).

The top principal components of a context's control cells are mostly cell cycle, depth, stress and other
variation that every knockdown group shares by chance. A predicted change that points along them looks alike
for many knockdowns, which blurs pds (the scorer ranks each prediction against all real changes by cosine).
Per context: PCs of the control cells in the scorer's space log1p(5 * CP10K) (genes with CP10K >= 0.05), then
each predicted change d (same space) -> d - V V^T d, and back to log2 fold changes.

Usage: [VCC_PANEL=...] python scripts/remove_control_pcs.py EFFECTS K       e.g. effects_ridgepk_h1 20
Writes PROCESSED/{EFFECTS}_pc{K}.npz (same layout, one table per context).
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTEXTS, CONTROLS, PROCESSED
from src.proxy import predicted_delta, to_space

EFFECTS, K = sys.argv[1], int(sys.argv[2])
MIN_EXPR, N_CELLS, SEED = 0.05, 10_000, 0

z = np.load(PROCESSED / f"{EFFECTS}.npz")
lfc_all = z["lfc"] if z["lfc"].ndim == 3 else z["lfc"][None]
out = []
for k, ctx in enumerate(CONTEXTS):
    a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
    assert list(a.var_names) == list(z["genes"]), "control genes are in a different order"
    X = sparse.csr_matrix(a.X)[np.random.default_rng(SEED).permutation(a.n_obs)[:N_CELLS]]
    lib = np.asarray(X.sum(1)).ravel()
    c = np.asarray(X.sum(0)).ravel() / lib.sum() * 1e4                      # context control CP10K
    cols = np.flatnonzero(c >= MIN_EXPR)
    cells = sparse.diags(5e4 / lib) @ X[:, cols]                            # CP50K per cell
    cells.data = np.log1p(cells.data)
    M = cells.toarray()
    M -= M.mean(0)
    _, sv, Vt = randomized_svd(M, K, random_state=SEED)
    V = Vt.T                                                                 # (genes kept, K)
    lfc = lfc_all[min(k, len(lfc_all) - 1)].astype(np.float64)
    d = predicted_delta(c[cols], lfc[:, cols])                              # (knockdowns, genes kept)
    d_new = d - (d @ V) @ V.T
    with np.errstate(divide="ignore", invalid="ignore"):
        new_level = np.maximum(np.expm1(to_space(c[cols]) + d_new) / 5.0, 0.0)
        lfc_new = np.log2(np.maximum(new_level, 1e-6) / c[cols])
    removed = 1 - (d_new ** 2).sum() / max((d ** 2).sum(), 1e-12)
    res = lfc.copy()
    res[:, cols] = lfc_new
    out.append(res.astype(np.float32))
    print(f"context {ctx}: top {K} control PCs explain {sv[:3].round(1)}...; removed {removed:.1%} of the predicted "
          f"change", flush=True)

dst = PROCESSED / f"{EFFECTS}_pc{K}.npz"
np.savez_compressed(dst, perts=z["perts"], genes=z["genes"], lfc=np.stack(out), contexts=np.array(CONTEXTS))
print("wrote", dst)
