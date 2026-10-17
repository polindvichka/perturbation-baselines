"""Turn predicted effects (scripts/make_effects.py) into a submission: 400 cells per gene per context.

For every context (A, B, C) and every target gene:
  1. draw 400 real control cells of that context (with replacement),
  2. change each gene's counts by the predicted fold change, f = 2 ** (SCALE * lfc):
       f < 1: keep each count with probability f       (binomial thinning: 10 counts, f=0.3 -> about 3)
       f > 1: add Poisson(count * (f - 1)) extra counts (10 counts, f=1.5 -> about 15)
     so the result stays whole numbers, keeps real cell-to-cell variation, and zeros stay zero
     (which keeps the matrix sparse).

MODE (optional):
  resample  (default) the factor is applied to the 400 drawn cells as they are, so each group's
            average also carries the chance difference between those 400 cells and all controls
  exact     each gene's group total is aimed at exactly 400 x (control average) x f, by correcting
            the factor for what the 400 drawn cells happen to contain; removes most sampling noise
  exactNN   like exact, but NN% of the cells are left untouched (cells where the knockdown failed);
            the other cells carry a stronger change so the group total is still the same

Usage: python scripts/make_submission.py [EFFECTS=effects_v1] [SCALE=1.0] [MODE=resample]
Writes PROCESSED/submissions/{EFFECTS}_x{SCALE}[_{MODE}].h5ad, ready for `vcc prep`.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED

EFFECTS = sys.argv[1] if len(sys.argv) > 1 else "effects_v1"
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
MODE = sys.argv[3] if len(sys.argv) > 3 else "resample"
assert MODE == "resample" or MODE.startswith("exact"), MODE
NONRESPONDING = int(MODE[5:] or 0) / 100 if MODE.startswith("exact") else 0.0
CONTEXTS = ["A", "B", "C"]
N_CELLS = 400
MAX_FACTOR = 16.0      # cap on any single gene's increase, so one noisy estimate cannot explode
SEED = 0

genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].to_numpy(str)
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].to_numpy(str)
z = np.load(PROCESSED / f"{EFFECTS}.npz")
assert list(z["genes"]) == list(genes) and list(z["perts"]) == list(targets), "effects file is in a different order"
factors = np.minimum(2.0 ** (SCALE * z["lfc"].astype(np.float64)), MAX_FACTOR)    # (300, 18533)
rng = np.random.default_rng(SEED)


def apply_effect(X, f):
    """X: sparse counts (cells x genes); f: one factor per gene. Returns new sparse counts."""
    X = X.copy()
    counts = X.data.astype(np.int64)
    f_of_value = f[X.indices]                       # the factor of each stored value's gene
    down, up = f_of_value < 1, f_of_value > 1
    counts[down] = rng.binomial(counts[down], f_of_value[down])
    counts[up] += rng.poisson(counts[up] * (f_of_value[up] - 1))
    X.data = counts.astype(np.float32)
    X.eliminate_zeros()                             # counts thinned to 0 are not stored
    return X


blocks, target_col, context_col = [], [], []
for ctx in CONTEXTS:
    controls = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
    assert list(controls.var_names) == list(genes), "control genes are in a different order"
    C = sparse.csr_matrix(controls.X)
    control_mean = np.asarray(C.mean(axis=0)).ravel()          # average counts per cell, per gene
    n_off = round(NONRESPONDING * N_CELLS)
    for k, gene in enumerate(targets):
        picked = C[rng.integers(0, C.shape[0], N_CELLS)]      # 400 random control cells
        if MODE == "resample":
            blocks.append(apply_effect(picked, factors[k]))
        else:
            target_total = N_CELLS * control_mean * factors[k]           # what the group total should be
            off, on = picked[:n_off], picked[n_off:]
            off_total = np.asarray(off.sum(axis=0)).ravel()
            on_total = np.asarray(on.sum(axis=0)).ravel()
            # factor for the responding cells that lands the group total on target
            f_on = np.where(on_total > 0, (target_total - off_total) / np.maximum(on_total, 1), factors[k])
            f_on = np.clip(f_on, 0.0, MAX_FACTOR)
            blocks.append(sparse.vstack([off, apply_effect(on, f_on)], format="csr"))
        target_col += [gene] * N_CELLS
        context_col += [ctx] * N_CELLS
    print(f"context {ctx} done", flush=True)

X = sparse.vstack(blocks, format="csr")
obs = pd.DataFrame({"target_gene": target_col, "context": context_col},
                   index=[f"cell{i}" for i in range(X.shape[0])])
sub = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes))

suffix = "" if MODE == "resample" else f"_{MODE}"
out = PROCESSED / "submissions" / f"{EFFECTS}_x{SCALE:g}{suffix}.h5ad"
out.parent.mkdir(parents=True, exist_ok=True)
sub.write_h5ad(out)
print(f"wrote {out}: {sub.shape[0]} cells x {sub.shape[1]} genes, "
      f"{X.nnz / sub.shape[0]:.0f} stored values per cell (cap ~13,200), "
      f"largest cell {int(X.sum(1).max())} counts (cap 1,000,000)")
