"""Turn predicted effects (scripts/make_effects.py) into a submission: 400 cells per gene per context.

For every context (A, B, C) and every target gene:
  1. draw 400 real control cells of that context (with replacement),
  2. change each gene's counts by the predicted fold change, f = 2 ** (SCALE * lfc):
       f < 1: keep each count with probability f       (binomial thinning: 10 counts, f=0.3 -> about 3)
       f > 1: add Poisson(count * (f - 1)) extra counts (10 counts, f=1.5 -> about 15)
     so the result stays whole numbers, keeps real cell-to-cell variation, and zeros stay zero
     (which keeps the matrix sparse).

Usage: python scripts/make_submission.py [EFFECTS=effects_v1] [SCALE=1.0]
Writes PROCESSED/submissions/{EFFECTS}_x{SCALE}.h5ad, ready for `vcc prep`.
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
    for k, gene in enumerate(targets):
        picked = C[rng.integers(0, C.shape[0], N_CELLS)]      # 400 random control cells
        blocks.append(apply_effect(picked, factors[k]))
        target_col += [gene] * N_CELLS
        context_col += [ctx] * N_CELLS
    print(f"context {ctx} done", flush=True)

X = sparse.vstack(blocks, format="csr")
obs = pd.DataFrame({"target_gene": target_col, "context": context_col},
                   index=[f"cell{i}" for i in range(X.shape[0])])
sub = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes))

out = PROCESSED / "submissions" / f"{EFFECTS}_x{SCALE:g}.h5ad"
out.parent.mkdir(parents=True, exist_ok=True)
sub.write_h5ad(out)
print(f"wrote {out}: {sub.shape[0]} cells x {sub.shape[1]} genes, "
      f"{X.nnz / sub.shape[0]:.0f} stored values per cell (cap ~13,200), "
      f"largest cell {int(X.sum(1).max())} counts (cap 1,000,000)")
