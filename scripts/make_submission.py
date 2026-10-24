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

The renderers live in src/render.py (shared with the stand-in for local scoring).

  tilt      per-cell average from EFFECTS x SCALE, group total from BULK_EFFECTS x BULK_SCALE
            (see src/render.py)

Usage: python scripts/make_submission.py [EFFECTS=effects_v1] [SCALE=1.0] [MODE=resample] [BULK_EFFECTS BULK_SCALE]
Writes PROCESSED/submissions/{EFFECTS}_x{SCALE}[_{MODE}].h5ad, ready for `vcc prep`.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTEXTS, CONTROLS, PROCESSED
from src.render import Renderer

EFFECTS = sys.argv[1] if len(sys.argv) > 1 else "effects_v1"
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
MODE = sys.argv[3] if len(sys.argv) > 3 else "resample"
BULK = (sys.argv[4], float(sys.argv[5])) if MODE == "tilt" else None
N_CELLS = 400
SEED = 0

genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].to_numpy(str)
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].to_numpy(str)
z = np.load(PROCESSED / f"{EFFECTS}.npz")
assert list(z["genes"]) == list(genes) and list(z["perts"]) == list(targets), "effects file is in a different order"
lfc = z["lfc"].astype(np.float64)        # (300, 18533), or (3, 300, 18533) with one table per context
factors = 2.0 ** (SCALE * lfc)
if BULK:
    zb = np.load(PROCESSED / f"{BULK[0]}.npz")
    assert list(zb["genes"]) == list(genes) and list(zb["perts"]) == list(targets), "bulk effects in a different order"
    bulk_factors = 2.0 ** (BULK[1] * zb["lfc"].astype(np.float64))
rng = np.random.default_rng(SEED)

blocks, target_col, context_col = [], [], []
for ctx in CONTEXTS:
    controls = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
    assert list(controls.var_names) == list(genes), "control genes are in a different order"
    renderer = Renderer(controls.X, MODE, rng, n_cells=N_CELLS)
    pick = lambda F: F[CONTEXTS.index(ctx)] if F.ndim == 3 else F
    ctx_factors = pick(factors)
    ctx_bulk = pick(bulk_factors) if BULK else None
    for k, gene in enumerate(targets):
        blocks.append(renderer.group(ctx_factors[k], None if ctx_bulk is None else ctx_bulk[k]))
        target_col += [gene] * N_CELLS
        context_col += [ctx] * N_CELLS
    print(f"context {ctx} done", flush=True)

X = sparse.vstack(blocks, format="csr")
obs = pd.DataFrame({"target_gene": target_col, "context": context_col},
                   index=[f"cell{i}" for i in range(X.shape[0])])
sub = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes))

suffix = "" if MODE == "resample" else f"_{MODE}" + (f"_{BULK[0]}_x{BULK[1]:g}" if BULK else "")
out = PROCESSED / "submissions" / f"{EFFECTS}_x{SCALE:g}{suffix}.h5ad"
out.parent.mkdir(parents=True, exist_ok=True)
sub.write_h5ad(out)
print(f"wrote {out}: {sub.shape[0]} cells x {sub.shape[1]} genes, "
      f"{X.nnz / sub.shape[0]:.0f} stored values per cell (cap ~13,200), "
      f"largest cell {int(X.sum(1).max())} counts (cap 1,000,000)")
