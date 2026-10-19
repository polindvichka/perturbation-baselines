"""Make a prediction for the K562 stand-in context, the same way make_submission.py does for A/B/C.

Uses the stand-in's own spare control cells (base_controls.h5ad, disjoint from the scorer's reference
controls) and an effects file built WITHOUT K562 (e.g. scripts/make_effects.py effects_noK562 hct116 ...),
so K562 plays an unseen context. Score the result with scripts/local_score.py.

Usage: python scripts/standin_submission.py EFFECTS SCALE MODE [LABEL] [BULK_EFFECTS BULK_SCALE]
  (BULK_* only for MODE tilt; see src/render.py)
Writes PROCESSED/standin/pred_{LABEL or EFFECTS_xSCALE_MODE}.h5ad
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROCESSED
from src.render import Renderer

EFFECTS, SCALE, MODE = sys.argv[1], float(sys.argv[2]), sys.argv[3]
LABEL = sys.argv[4] if len(sys.argv) > 4 else f"{EFFECTS}_x{SCALE:g}_{MODE}"
BULK = (sys.argv[5], float(sys.argv[6])) if MODE == "tilt" else None
STANDIN = PROCESSED / "standin"
N_CELLS, SEED = 400, 0

base = ad.read_h5ad(STANDIN / "base_controls.h5ad")
targets = pd.read_csv(STANDIN / "standin_perts.csv")["target_gene"].astype(str).tolist()
def standin_lfc(name):
    z = np.load(PROCESSED / f"{name}.npz")
    lfc = pd.DataFrame(z["lfc"], index=z["perts"].astype(str), columns=z["genes"].astype(str))
    # the stand-in has K562's gene list: map columns onto it (genes outside the challenge axis get no change)
    lfc = lfc.loc[:, ~lfc.columns.duplicated()]
    return lfc.reindex(index=targets, columns=base.var_names.astype(str)).fillna(0).to_numpy(np.float64)


factors = 2.0 ** (SCALE * standin_lfc(EFFECTS))
bulk_factors = 2.0 ** (BULK[1] * standin_lfc(BULK[0])) if BULK else None

rng = np.random.default_rng(SEED)
renderer = Renderer(base.X, MODE, rng, n_cells=N_CELLS)
blocks = [renderer.group(factors[k], None if bulk_factors is None else bulk_factors[k]) for k in range(len(targets))]
X = sparse.vstack(blocks, format="csr")
obs = pd.DataFrame({"target_gene": np.repeat(targets, N_CELLS)}, index=[f"p{i}" for i in range(X.shape[0])])
pred = ad.AnnData(X, obs=obs, var=base.var.copy())
out = STANDIN / f"pred_{LABEL}.h5ad"
pred.write_h5ad(out)
print(f"wrote {out}: {pred.shape}, {len(targets)} targets")
