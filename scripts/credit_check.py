"""How much of the mse noise credit a recipe would keep on the real contexts (D/E/F), without their real cells.

The scorer's mse subtracts a credit for our cells' sampling noise, but scales it down when our predictions differ
too little between knockdowns (metrics/delta.py, issue #348: r = min(1, budget / claim), budget = spread of our
pseudobulks across knockdowns, claim = our noise, capped at the real side's). Both mostly depend on our own cells,
so scoring a render against a second render of the same recipe (another seed) shows r. Other teams aim for
r just under 1 ("amplitude at 0.97x the MSE sampling claim"); on HEK293T r was 0.30 / 0.56 / 0.89 at x5 / x7 / x9.
Checked 2026-10-26: HEK293T x5 gives 0.28 here against 0.30 with its real cells.
Caveat: the real side here has as many cells as ours; with fewer real cells the cap binds less, r about the same.

Usage: VCC_PANEL=test python scripts/credit_check.py EFFECTS MODE CONTEXT SCALE [SCALE ...]
  e.g. VCC_PANEL=test python scripts/credit_check.py effects_ridgepk_test exact85dsteer6000n0r D 7 9 11
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import CONTEXTS, CONTROLS, PROCESSED
from src.render import Renderer
from local_score import CELL_EVAL_NOTES, COMMON, sh

EFFECTS, MODE, CTX, SCALES = sys.argv[1], sys.argv[2], sys.argv[3], [float(s) for s in sys.argv[4:]]
N_CELLS, N_CONTROLS = 400, 5000
# without the _norm metric: scored against itself the distance is about 0 and the scorer refuses to divide by it
METRICS = "metrics=[expr_mse_unbiased_capped]"
WORK = PROCESSED / "credit_work"
WORK.mkdir(parents=True, exist_ok=True)

genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].to_numpy(str)
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].to_numpy(str)
z = np.load(PROCESSED / f"{EFFECTS}.npz")
lfc = z["lfc"][CONTEXTS.index(CTX)] if z["lfc"].ndim == 3 else z["lfc"]
controls = ad.read_h5ad(CONTROLS / f"context_{CTX}.h5ad")
C = sparse.csr_matrix(controls.X)
ctrl = C[np.random.default_rng(99).permutation(C.shape[0])[:N_CONTROLS]]


def render(scale, seed):
    """One context's groups plus the same control cells, as the scorer reads them."""
    renderer = Renderer(C, MODE, np.random.default_rng(seed), n_cells=N_CELLS)
    factors = 2.0 ** (scale * lfc.astype(np.float64))
    X = sparse.vstack([renderer.group(factors[k]) for k in range(len(targets))] + [ctrl], format="csr")
    labels = [g for g in targets for _ in range(N_CELLS)] + ["non-targeting"] * N_CONTROLS
    obs = pd.DataFrame({"target_gene": labels}, index=[f"cell{i}" for i in range(X.shape[0])])
    return ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes))


for scale in SCALES:
    label = f"{EFFECTS}_x{scale:g}_{MODE}_{CTX}"
    pred, real = WORK / f"pred_{label}.h5ad", WORK / f"real_{label}.h5ad"
    render(scale, 1).write_h5ad(pred)
    render(scale, 2).write_h5ad(real)
    notes = sh([*CELL_EVAL_NOTES, "run", "-ap", pred, "-ar", real, "-o", WORK / f"run_{label}", *COMMON,
                "--set", METRICS])
    pred.unlink(); real.unlink()
    ratio = [line.split(" to ")[1].split("x ")[0] for line in notes.splitlines() if "scaled the prediction" in line]
    print(f"{label}: credit kept {ratio[0] if ratio else '1 (not scaled)'}", flush=True)
    for line in notes.splitlines():
        if "sampling correction" in line:
            print("  scorer:", line.strip(), flush=True)
