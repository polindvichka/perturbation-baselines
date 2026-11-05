"""Fast mse check on a stand-in panel: raw mse of predicted group totals vs the stand-in's real cells, computed
directly from effect files with src/proxy.py (seconds, no cells and no official scorer). The official
scorer's mse only sees group totals too, so this should rank effect files the way local_score.py does.

Works on cell panels (h1, k562: truth = the stand-in's real cells) and table-only panels (rpe1, jurkat, hepg2:
truth = the line's effect table).

Usage: VCC_PANEL=h1 python scripts/mse_check.py EFFECTS:SCALE [EFFECTS:SCALE ...]
  e.g. effects_ridgepk_h1:7 effects_v1_h1:1.75     (prints raw mse and the best single scale for each)
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PANEL, PROCESSED, STANDIN
from src.proxy import best_scale, scores


def truth_from_cells():
    """Group totals of the stand-in's real cells (the scorer's pseudobulk: summed counts / summed depth)."""
    real = ad.read_h5ad(STANDIN / "real.h5ad")
    labels = real.obs["target_gene"].astype(str).to_numpy()
    X = real.X.tocsr()
    lib = np.asarray(X.sum(1)).ravel()
    mean_cp10k = lambda rows: 1e4 * np.asarray(X[rows].sum(0)).ravel() / lib[rows].sum()
    perts = [p for p in dict.fromkeys(labels) if p != "non-targeting"]
    return perts, real.var_names.astype(str), np.stack([mean_cp10k(labels == p) for p in perts]), \
        mean_cp10k(labels == "non-targeting")


def truth_from_table(file):
    """Group averages from a public line's effect table (table-only panels, build_table_panel.py)."""
    z = np.load(PROCESSED / file)
    perts = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str).tolist()
    row = {p: i for i, p in enumerate(z["perts"].astype(str))}
    genes = pd.Index(z["genes"].astype(str))
    first = ~genes.duplicated()
    control = z["basal_cp10k"] if "basal_cp10k" in z.files else z["control_cp10k"]
    return perts, genes[first], z["mean_cp10k"][[row[p] for p in perts]][:, first], control[first]


manifest = json.loads((CONTROLS / "manifest.json").read_text())
perts, genes, truth, control = truth_from_table(manifest["table"]) if "table" in manifest else truth_from_cells()
with np.errstate(divide="ignore", invalid="ignore"):   # exact in scorer space: control * 2**lfc = truth
    real_lfc = np.where(control > 0, np.log2(np.maximum(truth, 1e-12) / control), 0.0)
own = np.array([[g == p for g in genes] for p in perts])
print(f"panel {PANEL}: {len(perts)} knockdowns, {len(genes)} genes", flush=True)

for arg in sys.argv[1:]:
    name, scale = arg.rsplit(":", 1)
    z = np.load(PROCESSED / f"{name}.npz")
    lfc = z["lfc"][0] if z["lfc"].ndim == 3 else z["lfc"]
    pred = pd.DataFrame(lfc, index=z["perts"].astype(str), columns=z["genes"].astype(str))
    pred = pred.loc[:, ~pred.columns.duplicated()].reindex(index=perts, columns=genes).fillna(0).to_numpy()
    pred = float(scale) * pred
    raw, pds = scores(pred, real_lfc, control, own)
    s = best_scale(pred, real_lfc, control, own)
    print(f"{arg:28s} raw mse {raw:.3f}   best extra scale x{s:.2f} -> raw {scores(s * pred, real_lfc, control, own)[0]:.3f}"
          f"   pds rank {pds:.4f} (0 = perfect, 0.5 = chance)", flush=True)
