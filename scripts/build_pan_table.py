"""Knockdown-effect table from the H1 hESC CRISPRi Perturb-seq screens of Pan et al. (Weissman / Zhuang labs,
GEO GSE295214), taken from the authors' pseudo-bulk database.

Source: pseudo_bulk_perturbation_database.h5ad (4.4 GB, MIT licence), from
https://huggingface.co/datasets/xingjiepan/SCMG_data/resolve/main/data/pseudo_bulk_perturbation_database.h5ad
Rows are (dataset, knockdown); only the hESC ones are used ('hESC_perturb_seq', 2,354 knockdowns, and
'hESC_TF_screen', 510). Per row: layer 'control' = the control profile in log1p(CP10K) (expm1 sums to 10,000),
X = the knockdown's shift in that space, layer 'measure_mask' = 1 where the gene was measured.
Log2 fold change as in the other tables: log2(CP10K knockdown + 0.1) - log2(CP10K control + 0.1).
A knockdown in both screens gets the mean of the two.

Writes PROCESSED/pan_lfc.npz like feng_lfc.npz: perts (P,), genes (18533,) on the challenge axis,
lfc (P, 18533) with NaN = not measured, n_cells (P,) set to a placeholder (the table has no cell counts).

Usage: python scripts/build_pan_table.py
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, DATA, PROCESSED

SOURCE = DATA / "pan" / "pseudo_bulk_perturbation_database.h5ad"
EPS = 0.1
genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = set(pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"])


def dense(m):
    return np.asarray(m.todense() if hasattr(m, "todense") else m, dtype=np.float64)


a = ad.read_h5ad(SOURCE, backed="r")
rows = np.flatnonzero(a.obs["condition"].astype(str).str.startswith("hESC").to_numpy())
a = a[rows].to_memory()
print(f"{a.n_obs} hESC rows:", a.obs["condition"].value_counts().to_dict(), flush=True)

ctrl = np.expm1(dense(a.layers["control"]))
pert = np.expm1(dense(a.layers["control"]) + dense(a.X))
lfc = np.log2(np.maximum(pert, 0) + EPS) - np.log2(ctrl + EPS)
lfc[dense(a.layers["measure_mask"]) == 0] = np.nan

col = pd.Index(genes).get_indexer(a.var["gene_name"].astype(str))     # source gene -> challenge column
ok = col >= 0
names = a.obs["perturbed_gene_name"].astype(str).to_numpy()
perts = sorted(set(names))
out = np.full((len(perts), len(genes)), np.nan, np.float32)
pos = {g: i for i, g in enumerate(genes)}
for i, p in enumerate(perts):
    with np.errstate(all="ignore"):
        v = np.nanmean(lfc[names == p], axis=0)
    out[i, col[ok]] = v[ok]
    if p in pos:
        if i < 3:                                       # sanity check: the knocked-down gene should be low
            print(f"  {p}: own log2 FC {out[i, pos[p]]:+.2f}", flush=True)
        out[i, pos[p]] = np.nan                         # the knocked-down gene itself
np.savez_compressed(PROCESSED / "pan_lfc.npz", perts=np.array(perts), genes=genes.to_numpy(str), lfc=out,
                    n_cells=np.full(len(perts), 100))
print(f"wrote {PROCESSED / 'pan_lfc.npz'}: {len(perts)} knockdowns ({len(set(perts) & targets)} challenge targets), "
      f"{int(ok.sum())} challenge genes measured")
