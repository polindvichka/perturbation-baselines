"""Knockdown-effect table from the primary human CD4+ T cell CRISPRi screen (Zhu ... Marson, Cell 2026).

Source: GWCD4i.DE_stats.h5ad from https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/
(MIT licence per the CZI dataset page). It holds DESeq2 log2 fold changes per (knocked-down gene x culture
condition: Rest, Stim8hr, Stim48hr), measured with 10x Flex like the challenge data.

Keeps a knockdown/condition only if the knockdown itself was significant and it has no off-target or
low-expression flag, then averages the passing conditions. Writes PROCESSED/cd4_lfc.npz with
    perts (P,)   genes (18533,) on the challenge axis   lfc (P, 18533) log2 FC, NaN = not measured
    n_cells (P,) summed over the kept conditions   n_conditions (P,)
Only the challenge targets are kept (pass ALL=1 as an argument to keep every knockdown).

Usage: python scripts/build_cd4_table.py [ALL]
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, DATA, PROCESSED, all_targets

KEEP_ALL = len(sys.argv) > 1
genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = set(all_targets())               # the knockdowns of both panels (A/B/C and D/E/F)

a = ad.read_h5ad(DATA / "cd4" / "GWCD4i.DE_stats.h5ad")
obs = a.obs
name = obs["target_contrast_gene_name"].astype(str)
good = (obs["ontarget_significant"].astype(bool) & ~obs["distal_offtarget_flag"].astype(bool)
        & ~obs["low_target_gex"].astype(bool))
print(f"{len(obs)} knockdown x condition rows, {good.sum()} pass the quality filters")
rows = good & (name.isin(targets) | KEEP_ALL)
print(f"challenge targets present: {name[name.isin(targets)].nunique()} of {len(targets)}, "
      f"passing filters: {name[good & name.isin(targets)].nunique()}")

lfc = pd.DataFrame(np.asarray(a.layers["log_fc"][rows.to_numpy()]), columns=a.var["gene_name"].astype(str))
lfc = lfc.T.groupby(level=0).mean().T                          # duplicate symbols -> one column
lfc.index = name[rows].to_numpy()
mean = lfc.groupby(level=0).mean()                              # average the kept conditions
n_cells = obs.loc[rows, "n_cells_target"].groupby(name[rows]).sum().reindex(mean.index)
n_cond = name[rows].value_counts().reindex(mean.index)
mean = mean.reindex(columns=genes)                              # challenge gene axis, NaN = not measured
for p in mean.index:                                            # mask each knockdown's own gene
    if p in mean.columns:
        mean.loc[p, p] = np.nan

np.savez_compressed(PROCESSED / "cd4_lfc.npz", perts=mean.index.to_numpy(str), genes=genes.to_numpy(str),
                    lfc=mean.to_numpy(np.float32), n_cells=n_cells.to_numpy(int), n_conditions=n_cond.to_numpy(int))
print(f"wrote {PROCESSED / 'cd4_lfc.npz'}: {mean.shape[0]} knockdowns, "
      f"{int(mean.notna().any().sum())} challenge genes measured; conditions per knockdown: "
      f"{n_cond.value_counts().sort_index().to_dict()}")
