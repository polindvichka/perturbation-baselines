"""Knockdown-effect table from the HipSci iPSC CRISPRi map (Feng et al., Cell Genomics 2026).

Source: GenomeWideScreen_LFC_byGene.tsv.gz (MIT licence), from
https://huggingface.co/datasets/Boom5426/iPSC-CRISPRi-PerturbSeq/resolve/main/feng2026_ipsc_crispri_map/lfc/
Columns: Target, Expressed_Gene_Symbol, Expressed_Gene_Ens_ID, lfc, pval_adj. One log2 fold change per
(knocked-down gene, measured gene), pooled over the iPSC lines of 26 donors.

Writes PROCESSED/feng_lfc.npz like cd4_lfc.npz: perts (P,), genes (18533,) on the challenge axis,
lfc (P, 18533) with NaN = not measured, n_cells (P,) set to a placeholder (the table has no cell counts).
Only the challenge targets are kept (pass ALL=1 as an argument to keep every knockdown).

Usage: python scripts/build_feng_table.py [ALL]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, DATA, PROCESSED

KEEP_ALL = len(sys.argv) > 1
SOURCE = DATA / "feng" / "GenomeWideScreen_LFC_byGene.tsv.gz"
genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = set(pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"])
col_of = {g: i for i, g in enumerate(genes)}

rows = {}                                        # target -> {challenge column: lfc}
for chunk in pd.read_csv(SOURCE, sep="\t", usecols=["Target", "Expressed_Gene_Symbol", "lfc"], chunksize=5_000_000):
    if not KEEP_ALL:
        chunk = chunk[chunk["Target"].isin(targets)]
    chunk = chunk[chunk["Expressed_Gene_Symbol"].isin(col_of)]
    for t, part in chunk.groupby("Target"):
        d = rows.setdefault(t, {})
        d.update(zip(part["Expressed_Gene_Symbol"].map(col_of), part["lfc"]))
    print(f"  read a chunk, {len(rows)} knockdowns so far", flush=True)

perts = sorted(rows)
lfc = np.full((len(perts), len(genes)), np.nan, np.float32)
for i, t in enumerate(perts):
    cols = np.fromiter(rows[t].keys(), int)
    lfc[i, cols] = np.fromiter(rows[t].values(), float)
    if t in col_of:
        lfc[i, col_of[t]] = np.nan               # the knocked-down gene itself
np.savez_compressed(PROCESSED / "feng_lfc.npz", perts=np.array(perts), genes=genes.to_numpy(str), lfc=lfc,
                    n_cells=np.full(len(perts), 100))
print(f"wrote {PROCESSED / 'feng_lfc.npz'}: {len(perts)} knockdowns ({len(set(perts) & targets)} challenge targets), "
      f"median {int(np.median((~np.isnan(lfc)).sum(1)))} measured genes per knockdown")
