"""One predicted effect per challenge gene: its log2 fold change, averaged over the public lines that measured it.

For every source table and every challenge gene it knocked down (with enough cells):
    lfc[gene j] = log2(mean expression in knockdown cells + EPS) - log2(mean in control cells + EPS)
(expression = counts per 10,000, so cells sequenced deeper or shallower are comparable).
Then, per challenge gene, the mean over the sources that have it.

Writes PROCESSED/effects_v1.npz with
    perts      (300,)          the challenge genes, in pert_counts.csv order
    genes      (18533,)        the challenge gene axis, in gene_names.csv order
    lfc        (300, 18533)    float32; 0 where no source measured that gene
    n_sources  (300,)          how many sources went into each row

Usage: python scripts/make_effects.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED

EPS = 0.1          # keeps genes with almost no counts from giving huge fake fold changes
MIN_CELLS = 20     # fewer cells than this = too noisy to use

# the lines that actually knocked down some of our 300 genes (see scripts/coverage.py)
SOURCES = {
    "k562": "k562_delta.npz",
    "hct116": "xatlas_HCT116_delta.npz",
    "hek293t": "xatlas_HEK293T_delta.npz",
    "h1_train": "h1_train_delta.npz",
    "h1_val": "h1_val_delta.npz",
    "h1_test": "h1_test_delta.npz",
}

challenge_genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"]


def load_lfc(file):
    """Log2 fold changes of one source: rows = our challenge genes it knocked down, columns = 18,533 challenge genes."""
    z = np.load(PROCESSED / file)
    # 1. which rows to keep: our genes, with enough cells
    keep = np.isin(z["perts"], targets) & (z["n_cells"] >= MIN_CELLS)
    knockdown = z["mean_cp10k"][keep]                              # (rows kept, genes of this source)
    # the two kinds of table name the control average differently
    control = z["basal_cp10k"] if "basal_cp10k" in z.files else z["control_cp10k"]

    # 2. log2 fold change; every knockdown row minus the one control row ("broadcasting")
    lfc = np.log2(knockdown + EPS) - np.log2(control + EPS)
    lfc[:, control == 0] = np.nan                                  # gene never seen in controls = not measured

    # 3. label rows and columns, then put the columns in the challenge order (NaN where not measured)
    df = pd.DataFrame(lfc, index=z["perts"][keep], columns=z["genes"])
    df = df.loc[:, ~df.columns.duplicated()]                       # reindex needs unique column names
    return df.reindex(columns=challenge_genes)


# one table per source, stacked; a gene can now appear in several rows (one per source)
parts = []
for name, file in SOURCES.items():
    df = load_lfc(file)
    print(f"{name}: {len(df)} of our genes, {df.notna().any().sum()} challenge genes measured")
    parts.append(df)
stacked = pd.concat(parts)

# average the rows that belong to the same gene; .mean() skips NaN
effects = stacked.groupby(level=0).mean().reindex(targets)
n_sources = stacked.index.value_counts().reindex(targets).fillna(0).astype(int)
lfc = effects.fillna(0).to_numpy(np.float32)                       # nothing measured -> no change

np.savez_compressed(PROCESSED / "effects_v1.npz", perts=targets.to_numpy(str),
                    genes=challenge_genes.to_numpy(str), lfc=lfc, n_sources=n_sources.to_numpy())
print("\nsources per gene:", n_sources.value_counts().sort_index().to_dict())
print("shape:", lfc.shape)
print("ACLY ->", effects.loc["ACLY", ["HMGCS1", "IDI1", "LDLR", "ACLY"]].round(2).to_dict())
print("wrote", PROCESSED / "effects_v1.npz")
