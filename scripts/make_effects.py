"""One predicted effect per challenge gene: its log2 fold change, averaged over the public lines that measured it.

For every source table and every challenge gene it knocked down (with enough cells):
    lfc[gene j] = log2(mean expression in knockdown cells + EPS) - log2(mean in control cells + EPS)
(expression = counts per 10,000, so cells sequenced deeper or shallower are comparable).
Then, per challenge gene, the mean over the sources that have it.

Writes PROCESSED/{OUT}.npz (default effects_v1) with
    perts      (300,)          the challenge genes, in pert_counts.csv order
    genes      (18533,)        the challenge gene axis, in gene_names.csv order
    lfc        (300, 18533)    float32; 0 where no source measured that gene
    n_sources  (300,)          how many sources went into each row
    n_up, n_down (300, 18533)  how many sources saw that gene go up / down

Usage: python scripts/make_effects.py [OUT=effects_v1] [SOURCE ...]
  SOURCE  names from SOURCES below (default: all), e.g. leave out k562 to use K562 as a stand-in context
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED, SUFFIX, is_holdout

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
    "cd4": "cd4_lfc.npz",               # scripts/build_cd4_table.py; already log2 FC, 10x Flex
}

OUT = sys.argv[1] if len(sys.argv) > 1 else "effects_v1"
if len(sys.argv) > 2:
    SOURCES = {name: SOURCES[name] for name in sys.argv[2:]}
SOURCES = {name: f for name, f in SOURCES.items() if not is_holdout(name)}   # a stand-in never sees its own data
OUT += SUFFIX

challenge_genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"]


def load_lfc(file):
    """Log2 fold changes of one source: rows = our challenge genes it knocked down, columns = 18,533 challenge genes."""
    z = np.load(PROCESSED / file)
    # 1. which rows to keep: our genes, with enough cells
    keep = np.isin(z["perts"], targets) & (z["n_cells"] >= MIN_CELLS)
    if "lfc" in z.files:                                           # source that already holds log2 FCs (CD4)
        df = pd.DataFrame(z["lfc"][keep], index=z["perts"][keep], columns=z["genes"])
        return df.reindex(columns=challenge_genes)
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
# per gene and measured gene: how many sources saw it go up, and how many down (for confidence)
n_up = (stacked > 0).groupby(level=0).sum().reindex(targets).fillna(0).to_numpy(np.int8)
n_down = (stacked < 0).groupby(level=0).sum().reindex(targets).fillna(0).to_numpy(np.int8)

np.savez_compressed(PROCESSED / f"{OUT}.npz", perts=targets.to_numpy(str),
                    genes=challenge_genes.to_numpy(str), lfc=lfc, n_sources=n_sources.to_numpy(),
                    n_up=n_up, n_down=n_down)
print("\nsources per gene:", n_sources.value_counts().sort_index().to_dict())
print("shape:", lfc.shape)
print("wrote", PROCESSED / f"{OUT}.npz")
