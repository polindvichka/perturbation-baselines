"""Which public cell line do the challenge contexts A, B and C look like?

For each dataset we average log1p(counts per 10,000) over its control cells,
which gives one number per gene. Then we correlate those profiles.

Usage: python scripts/context_fingerprint.py
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

CONTROL_DIR = Path("/root/vcc/controls")
DATA = Path("/workspace/data")
OUT_CSV = "/workspace/processed/control_profiles.csv"
CHUNK = 50_000

challenge_genes = pd.read_csv(CONTROL_DIR / "gene_names.csv")["gene_name"].tolist()
challenge_set = set(challenge_genes)

# (name, file, obs column with the perturbation label, label of the control cells)
PUBLIC = [
    ("K562", DATA / "replogle_2022_k562_gwps.h5ad", "gene", "non-targeting"),
    ("RPE1", DATA / "replogle_2022_rpe1.h5ad", "gene", "non-targeting"),
    ("HepG2", DATA / "NadigOConner2024_hepg2.h5ad", "perturbation", "control"),
    ("Jurkat", DATA / "NadigOConner2024_jurkat.h5ad", "perturbation", "control"),
]


def pick_gene_names(adata):
    """Use whichever naming (index or a var column) matches the challenge gene symbols best."""
    options = {"index": pd.Series(adata.var_names.astype(str))}
    for col in adata.var.columns:
        options[col] = adata.var[col].astype(str).reset_index(drop=True)
    best = max(options, key=lambda k: len(challenge_set & set(options[k])))
    names = options[best].tolist()
    print(f"   gene names taken from '{best}': {len(challenge_set & set(names))} match the challenge genes")
    return names


def log_cp10k_sum(X):
    """For each gene: sum over cells of log1p(counts per 10,000). Also returns the number of cells."""
    X = sparse.csr_matrix(X, dtype=np.float64)
    totals = np.asarray(X.sum(axis=1)).ravel()
    X = sparse.csr_matrix(sparse.diags(1e4 / np.maximum(totals, 1)) @ X)
    X.data = np.log1p(X.data)
    return np.asarray(X.sum(axis=0)).ravel(), X.shape[0]


def control_profile(name, path, column, control):
    print(f"{name}: reading {path.name}", flush=True)
    adata = ad.read_h5ad(path, backed="r")
    labels = adata.obs[column].astype(str).to_numpy()
    is_control = labels == control
    print(f"   {int(is_control.sum())} control cells out of {adata.n_obs}")
    if is_control.sum() == 0:
        print(f"   WARNING: no cells labelled '{control}' in column '{column}', skipping")
        return None
    names = pick_gene_names(adata)
    total, n_cells = np.zeros(adata.n_vars), 0
    for start in range(0, adata.n_obs, CHUNK):
        stop = min(start + CHUNK, adata.n_obs)
        keep = is_control[start:stop]
        if not keep.any():
            continue
        chunk = adata.X[start:stop]
        s, k = log_cp10k_sum(chunk[keep])
        total += s
        n_cells += k
    return pd.Series(total / n_cells, index=names).groupby(level=0).mean()


def context_profile(label):
    print(f"context {label}: reading controls", flush=True)
    adata = ad.read_h5ad(CONTROL_DIR / f"context_{label}.h5ad")
    s, n = log_cp10k_sum(adata.X)
    return pd.Series(s / n, index=adata.var_names.astype(str)).groupby(level=0).mean()


profiles = {f"ctx_{label}": context_profile(label) for label in "ABC"}
for name, path, column, control in PUBLIC:
    p = control_profile(name, path, column, control)
    if p is not None:
        profiles[name] = p

table = pd.DataFrame(profiles).dropna()                     # genes present in every dataset
table = table[table.min(axis=1) > 0.05]                     # expressed in every dataset
print(f"\ngenes used for the comparison: {len(table)}")
table.to_csv(OUT_CSV)

ctx_cols = [c for c in table.columns if c.startswith("ctx_")]
pub_cols = [c for c in table.columns if not c.startswith("ctx_")]


def corr_table(rows, cols, centred=False):
    a, b = table[rows], table[cols]
    if centred:                                             # remove each group's average profile
        a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    return pd.DataFrame({c: [spearmanr(a[r], b[c])[0] for r in rows] for c in cols}, index=rows).round(3)


print("\n(1) context vs public line, Spearman correlation of average expression")
print(corr_table(ctx_cols, pub_cols))
print("\n(2) same, after removing each group's average profile (looks only at differences)")
print(corr_table(ctx_cols, pub_cols, centred=True))
print("\n(3) context vs context")
print(corr_table(ctx_cols, ctx_cols))
print("\n(4) public line vs public line")
print(corr_table(pub_cols, pub_cols))
print("\nprofiles saved to", OUT_CSV)