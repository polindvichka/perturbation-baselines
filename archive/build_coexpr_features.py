"""Co-expression features: for every gene, how strongly does it move together with a fixed set of
anchor genes across control cells?

Why: when a gene is knocked down, the genes that change are mostly the ones that normally rise and
fall together with it. Control cells (non-targeting guide) show that pattern without any knockdown.

For each source of control cells (K562, context A, B, C) this writes
PROCESSED/coexpr_{NAME}.npz with
    genes   gene symbols measured in that source
    anchors the anchor gene symbols (the SAME list for every source)
    feat    (len(genes) x len(anchors)) Pearson correlation of log1p(CP10K) across control cells;
            a gene's correlation with itself is set to 0

Anchors = the 2000 most variable genes that are expressed in all four sources, so a feature column
means the same thing in every source.

Usage: python archive/build_coexpr_features.py
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, DATA, PROCESSED

OUT = PROCESSED
CHUNK = 50_000
N_ANCHORS = 2000
MIN_MEAN = 0.1          # anchor genes must have mean log1p(CP10K) above this in every source

K562 = (DATA / "replogle_2022_k562_gwps.h5ad", "gene", "non-targeting")


def log_cp10k(X):
    X = sparse.csr_matrix(X, dtype=np.float32)
    totals = np.asarray(X.sum(axis=1)).ravel()
    X = sparse.csr_matrix(sparse.diags((1e4 / np.maximum(totals, 1)).astype(np.float32)) @ X)
    X.data = np.log1p(X.data)
    return X


def gene_names(adata, challenge_set):
    options = {"index": pd.Series(adata.var_names.astype(str))}
    for col in adata.var.columns:
        options[col] = adata.var[col].astype(str).reset_index(drop=True)
    best = max(options, key=lambda k: len(challenge_set & set(options[k])))
    print(f"   gene names taken from '{best}'")
    return options[best].tolist()


def load_controls(label, challenge_set):
    """Returns (dense cells x genes log1p(CP10K) float32, gene names) with duplicate gene names dropped."""
    if label == "K562":
        path, column, control = K562
        print(f"K562: reading controls from {path.name}", flush=True)
        adata = ad.read_h5ad(path, backed="r")
        is_control = adata.obs[column].astype(str).to_numpy() == control
        print(f"   {int(is_control.sum())} control cells out of {adata.n_obs}")
        parts = []
        for start in range(0, adata.n_obs, CHUNK):
            stop = min(start + CHUNK, adata.n_obs)
            keep = is_control[start:stop]
            if keep.any():
                parts.append(log_cp10k(adata.X[start:stop][keep]))
        X = sparse.vstack(parts).tocsr()
    else:
        print(f"context {label}: reading controls", flush=True)
        adata = ad.read_h5ad(CONTROLS / f"context_{label}.h5ad")
        X = log_cp10k(adata.X)
    names = gene_names(adata, challenge_set)
    first = ~pd.Series(names).duplicated().to_numpy()
    X = X[:, np.flatnonzero(first)]
    names = [n for n, f in zip(names, first) if f]
    print(f"   {X.shape[0]} cells x {X.shape[1]} genes")
    return X.toarray().astype(np.float32), names


def main():
    challenge_set = set(pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"])
    sources = {}
    for label in ("K562", "A", "B", "C"):
        sources[label] = load_controls(label, challenge_set)

    # anchors: expressed in every source, ranked by mean variance across sources
    stats = {}
    for label, (X, names) in sources.items():
        stats[label] = pd.DataFrame({"mean": X.mean(axis=0), "var": X.var(axis=0)}, index=names)
    common = set.intersection(*[set(s.index) for s in stats.values()])
    ok = [g for g in common if all(stats[l].at[g, "mean"] > MIN_MEAN for l in stats)]
    score = pd.Series({g: np.mean([stats[l].at[g, "var"] for l in stats]) for g in ok})
    anchors = score.sort_values(ascending=False).index[:N_ANCHORS].tolist()
    print(f"\n{len(common)} genes in all sources, {len(ok)} expressed, {len(anchors)} anchors chosen")

    for label, (X, names) in sources.items():
        pos = {g: i for i, g in enumerate(names)}
        anchor_cols = np.array([pos[a] for a in anchors])
        Z = X - X.mean(axis=0)
        sd = np.sqrt((Z ** 2).mean(axis=0))
        Z = Z / np.where(sd > 1e-6, sd, 1)                 # constant genes become all zeros
        feat = (Z.T @ Z[:, anchor_cols]) / Z.shape[0]      # genes x anchors
        for a_idx, col in enumerate(anchor_cols):          # a gene vs itself -> 0
            feat[col, a_idx] = 0
        path = OUT / f"coexpr_{label}.npz"
        np.savez(path, genes=np.array(names, dtype=object), anchors=np.array(anchors, dtype=object),
                 feat=feat.astype(np.float32))
        print(f"{label}: wrote {path}  ({feat.shape[0]} genes x {feat.shape[1]} anchors)")


if __name__ == "__main__":
    main()