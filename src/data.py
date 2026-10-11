"""Load perturbation datasets and turn them into pseudobulk delta tables."""
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CONTROL_LABEL = "non-targeting"


@dataclass
class Context:
    """One cell context: its basal profile and (if known) perturbation deltas."""
    name: str
    genes: np.ndarray        # (G,) gene symbols, challenge order
    basal: np.ndarray        # (G,) mean log1p(CP10K) of control cells
    delta: pd.DataFrame      # (n_perts, G) perturbed mean - basal; empty for challenge contexts
    n_cells: pd.Series       # cells behind each delta row


def load_genes(path):
    return pd.read_csv(path)["gene_name"].to_numpy()


def harmonise_genes(adata, genes):
    """Return a CSR matrix of adata.X with columns in `genes` order.
    Genes missing from adata become zero columns; extra genes are dropped."""
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    target_idx = pd.Index(genes).get_indexer(adata.var_names)   # -1 if absent
    keep = np.where(target_idx >= 0)[0]
    # selection matrix: source column keep[i] -> target column target_idx[keep[i]]
    S = sp.csr_matrix(
        (np.ones(len(keep), dtype=np.float32), (np.arange(len(keep)), target_idx[keep])),
        shape=(len(keep), len(genes)),
    )
    return (X.tocsc()[:, keep] @ S).tocsr()


def log_cp10k(X):
    """Library-size normalise to 10k counts per cell, then log1p. Keeps sparsity."""
    X = sp.csr_matrix(X, dtype=np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1.0
    X = sp.diags(1e4 / lib) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def group_means(X, labels):
    """Mean of rows of X per label, via a sparse indicator matrix. Returns (DataFrame, counts)."""
    labels = pd.Series(labels).astype(str).to_numpy()
    cats, inv = np.unique(labels, return_inverse=True)
    counts = np.bincount(inv)
    G = sp.csr_matrix(
        (1.0 / counts[inv], (inv, np.arange(len(inv)))), shape=(len(cats), len(inv))
    )
    M = np.asarray((G @ X).todense())
    return pd.DataFrame(M, index=cats), pd.Series(counts, index=cats)


def pseudobulk_delta(adata, genes, pert_col="target_gene", control=CONTROL_LABEL, min_cells=20):
    """Per-perturbation mean log1p(CP10K) minus control mean, in `genes` order."""
    X = log_cp10k(harmonise_genes(adata, genes))
    means, counts = group_means(X, adata.obs[pert_col])
    basal = means.loc[control].to_numpy()
    means = means.drop(index=control)
    counts = counts.drop(index=control)
    ok = counts >= min_cells
    delta = means[ok] - basal
    delta.columns = genes
    return basal, delta, counts[ok]


def load_control_context(name, root="/root/vcc/controls", genes=None):
    """Challenge context: controls only, so the delta table is empty."""
    genes = load_genes(f"{root}/gene_names.csv") if genes is None else genes
    a = ad.read_h5ad(f"{root}/context_{name}.h5ad")
    X = log_cp10k(harmonise_genes(a, genes))
    basal = np.asarray(X.mean(0)).ravel()
    return Context(name, genes, basal, pd.DataFrame(columns=genes), pd.Series(dtype=int))