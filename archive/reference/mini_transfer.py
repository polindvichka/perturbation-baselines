"""
mini_transfer.py -- the smallest "virtual cell" experiment.

Question: if I know what knocking down gene X does in cell line A (K562),
can I predict what it does in cell line B (RPE1)?

This mirrors the 2026 Virtual Cell Challenge idea (transfer across contexts),
at toy scale. Steps:
  1. Turn each dataset into a table: one row per knocked-down gene,
     one column per gene = "perturbation effect" (mean expression after
     knockdown minus mean of control cells, on log scale).
  2. Hold out 20% of knocked-down genes (never used for training).
  3. Predict their effect in line B with 4 models, score with 2 metrics.

Run:
  python mini_transfer.py --synthetic                 # 10 s smoke test
  python mini_transfer.py --download                  # real data via pertpy (big download)
  python mini_transfer.py --a K562.h5ad --b RPE1.h5ad # your own files
"""
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


# ---------- 1. data -> perturbation-effect table ----------

def effect_table(adata, pert_col="perturbation", ctrl="control",
                 max_cells=100, seed=0):
    """Return DataFrame (perturbation x gene) of mean log-expression minus control mean."""
    rng = np.random.default_rng(seed)
    X = adata.X
    # crude check: raw counts are non-negative integers with large max
    sample = X[:200].toarray() if sp.issparse(X) else np.asarray(X[:200])
    looks_like_counts = sample.max() > 30 and np.allclose(sample, np.round(sample))

    labels = adata.obs[pert_col].astype(str).values
    if ctrl not in set(labels):
        raise ValueError(f"control label '{ctrl}' not found. Labels look like: "
                         f"{sorted(set(labels))[:10]}. Pass --ctrl / --pert-col.")

    def prep(idx):
        M = X[idx]
        M = M.toarray() if sp.issparse(M) else np.asarray(M)
        M = M.astype(np.float32)
        if looks_like_counts:
            M = np.log1p(M / np.maximum(M.sum(1, keepdims=True), 1) * 1e4)
        return M.mean(0)

    def pick(idx):
        return rng.choice(idx, size=min(max_cells, len(idx)), replace=False)

    base = prep(pick(np.where(labels == ctrl)[0]))
    rows, names = [], []
    for p in sorted(set(labels) - {ctrl}):
        idx = np.where(labels == p)[0]
        if len(idx) < 10:        # too few cells -> noisy
            continue
        rows.append(prep(pick(idx)) - base)
        names.append(p)
    return pd.DataFrame(np.vstack(rows), index=names, columns=adata.var_names)


def synthetic_pair(n_genes=600, n_perts=200, seed=0):
    """Two fake 'cell lines' that share most (not all) perturbation biology."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n_perts, 8)) @ rng.normal(size=(8, n_genes))   # shared programs
    genes = [f"G{i}" for i in range(n_genes)]
    perts = [f"G{i}" for i in range(n_perts)]
    A = shared + 0.5 * rng.normal(size=shared.shape)
    B = 0.8 * shared + 0.5 * rng.normal(size=shared.shape) \
        + 0.5 * rng.normal(size=(n_perts, 3)) @ rng.normal(size=(3, n_genes))  # B-specific part
    return (pd.DataFrame(A, index=perts, columns=genes),
            pd.DataFrame(B, index=perts, columns=genes))


# ---------- 2. models ----------

def fit_predict(name, A_tr, B_tr, A_te):
    """Each model predicts line-B effects for the held-out perturbations."""
    n = len(A_te)
    if name == "no_change":                       # predict 'nothing happens'
        return np.zeros((n, B_tr.shape[1]))
    if name == "mean_effect":                     # same average B effect for every gene
        return np.tile(B_tr.mean(axis=0).values, (n, 1))
    if name == "copy_from_A":                     # B effect = A effect for the same gene
        return A_te.values
    if name == "ridge_A_to_B":                    # learn how A effects map onto B effects
        pca_a = PCA(n_components=min(30, len(A_tr) - 1)).fit(A_tr.values)
        pca_b = PCA(n_components=min(30, len(B_tr) - 1)).fit(B_tr.values)
        reg = Ridge(alpha=10.0).fit(pca_a.transform(A_tr.values),
                                    pca_b.transform(B_tr.values))
        return pca_b.inverse_transform(reg.predict(pca_a.transform(A_te.values)))
    raise ValueError(name)


# ---------- 3. metrics ----------

def score(pred, truth):
    """Return (mean per-perturbation Pearson of effects, mean squared error)."""
    pr = [np.corrcoef(p, t)[0, 1] if p.std() > 0 else 0.0
          for p, t in zip(pred, truth)]
    return float(np.nanmean(pr)), float(((pred - truth) ** 2).mean())


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--download", action="store_true",
                    help="fetch Replogle K562 essential + RPE1 via pertpy")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--ctrl", default="control")
    ap.add_argument("--n-genes", type=int, default=2000, help="top-variance genes to keep")
    ap.add_argument("--max-cells", type=int, default=100)
    args = ap.parse_args()

    if args.synthetic:
        A, B = synthetic_pair()
    else:
        if args.download:
            import pertpy as pt
            adA = pt.data.replogle_2022_k562_essential()
            adB = pt.data.replogle_2022_rpe1()
        else:
            adA, adB = ad.read_h5ad(args.a), ad.read_h5ad(args.b)
        print("obs columns (A):", list(adA.obs.columns))
        A = effect_table(adA, args.pert_col, args.ctrl, args.max_cells)
        B = effect_table(adB, args.pert_col, args.ctrl, args.max_cells)

    # keep only genes + perturbations present in BOTH lines
    genes = A.columns.intersection(B.columns)
    perts = A.index.intersection(B.index)
    A, B = A.loc[perts, genes], B.loc[perts, genes]
    top = B.var(axis=0).sort_values(ascending=False).index[:args.n_genes]   # most-changing genes
    A, B = A[top], B[top]
    print(f"{len(perts)} shared perturbations x {A.shape[1]} genes")

    # hold out 20% of perturbations
    rng = np.random.default_rng(0)
    test = rng.choice(len(perts), size=max(5, len(perts) // 5), replace=False)
    mask = np.zeros(len(perts), bool); mask[test] = True
    A_tr, B_tr, A_te, B_te = A[~mask], B[~mask], A[mask], B[mask]

    print(f"\n{'model':<14}{'Pearson(effect)':>18}{'MSE':>10}")
    for m in ["no_change", "mean_effect", "copy_from_A", "ridge_A_to_B"]:
        r, mse = score(fit_predict(m, A_tr, B_tr, A_te), B_te.values)
        print(f"{m:<14}{r:>18.3f}{mse:>10.4f}")


if __name__ == "__main__":
    main()
