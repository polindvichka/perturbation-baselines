"""
linear_baseline.py -- the simple linear model in the spirit of
Ahlmann-Eltze, Huber & Anders (Nature Methods 2025), single-dataset version.

Task: in ONE cell line, predict the effect of knocking down genes you have never
seen knocked down (held-out perturbations).

Model (all linear, no GPU needed):
    effect(g)  ~=  G[g] @ W @ P  +  b
  P : top PCA directions of the effect matrix (what "expression programs" move)
  G : an embedding of each gene g, from PCA of the transposed training matrix
      (genes described by how they respond to the training knockdowns)
  W : small matrix learned by ridge regression
  b : mean training effect

Run:
  python linear_baseline.py --synthetic
  python linear_baseline.py --download            # Replogle K562 essential via pertpy
  python linear_baseline.py --h5ad K562.h5ad
"""
import argparse
import re
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from mini_transfer import effect_table, score


def synthetic_single(n_genes=800, n_perts=300, rank=10, seed=0):
    """Co-regulated genes: effect of knocking down p on gene j ~ Z[p].Z[j]."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_genes, rank))
    genes = [f"G{i}" for i in range(n_genes)]
    Y = Z[:n_perts] @ Z.T + 0.7 * rng.normal(size=(n_perts, n_genes))
    return pd.DataFrame(Y, index=genes[:n_perts], columns=genes)


def linear_model(Y_tr, test_genes, k=10, alpha=1.0):
    """Fit on Y_tr (train perts x genes); predict effects for test_genes."""
    b = Y_tr.mean(axis=0).values
    Yc = Y_tr.values - b
    P = PCA(n_components=k).fit(Yc).components_                  # k x genes
    # gene embedding: genes as rows, training knockdowns as features
    emb = PCA(n_components=k).fit_transform(Yc.T)                # genes x k
    emb = pd.DataFrame(emb, index=Y_tr.columns)
    W = Ridge(alpha=alpha, fit_intercept=False).fit(
        emb.loc[Y_tr.index].values, Yc @ P.T)
    return emb.loc[test_genes].values @ W.coef_.T @ P + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--dataset", default="adamson",
                    choices=["adamson", "k562_essential"],
                    help="adamson = small (K562, ~80 genes); k562_essential = big")
    ap.add_argument("--h5ad")
    ap.add_argument("--pert-col", default="perturbation")
    ap.add_argument("--ctrl", default="control")
    ap.add_argument("--max-cells", type=int, default=100)
    ap.add_argument("--k", type=int, default=10, help="PCA dimensions")
    args = ap.parse_args()

    if args.synthetic:
        Y = synthetic_single()
    else:
        if args.download:
            import pertpy as pt
            a = (pt.data.adamson_2016_upr_perturb_seq() if args.dataset == "adamson"
                 else pt.data.replogle_2022_k562_essential())
        else:
            a = ad.read_h5ad(args.h5ad)
        print("obs columns:", list(a.obs.columns))
        ctrl = args.ctrl
        if args.ctrl == "control" and args.dataset == "adamson":
            ctrl = "*"   # Adamson marks non-targeting-guide cells '*', not 'control'
        Y = effect_table(a, args.pert_col, ctrl, args.max_cells)
        if args.dataset == "adamson":
            # Adamson perturbation labels are "GENE_guideID" (e.g. ATF4_pBA576);
            # several guides target the same gene, so collapse them by averaging.
            Y.index = [re.sub(r"_p[A-Za-z0-9]+$", "", i) for i in Y.index]
            Y = Y.groupby(level=0).mean()

    Y = Y.loc[Y.index.intersection(Y.columns)]   # perturbed gene must also be a measured gene
    print(f"{Y.shape[0]} perturbations x {Y.shape[1]} genes")

    rng = np.random.default_rng(0)
    te = rng.choice(len(Y), size=max(5, len(Y) // 5), replace=False)
    mask = np.zeros(len(Y), bool); mask[te] = True
    Y_tr, Y_te = Y[~mask], Y[mask]

    preds = {
        "no_change":   np.zeros(Y_te.shape),
        "mean_effect": np.tile(Y_tr.mean(axis=0).values, (len(Y_te), 1)),
        "linear":      linear_model(Y_tr, Y_te.index, k=args.k),
    }
    print(f"\n{'model':<14}{'Pearson(effect)':>18}{'MSE':>10}")
    for name, p in preds.items():
        r, mse = score(p, Y_te.values)
        print(f"{name:<14}{r:>18.3f}{mse:>10.4f}")


if __name__ == "__main__":
    main()
