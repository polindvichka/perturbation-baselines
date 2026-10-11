"""Basal-profile baseline: sample NEW cells from each context's average control expression.
No perturbation effect is modelled, and no control cell is copied."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

CONTROLS = "/root/vcc/controls"
OUT = "/workspace/processed/basal_baseline.h5ad"

genes = pd.read_csv(f"{CONTROLS}/gene_names.csv")["gene_name"].astype(str).to_numpy()
perts = pd.read_csv(f"{CONTROLS}/pert_counts.csv")["target_gene"].astype(str).to_numpy()
rng = np.random.default_rng(0)

blocks = []      # sparse count blocks, 400 cells each
label_parts = []

for ctx in ["A", "B", "C"]:
    a = ad.read_h5ad(f"{CONTROLS}/context_{ctx}.h5ad")
    assert list(a.var_names) == list(genes), "gene order differs from gene_names.csv!"
    X = sp.csr_matrix(a.X)

    # Two summaries learned from the controls (not the cells themselves)
    gene_totals = np.asarray(X.sum(0)).ravel().astype(np.float64)
    share = gene_totals / gene_totals.sum()                 # each gene's share of all counts
    library_sizes = np.asarray(X.sum(1)).ravel().astype(np.int64)

    for pert in perts:
        sizes = rng.choice(library_sizes, size=400)          # a total count for each new cell
        counts = rng.multinomial(sizes, share)               # shape (400, 18533), whole numbers
        blocks.append(sp.csr_matrix(counts.astype(np.float32)))

    label_parts.append(pd.DataFrame({"target_gene": np.repeat(perts, 400), "context": ctx}))
    print("context", ctx, "done", flush=True)

X_all = sp.vstack(blocks, format="csr")
obs = pd.concat(label_parts, ignore_index=True)
obs.index = [f"cell_{i:06d}" for i in range(len(obs))]
out = ad.AnnData(X=X_all, obs=obs, var=pd.DataFrame(index=genes))

print("cells x genes:", out.shape)
print("stored values per cell:", round(X_all.nnz / X_all.shape[0]), "(cap is about 13,200)")
print("perturbations per context:", obs.groupby("context")["target_gene"].nunique().to_dict())
out.write_h5ad(OUT)
print("wrote", OUT)