import sys
import anndata as ad
import numpy as np
import pandas as pd

path = sys.argv[1]
adata = ad.read_h5ad(path, backed="r")
targets = set(pd.read_csv("/root/vcc/controls/pert_counts.csv")["target_gene"])

print("shape (cells, genes):", adata.shape)
print("obs columns:", list(adata.obs.columns))
print("first gene names:", list(adata.var_names[:5]))

# find which column holds the perturbed gene, and how many challenge genes it has
for col in adata.obs.columns:
    values = adata.obs[col]
    if values.dtype.name in ("category", "object", "str", "string"):
        unique = set(values.astype(str).unique())
        hit = len(targets & unique)
        print(f"{col}: {len(unique)} unique, e.g. {list(unique)[:3]}, challenge genes found: {hit}/300")

# are the counts whole numbers or already normalised?
chunk = adata.X[:200]
chunk = chunk.toarray() if hasattr(chunk, "toarray") else np.asarray(chunk)
print("X dtype:", chunk.dtype, "| max:", chunk.max(), "| all whole numbers:", bool(np.all(chunk == np.round(chunk))))