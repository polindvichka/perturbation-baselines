import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import anndata as ad
from data import DeltaTable, log_cp10k

# 1. Which K562 rows are challenge genes, and how many cells did each have?
table = DeltaTable.load("/workspace/processed/k562_delta.npz")
challenge = pd.read_csv("/root/vcc/controls/pert_counts.csv")["target_gene"].to_numpy()
rows = np.where(np.isin(table.perts, challenge))[0]
cell_counts = table.n_cells[rows]

# 2. Load all the non-targeting (control) cells from K562
a = ad.read_h5ad("/workspace/data/replogle_2022_k562_gwps.h5ad", backed="r")
is_control = (a.obs["gene"].astype(str).to_numpy() == "non-targeting")
controls = a[np.where(is_control)[0]].to_memory()
X = log_cp10k(controls.X)
print("control cells loaded:", X.shape[0])

overall_mean = np.asarray(X.mean(0)).ravel()      # the "normal cell" profile

# 3. For each challenge gene, build a FAKE perturbation of the same size from controls
rng = np.random.default_rng(0)
noise_error = []
for n in cell_counts:
    pick = rng.choice(X.shape[0], size=int(n), replace=False)
    fake_mean = np.asarray(X[pick].mean(0)).ravel()
    noise_error.append(np.mean((fake_mean - overall_mean) ** 2))

# 4. Compare with the real challenge genes
real_error = np.mean(np.nanmean(table.delta[rows] ** 2, axis=1))
noise = np.mean(noise_error)
print(f"real challenge genes, squared change: {real_error:.5f}")
print(f"fake perturbations (pure noise):      {noise:.5f}")
print(f"share that looks like real signal:    {1 - noise / real_error:.1%}")