"""Build a stand-in 'context' from K562 so we can score our predictions locally with the official scorer.

real.h5ad        K562 cells for the challenge target genes that K562 measured (up to 400 cells each),
                 plus REFERENCE control cells (non-targeting, 18,400).
base_controls.h5ad  a different, disjoint set of K562 control cells (18,400) that the prediction
                 scripts use as their starting cells. Disjoint on purpose, so a prediction cannot
                 look good by copying the reference controls.

Labels are written to obs['target_gene'] ('non-targeting' for controls). Counts stay raw integers.

Usage: python scripts/build_k562_standin.py
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

CONTROL_DIR = Path("/root/vcc/controls")
DATA = Path("/workspace/data")
OUT = Path("/workspace/processed/standin")
K562_FILE = DATA / "replogle_2022_k562_gwps.h5ad"
LABEL_COL, CONTROL_LABEL = "gene", "non-targeting"
MAX_CELLS_PER_PERT = 400
MIN_CELLS_PER_PERT = 20
N_CONTROLS = 18_400
CHUNK = 50_000
SEED = 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    targets = set(pd.read_csv(CONTROL_DIR / "pert_counts.csv")["target_gene"])
    print(f"reading {K562_FILE.name} (backed)", flush=True)
    adata = ad.read_h5ad(K562_FILE, backed="r")
    labels = adata.obs[LABEL_COL].astype(str).to_numpy()

    # which cells
    chosen = []
    kept_perts = []
    for g in sorted(targets & set(labels)):
        idx = np.flatnonzero(labels == g)
        if len(idx) < MIN_CELLS_PER_PERT:
            continue
        chosen.append(rng.choice(idx, size=min(len(idx), MAX_CELLS_PER_PERT), replace=False))
        kept_perts.append(g)
    ctrl = rng.permutation(np.flatnonzero(labels == CONTROL_LABEL))
    ref_ctrl, base_ctrl = ctrl[:N_CONTROLS], ctrl[N_CONTROLS:2 * N_CONTROLS]
    print(f"{len(kept_perts)} of {len(targets)} challenge genes have >= {MIN_CELLS_PER_PERT} K562 cells; "
          f"{sum(len(c) for c in chosen)} perturbed cells; controls: {len(ref_ctrl)} reference, {len(base_ctrl)} base")
    pert_idx = np.concatenate(chosen)
    groups = {"real_pert": pert_idx, "ref_ctrl": ref_ctrl, "base_ctrl": base_ctrl}

    # read them chunk by chunk
    wanted = {name: np.sort(idx) for name, idx in groups.items()}
    parts = {name: [] for name in groups}
    for start in range(0, adata.n_obs, CHUNK):
        stop = min(start + CHUNK, adata.n_obs)
        block = None
        for name, idx in wanted.items():
            sel = idx[(idx >= start) & (idx < stop)] - start
            if len(sel) == 0:
                continue
            if block is None:
                block = sparse.csr_matrix(adata.X[start:stop])
            parts[name].append((sel + start, block[sel]))
        print(f"  read {stop}/{adata.n_obs}", flush=True)

    names = adata.var_names.astype(str).to_numpy()
    for col in adata.var.columns:                      # prefer a column that matches challenge symbols better
        cand = adata.var[col].astype(str).to_numpy()
        if len(targets & set(cand)) > len(targets & set(names)):
            names = cand
    first = ~pd.Series(names).duplicated().to_numpy()
    var = pd.DataFrame(index=names[first])

    def assemble(name, label_of):
        idx = np.concatenate([p[0] for p in parts[name]])
        X = sparse.vstack([p[1] for p in parts[name]]).tocsr()[:, np.flatnonzero(first)]
        lab = np.array([label_of(i) for i in idx])
        return ad.AnnData(X.astype(np.float32), obs=pd.DataFrame({"target_gene": lab}, index=[f"c{i}" for i in idx]), var=var.copy())

    real_pert = assemble("real_pert", lambda i: labels[i])
    ref = assemble("ref_ctrl", lambda i: CONTROL_LABEL)
    base = assemble("base_ctrl", lambda i: CONTROL_LABEL)
    real = ad.concat([real_pert, ref])
    sample = real.X[:2000].toarray() if real.n_obs else np.zeros((1, 1))
    print("values are whole numbers:", bool(np.allclose(sample, np.round(sample))), "(must be True)")
    real.write_h5ad(OUT / "real.h5ad")
    base.write_h5ad(OUT / "base_controls.h5ad")
    pd.Series(kept_perts, name="target_gene").to_csv(OUT / "standin_perts.csv", index=False)
    print(f"wrote {OUT/'real.h5ad'}: {real.n_obs} cells x {real.n_vars} genes, {len(kept_perts)} perturbations")
    print(f"wrote {OUT/'base_controls.h5ad'}: {base.n_obs} cells")


if __name__ == "__main__":
    main()