"""Build a local stand-in panel that looks like a challenge context, so past submissions can be re-made on it
and scored with the official scorer (scripts/local_score.py), and the local scores compared with the leaderboard.

A panel is shaped like the challenge download, so every prediction script runs on it unchanged (VCC_PANEL=NAME):
    controls_NAME/context_CTX.h5ad   18,400 BASE control cells on the challenge gene axis (the prediction's
                                     starting cells, like context_A.h5ad)
    controls_NAME/gene_names.csv     the challenge gene axis
    controls_NAME/pert_counts.csv    the stand-in's knockdowns
    controls_NAME/manifest.json      {"contexts": [CTX]}
    processed/standin_NAME/real.h5ad the hidden answer: up to N_PER real cells per knockdown + 18,400 REFERENCE
                                     controls (disjoint from the base ones), on the genes the dataset measured
Cells are thinned to about TARGET_UMI counts (the challenge's controls have a median of ~20,000), so the scorer
has the same power to call genes as on the leaderboard.

Panels
  h1    VCC 2025 H1 hESC (10x Flex, like the challenge): 300 knockdowns, ~1,100 cells each, ~45,000 counts per cell
  k562  Replogle 2022 genome-wide K562 (the D look-alike), the A/B/C knockdowns it measured; ~11,000 counts per cell

Usage: python scripts/build_standin.py h1|k562
"""
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import harmonise_genes
from src.paths import DATA, PANEL_DIRS, PROCESSED

NAME = sys.argv[1]
N_PER, MIN_CELLS, N_CTRL, TARGET_UMI = 460, 50, 18_400, 20_000
CONTROL_LABEL, CHUNK, SEED = "non-targeting", 100_000, 0
PANELS = {  # context name, files, obs column with the knockdown, var column with gene symbols (None = index)
    "h1": ("H1", [DATA / "h1" / f"adata_{s}.h5ad" for s in ("Training", "Validation", "Test")], "target_gene", None),
    "k562": ("K562", [DATA / "replogle_2022_k562_gwps.h5ad"], "gene", None),
}
CTX, FILES, PERT_COL, VAR_COL = PANELS[NAME]
OUT_CONTROLS, OUT_REAL = PANEL_DIRS[NAME], PROCESSED / f"standin_{NAME}"
rng = np.random.default_rng(SEED)
challenge_genes = pd.read_csv(PANEL_DIRS["val"] / "gene_names.csv")["gene_name"].to_numpy(str)


def read_rows(adata, idx):
    """Rows `idx` (sorted) of a backed AnnData as CSR, read in chunks."""
    parts = []
    for start in range(0, adata.n_obs, CHUNK):
        sel = idx[(idx >= start) & (idx < start + CHUNK)]
        if len(sel):
            parts.append(sparse.csr_matrix(adata.X[start:min(start + CHUNK, adata.n_obs)])[sel - start])
    return sparse.vstack(parts).tocsr()


def thin(X, p):
    """Keep each count with probability p (binomial thinning), so cells look sequenced less deeply."""
    X = X.astype(np.float32).tocsr()
    if p < 1:
        X.data = rng.binomial(np.round(X.data).astype(np.int64), p).astype(np.float32)
        X.eliminate_zeros()
    return X


def as_anndata(X, labels, var_names):
    obs = pd.DataFrame({"target_gene": labels}, index=[f"{NAME}_{i}" for i in range(X.shape[0])])
    return ad.AnnData(X, obs=obs, var=pd.DataFrame(index=var_names))


pert_blocks, pert_labels, done, ctrl = [], [], set(), None
for k, file in enumerate(FILES):
    a = ad.read_h5ad(file, backed="r")
    labels = a.obs[PERT_COL].astype(str).to_numpy()
    symbols = (a.var[VAR_COL] if VAR_COL else a.var_names).astype(str).to_numpy()
    allowed = set(pd.read_csv(PANEL_DIRS["val"] / "pert_counts.csv")["target_gene"]) if NAME == "k562" else None
    counts = pd.Series(labels).value_counts()
    perts = sorted(p for p, n in counts.items() if p != CONTROL_LABEL and n >= MIN_CELLS
                   and (allowed is None or p in allowed) and p not in done)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == p), min(N_PER, counts[p]), replace=False) for p in perts])
    order = np.argsort(idx)
    if k == 0:   # the H1 files share one set of control cells; take them once
        c = rng.permutation(np.flatnonzero(labels == CONTROL_LABEL))
        assert len(c) >= 2 * N_CTRL, f"only {len(c)} control cells"
        c_ref, c_base = np.sort(c[:N_CTRL]), np.sort(c[N_CTRL:2 * N_CTRL])
        sample = read_rows(a, np.sort(rng.choice(a.n_obs, 2000, replace=False)))
        p_keep = min(1.0, TARGET_UMI / float(np.median(np.asarray(sample.sum(1)).ravel())))
        print(f"thinning to {p_keep:.2f} of the counts (target median {TARGET_UMI})", flush=True)
        ref, base = read_rows(a, c_ref), read_rows(a, c_base)
        first = ~pd.Series(symbols).duplicated().to_numpy()
        keep_cols = np.flatnonzero(first & np.isin(symbols, challenge_genes))
        real_genes = [g for g in challenge_genes if g in set(symbols[keep_cols])]
        col_of = {g: i for i, g in zip(keep_cols, symbols[keep_cols])}
        real_cols = np.array([col_of[g] for g in real_genes])
        ctrl = (ref, base, symbols, first)
    X = read_rows(a, idx[order])
    pert_blocks.append(X)
    pert_labels += list(labels[idx[order]])
    done.update(perts)
    print(f"{file.name}: {len(perts)} knockdowns, {len(idx)} cells", flush=True)

ref, base, symbols, first = ctrl
real = as_anndata(thin(sparse.vstack(pert_blocks + [ref]).tocsr()[:, real_cols], p_keep),
                  pert_labels + [CONTROL_LABEL] * N_CTRL, real_genes)
src = as_anndata(thin(base[:, np.flatnonzero(first)], p_keep), [CONTROL_LABEL] * N_CTRL, symbols[first])
base_axis = as_anndata(harmonise_genes(src, challenge_genes).astype(np.float32), src.obs["target_gene"].to_numpy(),
                       challenge_genes)
perts = list(dict.fromkeys(pert_labels))

OUT_CONTROLS.mkdir(parents=True, exist_ok=True); OUT_REAL.mkdir(parents=True, exist_ok=True)
base_axis.write_h5ad(OUT_CONTROLS / f"context_{CTX}.h5ad")
shutil.copy(PANEL_DIRS["val"] / "gene_names.csv", OUT_CONTROLS / "gene_names.csv")
pd.Series(perts, name="target_gene").to_csv(OUT_CONTROLS / "pert_counts.csv", index=False)
(OUT_CONTROLS / "manifest.json").write_text(json.dumps({"contexts": [CTX]}))
real.write_h5ad(OUT_REAL / "real.h5ad")
umi = np.asarray(real.X[-N_CTRL:].sum(1)).ravel()
cells = pd.Series(pert_labels).value_counts()
print(f"wrote panel {NAME}: {len(perts)} knockdowns (cells min/median {cells.min()}/{int(cells.median())}), "
      f"real on {len(real_genes)} genes, control median counts {int(np.median(umi))}")
