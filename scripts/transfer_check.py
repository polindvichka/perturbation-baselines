"""How well does a gene's knockdown effect in one cell line identify the same gene in another line?

For every pair (donor line -> recipient line) and every challenge gene g present in both:
  - correlate g's donor row with the recipient rows of ALL shared genes,
  - see where the recipient's own row for g ranks (1.0 = best possible, 0.5 = pure chance).
This resembles the discrimination score used by the challenge.

Usage: python scripts/transfer_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DeltaTable
from src.paths import CONTROLS, PROCESSED

MIN_CELLS = 20

genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].tolist()
targets = set(pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"])


def load_k562():
    table = DeltaTable.load(str(PROCESSED / "k562_delta.npz"))
    rows = list(table.perts)
    keep = [i for i, r in enumerate(rows) if r in targets]               # only the challenge genes
    position = pd.Index(genes).get_indexer(table.genes)
    found = position >= 0
    full = np.full((len(keep), len(genes)), np.nan, dtype=np.float32)
    full[:, position[found]] = np.asarray(table.delta)[keep][:, found]    # K562 columns -> challenge columns
    return [rows[i] for i in keep], full


def load_xatlas(split):
    z = np.load(PROCESSED / f"xatlas_{split}_delta.npz")
    perts = list(z["perts"])
    ok = z["n_cells"] >= MIN_CELLS
    return [p for p, k in zip(perts, ok) if k], z["delta"][ok]


lines = {"K562": load_k562(), "HCT116": load_xatlas("HCT116"), "HEK293T": load_xatlas("HEK293T")}
for name, (perts, matrix) in lines.items():
    print(f"{name}: {len(perts)} challenge genes with enough cells")


def zscore_rows(M):
    M = M - M.mean(axis=1, keepdims=True)
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)


def compare(donor, recipient):
    d_perts, d_mat = lines[donor]
    r_perts, r_mat = lines[recipient]
    d_idx = {p: i for i, p in enumerate(d_perts)}
    r_idx = {p: i for i, p in enumerate(r_perts)}
    shared = [p for p in d_perts if p in r_idx]
    D = d_mat[[d_idx[p] for p in shared]]
    R = r_mat[[r_idx[p] for p in shared]]
    measured = ~np.isnan(D).all(axis=0) & ~np.isnan(R).all(axis=0)        # gene measured in both lines
    D, R = np.nan_to_num(D[:, measured]), np.nan_to_num(R[:, measured])
    C = zscore_rows(D) @ zscore_rows(R).T                                  # C[g, s] = corr(donor g, recipient s)
    n = len(shared)
    own = np.diag(C)
    score = (C < own[:, None]).sum(axis=1) / (n - 1)                       # share of wrong genes ranked below the right one
    top1 = (C.argmax(axis=1) == np.arange(n)).mean()
    strength = np.linalg.norm(R, axis=1)
    strong = strength >= np.quantile(strength, 0.75)                       # the 25% strongest responders in the recipient
    off = C[~np.eye(n, dtype=bool)]
    return dict(donor=donor, recipient=recipient, genes=n, measured=int(measured.sum()),
                score=round(score.mean(), 3), score_strong=round(score[strong].mean(), 3),
                score_weak=round(score[~strong].mean(), 3), top1=round(top1, 3), chance_top1=round(1 / n, 3),
                corr_same_gene=round(float(np.median(own)), 3), corr_other_genes=round(float(np.median(off)), 3))


rows = [compare(a, b) for a in lines for b in lines if a != b]
result = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print()
print(result.to_string(index=False))
print("\nscore: 1.0 = the same gene is always the best match, 0.5 = no better than chance")