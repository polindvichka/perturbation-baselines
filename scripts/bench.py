# Goal: pretend one public line is the new context, predict it from the other lines, and score the prediction. It's what I just did roughly, but done properly and reusable.

# A. Load the effects. Copy the load_lfc function from make_effects.py. Build a dictionary L = {"k562": ..., "hct116": ..., "hek293t": ...}, each a DataFrame (our genes × 18,533).

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED

EPS = 0.1
MIN_CELLS = 20
MIN_EXPR = 0.5          # only score genes with at least this control CP10K in the held-out line

genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"]


def load_lfc(file):
    """Like make_effects.load_lfc, but also returns the control expression per challenge gene."""
    z = np.load(PROCESSED / file)
    keep = np.isin(z["perts"], targets) & (z["n_cells"] >= MIN_CELLS)
    control = z["basal_cp10k"] if "basal_cp10k" in z.files else z["control_cp10k"]
    lfc = np.log2(z["mean_cp10k"][keep] + EPS) - np.log2(control + EPS)
    lfc[:, control == 0] = np.nan
    df = pd.DataFrame(lfc, index=z["perts"][keep], columns=z["genes"])
    unique = ~df.columns.duplicated()
    expr = pd.Series(control[unique], index=z["genes"][unique]).reindex(genes)
    return df.loc[:, unique].reindex(columns=genes), expr


def pds_like(P, T):
    """P, T: (n_genes_predicted, n_columns) arrays, row i of P is the prediction for row i of T.
    Returns the average share of OTHER true rows that are more similar to prediction i than its own true row.
    0 = always the best match, about 0.5 = chance."""
    # TODO 1: normalise every row of P and of T to length 1
    #   hint: P / np.linalg.norm(P, axis=1, keepdims=True)
    # TODO 2: S = cosine similarity of every prediction with every true row, shape (n, n)
    #   hint: matrix product of the normalised P with the
    # TODO 3: for each i, count how many entries of S[i] are larger than S[i, i]
    #   hint: (S > np.diag(S)[:, None]).sum(axis=1)  -- npilarities
    # TODO 4: divide by (n - 1) and return the mean
    ...


def mse_like(P, T):
    """1 - error of P / error of the baseline that predictevery gene.
    Above 0 = better than that baseline."""
    # TODO 5: baseline = the mean of T over its rows (axisne
    # TODO 6: return 1 - mean((P - T)**2) / mean((baseline - T)**2)
    ...


lines = {
    "k562": load_lfc("k562_delta.npz"),
    "hct116": load_lfc("xatlas_HCT116_delta.npz"),
    "hek293t": load_lfc("xatlas_HEK293T_delta.npz"),
}

for held in lines:
    true, expr = lines[held]
    pred = pd.concat().mean()
    # TODO 7: pred = the mean of the OTHER lines' tables, per gene
    #   hint: pd.concat([...]).groupby(level=0).mean(), li
    # TODO 8: g = the genes both pred and true have (index intersection)
    # TODO 9: cols = boolean mask of genes with expr >= MI
    # TODO 10: P = pred.loc[g] with only cols, as a numpy array; NaN -> 0 (np.nan_to_num); same for T
    # TODO 11: print held, len(g), pds_like(P, T), and mse0.25, 0.5, 1, 1.5]
# B. Loop over held-out lines. For each held in L:
# - pred = mean of the other lines: pd.concat([...]).groupby(level=0).mean(), as in make_effects;
# - true = L[held];
# - keep the genes both have: g = pred.index.intersection(true.index).

# C. Pick the columns to score: genes reasonably expressed in the held line. That's the ~2,000 genes with control CP10K ≥ 0.5; load_lfc already reads that value, so return it too. Replace the remaining NaN with 0.

# D. Score 1, discrimination (like pds).
# - Compute the cosine similarity between every predicted row and every true row. Normalize each row to length 1 (X / np.linalg.norm(X, axis=1, keepdims=True)), then S = P_norm @ T_norm.T, a 294 × 294 table.
# - For each gene i, count how many true rows are more similar to prediction i than its own true row: (S[i] > S[i, i]).sum().
# - Divide by the number of genes. 0 = always the best match, ~0.5 = chance. Print the average.

# E. Score 2, error vs a simple baseline (like mse).
# - Baseline: predict the held line's own average effect for every gene. That's what the real scorer's 0 means.
# - score = 1 − mean((pred − true)²) / mean((baseline − true)²). Above 0 means better than the baseline.
# - Do it for pred × s with s in [0.25, 0.5, 1, 1.5], to see which scale is best.

# Start with A–D on one held-out line and send me what you get. Then add E.