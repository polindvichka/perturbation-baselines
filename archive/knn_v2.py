import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data import DeltaTable
from src.paths import CONTROLS, PROCESSED

# 1. Load the K562 table and the gene cards
table = DeltaTable.load(PROCESSED / "k562_delta.npz")
delta = table.delta
n_rows, n_genes = delta.shape
z = np.load(PROCESSED / "esm2_35m.npz", allow_pickle=True)
card_of = {g: i for i, g in enumerate(z["genes"])}
cards = z["emb"]

features = np.zeros((n_rows, cards.shape[1]), dtype=np.float32)
has_card = np.zeros(n_rows, dtype=bool)
for r, gene in enumerate(table.perts):
    if gene in card_of:
        features[r] = cards[card_of[gene]]
        has_card[r] = True

# make every card length 1, so a dot product = how similar two genes are
norms = np.linalg.norm(features, axis=1, keepdims=True)
norms[norms == 0] = 1
unit = features / norms

# 2. The same hidden groups as before
challenge = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].to_numpy()
challenge_rows = np.where(np.isin(table.perts, challenge))[0]
rng = np.random.default_rng(0)
groups = np.array_split(rng.permutation(challenge_rows), 5)

TOP = 50   # we check the direction on the 50 genes that changed most in the true row

def score_one(guess, truth):
    known = ~np.isnan(truth)
    g, t = guess[known], truth[known]
    error = np.mean((g - t) ** 2)
    zero_error = np.mean(t ** 2)
    corr = np.corrcoef(g, t)[0, 1] if g.std() > 0 else np.nan
    top = np.argsort(-np.abs(t))[:TOP]                      # the most-changed genes in the truth
    direction = np.mean(np.sign(g[top]) == np.sign(t[top])) if g.std() > 0 else np.nan
    return error, zero_error, corr, direction

ks = [10, 30, 100, 300]
names = ["zero (no change)", "mean of all perts"] + [f"nearest {k} genes" for k in ks]
results = {n: {"error": [], "zero": [], "corr": [], "dir": []} for n in names}

def record(name, guess, truth):
    error, zero_error, corr, direction = score_one(guess, truth)
    results[name]["error"].append(error)
    results[name]["zero"].append(zero_error)
    if not np.isnan(corr):
        results[name]["corr"].append(corr)
    if not np.isnan(direction):
        results[name]["dir"].append(direction)

# 3. Hide a group, predict it, score it
for hidden in groups:
    hidden = hidden[has_card[hidden]]
    other_rows = np.setdiff1d(np.arange(n_rows), hidden)
    train_rows = np.setdiff1d(np.where(has_card)[0], hidden)
    mean_row = np.nanmean(delta[other_rows], axis=0)

    similarity = unit[hidden] @ unit[train_rows].T          # (hidden genes, training genes)

    for j, row in enumerate(hidden):
        truth = delta[row]
        record("zero (no change)", np.zeros(n_genes), truth)
        record("mean of all perts", mean_row, truth)
        order = np.argsort(-similarity[j])                  # most similar genes first
        for k in ks:
            neighbours = train_rows[order[:k]]
            guess = np.nan_to_num(np.nanmean(delta[neighbours], axis=0))   # average their rows
            record(f"nearest {k} genes", guess, truth)

# 4. Results
for name in names:
    r = results[name]
    better = 1 - np.mean(r["error"]) / np.mean(r["zero"])
    pattern = f"{np.mean(r['corr']):.3f}" if r["corr"] else "n/a"
    direction = f"{np.mean(r['dir']):.1%}" if r["dir"] else "n/a"
    print(f"{name:20s} better than zero {better:+.2%} | pattern {pattern} | direction on top-{TOP} genes {direction}")