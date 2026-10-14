import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.data import DeltaTable
from src.paths import CONTROLS, PROCESSED

# 1. Load the K562 table and the gene cards (embeddings)
table = DeltaTable.load(PROCESSED / "k562_delta.npz")
delta = table.delta                                   # one row per knockdown, NaN for the knocked-down gene itself
n_rows, n_genes = delta.shape
z = np.load(PROCESSED / "esm2_35m.npz", allow_pickle=True)
card_of = {g: i for i, g in enumerate(z["genes"])}
cards = z["emb"]

# 2. A card for every K562 row (rows without a protein get zeros and are skipped)
features = np.zeros((n_rows, cards.shape[1]), dtype=np.float32)
has_card = np.zeros(n_rows, dtype=bool)
for r, gene in enumerate(table.perts):
    if gene in card_of:
        features[r] = cards[card_of[gene]]
        has_card[r] = True
print("K562 knockdowns with a card:", int(has_card.sum()), "of", n_rows)

# 3. The same hidden groups as baseline_v0.py
challenge = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].to_numpy()
challenge_rows = np.where(np.isin(table.perts, challenge))[0]
rng = np.random.default_rng(0)
groups = np.array_split(rng.permutation(challenge_rows), 5)

# 4. Score one prediction against the truth (same as before)
def score_one(guess, truth):
    known = ~np.isnan(truth)
    g, t = guess[known], truth[known]
    error = np.mean((g - t) ** 2)
    zero_error = np.mean(t ** 2)
    corr = np.corrcoef(g, t)[0, 1] if g.std() > 0 else np.nan
    return error, zero_error, corr

alphas = [10, 100, 1000, 10000]          # how strongly ridge is held back (bigger = more cautious)
names = ["zero (no change)", "mean of all perts"] + [f"ridge alpha={a}" for a in alphas]
errors = {n: [] for n in names}
zero_errors = {n: [] for n in names}
corrs = {n: [] for n in names}

def record(name, guess, truth):
    error, zero_error, corr = score_one(guess, truth)
    errors[name].append(error)
    zero_errors[name].append(zero_error)
    if not np.isnan(corr):
        corrs[name].append(corr)

# 5. For each group: hide it, learn from the rest, predict the hidden ones
for hidden in groups:
    hidden = hidden[has_card[hidden]]                              # we can only predict genes that have a card
    other_rows = np.setdiff1d(np.arange(n_rows), hidden)
    train_rows = np.setdiff1d(np.where(has_card)[0], hidden)

    mean_row = np.nanmean(delta[other_rows], axis=0)               # the "average knockdown" baseline

    # standardise the cards using the training rows only
    mu = features[train_rows].mean(axis=0)
    sd = features[train_rows].std(axis=0) + 1e-6
    X_train = (features[train_rows] - mu) / sd
    X_test = (features[hidden] - mu) / sd

    # training targets: fill the NaN (knocked-down gene's own entry) with the column average
    Y_train = delta[train_rows]
    col_mean = np.nanmean(Y_train, axis=0)
    Y_train = np.where(np.isnan(Y_train), col_mean, Y_train)

    predictions = {}
    for a in alphas:
        model = Ridge(alpha=a).fit(X_train, Y_train)
        predictions[a] = model.predict(X_test)                     # one predicted row per hidden gene

    for j, row in enumerate(hidden):
        truth = delta[row]
        record("zero (no change)", np.zeros(n_genes), truth)
        record("mean of all perts", mean_row, truth)
        for a in alphas:
            record(f"ridge alpha={a}", predictions[a][j], truth)

# 6. Results
for name in names:
    mean_error = np.mean(errors[name])
    better = 1 - mean_error / np.mean(zero_errors[name])
    pattern = f"{np.mean(corrs[name]):.3f}" if corrs[name] else "n/a"
    print(f"{name:20s} error {mean_error:.5f} | better than zero by {better:+.2%} | pattern match {pattern}")