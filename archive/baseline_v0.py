import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.data import DeltaTable
from src.paths import CONTROLS, PROCESSED

# ---------- 1. Load the K562 table ----------
# One row per knocked-down gene, one column per measured gene.
# A cell holds "how much did this gene's expression change?"
table = DeltaTable.load(PROCESSED / "k562_delta.npz")
delta = table.delta
n_perts, n_genes = delta.shape

# ---------- 2. Find the rows that are challenge genes ----------
challenge_genes = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].to_numpy()
is_challenge = np.isin(table.perts, challenge_genes)
challenge_rows = np.where(is_challenge)[0]          # about 269 rows
print("challenge genes found in K562:", len(challenge_rows))

# ---------- 3. Split the challenge genes into 5 groups ----------
# Each round we hide one group, "train" on everything else, then predict the hidden group.
rng = np.random.default_rng(0)
shuffled = rng.permutation(challenge_rows)
groups = np.array_split(shuffled, 5)

# ---------- 4. A helper that scores one prediction against the truth ----------
def score_one(guess, truth):
    known = ~np.isnan(truth)                        # skip the knocked-down gene itself
    g = guess[known]
    t = truth[known]
    error = np.mean((g - t) ** 2)                   # how far off the guess is
    zero_error = np.mean(t ** 2)                    # how far off "no change" would be
    if g.std() > 0:
        corr = np.corrcoef(g, t)[0, 1]              # does the guess have the right pattern?
    else:
        corr = np.nan                               # a flat guess has no pattern
    return error, zero_error, corr

# ---------- 5. Try three simple guesses ----------
names = ["zero (no change)", "mean of all perts", "mean of challenge"]
errors = {n: [] for n in names}
zero_errors = {n: [] for n in names}
corrs = {n: [] for n in names}

for hidden in groups:
    all_rows = np.arange(n_perts)
    train_rows = np.setdiff1d(all_rows, hidden)                 # everything except the hidden genes
    train_challenge_rows = np.setdiff1d(challenge_rows, hidden) # same, challenge genes only

    guesses = {
        "zero (no change)": np.zeros(n_genes),
        "mean of all perts": np.nanmean(delta[train_rows], axis=0),
        "mean of challenge": np.nanmean(delta[train_challenge_rows], axis=0),
    }

    for row in hidden:
        truth = delta[row]
        for name in names:
            error, zero_error, corr = score_one(guesses[name], truth)
            errors[name].append(error)
            zero_errors[name].append(zero_error)
            if not np.isnan(corr):
                corrs[name].append(corr)

# ---------- 6. Print the results ----------
for name in names:
    mean_error = np.mean(errors[name])
    mean_zero_error = np.mean(zero_errors[name])
    better_than_zero = 1 - mean_error / mean_zero_error
    if len(corrs[name]) > 0:
        pattern = f"{np.mean(corrs[name]):.3f}"
    else:
        pattern = "n/a"
    print(f"{name:20s} error {mean_error:.5f} | better than zero by {better_than_zero:+.1%} | pattern match {pattern}")