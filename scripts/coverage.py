"""Which public lines knocked down each of the 300 challenge genes, and with how many cells.

A gene measured in several lines can be predicted from a cross-line average; a gene measured
nowhere needs a fallback. Only lines with at least MIN_CELLS cells for a gene count as evidence.

Usage: python scripts/coverage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED

MIN_CELLS = 20

datasets = {
    "k562": "k562_delta.npz",
    "rpe1": "rpe1_delta.npz",
    "hct116": "xatlas_HCT116_delta.npz",
    "hek293t": "xatlas_HEK293T_delta.npz",
    "jurkat": "jurkat_delta.npz",
    "hepg2": "hepg2_delta.npz",
    "h1_train": "h1_train_delta.npz",
    "h1_val": "h1_val_delta.npz",
    "h1_test": "h1_test_delta.npz",
}

# one Series per line: cells per knocked-down gene
columns = {}
for name, file in datasets.items():
    with np.load(PROCESSED / file) as z:
        columns[name] = pd.Series(z["n_cells"], index=z["perts"])

predict_genes = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"]

# rows = the 300 challenge genes, columns = lines, values = cells (0 if the line never knocked it down)
table = pd.concat(columns, axis=1).reindex(predict_genes).fillna(0)
table["n_lines"] = (table[list(datasets)] >= MIN_CELLS).sum(axis=1)

print("genes by number of lines:\n", table["n_lines"].value_counts().sort_index().to_string())
print("\ngenes with no line:", list(table.index[table["n_lines"] == 0]))
print("\ngenes covered per line:\n", (table[list(datasets)] >= MIN_CELLS).sum().sort_values(ascending=False).to_string())
print("\ngenes with only one line:\n", table[table["n_lines"] == 1].to_string())
