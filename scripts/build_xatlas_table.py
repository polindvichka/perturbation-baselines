"""Build a knockdown-effect table from X-Atlas/Orion for our 300 challenge genes.

Streams the dataset (nothing big is saved to disk), keeps only the cells that carry one
of the 300 challenge genes or the control label, and for each gene computes
    mean log1p(counts per 10,000) over its cells  minus  the same over control cells.

Usage: python scripts/build_xatlas_table.py SPLIT [MAX_BATCHES]
  SPLIT        HCT116 or HEK293T
  MAX_BATCHES  optional, for a quick trial run (for example 5)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from datasets import load_dataset
from scipy import sparse

SPLIT = sys.argv[1]
MAX_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else None
REPO = "Xaira-Therapeutics/X-Atlas-Orion"
BATCH = 20_000
CONTROL_LABEL = "Non-Targeting"
CONTROL_DIR = Path("/root/vcc/controls")
OUT = f"/workspace/processed/xatlas_{SPLIT}_delta.npz" if MAX_BATCHES is None else f"/workspace/processed/xatlas_{SPLIT}_trial.npz"

genes = pd.read_csv(CONTROL_DIR / "gene_names.csv")["gene_name"].tolist()
targets = pd.read_csv(CONTROL_DIR / "pert_counts.csv")["target_gene"].tolist()
gene_index = {g: i for i, g in enumerate(genes)}

# ---- gene id (token) -> column of our 18,533 challenge genes -------------------
meta = load_dataset(REPO, "gene_metadata")["train"]
token_to_col = np.full(max(meta["gene_token_id"]) + 1, -1, dtype=np.int64)
for token, name in zip(meta["gene_token_id"], meta["gene_name"]):
    if name in gene_index:
        token_to_col[token] = gene_index[name]
print(f"gene ids that map to a challenge gene: {(token_to_col >= 0).sum()} of {len(token_to_col)}")

# ---- one accumulator row per target gene, plus one for the controls ------------
wanted = targets + [CONTROL_LABEL]
group_of = {label: k for k, label in enumerate(wanted)}
wanted_array = pa.array(wanted, pa.string())
sums = np.zeros((len(wanted), len(genes)))
n_cells = np.zeros(len(wanted))


def process(table):
    """Add the cells of one batch to the accumulators. Returns how many cells were kept."""
    labels = table["gene_target"].cast(pa.string())
    mask = pc.is_in(labels, value_set=wanted_array)
    if not pc.any(mask).as_py():
        return 0
    sub = table.filter(mask)
    group = np.array([group_of[label] for label in sub["gene_target"].to_pylist()])
    tokens = sub["gene_token_id"].combine_chunks()
    expression = sub["gene_expression"].combine_chunks()
    lengths = pc.list_value_length(tokens).to_numpy(zero_copy_only=False)
    flat_tokens = pc.list_flatten(tokens).to_numpy(zero_copy_only=False)
    flat_expr = pc.list_flatten(expression).to_numpy(zero_copy_only=False).astype(np.float64)
    n = len(group)
    row_of_value = np.repeat(np.arange(n), lengths)
    totals = np.bincount(row_of_value, weights=flat_expr, minlength=n)       # counts per cell
    value = np.log1p(1e4 * flat_expr / np.maximum(totals[row_of_value], 1))   # log1p(CP10K)
    col = token_to_col[flat_tokens]
    keep = col >= 0
    X = sparse.csr_matrix((value[keep], (row_of_value[keep], col[keep])), shape=(n, len(genes)))
    G = sparse.csr_matrix((np.ones(n), (group, np.arange(n))), shape=(len(wanted), n))
    sums[:] += (G @ X).toarray()
    n_cells[:] += np.bincount(group, minlength=len(wanted))
    return n


# ---- stream the dataset ----------------------------------------------------------
ds = load_dataset(REPO, split=SPLIT, streaming=True,
                  columns=["gene_target", "gene_token_id", "gene_expression"])
ds = ds.with_format("arrow")
start, seen, kept = time.time(), 0, 0
for i, batch in enumerate(ds.iter(batch_size=BATCH)):
    table = batch if isinstance(batch, pa.Table) else pa.table(batch)
    if i == 0:
        print("batch type:", type(batch).__name__, "| columns:", table.column_names, flush=True)
    seen += table.num_rows
    kept += process(table)
    if i % 10 == 0:
        speed = seen / (time.time() - start)
        print(f"batch {i}: {seen} cells read, {kept} kept, {speed:.0f} cells/s", flush=True)
    if MAX_BATCHES is not None and i + 1 >= MAX_BATCHES:
        break
print(f"finished: {seen} cells read, {kept} kept, {time.time() - start:.0f} s")

# ---- turn sums into effects --------------------------------------------------------
means = sums / np.maximum(n_cells, 1)[:, None]
control = means[-1]
delta = means[:-1] - control
for i, target in enumerate(targets):
    if target in gene_index:
        delta[i, gene_index[target]] = np.nan                  # the knocked-down gene itself
delta[:, control == 0] = np.nan                                # gene not measured in this dataset
np.savez_compressed(OUT, perts=np.array(targets), genes=np.array(genes), delta=delta.astype(np.float32),
                    n_cells=n_cells[:-1].astype(int), control_cells=int(n_cells[-1]),
                    control_profile=control.astype(np.float32))
cells = n_cells[:-1]
print(f"control cells: {int(n_cells[-1])}")
print(f"targets with at least 20 cells: {(cells >= 20).sum()} of {len(targets)}; cells per target min/median/max: "
      f"{int(cells.min())}/{int(np.median(cells))}/{int(cells.max())}")
print("wrote", OUT)