"""Cell stand-in panel from X-Atlas/Orion (HCT116 or HEK293T) on the A/B/C challenge knockdowns, scored with the
official scorer (scripts/local_score.py) and examined with scripts/exam.py, like build_standin.py's panels.

Why: these lines measured almost the same knockdowns as the challenge (290 of the 300 A/B/C genes), so past
submissions' recipes can be re-made on the very same genes, with the line held out of the evidence.
Streams the dataset once (nothing big is downloaded) and keeps
    up to N_PER cells of every A/B/C knockdown with >= MIN_CELLS cells
    2 x 18,400 control cells, sampled evenly over the whole stream (REFERENCE for the scorer, BASE for predictions)
Cells are thinned to about TARGET_UMI counts like the challenge's controls. Writes the same files as
build_standin.py (controls_NAME/, processed/standin_NAME/real.h5ad); replaces a table-only panel of the same name.

Usage: python scripts/build_xatlas_standin.py HCT116|HEK293T [MAX_BATCHES]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from datasets import load_dataset
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PANEL_DIRS, PROCESSED

SPLIT = sys.argv[1]
MAX_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else None
NAME = SPLIT.lower()
REPO, BATCH, CONTROL_LABEL = "Xaira-Therapeutics/X-Atlas-Orion", 20_000, "Non-Targeting"
N_PER, MIN_CELLS, N_CTRL, TARGET_UMI, SEED = 460, 50, 18_400, 20_000, 0
CONTROL_SHARE = {"HCT116": 0.25, "HEK293T": 0.2}[SPLIT]   # ~165k / ~219k controls in the stream -> ~40k kept
rng = np.random.default_rng(SEED)

genes = pd.read_csv(PANEL_DIRS["val"] / "gene_names.csv")["gene_name"].to_numpy(str)
targets = pd.read_csv(PANEL_DIRS["val"] / "pert_counts.csv")["target_gene"].astype(str).tolist()
gene_index = {g: i for i, g in enumerate(genes)}
meta = load_dataset(REPO, "gene_metadata")["train"]
token_to_col = np.full(max(meta["gene_token_id"]) + 1, -1, dtype=np.int64)
for token, name in zip(meta["gene_token_id"], meta["gene_name"]):
    if name in gene_index:
        token_to_col[token] = gene_index[name]

wanted = pa.array(targets + [CONTROL_LABEL], pa.string())
kept = {t: [] for t in targets}           # per knockdown: list of CSR rows (kept until N_PER)
seen = {t: 0 for t in targets}            # reservoir sampling, so the N_PER cells are a random subset
controls = []


def rows_of(sub):
    """CSR counts (cells x challenge genes) of an arrow table of cells."""
    tokens, expr = sub["gene_token_id"].combine_chunks(), sub["gene_expression"].combine_chunks()
    lengths = pc.list_value_length(tokens).to_numpy(zero_copy_only=False)
    flat_t = pc.list_flatten(tokens).to_numpy(zero_copy_only=False)
    flat_e = pc.list_flatten(expr).to_numpy(zero_copy_only=False).astype(np.float32)
    row = np.repeat(np.arange(len(lengths)), lengths)
    col = token_to_col[flat_t]
    ok = col >= 0
    return sparse.csr_matrix((flat_e[ok], (row[ok], col[ok])), shape=(len(lengths), len(genes)))


ds = load_dataset(REPO, split=SPLIT, streaming=True, columns=["gene_target", "gene_token_id", "gene_expression"])
start = time.time()
for i, batch in enumerate(ds.with_format("arrow").iter(batch_size=BATCH)):
    table = batch if isinstance(batch, pa.Table) else pa.table(batch)
    labels = table["gene_target"].cast(pa.string())
    sub = table.filter(pc.is_in(labels, value_set=wanted))
    if sub.num_rows:
        lab = np.array(sub["gene_target"].to_pylist())
        is_ctrl = lab == CONTROL_LABEL
        take = ~is_ctrl | (rng.random(len(lab)) < CONTROL_SHARE)
        if take.any():
            X = rows_of(sub.filter(pa.array(take)))
            lab = lab[take]
            for r, t in enumerate(lab):
                if t == CONTROL_LABEL:
                    controls.append(X[r])
                    continue
                seen[t] += 1
                if len(kept[t]) < N_PER:
                    kept[t].append(X[r])
                elif (j := rng.integers(seen[t])) < N_PER:
                    kept[t][j] = X[r]
    if i % 20 == 0:
        print(f"batch {i}: {sum(map(len, kept.values()))} knockdown cells, {len(controls)} controls, "
              f"{(i + 1) * BATCH / (time.time() - start):.0f} cells/s", flush=True)
    if MAX_BATCHES is not None and i + 1 >= MAX_BATCHES:
        break

perts = [t for t in targets if len(kept[t]) >= MIN_CELLS]
assert len(controls) >= 2 * N_CTRL, f"only {len(controls)} controls"
order = rng.permutation(len(controls))
C = sparse.vstack(controls).tocsr()[order]
ref, base = C[:N_CTRL], C[N_CTRL:2 * N_CTRL]
umi = np.asarray(ref.sum(1)).ravel()
p_keep = min(1.0, TARGET_UMI / float(np.median(umi)))
print(f"control median counts {int(np.median(umi))} -> thinning to {p_keep:.2f}", flush=True)


def thin(X):
    X = X.astype(np.float32).tocsr()
    if p_keep < 1:
        X.data = rng.binomial(np.round(X.data).astype(np.int64), p_keep).astype(np.float32)
        X.eliminate_zeros()
    return X


def as_anndata(X, labels):
    obs = pd.DataFrame({"target_gene": labels}, index=[f"{NAME}_{k}" for k in range(X.shape[0])])
    return ad.AnnData(X, obs=obs, var=pd.DataFrame(index=genes))


pert_X = sparse.vstack([sparse.vstack(kept[t]) for t in perts]).tocsr()
labels = [t for t in perts for _ in kept[t]]
measured = np.asarray((C != 0).sum(0)).ravel() > 0          # scorer only sees genes this dataset measured
real = as_anndata(thin(sparse.vstack([pert_X, ref]).tocsr()), labels + ["non-targeting"] * N_CTRL)[:, measured]
out_controls, out_real = PANEL_DIRS[NAME], PROCESSED / f"standin_{NAME}"
shutil.rmtree(out_controls, ignore_errors=True)
out_controls.mkdir(parents=True); out_real.mkdir(parents=True, exist_ok=True)
as_anndata(thin(base), ["non-targeting"] * N_CTRL).write_h5ad(out_controls / f"context_{SPLIT}.h5ad")
shutil.copy(PANEL_DIRS["val"] / "gene_names.csv", out_controls / "gene_names.csv")
pd.Series(perts, name="target_gene").to_csv(out_controls / "pert_counts.csv", index=False)
(out_controls / "manifest.json").write_text(json.dumps({"contexts": [SPLIT]}))
real.copy().write_h5ad(out_real / "real.h5ad")
cells = pd.Series(labels).value_counts()
print(f"wrote panel {NAME}: {len(perts)} knockdowns (cells min/median {cells.min()}/{int(cells.median())}), "
      f"real on {int(measured.sum())} genes, {time.time() - start:.0f} s")
