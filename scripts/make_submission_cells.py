"""Make a full challenge submission file from resampled real control cells (no lookup of any gene's own row).

For every context (A, B, C) and every one of the 300 target genes it draws 400 real control cells of
that context (with replacement) and, optionally, applies ONE shared effect: the average perturbation
response over HCT116 and HEK293T (the same recipe that was scored locally as 'bootstrap avg').

Usage: python scripts/make_submission_cells.py EFFECT [SCALE=1.0] [TOPK=0] [MODEL_NAME]
    EFFECT  none | avg | model | modeldev
            model     effect predicted by archive/heldout_model.py (trained on K562 knockdowns WITHOUT the 300
                      challenge genes); genes without a prediction, and columns K562 did not measure, get the
                      average effect
            modeldev  like model, but the model's mean over all genes is replaced by the average effect, so only
                      the gene-specific part is used
    SCALE   multiplies the effect before it is applied
    TOPK    if > 0, keep only the TOPK genes with the largest |effect|
Writes PROCESSED/cells_{EFFECT}_x{SCALE}_k{TOPK}.h5ad  (360,000 cells, raw integer counts, sparse)
Then:  vcc prep -i <that file> -o <out.vcc>  and  vcc submit <out.vcc> --name <your vague name>
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED

CONTEXTS = ["A", "B", "C"]
N_CELLS = 400
MIN_CELLS = 20
SEED = 0


def average_effect(genes):
    """Mean log-expression change over HCT116 + HEK293T perturbations, in the challenge gene order."""
    pos = {g: i for i, g in enumerate(genes)}
    rows = []
    for split in ("HCT116", "HEK293T"):
        z = np.load(PROCESSED / f"xatlas_{split}_delta.npz", allow_pickle=True)
        keep = z["n_cells"] >= MIN_CELLS
        names = [str(g) for g in z["genes"]]
        cols = np.array([pos.get(g, -1) for g in names])
        D = np.full((int(keep.sum()), len(genes)), np.nan, dtype=np.float32)
        ok = cols >= 0
        D[:, cols[ok]] = z["delta"][keep][:, ok]
        rows.append(D)
    with np.errstate(all="ignore"):
        return np.nan_to_num(np.nanmean(np.vstack(rows), axis=0)).astype(np.float32)


def model_effects(genes, avg, mode, model_name):
    """{gene: effect over all challenge genes} from the saved held-out model predictions."""
    z = np.load(PROCESSED / "model_predictions.npz", allow_pickle=True)
    name = model_name or str(z["best"])
    pos = {g: i for i, g in enumerate(genes)}
    cols = np.array([pos.get(str(g), -1) for g in z["k_genes"]])
    ok = cols >= 0
    P = z[f"pred_{name}"]
    centre = P.mean(axis=0)
    rows = {}
    for g, row in zip(z["test_genes"], P):
        eff = avg.copy()
        if mode == "model":
            eff[cols[ok]] = row[ok]
        else:
            eff[cols[ok]] = avg[cols[ok]] + (row[ok] - centre[ok])
        rows[str(g)] = eff
    print(f"using saved model '{name}': {len(rows)} of the target genes have a prediction")
    return rows, name


def apply_effect(X, eff, scale, rng):
    f = np.exp(scale * eff)
    down = np.flatnonzero(f < 1 - 1e-6)
    up = np.flatnonzero(f > 1 + 1e-6)
    if len(down):
        X[:, down] = rng.binomial(X[:, down].astype(np.int64), f[down])
    if len(up):
        X[:, up] += rng.poisson(X[:, up] * (f[up] - 1)).astype(X.dtype)
    return X


def main():
    effect = sys.argv[1]
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    assert effect in ("none", "avg", "model", "modeldev")
    model_name = sys.argv[4] if len(sys.argv) > 4 else None
    rng = np.random.default_rng(SEED)
    genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].tolist()
    targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].tolist()
    pos = {g: i for i, g in enumerate(genes)}
    avg = None if effect == "none" else average_effect(genes)
    rows = {}
    if effect in ("model", "modeldev"):
        rows, model_name = model_effects(genes, avg, effect, model_name)

    blocks, target_col, context_col = [], [], []
    for ctx in CONTEXTS:
        a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
        if list(a.var_names.astype(str)) != genes:                      # put genes in challenge order
            a = a[:, genes].copy()
        B = sparse.csr_matrix(a.X)
        print(f"context {ctx}: {B.shape[0]} control cells", flush=True)
        for k, g in enumerate(targets):
            X = B[rng.integers(0, B.shape[0], N_CELLS)].toarray().astype(np.int32)
            if effect != "none":
                eff = rows.get(g, avg).copy()
                if g in pos:
                    eff[pos[g]] = 0
                if 0 < topk < len(eff):
                    eff[np.argsort(-np.abs(eff))[topk:]] = 0
                X = apply_effect(X, eff, scale, rng)
            blocks.append(sparse.csr_matrix(X.astype(np.float32)))
            target_col += [g] * N_CELLS
            context_col += [ctx] * N_CELLS
            if (k + 1) % 100 == 0:
                print(f"  {ctx} {k + 1}/{len(targets)}", flush=True)

    obs = pd.DataFrame({"target_gene": target_col, "context": context_col})
    obs.index = [f"cell{i}" for i in range(len(obs))]
    sub = ad.AnnData(sparse.vstack(blocks).tocsr(), obs=obs, var=pd.DataFrame(index=genes))
    tag = f"_{model_name}" if effect in ("model", "modeldev") else ""
    out = PROCESSED / f"cells_{effect}{tag}_x{scale:g}_k{topk}.h5ad"
    sub.write_h5ad(out)
    print(f"wrote {out}: {sub.shape}")
    print("cells per (context, gene):", obs.groupby(["context", "target_gene"]).size().unique())
    print("whole numbers:", bool(np.allclose(sub.X[:1000].toarray(), np.round(sub.X[:1000].toarray()))))


if __name__ == "__main__":
    main()