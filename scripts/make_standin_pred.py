"""Make a prediction for the K562 stand-in context, to be scored with scripts/local_score.py.

Starting cells come from base_controls.h5ad (K562 controls that are NOT the reference controls).
No K562 knockdown is used for the predicted effects: effects come from HCT116 and HEK293T tables.

Usage:
    python scripts/make_standin_pred.py CELLS EFFECT [SCALE=1.0] [TOPK=0] [MODEL_NAME]
        CELLS   bootstrap | multinomial
                bootstrap    draw real control cells with replacement, then apply the effect
                multinomial  draw new cells from the controls' average gene shares (what v1-v3 did)
        EFFECT  none   no effect (cells only)
                modeldev  like model, but only the gene-specific part: the model's mean over all genes is
                       removed and the donor average effect is used in its place. Use this to judge the model
                       fairly, because the plain 'model' effect carries K562's own average response (it was
                       trained on K562 rows), which a new cell context would not have.
                model  effect predicted by scripts/heldout_model.py (trained without any challenge gene's own row);
                       genes it has no prediction for get the average effect. Optional 5th argument: which
                       saved model to use (default: the one that scored best on held-back TRAINING genes)
                avg    the same average effect for every perturbation (mean over HCT116 and HEK293T)
                look   each perturbation's own row from HCT116 and HEK293T (mean of the two), average if absent
        SCALE   multiplies the effect before it is applied (effect is a log-expression change)
        TOPK    if > 0, keep only the TOPK genes with the largest |effect| for each perturbation
Writes /workspace/processed/standin/pred_{CELLS}_{EFFECT}_x{SCALE}_k{TOPK}.h5ad
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

PROCESSED = Path("/workspace/processed")
STANDIN = PROCESSED / "standin"
MIN_CELLS = 20
N_CELLS = 400
SEED = 0


def donor_tables(var_names):
    """Returns (avg effect over K562 genes, {pert: effect over K562 genes}) from HCT116 + HEK293T."""
    pos = {g: i for i, g in enumerate(var_names)}
    lines = []
    for split in ("HCT116", "HEK293T"):
        z = np.load(PROCESSED / f"xatlas_{split}_delta.npz", allow_pickle=True)
        keep = z["n_cells"] >= MIN_CELLS
        genes = [str(g) for g in z["genes"]]
        cols = np.array([pos.get(g, -1) for g in genes])
        D = np.full((int(keep.sum()), len(var_names)), np.nan, dtype=np.float32)
        ok = cols >= 0
        D[:, cols[ok]] = z["delta"][keep][:, ok]
        lines.append(({str(p): i for i, p in enumerate(np.array(z["perts"])[keep])}, D))
    all_rows = np.vstack([D for _, D in lines])
    with np.errstate(all="ignore"):
        avg = np.nan_to_num(np.nanmean(all_rows, axis=0))
    per_pert = {}
    for p in set().union(*[set(m) for m, _ in lines]):
        rows = np.stack([D[m[p]] for m, D in lines if p in m])
        with np.errstate(all="ignore"):
            row = np.nanmean(rows, axis=0)
        per_pert[p] = np.where(np.isnan(row), avg, row)
    return avg, per_pert


def apply_effect_counts(X, eff, scale, rng):
    """X: dense int counts (cells x genes). Thin down-regulated genes, add Poisson counts to up-regulated ones."""
    f = np.exp(scale * eff)
    down = np.flatnonzero(f < 1 - 1e-6)
    up = np.flatnonzero(f > 1 + 1e-6)
    if len(down):
        X[:, down] = rng.binomial(X[:, down].astype(np.int64), f[down])
    if len(up):
        X[:, up] += rng.poisson(X[:, up] * (f[up] - 1)).astype(X.dtype)
    return X


def main():
    cells, effect = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    topk = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    assert cells in ("bootstrap", "multinomial") and effect in ("none", "avg", "look", "model", "modeldev")
    model_name = sys.argv[5] if len(sys.argv) > 5 else None
    rng = np.random.default_rng(SEED)

    base = ad.read_h5ad(STANDIN / "base_controls.h5ad")
    var = base.var_names.astype(str)
    B = sparse.csr_matrix(base.X)
    perts = pd.read_csv(STANDIN / "standin_perts.csv")["target_gene"].tolist()
    avg, per_pert = (None, {}) if effect == "none" else donor_tables(list(var))
    model_rows = {}
    if effect in ("model", "modeldev"):
        z = np.load(PROCESSED / "model_predictions.npz", allow_pickle=True)
        model_name = model_name or str(z["best"])
        col_of = {str(g): i for i, g in enumerate(var)}
        src = np.array([col_of.get(str(g), -1) for g in z["k_genes"]])
        P = z[f"pred_{model_name}"]
        ok = src >= 0
        for g, row in zip(z["test_genes"], P):
            full = np.zeros(len(var), dtype=np.float32)
            full[src[ok]] = row[ok]
            model_rows[str(g)] = full
        if effect == "modeldev":
            centre = np.mean(list(model_rows.values()), axis=0)
            model_rows = {g: avg + (row - centre) for g, row in model_rows.items()}
        print(f"using saved model '{model_name}': {len(model_rows)} perturbations have a prediction")

    sizes = np.asarray(B.sum(axis=1)).ravel()
    share = np.asarray(B.sum(axis=0)).ravel()
    share = share / share.sum()
    own = {g: i for i, g in enumerate(var)}

    blocks, labels = [], []
    for k, g in enumerate(perts):
        eff = np.zeros(len(var), dtype=np.float32)
        if effect == "avg":
            eff = avg.copy()
        elif effect == "look":
            eff = per_pert.get(g, avg).copy()
        elif effect in ("model", "modeldev"):
            eff = model_rows.get(g, avg).copy()
        if effect != "none":
            if g in own:
                eff[own[g]] = 0
            if topk > 0 and topk < len(eff):
                drop = np.argsort(-np.abs(eff))[topk:]
                eff[drop] = 0
        if cells == "bootstrap":
            X = B[rng.integers(0, B.shape[0], N_CELLS)].toarray().astype(np.int32)
            if effect != "none":
                X = apply_effect_counts(X, eff, scale, rng)
        else:
            s = share * np.exp(scale * eff)
            s = s / s.sum()
            X = np.stack([rng.multinomial(int(n), s) for n in rng.choice(sizes, N_CELLS)]).astype(np.int32)
        blocks.append(sparse.csr_matrix(X.astype(np.float32)))
        labels += [g] * N_CELLS
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(perts)}", flush=True)

    pred = ad.AnnData(sparse.vstack(blocks).tocsr(), obs=pd.DataFrame({"target_gene": labels}), var=pd.DataFrame(index=var))
    pred.obs_names = [f"p{i}" for i in range(pred.n_obs)]
    tag = f"_{model_name}" if effect in ("model", "modeldev") else ""
    out = STANDIN / f"pred_{cells}_{effect}{tag}_x{scale:g}_k{topk}.h5ad"
    pred.write_h5ad(out)
    print("wrote", out, pred.shape)


if __name__ == "__main__":
    main()