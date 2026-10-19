"""Learned transfer model: predict a knockdown's effect in an unseen cell line from its measured
effects in other lines, with gradient-boosted trees (LightGBM).

Training examples come from hiding one public line at a time (K562, HCT116, HEK293T) and using
the remaining lines as evidence. One example = (knocked-down gene p, measured gene j):
    features  summary of the source lines' log2 FCs for (p, j): mean, count, spread, up/down
              votes, largest change; the CD4 and H1 values on their own (10x Flex like the
              challenge); how strongly j (and p) are expressed in the hidden line's controls;
              how strongly j is expressed in the source lines
    label     the log2 FC of (p, j) measured in the hidden line
The model is fitted by minimizing squared error, so it learns how much to trust which evidence.
Each example is weighted by how much its gene counts in the scorer's expression space
(log1p of counts per 50,000): a log2 FC of a gene with control CP10K c moves that value by about
ln2 * 5c / (1 + 5c), so weakly expressed genes, whose fold changes are mostly noise, count little.

Modes
  cv        train on two hidden lines, test on the third; compare with the plain source average
  standin   train without any K562 data, predict K562 (for scripts/standin_submission.py)
            -> PROCESSED/effects_ml_noK562.npz
  final     train on all hidden-line folds, predict contexts A, B, C from all sources, using each
            context's own control expression -> PROCESSED/effects_ml.npz, lfc shape (3, 300, 18533)

  ridge_cv  like cv, but with a weighted ridge regression on a few interpretable inputs
            (less flexible than the trees, so it cannot memorize the two training lines)

Usage: python scripts/transfer_model.py cv|ridge_cv|standin|final
"""
import sys
from pathlib import Path

import anndata as ad
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED

MODE = sys.argv[1]
EPS, MIN_CELLS = 0.1, 20
MIN_EXPR = 0.05            # score / train only genes with at least this control CP10K in the hidden line
ROWS_PER_FOLD = 3_000_000  # random subsample of training examples per hidden line
SEED = 0
FILES = {"k562": "k562_delta.npz", "hct116": "xatlas_HCT116_delta.npz", "hek293t": "xatlas_HEK293T_delta.npz",
         "h1": ["h1_train_delta.npz", "h1_val_delta.npz", "h1_test_delta.npz"], "cd4": "cd4_lfc.npz"}
LABEL_LINES = ["k562", "hct116", "hek293t"]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED)
N_TREES = 400

genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"]
rng = np.random.default_rng(SEED)


def load(name):
    """(lfc DataFrame targets x challenge genes, control CP10K Series or None)."""
    files = FILES[name] if isinstance(FILES[name], list) else [FILES[name]]
    frames, expr = [], []
    for f in files:
        z = np.load(PROCESSED / f)
        keep = np.isin(z["perts"], targets) & (z["n_cells"] >= MIN_CELLS)
        if "lfc" in z.files:
            frames.append(pd.DataFrame(z["lfc"][keep], index=z["perts"][keep], columns=z["genes"]))
            continue
        c = z["basal_cp10k"] if "basal_cp10k" in z.files else z["control_cp10k"]
        l = np.log2(z["mean_cp10k"][keep] + EPS) - np.log2(c + EPS)
        l[:, c == 0] = np.nan
        d = pd.DataFrame(l, index=z["perts"][keep], columns=z["genes"])
        frames.append(d.loc[:, ~d.columns.duplicated()])
        expr.append(pd.Series(c, index=z["genes"]).groupby(level=0).mean())
    lfc = pd.concat([f.reindex(columns=genes) for f in frames]).groupby(level=0).mean()
    e = pd.concat(expr, axis=1).mean(axis=1).reindex(genes) if expr else None
    return lfc.reindex(targets), e


def features(sources, tgt_expr):
    """sources: {name: (lfc (300, G) array, expr (G,) or None)}; tgt_expr: (G,) control CP10K of the target context.
    Returns a dict of (300, G) feature arrays."""
    S = np.stack([v[0] for v in sources.values()])                  # (n_src, 300, G)
    have = ~np.isnan(S)
    n = have.sum(0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(S, 0)
        std = np.nanstd(S, 0)
        maxabs = np.nanmax(np.abs(S), 0)
    up, down = (S > 0).sum(0), (S < 0).sum(0)
    src_expr = [v[1] for v in sources.values() if v[1] is not None]
    src_e = np.log1p(np.nanmean(np.stack([e.to_numpy() for e in src_expr]), 0)) if src_expr else np.full(S.shape[2], np.nan)
    te = np.log1p(np.nan_to_num(tgt_expr))
    pos = {g: i for i, g in enumerate(genes)}
    p_expr = np.array([te[pos[t]] if t in pos else np.nan for t in targets])
    G = S.shape[2]
    f = {"mean": mean, "n": n.astype(np.float32), "std": std, "maxabs": maxabs,
         "up": up.astype(np.float32), "down": down.astype(np.float32),
         "agree": (np.maximum(up, down) / np.maximum(n, 1)).astype(np.float32),
         "cd4": sources["cd4"][0] if "cd4" in sources else np.full_like(mean, np.nan),
         "h1": sources["h1"][0] if "h1" in sources else np.full_like(mean, np.nan),
         "tgt_expr_j": np.broadcast_to(te, (len(targets), G)),
         "src_expr_j": np.broadcast_to(src_e, (len(targets), G)),
         "tgt_expr_p": np.broadcast_to(p_expr[:, None], (len(targets), G))}
    return f


def to_rows(f, mask):
    return np.column_stack([np.asarray(v, dtype=np.float32)[mask] for v in f.values()])


def own_gene_mask():
    pos = {g: i for i, g in enumerate(genes)}
    m = np.zeros((len(targets), len(genes)), bool)
    for i, t in enumerate(targets):
        if t in pos:
            m[i, pos[t]] = True
    return m


DATA = {k: load(k) for k in FILES}
DATA = {k: (v[0].to_numpy(np.float32), v[1]) for k, v in DATA.items()}
OWN = own_gene_mask()
print("loaded:", {k: int((~np.isnan(v[0])).any(1).sum()) for k, v in DATA.items()}, "target genes per source", flush=True)


def fold(held, exclude=()):
    """Training rows with `held` as the hidden line; sources = all other lines except `exclude`."""
    sources = {k: v for k, v in DATA.items() if k != held and k not in exclude}
    label, expr = DATA[held]
    f = features(sources, expr.to_numpy())
    mask = ~np.isnan(label) & (f["n"] > 0) & (expr.to_numpy() >= MIN_EXPR)[None, :] & ~OWN
    idx = np.flatnonzero(mask.ravel())
    if len(idx) > ROWS_PER_FOLD:
        idx = np.sort(rng.choice(idx, ROWS_PER_FOLD, replace=False))
    m = np.zeros(mask.size, bool); m[idx] = True; m = m.reshape(mask.shape)
    w = np.broadcast_to(bulk_weight(expr.to_numpy()), mask.shape)[m]
    return to_rows(f, m), label[m], f, mask, w


def bulk_weight(c):
    """Squared sensitivity of the scorer's log1p(5 * CP10K) value to a log2 FC, per gene."""
    c = np.nan_to_num(c)
    return ((np.log(2) * 5 * c / (1 + 5 * c)) ** 2).astype(np.float32)


def train(folds):
    X = np.vstack([fo[0] for fo in folds]); y = np.concatenate([fo[1] for fo in folds])
    w = np.concatenate([fo[4] for fo in folds])
    print(f"training on {len(y):,} examples, {X.shape[1]} features", flush=True)
    return lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, feature_name=list(folds[0][2].keys())), N_TREES)


def predict(model, f, fill_mean=True):
    """Model prediction for every (p, j); where no source measured (p, j) there is nothing to say -> 0."""
    X = to_rows(f, np.ones(f["mean"].shape, bool))
    out = model.predict(X, num_threads=0).reshape(f["mean"].shape).astype(np.float32)
    out[f["n"] == 0] = 0.0
    # the knocked-down gene itself: keep the plain average (the scorer ignores it, but it drives the cells)
    out[OWN] = np.nan_to_num(f["mean"][OWN])
    return out


if MODE == "cv":
    for test in LABEL_LINES:
        model = train([fold(h) for h in LABEL_LINES if h != test])
        _, _, f, mask, _ = fold(test)
        label, expr = DATA[test]
        pred = predict(model, f)
        base = np.nan_to_num(f["mean"])
        c = np.broadcast_to(np.nan_to_num(expr.to_numpy()), mask.shape)[mask]
        to_bulk = lambda l: np.log1p(5 * c * 2.0 ** l) - np.log1p(5 * c)      # the scorer's space
        y = label[mask]; yb = to_bulk(y)
        for name, p in [("average x1", base[mask]), ("average x0.5", 0.5 * base[mask]),
                        ("average x0.25", 0.25 * base[mask]), ("model", pred[mask]), ("model x2", 2 * pred[mask])]:
            pb = to_bulk(p)
            r2b = 1 - np.mean((yb - pb) ** 2) / np.mean((yb - yb.mean()) ** 2)
            print(f"  test {test:8s} {name:13s}: scorer-space corr {np.corrcoef(pb, yb)[0, 1]:.3f}  R2 {r2b:+.4f}")
        imp = pd.Series(model.feature_importance("gain"), index=list(f.keys())).sort_values(ascending=False)
        print("  feature importance (gain share):", (imp / imp.sum()).round(3).head(6).to_dict(), flush=True)

def ridge_inputs(f):
    """A few interpretable inputs per (p, j); missing values -> 0 (no evidence)."""
    mean = np.nan_to_num(f["mean"]); agree = f["agree"]; te = np.asarray(f["tgt_expr_j"])
    return {"mean": mean, "mean*agree": mean * agree, "mean*tgt_expr": mean * te,
            "mean*n": mean * f["n"], "cd4": np.nan_to_num(f["cd4"]), "h1": np.nan_to_num(f["h1"])}


def fit_ridge(label_lines, exclude=()):
    from sklearn.linear_model import Ridge
    X, y, w = [], [], []
    for h in label_lines:
        _, _, f, mask, _ = fold(h, exclude=exclude)
        X.append(np.column_stack([v[mask] for v in ridge_inputs(f).values()]))
        y.append(DATA[h][0][mask])
        w.append(np.broadcast_to(bulk_weight(DATA[h][1].to_numpy()), mask.shape)[mask])
    model = Ridge(alpha=1.0, fit_intercept=False).fit(np.vstack(X), np.concatenate(y), sample_weight=np.concatenate(w))
    print("ridge weights:", dict(zip(ridge_inputs(f).keys(), np.round(model.coef_, 3))), flush=True)
    return model


def predict_ridge(model, f):
    cols = ridge_inputs(f)
    out = (np.stack(list(cols.values()), -1) @ model.coef_).astype(np.float32)
    out[f["n"] == 0] = 0.0
    out[OWN] = np.nan_to_num(f["mean"][OWN])          # the knocked-down gene itself: keep the plain average
    return out


if MODE == "ridge_standin":
    model = fit_ridge([h for h in LABEL_LINES if h != "k562"], exclude=("k562",))
    sources = {k: v for k, v in DATA.items() if k != "k562"}
    pred = predict_ridge(model, features(sources, DATA["k562"][1].to_numpy()))
    np.savez_compressed(PROCESSED / "effects_ridge_noK562.npz", perts=targets.to_numpy(str),
                        genes=genes.to_numpy(str), lfc=pred)
    print("wrote", PROCESSED / "effects_ridge_noK562.npz", "mean|lfc|", float(np.abs(pred).mean()))

elif MODE == "ridge_final":
    model = fit_ridge(LABEL_LINES)
    preds = []
    for ctx in "ABC":
        a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
        s = np.asarray(a.X.sum(0)).ravel()
        expr = pd.Series(1e4 * s / s.sum(), index=a.var_names.astype(str)).reindex(genes).to_numpy()
        preds.append(predict_ridge(model, features(DATA, expr)))
        print(f"context {ctx}: mean|lfc| {np.abs(preds[-1]).mean():.4f}", flush=True)
    np.savez_compressed(PROCESSED / "effects_ridge.npz", perts=targets.to_numpy(str), genes=genes.to_numpy(str),
                        lfc=np.stack(preds), contexts=np.array(list("ABC")))
    print("wrote", PROCESSED / "effects_ridge.npz")

elif MODE == "ridge_cv":
    from sklearn.linear_model import Ridge

    for test in LABEL_LINES:
        rows_X, rows_y, rows_w = [], [], []
        for h in LABEL_LINES:
            if h == test:
                continue
            _, _, f, mask, _ = fold(h)
            cols = ridge_inputs(f)
            rows_X.append(np.column_stack([v[mask] for v in cols.values()]))
            rows_y.append(DATA[h][0][mask])
            rows_w.append(np.broadcast_to(bulk_weight(DATA[h][1].to_numpy()), mask.shape)[mask])
        model = Ridge(alpha=1.0, fit_intercept=False).fit(np.vstack(rows_X), np.concatenate(rows_y),
                                                          sample_weight=np.concatenate(rows_w))
        _, _, f, mask, _ = fold(test)
        cols = ridge_inputs(f)
        label, expr = DATA[test]
        c = np.broadcast_to(np.nan_to_num(expr.to_numpy()), mask.shape)[mask]
        to_bulk = lambda l: np.log1p(5 * c * 2.0 ** l) - np.log1p(5 * c)
        y = label[mask]; yb = to_bulk(y)
        pred = model.predict(np.column_stack([v[mask] for v in cols.values()]))
        base = np.nan_to_num(f["mean"])[mask]
        for name, pr in [("average x0.25", 0.25 * base), ("average x0.5", 0.5 * base), ("ridge", pred),
                         ("ridge x2", 2 * pred), ("ridge x4", 4 * pred)]:
            pb = to_bulk(pr)
            r2b = 1 - np.mean((yb - pb) ** 2) / np.mean((yb - yb.mean()) ** 2)
            print(f"  test {test:8s} {name:13s}: scorer-space corr {np.corrcoef(pb, yb)[0, 1]:.3f}  R2 {r2b:+.4f}")
        print("  ridge weights:", dict(zip(cols.keys(), np.round(model.coef_, 3))), flush=True)

elif MODE == "standin":
    model = train([fold(h, exclude=("k562",)) for h in LABEL_LINES if h != "k562"])
    sources = {k: v for k, v in DATA.items() if k != "k562"}
    pred = predict(model, features(sources, DATA["k562"][1].to_numpy()))
    np.savez_compressed(PROCESSED / "effects_ml_noK562.npz", perts=targets.to_numpy(str),
                        genes=genes.to_numpy(str), lfc=pred)
    print("wrote", PROCESSED / "effects_ml_noK562.npz", "mean|lfc|", float(np.abs(pred).mean()))

elif MODE == "final":
    model = train([fold(h) for h in LABEL_LINES])
    preds = []
    for ctx in "ABC":
        a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
        s = np.asarray(a.X.sum(0)).ravel()
        expr = pd.Series(1e4 * s / s.sum(), index=a.var_names.astype(str)).reindex(genes).to_numpy()
        preds.append(predict(model, features(DATA, expr)))
        print(f"context {ctx}: mean|lfc| {np.abs(preds[-1]).mean():.3f}", flush=True)
    np.savez_compressed(PROCESSED / "effects_ml.npz", perts=targets.to_numpy(str), genes=genes.to_numpy(str),
                        lfc=np.stack(preds), contexts=np.array(list("ABC")))
    print("wrote", PROCESSED / "effects_ml.npz")
