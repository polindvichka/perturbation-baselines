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

  dir_cv    direction-confidence head (src/direction.py): a classifier predicts whether a gene really moves in
            the predicted direction in the hidden line; reports how many genes pass the 90%-precision threshold
            and how much of the ridge model's scorer-space accuracy survives when only those are amplified
  dir_final ridge + direction head for contexts A, B, C: confident genes keep the ridge change, the others get
            OTHER_SHARE of it (default 1/7) -> PROCESSED/effects_ridge_gated.npz (use with SCALE 7)

Usage: python scripts/transfer_model.py cv|ridge_cv|ridge_standin|ridge_final|standin|final|dir_cv|dir_final [OTHER_SHARE]
"""
import sys
from pathlib import Path

import anndata as ad
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTEXTS, CONTROLS, PROCESSED, SUFFIX, is_holdout

MODE = sys.argv[1]
EPS, MIN_CELLS = 0.1, 20
MIN_EXPR = 0.05            # score / train only genes with at least this control CP10K in the hidden line
ROWS_PER_FOLD = 3_000_000  # random subsample of training examples per hidden line
SEED = 0
FILES = {"k562": "k562_delta.npz", "hct116": "xatlas_HCT116_delta.npz", "hek293t": "xatlas_HEK293T_delta.npz",
         "h1": ["h1_train_delta.npz", "h1_val_delta.npz", "h1_test_delta.npz"], "cd4": "cd4_lfc.npz"}
FILES = {k: v for k, v in FILES.items() if not is_holdout(k)}      # a stand-in panel never sees its own data
LABEL_LINES = [k for k in ["k562", "hct116", "hek293t"] if k in FILES]
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
    for ctx in CONTEXTS:
        a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
        s = np.asarray(a.X.sum(0)).ravel()
        expr = pd.Series(1e4 * s / s.sum(), index=a.var_names.astype(str)).reindex(genes).to_numpy()
        preds.append(predict_ridge(model, features(DATA, expr)))
        print(f"context {ctx}: mean|lfc| {np.abs(preds[-1]).mean():.4f}", flush=True)
    np.savez_compressed(PROCESSED / f"effects_ridge{SUFFIX}.npz", perts=targets.to_numpy(str), genes=genes.to_numpy(str),
                        lfc=np.stack(preds), contexts=np.array(CONTEXTS))
    print("wrote", PROCESSED / f"effects_ridge{SUFFIX}.npz")

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
    for ctx in CONTEXTS:
        a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
        s = np.asarray(a.X.sum(0)).ravel()
        expr = pd.Series(1e4 * s / s.sum(), index=a.var_names.astype(str)).reindex(genes).to_numpy()
        preds.append(predict(model, features(DATA, expr)))
        print(f"context {ctx}: mean|lfc| {np.abs(preds[-1]).mean():.3f}", flush=True)
    np.savez_compressed(PROCESSED / f"effects_ml{SUFFIX}.npz", perts=targets.to_numpy(str), genes=genes.to_numpy(str),
                        lfc=np.stack(preds), contexts=np.array(CONTEXTS))
    print("wrote", PROCESSED / f"effects_ml{SUFFIX}.npz")

elif MODE in ("dir_cv", "dir_final"):
    # direction-confidence head (src/direction.py): which genes may be pushed hard in the generated cells
    from src.direction import direction_inputs, direction_target, fit_direction, pick_threshold

    TARGET_PRECISION = 0.9
    OTHER_SHARE = float(sys.argv[2]) if len(sys.argv) > 2 else 1 / 7   # strength of non-confident genes, relative

    def dir_rows(lines, exclude=()):
        X, y = [], []
        for h in lines:
            _, _, f, mask, _ = fold(h, exclude=exclude)
            X.append(np.column_stack([v[mask] for v in direction_inputs(f).values()]))
            y.append(direction_target(np.nan_to_num(f["mean"])[mask], DATA[h][0][mask]))
        return np.vstack(X), np.concatenate(y)

    if MODE == "dir_cv":
        for test in LABEL_LINES:
            train_lines = [h for h in LABEL_LINES if h != test]
            X, y = dir_rows(train_lines, exclude=(test,))
            clf = fit_direction(X, y)
            t = pick_threshold(clf.predict_proba(X)[:, 1], y, TARGET_PRECISION)
            ridge = fit_ridge(train_lines, exclude=(test,))
            _, _, f, mask, _ = fold(test)
            label, expr = DATA[test]
            prob = clf.predict_proba(np.column_stack([v[mask] for v in direction_inputs(f).values()]))[:, 1]
            yt = direction_target(np.nan_to_num(f["mean"])[mask], label[mask])
            gate = prob >= t
            simple = (f["agree"][mask] == 1) & (f["n"][mask] >= 2)
            print(f"  test {test:8s} threshold {t:.3f}: flagged {gate.mean():.1%} of genes, precision {yt[gate].mean():.3f} "
                  f"(all genes {yt.mean():.3f}; simple 'all sources agree' gate: flagged {simple.mean():.1%}, "
                  f"precision {yt[simple].mean():.3f})")
            c = np.broadcast_to(np.nan_to_num(expr.to_numpy()), mask.shape)[mask]
            to_bulk = lambda l: np.log1p(5 * c * 2.0 ** l) - np.log1p(5 * c)
            yb = to_bulk(label[mask]); r = predict_ridge(ridge, f)[mask]
            for name, p in [("ridge x7, all genes", 7 * r), ("ridge x7 gated, others x1", np.where(gate, 7 * r, r)),
                            ("ridge x7 gated, others 0", np.where(gate, 7 * r, 0.0))]:
                pb = to_bulk(p)
                print(f"    {name:26s}: scorer-space R2 {1 - np.mean((yb - pb) ** 2) / np.mean((yb - yb.mean()) ** 2):+.4f}")
    else:
        X, y = dir_rows(LABEL_LINES)
        clf = fit_direction(X, y)
        t = pick_threshold(clf.predict_proba(X)[:, 1], y, TARGET_PRECISION)
        ridge = fit_ridge(LABEL_LINES)
        preds = []
        for ctx in CONTEXTS:
            a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
            s = np.asarray(a.X.sum(0)).ravel()
            expr = pd.Series(1e4 * s / s.sum(), index=a.var_names.astype(str)).reindex(genes).to_numpy()
            f = features(DATA, expr)
            r = predict_ridge(ridge, f)
            X_all = np.column_stack([v.ravel() for v in direction_inputs(f).values()])
            gate = (clf.predict_proba(X_all)[:, 1] >= t).reshape(r.shape) | OWN
            preds.append(np.where(gate, r, OTHER_SHARE * r).astype(np.float32))
            print(f"context {ctx}: threshold {t:.3f}, confident genes {gate.mean():.2%}", flush=True)
        out = PROCESSED / f"effects_ridge_gated{SUFFIX}.npz"
        np.savez_compressed(out, perts=targets.to_numpy(str), genes=genes.to_numpy(str), lfc=np.stack(preds),
                            contexts=np.array(CONTEXTS))
        print("wrote", out, "-> make_submission.py effects_ridge_gated 7 exact15")

elif MODE in ("pk_cv", "pk_standin", "pk_final"):
    # ridge pattern + a trained per-knockdown strength: knockdowns whose sources agree get stronger predictions.
    # Squared error in scorer space = sum over knockdowns of |d_p|^2 * (alpha_p - best alpha_p)^2 + const, so
    # alpha is fitted by weighted least squares on per-knockdown inputs, which is exactly the mse objective.
    import itertools

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.proxy import predicted_delta, scores

    pos = {x: i for i, x in enumerate(genes)}
    tcol = np.array([pos.get(t, -1) for t in targets])

    def knockdown_inputs(sources, expr, ridge_pred):
        """(300, k) per-knockdown inputs: 1, log size of the predicted change, source consistency (mean pairwise
        cosine of the sources' patterns), number of sources, knockdown of the target in the sources,
        target expression in the target context."""
        e = np.nan_to_num(expr); cols = e >= MIN_EXPR
        rows = {s: v[0] for s, v in sources.items()}
        pres = {s: ~np.isnan(v).all(1) for s, v in rows.items()}
        cons, cnt = np.zeros(len(targets)), np.zeros(len(targets))
        for a, b in itertools.combinations(rows, 2):
            A, B = np.nan_to_num(rows[a][:, cols]), np.nan_to_num(rows[b][:, cols])
            both = pres[a] & pres[b]
            c = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
            cons[both] += c[both]; cnt[both] += 1
        cons = np.where(cnt > 0, cons / np.maximum(cnt, 1), 0.0)
        nsrc = np.sum(list(pres.values()), 0)
        own_kd = np.array([np.nanmean([rows[s][i, tcol[i]] for s in rows
                                       if tcol[i] >= 0 and not np.isnan(rows[s][i, tcol[i]])] or [0.0])
                           for i in range(len(targets))])
        p_expr = np.array([np.log1p(e[c]) if c >= 0 else 0.0 for c in tcol])
        d = np.where(OWN, 0.0, predicted_delta(e, ridge_pred))[:, cols]
        size = np.log1p(np.linalg.norm(d, axis=1))
        return np.column_stack([np.ones(len(targets)), size, cons, nsrc, own_kd, p_expr])

    def knockdown_examples(held, exclude=()):
        """Per-knockdown inputs and the best strength for each knockdown of the hidden line `held`."""
        lab_lines = [h for h in LABEL_LINES if h != held and h not in exclude]
        ridge = fit_ridge(lab_lines, exclude=(held,) + tuple(exclude))
        sources = {k: v for k, v in DATA.items() if k != held and k not in exclude}
        label, expr = DATA[held]
        e = np.nan_to_num(expr.to_numpy()); cols = e >= MIN_EXPR
        r = predict_ridge(ridge, features(sources, e))
        X = knockdown_inputs(sources, e, r)
        d = np.where(OWN, 0.0, predicted_delta(e, r))[:, cols]
        t = np.where(np.isnan(label) | OWN, 0.0, predicted_delta(e, label))[:, cols]
        a, b = (d * t).sum(1), (d * d).sum(1)
        ok = ~np.isnan(label).all(1) & (b > 0)
        return X[ok], a[ok] / b[ok], b[ok], r, label, e

    def fit_strength(examples):
        X = np.vstack([x[0] for x in examples]); y = np.concatenate([x[1] for x in examples])
        w = np.concatenate([x[2] for x in examples])
        coef = np.linalg.solve((X * w[:, None]).T @ X + 1e-3 * np.eye(X.shape[1]), (X * w[:, None]).T @ y)
        glob = float((y * w).sum() / w.sum())
        print("  strength model:", dict(zip(["const", "size", "consistency", "n_sources", "target_kd", "target_expr"],
                                            np.round(coef, 3))), "| global strength", round(glob, 3), flush=True)
        return coef, glob

    def relative_strength(coef, glob, X):
        """Per-knockdown strength divided by the global one, so that SCALE keeps its meaning (v14: x7)."""
        return np.maximum(X @ coef, 0.0) / glob

    if MODE == "pk_cv":
        for test in LABEL_LINES:
            coef, glob = fit_strength([knockdown_examples(h, exclude=(test,)) for h in LABEL_LINES if h != test])
            _, _, _, r, label, e = knockdown_examples(test)
            sources = {k: v for k, v in DATA.items() if k != test}
            rel = relative_strength(coef, glob, knockdown_inputs(sources, e, r))
            for name, p in [("ridge", r), ("ridge x per-knockdown strength", rel[:, None] * r)]:
                from src.proxy import best_scale
                s = best_scale(p, label, e, OWN)
                raw, pds = scores(s * p, label, e, OWN)
                print(f"  test {test:8s} {name:31s}: share explained {1 - raw:.4f}  pds {pds:.3f}", flush=True)
    else:
        held_out = ("k562",) if MODE == "pk_standin" else ()
        lines = [h for h in LABEL_LINES if h not in held_out]
        coef, glob = fit_strength([knockdown_examples(h, exclude=held_out) for h in lines])
        ridge = fit_ridge(lines, exclude=held_out)
        sources = {k: v for k, v in DATA.items() if k not in held_out}
        if MODE == "pk_standin":
            e = DATA["k562"][1].to_numpy()
            r = predict_ridge(ridge, features(sources, e))
            pred = relative_strength(coef, glob, knockdown_inputs(sources, e, r))[:, None] * r
            out = PROCESSED / "effects_ridgepk_noK562.npz"
            np.savez_compressed(out, perts=targets.to_numpy(str), genes=genes.to_numpy(str), lfc=pred.astype(np.float32))
        else:
            preds = []
            for ctx in CONTEXTS:
                a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
                s = np.asarray(a.X.sum(0)).ravel()
                e = pd.Series(1e4 * s / s.sum(), index=a.var_names.astype(str)).reindex(genes).to_numpy()
                r = predict_ridge(ridge, features(DATA, e))
                rel = relative_strength(coef, glob, knockdown_inputs(DATA, e, r))
                preds.append((rel[:, None] * r).astype(np.float32))
                print(f"context {ctx}: strength range {rel.min():.2f}-{rel.max():.2f}, median {np.median(rel):.2f}", flush=True)
            out = PROCESSED / f"effects_ridgepk{SUFFIX}.npz"
            np.savez_compressed(out, perts=targets.to_numpy(str), genes=genes.to_numpy(str), lfc=np.stack(preds),
                                contexts=np.array(CONTEXTS))
        print("wrote", out, flush=True)
