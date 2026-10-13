"""Can a model that never sees a gene's own knockdown predict which gene is which?

Training: every K562 knockdown EXCEPT our 300 challenge genes.
Features (compared on exactly the same genes):
    esm      the gene's ESM-2 protein embedding
    coexpr   how the gene co-varies with 2000 anchor genes across K562 control cells
    go       Gene Ontology annotations (reduced)
    string   STRING interaction partners (reduced)
    pathway  go + string joined
    all      every set above joined
Testing:  the 300 challenge genes. Their predicted effects are compared with the real
          effects measured in K562, HCT116 and HEK293T.

Score (same as transfer_check.py): for each test gene, how many of the other genes' real
profiles does the prediction match worse than it matches the right one? 1.0 = always the
right one, 0.5 = chance. The average effect over all test genes is removed first, so the
score only measures what is specific to each gene.

Needs coexpr_K562.npz (scripts/build_coexpr_features.py) and pathway_features.npz (scripts/build_pathway_features.py).

Usage: python scripts/heldout_model.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DeltaTable

CONTROL_DIR = Path("/root/vcc/controls")
PROCESSED = Path("/workspace/processed")
MIN_CELLS = 20
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
KNN_KS = [5, 20, 50, 200]
FEATURE_SETS = ["esm", "coexpr", "go", "string", "pathway", "all"]

genes = pd.read_csv(CONTROL_DIR / "gene_names.csv")["gene_name"].tolist()
gene_pos = {g: i for i, g in enumerate(genes)}
targets = pd.read_csv(CONTROL_DIR / "pert_counts.csv")["target_gene"].tolist()
target_set = set(targets)

# ---- K562 table (training data) ---------------------------------------------------
table = DeltaTable.load(str(PROCESSED / "k562_delta.npz"))
rows = None
for name in ("perturbations", "perts", "perturbation", "rows", "index", "labels", "names"):
    if hasattr(table, name):
        rows = list(getattr(table, name))
        break
if rows is None:
    raise SystemExit(f"Cannot find the row names. Fields on DeltaTable: {list(vars(table))}")
k_genes = list(table.genes)
k_col = {g: i for i, g in enumerate(k_genes)}                       # gene -> K562 column
k_to_challenge = pd.Index(genes).get_indexer(k_genes)               # K562 column -> challenge column
k_row = {r: i for i, r in enumerate(rows)}
k_delta = np.nan_to_num(np.asarray(table.delta, dtype=np.float32))  # own-gene NaN -> 0

# ---- features ---------------------------------------------------------------------
z = np.load(PROCESSED / "esm2_35m.npz", allow_pickle=True)
esm_of = {g: i for i, g in enumerate(list(z["genes"]))}
esm = z["emb"].astype(np.float32)

c = np.load(PROCESSED / "coexpr_K562.npz", allow_pickle=True)
co_of = {g: i for i, g in enumerate(list(c["genes"]))}
coexpr = c["feat"].astype(np.float32)

p = np.load(PROCESSED / "pathway_features.npz", allow_pickle=True)   # from scripts/build_pathway_features.py
pw_of = {g: i for i, g in enumerate(list(p["genes"]))}
go_feat, string_feat = p["go"].astype(np.float32), p["string"].astype(np.float32)
zero_row = np.zeros(go_feat.shape[1], dtype=np.float32)               # genes with no annotation get zeros


def pathway_rows(table, gene_list):
    return np.stack([table[pw_of[g]] if g in pw_of else zero_row for g in gene_list])


has_both = lambda g: g in esm_of and g in co_of          # same gene set as before, so numbers stay comparable
train_all = [r for r in rows if has_both(r) and r not in target_set]
test_genes = [t for t in targets if has_both(t)]
print(f"K562 knockdowns for training (no challenge genes, ESM and co-expression available): {len(train_all)}")
print(f"challenge genes with those features (to predict): {len(test_genes)} of {len(targets)}")
print(f"  with GO annotation: train {sum(bool(p['has_go'][pw_of[g]]) for g in train_all if g in pw_of)}/{len(train_all)}, "
      f"test {sum(bool(p['has_go'][pw_of[g]]) for g in test_genes if g in pw_of)}/{len(test_genes)};  "
      f"with STRING partners: train {sum(bool(p['has_string'][pw_of[g]]) for g in train_all if g in pw_of)}, "
      f"test {sum(bool(p['has_string'][pw_of[g]]) for g in test_genes if g in pw_of)}")
print(f"  of those, with a K562 knockdown to compare against: {sum(g in k_row for g in test_genes)}")


def block(M, ref):
    """Centre on the training mean and scale so the average squared row length is 1."""
    mu = ref.mean(axis=0)
    scale = np.sqrt(((ref - mu) ** 2).sum(axis=1).mean())
    return (M - mu) / max(scale, 1e-9)


def make_features(train_genes):
    """Returns f(set_name, gene_list) -> matrix; scaling is fixed from train_genes only."""
    sources = {
        "esm": lambda gl: esm[[esm_of[g] for g in gl]],
        "coexpr": lambda gl: coexpr[[co_of[g] for g in gl]],
        "go": lambda gl: pathway_rows(go_feat, gl),
        "string": lambda gl: pathway_rows(string_feat, gl),
    }
    refs = {k: fn(train_genes) for k, fn in sources.items()}

    def f(kind, gene_list):
        blocks = {k: block(fn(gene_list), refs[k]) for k, fn in sources.items()}
        if kind == "all":
            return np.hstack(list(blocks.values()))
        if kind == "pathway":
            return np.hstack([blocks["go"], blocks["string"]])
        return blocks[kind]
    return f


def zero_own(P, gene_list, columns):
    """Set each gene's own column to 0 (the scorer ignores it). columns maps gene -> column or None."""
    for i, g in enumerate(gene_list):
        col = columns.get(g)
        if col is not None:
            P[i, col] = 0
    return P


def zrows(M):
    M = M - M.mean(axis=1, keepdims=True)
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)


def retrieval(P, T):
    """P, T: (genes x columns), same row order. Returns mean rank score and top-1 share."""
    P, T = P - P.mean(axis=0), T - T.mean(axis=0)
    C = zrows(P) @ zrows(T).T
    own = np.diag(C)
    n = len(own)
    return float(((C < own[:, None]).sum(axis=1) / (n - 1)).mean()), float((C.argmax(axis=1) == np.arange(n)).mean())


# ---- the two models ----------------------------------------------------------------
def fit_ridge(Xtr, Ytr, alpha):
    """Features are already centred; the targets are centred here and the mean added back."""
    ym = Ytr.mean(axis=0)
    Xc = Xtr.astype(np.float64)
    if Xc.shape[1] <= Xc.shape[0]:
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ (Ytr - ym).astype(np.float64))
    else:                                                    # dual form when features outnumber genes
        W = Xc.T @ np.linalg.solve(Xc @ Xc.T + alpha * np.eye(Xc.shape[0]), (Ytr - ym).astype(np.float64))
    return lambda X: (X @ W + ym).astype(np.float32)


def fit_knn(Xtr, Ytr, k):
    unit = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    Xn = unit(Xtr)

    def predict(X):
        S = unit(X) @ Xn.T
        nearest = np.argpartition(-S, k, axis=1)[:, :k]
        return np.stack([Ytr[idx].mean(axis=0) for idx in nearest])
    return predict


FIT = {"ridge": fit_ridge, "knn": fit_knn}
GRID = {"ridge": RIDGE_ALPHAS, "knn": KNN_KS}

# ---- pick the settings on held-back TRAINING genes (never on the 300) ------------------
rng = np.random.default_rng(0)
order = rng.permutation(len(train_all))
n_val = max(len(train_all) // 10, 50)
val_genes = [train_all[i] for i in order[:n_val]]
fit_genes = [train_all[i] for i in order[n_val:]]
feats_fit = make_features(fit_genes)
Yfit = k_delta[[k_row[g] for g in fit_genes]]
Tval = k_delta[[k_row[g] for g in val_genes]]
print(f"\nsettings are chosen on {n_val} held-back training genes (score among those genes):")
best, best_val = {}, {}
for kind in FEATURE_SETS:
    Xfit, Xval = feats_fit(kind, fit_genes), feats_fit(kind, val_genes)
    for model in FIT:
        results = []
        for setting in GRID[model]:
            P = zero_own(FIT[model](Xfit, Yfit, setting)(Xval), val_genes, k_col)
            results.append((retrieval(P, Tval)[0], setting))
        best[(kind, model)] = max(results)[1]
        best_val[(kind, model)] = max(results)[0]
        print(f"  {kind:6s} {model:5s}: " + "  ".join(f"{s}:{sc:.3f}" for sc, s in results)
              + f"   -> chosen {best[(kind, model)]}")

# ---- final models: train on ALL non-challenge genes, predict the 300 -----------------
feats_all = make_features(train_all)
Ytr = k_delta[[k_row[g] for g in train_all]]
predictions = {}
for kind in FEATURE_SETS:
    Xtr, Xte = feats_all(kind, train_all), feats_all(kind, test_genes)
    for model in FIT:
        predictions[f"{model}_{kind}"] = FIT[model](Xtr, Ytr, best[(kind, model)])(Xte)


def to_challenge(M_k562_space):
    full = np.full((len(M_k562_space), len(genes)), np.nan, dtype=np.float32)
    ok = k_to_challenge >= 0
    full[:, k_to_challenge[ok]] = M_k562_space[:, ok]
    return full


# save the predicted effects (K562 gene space) so they can be used to build predictions for the scorer.
# 'best' is picked ONLY from the held-back TRAINING genes, never from the challenge genes.
top = max(best_val, key=best_val.get)
best_name = f"{top[1]}_{top[0]}"
np.savez(PROCESSED / "model_predictions.npz", test_genes=np.array(test_genes, dtype=object),
         k_genes=np.array(k_genes, dtype=object), best=best_name,
         **{f"pred_{name}": P.astype(np.float32) for name, P in predictions.items()})
print(f"\nsaved predicted effects for {len(test_genes)} genes; best on held-back training genes: "
      f"{best_name} ({best_val[top]:.3f})")

pred_full = {name: to_challenge(P) for name, P in predictions.items()}


# real effects of the challenge genes in each line (challenge-gene columns, NaN = not measured)
def load_xatlas(split):
    zz = np.load(PROCESSED / f"xatlas_{split}_delta.npz")
    ok = zz["n_cells"] >= MIN_CELLS
    perts = [p for p, keep in zip(list(zz["perts"]), ok) if keep]
    return {p: row for p, row in zip(perts, zz["delta"][ok])}


k562_truth = {}
for g in targets:
    if g in k_row:
        row = np.full(len(genes), np.nan, dtype=np.float32)
        ok = k_to_challenge >= 0
        row[k_to_challenge[ok]] = k_delta[k_row[g]][ok]
        k562_truth[g] = row
lines = {"K562": k562_truth, "HCT116": load_xatlas("HCT116"), "HEK293T": load_xatlas("HEK293T")}
donor_for = {"K562": "HCT116", "HCT116": "K562", "HEK293T": "K562"}   # for the lookup reference

# ---- compare, on identical genes and columns for every method --------------------------
test_index = {g: i for i, g in enumerate(test_genes)}
out = []
for recipient, truth in lines.items():
    donor_name = donor_for[recipient]
    donor = lines[donor_name]
    shared = [g for g in test_genes if g in truth and g in donor]
    T = np.stack([truth[g] for g in shared])
    L = np.stack([donor[g] for g in shared])
    Ps = {name: M[[test_index[g] for g in shared]] for name, M in pred_full.items()}
    measured = ~np.isnan(T).all(axis=0) & ~np.isnan(L).all(axis=0)
    for M in Ps.values():
        measured &= ~np.isnan(M).all(axis=0)
    own_cols = {g: gene_pos[g] for g in shared}
    keep_pos = np.cumsum(measured) - 1                           # challenge column -> position among measured columns

    def prep(M):
        M = np.nan_to_num(M[:, measured])
        for i, g in enumerate(shared):
            col = own_cols[g]
            if measured[col]:
                M[i, keep_pos[col]] = 0
        return M
    Tm = prep(T)
    row = {"recipient": recipient, "genes": len(shared), "columns": int(measured.sum()),
           "lookup_from": donor_name}
    s, t = retrieval(prep(L), Tm)
    row.update({"lookup": round(s, 3), "lookup_top1": round(t, 3)})
    for name, M in Ps.items():
        row[name] = round(retrieval(prep(M), Tm)[0], 3)
    row["chance_top1"] = round(1 / len(shared), 3)
    out.append(row)

summary = pd.DataFrame(out).set_index("recipient")
methods = ["lookup"] + [c for c in summary.columns if c.startswith(("ridge_", "knn_"))]
print("\ngenes compared:", summary["genes"].to_dict(), " columns:", summary["columns"].to_dict())
print("lookup is copied from:", summary["lookup_from"].to_dict())
print("\nscore (1.0 = right gene always best match, 0.5 = chance; about +-0.02 is noise):")
print(summary[methods].T.to_string())
print(f"\nlookup top-1 share: {summary['lookup_top1'].to_dict()}   chance top-1: {summary['chance_top1'].to_dict()}")
print("lookup copies the same gene's row from another line; every other row never sees that gene's row.")