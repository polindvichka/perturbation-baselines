"""Neural transfer model: attention over source lines, trained to predict a knockdown's whole response in an unseen
line, with a loss that matches the scorer (scorer-space squared error + a pds-style contrastive term).

Why (compared with transfer_model.py, 12 weights): the ridge trusts every source the same for every gene and fits
each (knockdown, gene) on its own. Here
  - each source line's measured response to the knockdown is one token; the target context (its control expression)
    is another; a small transformer lets them look at each other, so trust can depend on the knockdown, the context
    and on how the sources agree
  - trust is given per gene: gate(source, gene) = softplus(<U h_source, E_gene> + bias), with a learned embedding
    E_gene, so e.g. K562 can be trusted for some gene programmes and not for others
  - the loss rewards what pds measures: each predicted pattern should be closer (cosine, scorer space) to its own
    real pattern than to the other knockdowns' in the batch (InfoNCE), next to the usual squared error
Training data: hide one line H (K562, HCT116, HEK293T) at a time; the other label lines are predicted from the rest
(H never used), on ALL knockdowns they share with at least one source except the 300 final-phase targets.
Evaluation: H's 300 final-phase targets (never trained on), with the offline proxies of src/proxy.py (raw mse at the
best single scale, pds rank; lower is better), next to the plain source mean.

Modes
  cv      as above
  export  train for the current panel and write PROCESSED/effects_attn{SUFFIX}.npz (same layout as the ridge's):
          a stand-in panel made from line L hides L (as a label line and as a source); test (D/E/F) trains on all
          three label lines. Each context's prediction is rescaled to the scorer-space size of the ridge's
          effects_ridgepk{SUFFIX}.npz (so make_submission.py SCALE keeps its meaning), and every knockdown's own
          gene is copied from it; genes outside the model's gene set stay 0. N_SEEDS > 1 averages that many
          models (seeds SEED, SEED+1, ...) -> effects_attn_ens{N}{SUFFIX}.npz.

Usage: [SEED=0] [N_SEEDS=1] VCC_PANEL=test python scripts/attn_model.py cv|export [EPOCHS=60] [LAMBDA=0.5]
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import CONTEXTS, CONTROLS, HOLDOUT, PROCESSED, SUFFIX
from src.proxy import best_scale, predicted_delta, scores
from map_model import DATA, EXPR, LABEL_LINES, final_targets, genes, load

MODE = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
LAMBDA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
D, D_GENE, K_BASIS, TAU, BATCH = 64, 16, 64, 0.1, 64
SEED = int(os.environ.get("SEED", 0))
N_SEEDS = int(os.environ.get("N_SEEDS", 1))                             # export: average this many models
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)

LAB = {k: v[0] for k, v in DATA.items()}
h1 = [load(f)[0] for f in ["h1_train_delta.npz", "h1_val_delta.npz", "h1_test_delta.npz"]]
LAB["h1"] = h1[0].combine_first(h1[1]).combine_first(h1[2])
SOURCES = list(LAB)                                                     # k562, hct116, hek293t, cd4, h1
G = np.flatnonzero(np.any([np.nan_to_num(e) >= 0.05 for e in EXPR.values()], 0))   # expressed in some label line
own_col = {g: i for i, g in enumerate(genes.to_numpy()[G])}
tcols = np.array([own_col[t] for t in final_targets if t in own_col])
print(f"genes {len(G)}, sources {SOURCES}, device {DEV}", flush=True)


def weight(e):
    """sqrt of the scorer's bulk weight per gene (sensitivity of log1p(5 CP10K) to a log2 FC)."""
    c = np.nan_to_num(e[G])
    return (np.log(2) * 5 * c / (1 + 5 * c)).astype(np.float32)


def batch_arrays(perts, srcs, H):
    """X (n, S, G) source values (0 = missing), M (n, S) source present, Y (n, G) label (NaN = missing)."""
    X = np.zeros((len(perts), len(SOURCES), len(G)), np.float32); M = np.zeros((len(perts), len(SOURCES)), bool)
    for k, s in enumerate(SOURCES):
        if s not in srcs:
            continue
        v = LAB[s].reindex(perts).to_numpy(np.float32)[:, G]
        M[:, k] = ~np.isnan(v).all(1)
        X[:, k] = np.clip(np.nan_to_num(v), -5, 5)                    # a few noisy genes reach +-10
    Y = LAB[H].reindex(perts).to_numpy(np.float32)[:, G]
    return X, M, Y


class Model(nn.Module):
    """v2: starts exactly at the source mean and learns only corrections (v1, free per-gene gates and a 11k x 128
    input layer, memorized the two training lines: worse than the plain mean on the hidden one)."""

    def __init__(self, basis, n_src):
        super().__init__()
        self.register_buffer("P", basis)                                # (G, K) main response patterns
        K = basis.shape[1]
        self.inp = nn.Linear(K, D)                                      # a source's response (K patterns) -> token
        self.src_emb = nn.Embedding(n_src, D)
        self.ctx = nn.Linear(K, D)                                      # target control expression -> token
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.gene = nn.Parameter(0.02 * torch.randn(basis.shape[0], D_GENE))   # learned gene embedding
        self.U = nn.Linear(D, D_GENE)
        nn.init.zeros_(self.U.weight); nn.init.zeros_(self.U.bias)      # gates start at exactly 1
        self.expr_w = nn.Parameter(torch.zeros(1))

    def forward(self, X, M, te, w):
        n, S, _ = X.shape
        tok = self.inp((X * w) @ self.P) + self.src_emb.weight[None]    # (n, S, D)
        ctx = self.ctx((te - te.mean(1, keepdim=True)) @ self.P)[:, None]
        h = self.enc(torch.cat([ctx, tok], 1), src_key_padding_mask=torch.cat(
            [torch.zeros(n, 1, dtype=torch.bool, device=X.device), ~M], 1))
        gate = 1 + 0.5 * torch.tanh(self.U(h[:, 1:]) @ self.gene.T + self.expr_w * te[:, None])
        gate = gate * M[..., None]
        return (gate * X).sum(1) / M.sum(1, keepdim=True).clamp(min=1)    # gate 1 everywhere = source mean


def losses(pred, Y, w, own):
    """Scorer-space weighted squared error and InfoNCE over the batch (own-target genes left out)."""
    ok = ~torch.isnan(Y) & ~own
    y = torch.nan_to_num(Y)
    se = (((pred - y) * w) ** 2 * ok).sum() / ((y * w) ** 2 * ok).sum().clamp(min=1e-6)
    P = F.normalize(pred * w * ok, dim=1); R = F.normalize(y * w * ok, dim=1)
    logits = P @ R.T / TAU
    nce = F.cross_entropy(logits, torch.arange(len(P), device=P.device))
    return se, nce


def own_mask(perts):
    m = np.zeros((len(perts), len(G)), bool)
    for i, p in enumerate(perts):
        if p in own_col:
            m[i, own_col[p]] = True
    return m


def fold_data(H_label, exclude):
    srcs = [s for s in SOURCES if s != H_label and s not in exclude]
    have = set().union(*[set(LAB[s].index) for s in srcs])
    perts = np.array(sorted(set(LAB[H_label].index) & have - set(final_targets)))
    X, M, Y = batch_arrays(perts, srcs, H_label)
    te = np.log1p(np.nan_to_num(EXPR[H_label][G])).astype(np.float32)
    return dict(X=X, M=M, Y=Y, own=own_mask(perts), w=weight(EXPR[H_label]), te=te, n=len(perts))


def train(folds):
    Yw = np.vstack([np.nan_to_num(f["Y"]) * f["w"] for f in folds])
    basis = np.linalg.svd(Yw - Yw.mean(0), full_matrices=False)[2][:K_BASIS].T.astype(np.float32)
    model = Model(torch.tensor(basis), len(SOURCES)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    T = {k: [torch.tensor(f[k], device=DEV) for f in folds] for k in ["X", "M", "Y", "own", "w", "te"]}
    for ep in range(EPOCHS):
        model.train(); tot = np.zeros(2); nb = 0
        for fi in rng.permutation(len(folds)):
            order = torch.randperm(folds[fi]["n"], device=DEV)
            for b in range(0, len(order), BATCH):
                i = order[b:b + BATCH]
                if len(i) < 8:
                    continue
                M = T["M"][fi][i].clone()
                drop = (torch.rand_like(M, dtype=torch.float) < 0.3) & M  # source dropout, keep at least one
                keep_one = M & (drop.sum(1, keepdim=True) == M.sum(1, keepdim=True))
                M = M & ~(drop & ~keep_one)
                te = T["te"][fi][None].expand(len(i), -1)
                pred = model(T["X"][fi][i] * M[..., None], M, te, T["w"][fi][None])
                se, nce = losses(pred, T["Y"][fi][i], T["w"][fi][None], T["own"][fi][i])
                loss = se + LAMBDA * nce
                opt.zero_grad(); loss.backward(); opt.step()
                tot += [se.item(), nce.item()]; nb += 1
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"    epoch {ep:3d}: train se {tot[0] / nb:.4f}  nce {tot[1] / nb:.3f}", flush=True)
    return model


def evaluate(model, H):
    perts = np.array([t for t in final_targets if t in LAB[H].index])
    srcs = [s for s in SOURCES if s != H]
    X, M, Y = batch_arrays(perts, srcs, H)
    e = EXPR[H][G]; own = own_mask(perts)
    keep = M.any(1)
    te = torch.tensor(np.log1p(np.nan_to_num(e)).astype(np.float32), device=DEV)
    model.eval()
    with torch.no_grad():
        w = torch.tensor(weight(EXPR[H]), device=DEV)[None]
        pred = model(torch.tensor(X, device=DEV), torch.tensor(M, device=DEV), te[None].expand(len(perts), -1),
                     w).cpu().numpy()
    with np.errstate(all="ignore"):
        mean = np.nan_to_num(np.nanmean(np.where(M[..., None], X, np.nan), 1))
    out = {}
    for name, p in [("source mean", mean), ("attention model", pred),
                    ("attention, common change removed", pred - pred[keep].mean(0, keepdims=True))]:
        s = best_scale(p[keep], Y[keep], e, own[keep])
        out[name] = scores(s * p[keep], Y[keep], e, own[keep])
        print(f"  {name:34s}: raw mse {out[name][0]:.4f}  pds {out[name][1]:.3f}  (scale {s:.2f})", flush=True)
    return out


if MODE == "cv":
    for H in LABEL_LINES:
        print(f"hidden {H}: training on", [L for L in LABEL_LINES if L != H], flush=True)
        folds = [fold_data(L, exclude=(H,)) for L in LABEL_LINES if L != H]
        print("  knockdowns per training line:", [f["n"] for f in folds], flush=True)
        model = train(folds)
        evaluate(model, H)

if MODE == "export":
    import anndata as ad
    import pandas as pd
    hide = (HOLDOUT,) if HOLDOUT else ()
    lines = [L for L in LABEL_LINES if L not in hide]
    print("training on", lines, "hiding", hide, f"seeds {SEED}..{SEED + N_SEEDS - 1}", flush=True)
    folds = [fold_data(L, exclude=hide) for L in lines]
    models = []
    for sd in range(SEED, SEED + N_SEEDS):
        torch.manual_seed(sd); rng = np.random.default_rng(sd)
        models.append(train(folds))
    ref = np.load(PROCESSED / f"effects_ridgepk{SUFFIX}.npz")
    perts = ref["perts"].astype(str)
    assert list(ref["genes"]) == list(genes), "ridge effects are on a different gene axis"
    srcs = [s for s in SOURCES if s not in hide]
    X, M, _ = batch_arrays(perts, srcs, LABEL_LINES[0])
    gpos = {g: i for i, g in enumerate(genes)}
    own_full = np.array([gpos.get(p, -1) for p in perts])
    out = []
    for m in models:
        m.eval()
    for k, ctx in enumerate(CONTEXTS):
        a = ad.read_h5ad(CONTROLS / f"context_{ctx}.h5ad")
        tot = np.asarray(a.X.sum(0)).ravel()
        e = pd.Series(1e4 * tot / tot.sum(), index=a.var_names.astype(str)).reindex(genes).to_numpy()
        te = torch.tensor(np.log1p(np.nan_to_num(e[G])).astype(np.float32), device=DEV)
        with torch.no_grad():                                           # mean over the ensemble
            p = np.mean([m(torch.tensor(X, device=DEV), torch.tensor(M, device=DEV), te[None].expand(len(perts), -1),
                           torch.tensor(weight(e), device=DEV)[None]).cpu().numpy() for m in models], 0)
        full = np.zeros((len(perts), len(genes)), np.float32); full[:, G] = p
        r = ref["lfc"][k].astype(np.float32)
        own = np.zeros_like(full, bool); own[np.arange(len(perts))[own_full >= 0], own_full[own_full >= 0]] = True
        size = lambda l: np.linalg.norm(np.where(own, 0.0, predicted_delta(e, l)))
        full *= size(r) / max(size(full), 1e-12)
        full[own] = r[own]
        out.append(full)
        print(f"context {ctx}: rescaled to the ridge's size; {int((~M.any(1)).sum())} knockdowns without a source",
              flush=True)
    path = PROCESSED / f"effects_attn{f'_ens{N_SEEDS}' if N_SEEDS > 1 else ''}{SUFFIX}.npz"
    np.savez_compressed(path, perts=perts, genes=genes.to_numpy(str), lfc=np.stack(out), contexts=np.array(CONTEXTS))
    print("wrote", path, flush=True)
