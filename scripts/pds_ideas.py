"""Quick offline check of two pds ideas on the 300 final-phase targets, hiding one public line at a time.

  neighbours  CRISPRi silences the promoter region, so genes whose start site lies close to the target's are often
              knocked down too. The scorer leaves out every target gene from pds (vcc2026.yaml, exclusion_scope
              panel), but not these neighbours, so they mark each knockdown. Prints how much neighbours drop, by
              distance, in the hidden line and in the average of the other lines; then sets every neighbour to
              the drop seen in the other label lines for its distance.
  common      subtract the change shared by all knockdowns (the mean over knockdowns, per gene), so each
              prediction differs more from the others.
pds as in src/proxy.py (rank of the right knockdown, lower is better), all target genes left out like the scorer.
Gene start sites: GENCODE v46 basic, $VCC_DATA/data/gencode/gencode.v46.basic.annotation.gtf.gz (from
ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/).

Usage: python scripts/pds_ideas.py
"""
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import ROOT
from src.proxy import scores
from map_model import DATA, EXPR, LABEL_LINES, final_targets, genes

BINS = [0, 2_000, 10_000, 50_000, 200_000]
GTF = ROOT / "data" / "gencode" / "gencode.v46.basic.annotation.gtf.gz"

tss = {}
with gzip.open(GTF, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        c = line.split("\t", 9)
        if c[2] != "gene":
            continue
        name = re.search(r'gene_name "([^"]+)"', c[8]).group(1)
        if name not in tss:
            tss[name] = (c[0], int(c[3]) if c[6] == "+" else int(c[4]))
g = genes.to_numpy()
chrom = np.array([tss.get(x, ("?", -1))[0] for x in g])
pos = np.array([tss.get(x, ("?", -1))[1] for x in g], dtype=np.int64)
col = {x: i for i, x in enumerate(g)}
tcols = np.array([col[t] for t in final_targets if t in col])
print(f"start sites for {(pos >= 0).sum()} of {len(g)} genes", flush=True)


def neighbour_bins(perts):
    """(knockdowns, genes) int: distance bin of each gene to the knockdown's target (-1 = not a neighbour)."""
    out = np.full((len(perts), len(g)), -1, np.int8)
    for i, p in enumerate(perts):
        if p not in tss:
            continue
        ch, x = tss[p]
        d = np.abs(pos - x)
        near = (chrom == ch) & (pos >= 0) & (d < BINS[-1]) & (g != p)
        out[i, near] = np.digitize(d[near], BINS) - 1
    return out


def bin_means(lfc, nb, expr):
    ok = np.nan_to_num(expr) >= 0.05
    return [float(np.nanmean(lfc[(nb == b) & ok[None, :]])) if ((nb == b) & ok[None, :]).any() else np.nan
            for b in range(len(BINS) - 1)]


LAB = {k: DATA[k][0] for k in DATA}
for H in LABEL_LINES:
    perts = np.array([t for t in final_targets if t in LAB[H].index])
    real = LAB[H].reindex(perts).to_numpy(np.float64)
    e = EXPR[H]
    others = [k for k in LAB if k != H]
    with np.errstate(all="ignore"):
        S = np.stack([LAB[k].reindex(perts).to_numpy(np.float64) for k in others])
        base = np.nan_to_num(np.nanmean(S, 0))
    nb = neighbour_bins(perts)
    print(f"\nhidden {H}: mean log2 FC of neighbours by distance {BINS[1:]} (bp)")
    print("  hidden line   ", np.round(bin_means(real, nb, e), 3))
    print("  others' mean  ", np.round(bin_means(base, nb, e), 3))
    # neighbour drop learned from the other label lines only
    learn = []
    for L in [l for l in LABEL_LINES if l != H]:
        pl = np.array([t for t in final_targets if t in LAB[L].index])
        learn.append(bin_means(LAB[L].reindex(pl).to_numpy(np.float64), neighbour_bins(pl), EXPR[L]))
    drop = np.nanmean(np.array(learn), 0)
    own = np.zeros_like(real, bool); own[:, tcols] = True                 # every target gene left out
    variants = {"others' mean": base}
    nbr = base.copy()
    for b in range(len(BINS) - 1):
        m = nb == b
        nbr[m] = np.minimum(nbr[m], drop[b]) if drop[b] < 0 else nbr[m]
    variants["+ neighbour drop"] = nbr
    variants["- common change"] = base - base.mean(0, keepdims=True)
    variants["both"] = nbr - nbr.mean(0, keepdims=True)
    for name, p in variants.items():
        _, pds = scores(p, real, e, own)
        print(f"  {name:18s}: pds {pds:.3f}", flush=True)
