"""Knockdown-effect table from the KOLF2.1J iPSC Perturbation Cell Atlas (Doctor, Nourreddine, Mali), CRISPRi.

The full single-cell data is 176 GB, so this uses the summary the authors publish for their web viewer
(github.com/y-doctor/KOLF2.1J_Perturbation_Cell_Atlas, MIT licence): for each of 10,167 knockdowns, the mean
z-score of every gene relative to non-targeting cells, stored as int8 with a step of 0.04. Only the rows of the
challenge targets (of every panel) are fetched (one HTTP range request each, ~38 KB).

These are z-scores, not log2 fold changes, so they are used as a separate model input (like CD4), never mixed
into the cross-line average. Writes PROCESSED/kolf_z.npz: perts (P,), genes (18533,) on the challenge axis,
lfc (P, 18533) holding the z-scores (NaN = gene not in the table), n_cells (P,) placeholder.

Usage: python scripts/build_kolf_table.py
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTROLS, PROCESSED, all_targets

BASE = "https://raw.githubusercontent.com/y-doctor/KOLF2.1J_Perturbation_Cell_Atlas/HEAD/docs/data/panel/all/"
genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"]
targets = all_targets()                                     # every panel's knockdowns, not only this one's

meta = json.load(urllib.request.urlopen(BASE + "meta.json"))
row_of = {p: i for i, p in enumerate(meta["perts"])}
have = [t for t in targets if t in row_of]
print(f"{len(have)} of {len(targets)} challenge targets are in the KOLF panel", flush=True)


def fetch(p):
    i = row_of[p]
    start = (i % meta["chunk_size"]) * meta["row_bytes"]
    req = urllib.request.Request(BASE + f"chunk_{i // meta['chunk_size']}.bin",
                                 headers={"Range": f"bytes={start}-{start + meta['row_bytes'] - 1}"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    assert len(raw) == meta["row_bytes"], (p, len(raw))
    return np.frombuffer(raw, dtype=np.int8).astype(np.float32) * meta["scale"]


with ThreadPoolExecutor(16) as pool:
    rows = list(pool.map(fetch, have))

col = pd.Index(genes).get_indexer(meta["genes"])            # KOLF gene -> challenge column (-1 = absent)
ok = col >= 0
z = np.full((len(have), len(genes)), np.nan, np.float32)
for r, v in enumerate(rows):
    z[r, col[ok]] = v[ok]
pos = {g: i for i, g in enumerate(genes)}
for r, p in enumerate(have):
    if p in pos:
        if r < 3:                                           # sanity check: the knocked-down gene should be low
            print(f"  {p}: own z {z[r, pos[p]]:+.2f}", flush=True)
        z[r, pos[p]] = np.nan                               # the knocked-down gene itself
np.savez_compressed(PROCESSED / "kolf_z.npz", perts=np.array(have), genes=genes.to_numpy(str), lfc=z,
                    n_cells=np.full(len(have), 100))
print(f"wrote {PROCESSED / 'kolf_z.npz'}: {len(have)} knockdowns, {int(ok.sum())} challenge genes measured")
