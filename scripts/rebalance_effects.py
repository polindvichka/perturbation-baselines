"""Scale the shared and the gene-specific part of predicted effects separately.

shared   = the average effect over all target genes (what any of these knockdowns does)
specific = each gene's effect minus that average
new lfc  = SHARED_SCALE * shared + SPECIFIC_SCALE * specific

Usage: python scripts/rebalance_effects.py IN OUT SHARED_SCALE SPECIFIC_SCALE
  e.g. python scripts/rebalance_effects.py effects_v1 effects_v1_s1.25_g2.5 1.25 2.5
Reads and writes PROCESSED/{name}.npz; use the result with make_submission.py at SCALE 1.0.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROCESSED

src, dst, shared_scale, specific_scale = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
z = dict(np.load(PROCESSED / f"{src}.npz"))
lfc = z["lfc"].astype(np.float64)                   # (300 targets, 18533 genes)
shared = lfc.mean(axis=0, keepdims=True)
specific = lfc - shared
z["lfc"] = (shared_scale * shared + specific_scale * specific).astype(np.float32)
np.savez_compressed(PROCESSED / f"{dst}.npz", **z)

print(f"shared part: mean |lfc| {np.abs(shared).mean():.4f}, gene-specific part: mean |lfc| {np.abs(specific).mean():.4f}")
print(f"wrote {PROCESSED / f'{dst}.npz'}")
