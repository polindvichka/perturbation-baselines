"""Subtract (part of) the change that every knockdown shares from the predicted effects (a pds idea).

Per context and gene, the common change is the mean predicted log2 FC over all knockdowns. Part of it is real
(the generic response to any knockdown), but in the prediction it makes every knockdown look alike, which blurs
pds (each prediction is ranked against all real changes by cosine). scripts/pds_ideas.py showed a small, steady
pds gain from removing it on the offline proxy; this file makes the effects so the real scorer can judge it.

Usage: [VCC_PANEL=...] python scripts/common_change.py EFFECTS FRACTION      e.g. effects_ridgepk_h1 0.5
Writes PROCESSED/{EFFECTS}_cc{100 x FRACTION}.npz (same layout).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROCESSED

EFFECTS, FRACTION = sys.argv[1], float(sys.argv[2])

z = np.load(PROCESSED / f"{EFFECTS}.npz")
lfc = z["lfc"].astype(np.float64)
common = np.nanmean(lfc, axis=-2, keepdims=True)                          # per context and gene
out = {k: z[k] for k in z.files}
out["lfc"] = (lfc - FRACTION * common).astype(z["lfc"].dtype)
path = PROCESSED / f"{EFFECTS}_cc{round(100 * FRACTION)}.npz"
np.savez_compressed(path, **out)
print(f"common change: mean |.| {np.nanmean(np.abs(common)):.4f}, largest {np.nanmax(np.abs(common)):.3f}; "
      f"wrote {path.name}")
