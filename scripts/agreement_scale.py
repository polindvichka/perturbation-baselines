"""Amplify predicted effects more where the source lines agree on the direction.

agree    = at least 2 sources measured the change and all of them saw the same direction
new lfc  = AGREE_SCALE * lfc where agree, OTHER_SCALE * lfc elsewhere

Usage: python scripts/agreement_scale.py IN OUT AGREE_SCALE OTHER_SCALE
  e.g. python scripts/agreement_scale.py effects_v1 effects_v1_agree2.25_other1 2.25 1.0
Needs n_up / n_down in IN (scripts/make_effects.py). Use the result with make_submission.py at SCALE 1.0.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROCESSED

src, dst, agree_scale, other_scale = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
z = dict(np.load(PROCESSED / f"{src}.npz"))
up, down = z["n_up"].astype(int), z["n_down"].astype(int)
agree = (up + down >= 2) & ((up == 0) | (down == 0))
lfc = z["lfc"].astype(np.float64)
z["lfc"] = np.where(agree, agree_scale * lfc, other_scale * lfc).astype(np.float32)
np.savez_compressed(PROCESSED / f"{dst}.npz", **z)

moved = lfc != 0
print(f"changes where sources agree: {agree[moved].mean():.0%} of all predicted changes, "
      f"carrying {np.abs(lfc[agree]).sum() / np.abs(lfc).sum():.0%} of the total change")
print(f"mean |lfc| after: {np.abs(z['lfc']).mean():.4f} (uniform x1.75 would be {1.75 * np.abs(lfc).mean():.4f})")
print(f"wrote {PROCESSED / f'{dst}.npz'}")
