"""Mix two effect sets: lfc = A_WEIGHT * A + B_WEIGHT * B (same knockdowns and genes; either may be per context).

Usage: python scripts/combine_effects.py A A_WEIGHT B B_WEIGHT OUT
  e.g. python scripts/combine_effects.py effects_ridgepk 7 effects_v2 0.5 effects_mix
Reads and writes PROCESSED/{name}.npz; use the result with make_submission.py at SCALE 1.0.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROCESSED

a_name, a_w, b_name, b_w, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5]
A, B = np.load(PROCESSED / f"{a_name}.npz"), np.load(PROCESSED / f"{b_name}.npz")
assert (A["perts"] == B["perts"]).all() and (A["genes"] == B["genes"]).all(), "different knockdowns or genes"
lfc = a_w * np.nan_to_num(A["lfc"]) + b_w * np.nan_to_num(B["lfc"])     # broadcasts (300, G) against (3, 300, G)
extra = {"contexts": A["contexts"]} if "contexts" in A.files else ({"contexts": B["contexts"]} if "contexts" in B.files else {})
np.savez_compressed(PROCESSED / f"{out}.npz", perts=A["perts"], genes=A["genes"], lfc=lfc.astype(np.float32), **extra)
print(f"wrote {PROCESSED / out}.npz: shape {lfc.shape}, mean |lfc| {np.abs(lfc).mean():.4f}")
