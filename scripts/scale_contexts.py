"""Give each context its own amplitude, by scaling its effects before rendering at one common SCALE.

make_submission.py takes one SCALE for all contexts. The mse credit a context keeps depends on its own cells
(scripts/credit_check.py: at x7, D keeps 52% of it), so the right amplitude can differ between D, E and F.
Rendering EFFECTS_sD_sE_sF at SCALE 7 gives context k the amplitude 7 x s_k.

Usage: VCC_PANEL=test python scripts/scale_contexts.py EFFECTS S_D S_E S_F      e.g. effects_attn_test_cc100 1 1.2 1.1
Writes PROCESSED/{EFFECTS}_s{S_D}_{S_E}_{S_F}.npz (same layout).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import CONTEXTS, PROCESSED

EFFECTS, FACTORS = sys.argv[1], [float(s) for s in sys.argv[2:]]
assert len(FACTORS) == len(CONTEXTS), f"one factor per context {CONTEXTS}"

z = np.load(PROCESSED / f"{EFFECTS}.npz")
out = {k: z[k] for k in z.files}
out["lfc"] = (z["lfc"] * np.array(FACTORS, z["lfc"].dtype)[:, None, None]).astype(z["lfc"].dtype)
path = PROCESSED / f"{EFFECTS}_s{'_'.join(f'{s:g}' for s in FACTORS)}.npz"
np.savez_compressed(path, **out)
print(f"wrote {path.name}")
