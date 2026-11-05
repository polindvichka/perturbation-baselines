"""Set each knockdown's own target gene to the knockdown depth seen in Arc's CRISPRi data.

The source lines (Replogle K562, X-Atlas HEK293T/HCT116) keep 35-55% of the target's expression, while Arc's
own H1 screen (10x Flex, like D/E/F) keeps about 13% (median over 243 knockdowns). The model averages the
source lines, so it under-calls the one gene that is unique to every knockdown.
make_submission.py renders f = 2 ** (SCALE * lfc), so the target is set to lfc = log2(FRAC) / SCALE,
giving a rendered fraction of FRAC at that SCALE.

Usage: VCC_PANEL=test python scripts/set_target.py EFFECTS SCALE FRAC      e.g. effects_attn_test_cc100 9 0.13
Writes PROCESSED/{EFFECTS}_t{FRAC}x{SCALE}.npz (same layout).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROCESSED

EFFECTS, SCALE, FRAC = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])

z = np.load(PROCESSED / f"{EFFECTS}.npz", allow_pickle=True)
out = {k: z[k] for k in z.files}
lfc = z["lfc"].copy()
col = {g: j for j, g in enumerate(z["genes"])}
hit = [(i, col[p]) for i, p in enumerate(z["perts"]) if p in col]
target = np.log2(FRAC) / SCALE
for i, j in hit:
    lfc[:, i, j] = np.minimum(lfc[:, i, j], target)   # only deepen; a prediction already deeper stays
before = 2 ** (SCALE * np.array([z["lfc"][:, i, j] for i, j in hit]))
print(f"{len(hit)} of {len(z['perts'])} targets measured; rendered fraction left before: median "
      f"{np.median(before):.2f}, after: {FRAC:g} or lower")
out["lfc"] = lfc.astype(z["lfc"].dtype)
path = PROCESSED / f"{EFFECTS}_t{FRAC:g}x{SCALE:g}.npz"
np.savez_compressed(path, **out)
print(f"wrote {path.name}")
