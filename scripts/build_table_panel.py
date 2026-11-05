"""Build a table-only stand-in panel for the fast mse check (scripts/mse_check.py) from a public line's effect table.

mse only looks at group totals, so a pseudobulk table is enough: no cells, no official scorer. The panel has the
same layout as a challenge download, so make_effects.py / transfer_model.py run on it unchanged (VCC_PANEL=NAME):
    controls_NAME/pert_counts.csv   up to N_PERTS of the line's knockdowns with >= MIN_CELLS cells
    controls_NAME/gene_names.csv    the challenge gene axis
    controls_NAME/context_CTX.h5ad  ONE row: the line's control CP10K on the challenge axis (0 = not measured);
                                    the prediction scripts only use it for the context's expression levels
    controls_NAME/manifest.json     {"contexts": [CTX], "table": "<name>_delta.npz"}

Panels
  jurkat, hepg2, rpe1  look-alikes of the challenge contexts (Jurkat ~ A, RPE1 ~ B and E, HepG2 ~ C); their screens
                       knock down essential genes, which transfer differently from the challenge genes
  hct116, hek293t      X-Atlas lines restricted to the A/B/C challenge knockdowns, so past submissions' recipes can
                       be re-made on the very same genes

Usage: python scripts/build_table_panel.py jurkat|hepg2|rpe1|hct116|hek293t [N_PERTS=500]
"""
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PANEL_DIRS, PROCESSED

NAME = sys.argv[1]
N_PERTS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
MIN_CELLS, SEED = 50, 0

TABLES = {"hct116": "xatlas_HCT116_delta.npz", "hek293t": "xatlas_HEK293T_delta.npz"}
CHALLENGE_GENES_ONLY = NAME in TABLES
table = TABLES.get(NAME, f"{NAME}_delta.npz")
z = np.load(PROCESSED / table)
challenge_genes = pd.read_csv(PANEL_DIRS["val"] / "gene_names.csv")["gene_name"].to_numpy(str)
ok = z["n_cells"] >= MIN_CELLS
if CHALLENGE_GENES_ONLY:
    ok &= np.isin(z["perts"].astype(str), pd.read_csv(PANEL_DIRS["val"] / "pert_counts.csv")["target_gene"])
ok = np.flatnonzero(ok)
pick = np.sort(np.random.default_rng(SEED).choice(ok, min(N_PERTS, len(ok)), replace=False))
perts = z["perts"][pick].astype(str)

control_key = "basal_cp10k" if "basal_cp10k" in z.files else "control_cp10k"
basal = pd.Series(z[control_key], index=z["genes"].astype(str)).groupby(level=0).mean()
expr = basal.reindex(challenge_genes).fillna(0).to_numpy(np.float32)
out = PANEL_DIRS[NAME]
out.mkdir(parents=True, exist_ok=True)
ctx = NAME.upper()
ad.AnnData(expr[None, :], obs=pd.DataFrame(index=["control_cp10k"]),
           var=pd.DataFrame(index=challenge_genes)).write_h5ad(out / f"context_{ctx}.h5ad")
shutil.copy(PANEL_DIRS["val"] / "gene_names.csv", out / "gene_names.csv")
pd.Series(perts, name="target_gene").to_csv(out / "pert_counts.csv", index=False)
(out / "manifest.json").write_text(json.dumps({"contexts": [ctx], "table": table}))
print(f"wrote panel {NAME}: {len(perts)} knockdowns (of {len(ok)} with >= {MIN_CELLS} cells), "
      f"{int((expr > 0).sum())} challenge genes measured")
