"""Build knockdown-effect tables (DeltaTable .npz) from public Perturb-seq h5ad files in DATA.

Usage: python scripts/build_tables.py [NAME ...]     (no names = every dataset below)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import build_delta_table
from src.paths import DATA, PROCESSED

# name: (file in DATA, obs column with the knocked-down gene, label of the control cells)
DATASETS = {
    "rpe1": ("replogle_2022_rpe1.h5ad", "gene", "non-targeting"),
    "k562": ("replogle_2022_k562_gwps.h5ad", "gene", "non-targeting"),
    "hepg2": ("NadigOConner2024_hepg2.h5ad", "perturbation", "control"),
    "jurkat": ("NadigOConner2024_jurkat.h5ad", "perturbation", "control"),
}

PROCESSED.mkdir(parents=True, exist_ok=True)
for name in sys.argv[1:] or DATASETS:
    file, pert_col, control = DATASETS[name]
    t = build_delta_table(DATA / file, name, pert_col=pert_col, control=control)
    t.save(PROCESSED / f"{name}_delta.npz")
    print(name, "perts:", len(t.perts), "genes:", len(t.genes), flush=True)
