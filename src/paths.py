"""Where the data lives, so the same scripts run on the laptop and on a rented machine.

Everything sits under one folder, set with the VCC_DATA environment variable
(for example `export VCC_DATA=/workspace` on vast.ai). If it is not set, the repo folder is used.

    $VCC_DATA/controls/    challenge download: context_*.h5ad, gene_names.csv, pert_counts.csv
    $VCC_DATA/data/        public datasets as downloaded (big .h5ad files)
    $VCC_DATA/processed/   everything we compute: effect tables, predictions, local scores
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("VCC_DATA", REPO))
CONTROLS = ROOT / "controls"
DATA = ROOT / "data"
PROCESSED = ROOT / "processed"
