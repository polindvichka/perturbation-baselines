"""Where the data lives, so the same scripts run on the laptop and on a rented machine.

Everything sits under one folder, set with the VCC_DATA environment variable
(for example `export VCC_DATA=/workspace` on vast.ai). If it is not set, the repo folder is used.

    $VCC_DATA/controls/         validation download (contexts A, B, C): context_*.h5ad, gene_names.csv,
                                pert_counts.csv, manifest.json
    $VCC_DATA/controls_final/   final-phase download (contexts D, E, F), same files
    $VCC_DATA/data/             public datasets as downloaded (big .h5ad files)
    $VCC_DATA/processed/        everything we compute: effect tables, predictions, local scores

Which panel a script works on is chosen with VCC_PANEL: `val` (default, controls/), `test` (controls_final/),
or a local stand-in panel (`h1`, `k562`; built by scripts/build_standin.py into controls_<name>/, scored with
scripts/local_score.py). CONTEXTS lists that panel's contexts, and SUFFIX ("" or "_<panel>") is added to the
names of panel-specific outputs (effects, submissions), so panels never overwrite each other.
HOLDOUT is the public source a stand-in panel was made from: it must not be used as evidence for itself.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("VCC_DATA", REPO))
DATA = ROOT / "data"
PROCESSED = ROOT / "processed"
STANDIN_PANELS = {"h1": "h1", "k562": "k562"}          # stand-in panel -> the source it is made from
PANEL_DIRS = {"val": ROOT / "controls", "test": ROOT / "controls_final",
              **{p: ROOT / f"controls_{p}" for p in STANDIN_PANELS}}

PANEL = os.environ.get("VCC_PANEL", "val")
CONTROLS = PANEL_DIRS[PANEL]
SUFFIX = "" if PANEL == "val" else f"_{PANEL}"
HOLDOUT = STANDIN_PANELS.get(PANEL)
STANDIN = PROCESSED / f"standin_{PANEL}"               # real cells + scorer files of a stand-in panel
_manifest = CONTROLS / "manifest.json"
CONTEXTS = json.loads(_manifest.read_text())["contexts"] if _manifest.exists() else ["A", "B", "C"]


def is_holdout(source):
    """True if `source` (e.g. 'h1_train', 'k562') comes from the current stand-in panel's own data."""
    return HOLDOUT is not None and source.split("_")[0] == HOLDOUT


def all_targets():
    """Knockdowns of every panel whose controls are present (source tables are built for all of them)."""
    import pandas as pd
    lists = [pd.read_csv(d / "pert_counts.csv")["target_gene"] for d in PANEL_DIRS.values()
             if (d / "pert_counts.csv").exists()]
    return sorted(set().union(*lists))
