import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import build_delta_table
from src.paths import DATA, PROCESSED

PROCESSED.mkdir(parents=True, exist_ok=True)
JOBS = [("rpe1", DATA / "replogle_2022_rpe1.h5ad"),
        ("k562", DATA / "replogle_2022_k562_gwps.h5ad")]
for name, path in JOBS:
    t = build_delta_table(path, name)
    t.save(PROCESSED / f"{name}_delta.npz")
    print(name, "perts:", len(t.perts), "genes:", len(t.genes), flush=True)