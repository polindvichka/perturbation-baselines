import os, sys
sys.path.insert(0, "src")
from data import build_delta_table

os.makedirs("/workspace/processed", exist_ok=True)
JOBS = [("rpe1", "/workspace/data/replogle_2022_rpe1.h5ad"),
        ("k562", "/workspace/data/replogle_2022_k562_gwps.h5ad")]
for name, path in JOBS:
    t = build_delta_table(path, name)
    t.save(f"/workspace/processed/{name}_delta.npz")
    print(name, "perts:", len(t.perts), "genes:", len(t.genes), flush=True)