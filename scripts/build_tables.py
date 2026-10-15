"""Build knockdown-effect tables (DeltaTable .npz) from public Perturb-seq h5ad files in DATA.

Usage: python scripts/build_tables.py [NAME ...]     (no names = every dataset below)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import build_delta_table
from src.paths import DATA, PROCESSED

# name: (file in DATA, obs column with the knocked-down gene, label of the control cells[, var column with gene symbols])
DATASETS = {
    "rpe1": ("replogle_2022_rpe1.h5ad", "gene", "non-targeting"),
    "k562": ("replogle_2022_k562_gwps.h5ad", "gene", "non-targeting"),
    # Nadig et al. 2025, raw files from GEO GSE264667; genes are indexed by Ensembl id, symbols in var["gene_name"]
    "hepg2": ("GSE264667_hepg2_raw_singlecell_01.h5ad", "gene", "non-targeting", "gene_name"),
    "jurkat": ("GSE264667_jurkat_raw_singlecell_01.h5ad", "gene", "non-targeting", "gene_name"),
    # VCC 2025 H1 hESC (CRISPRi + 10x Flex, like this year); the 3 files share the same control cells
    "h1_train": ("h1/adata_Training.h5ad", "target_gene", "non-targeting"),
    "h1_val": ("h1/adata_Validation.h5ad", "target_gene", "non-targeting"),
    "h1_test": ("h1/adata_Test.h5ad", "target_gene", "non-targeting"),
}

PROCESSED.mkdir(parents=True, exist_ok=True)
for name in sys.argv[1:] or DATASETS:
    file, pert_col, control, *var_col = DATASETS[name]
    t = build_delta_table(DATA / file, name, pert_col=pert_col, control=control, var_col=var_col[0] if var_col else None)
    t.save(PROCESSED / f"{name}_delta.npz")
    print(name, "perts:", len(t.perts), "genes:", len(t.genes), flush=True)
