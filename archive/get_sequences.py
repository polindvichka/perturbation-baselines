"""Download human protein sequences from UniProt and match them to gene symbols."""
import gzip
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import DeltaTable
from src.paths import CONTROLS, DATA, PROCESSED

URL = ("https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta"
       "&query=%28proteome%3AUP000005640%29+AND+%28reviewed%3Atrue%29")
FASTA = DATA / "uniprot_human.fasta.gz"
OUT = PROCESSED / "gene_sequences.csv"

# 1. Download the file (wget -c resumes if interrupted)
if not FASTA.exists():
    subprocess.run(["wget", "-c", "-O", str(FASTA), URL], check=True)

# 2. Read it. Each entry is a header line starting with ">" followed by sequence lines.
best = {}   # gene symbol -> longest protein sequence found for it

def keep(header, chunks):
    if header is None:
        return
    match = re.search(r"GN=(\S+)", header)      # the gene symbol, e.g. GN=ABCD1
    if match is None:
        return
    gene = match.group(1)
    seq = "".join(chunks)
    if gene not in best or len(seq) > len(best[gene]):
        best[gene] = seq

header, chunks = None, []
with gzip.open(FASTA, "rt") as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            keep(header, chunks)
            header, chunks = line, []
        else:
            chunks.append(line)
keep(header, chunks)

table = pd.DataFrame({"gene": list(best), "sequence": list(best.values())})
table["length"] = table["sequence"].str.len()
table.to_csv(OUT, index=False)
print("genes with a protein sequence:", len(table))
print("median protein length:", int(table["length"].median()))

# 3. How many of OUR genes are covered?
vocab = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].astype(str)
challenge = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].astype(str)
k562 = pd.Series(DeltaTable.load(PROCESSED / "k562_delta.npz").perts)
rpe1 = pd.Series(DeltaTable.load(PROCESSED / "rpe1_delta.npz").perts)

have = set(table["gene"])
for name, genes in [("challenge genes (300)", challenge), ("K562 knockdowns", k562),
                    ("RPE1 knockdowns", rpe1), ("all 18,533 challenge-vocab genes", vocab)]:
    covered = genes.isin(have).sum()
    print(f"{name}: {covered} of {len(genes)} have a sequence ({covered / len(genes):.0%})")

missing = challenge[~challenge.isin(have)].tolist()
print("challenge genes WITHOUT a sequence:", missing[:30])