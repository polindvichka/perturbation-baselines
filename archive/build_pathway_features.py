"""Pathway features per gene, from public annotation files (no perturbation data involved).

go      which Gene Ontology terms the gene is annotated with (terms with 5-500 genes), reduced to N_DIM numbers
string  which genes it interacts with in STRING (combined score >= 700), reduced to N_DIM numbers

Genes that are knocked down in the same pathway or complex tend to change the same genes, so these are the
features most likely to say something about WHICH gene was perturbed.

Downloads (once) to DATA/pathway/: goa_human.gaf.gz, 9606.protein.info.v12.0.txt.gz,
9606.protein.links.v12.0.txt.gz.  Writes PROCESSED/pathway_features.npz with
    genes (n,)   gene symbols
    go (n, N_DIM), string (n, N_DIM)   float32; all zeros for a gene that has no annotation / no interaction
    has_go (n,), has_string (n,)       bool

Usage: python archive/build_pathway_features.py
"""
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATA, PROCESSED

PATHWAY_DIR = DATA / "pathway"
OUT = PROCESSED / "pathway_features.npz"
URLS = {
    "goa_human.gaf.gz": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
    "9606.protein.info.v12.0.txt.gz": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
    "9606.protein.links.v12.0.txt.gz": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
}
N_DIM = 128
GO_MIN, GO_MAX = 5, 500
STRING_MIN_SCORE = 700


def fetch():
    PATHWAY_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = PATHWAY_DIR / name
        if not path.exists():
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(url, path)
        print(f"{name}: {path.stat().st_size / 1e6:.0f} MB")


def read_go():
    """gene symbol -> set of GO ids (GAF 2.x: column 2 = symbol, 3 = qualifier, 4 = GO id)."""
    pairs = set()
    with gzip.open(PATHWAY_DIR / "goa_human.gaf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5 or "NOT" in f[3]:
                continue
            pairs.add((f[2], f[4]))
    df = pd.DataFrame(sorted(pairs), columns=["gene", "go"])
    print(f"GO: {df.gene.nunique()} genes, {df.go.nunique()} terms, {len(df)} annotations")
    return df


def read_string():
    info = pd.read_csv(PATHWAY_DIR / "9606.protein.info.v12.0.txt.gz", sep="\t", usecols=[0, 1])
    info.columns = ["id", "name"]
    name_of = dict(zip(info.id, info.name))
    parts = []
    for chunk in pd.read_csv(PATHWAY_DIR / "9606.protein.links.v12.0.txt.gz", sep=" ", chunksize=2_000_000):
        chunk = chunk[chunk["combined_score"] >= STRING_MIN_SCORE]
        parts.append(pd.DataFrame({"a": chunk["protein1"].map(name_of), "b": chunk["protein2"].map(name_of),
                                   "w": chunk["combined_score"] / 1000.0}))
    edges = pd.concat(parts).dropna()
    edges = edges[edges.a != edges.b]
    print(f"STRING: {len(edges)} directed edges with score >= {STRING_MIN_SCORE}, "
          f"{len(set(edges.a) | set(edges.b))} genes")
    return edges


def reduce(M, n_dim):
    """Truncated SVD of a sparse matrix; rows with no entries stay all zeros."""
    k = min(n_dim, min(M.shape) - 1)
    Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(M)
    if k < n_dim:
        Z = np.hstack([Z, np.zeros((Z.shape[0], n_dim - k))])
    return Z.astype(np.float32)


def main():
    fetch()
    go, edges = read_go(), read_string()
    genes = sorted(set(go.gene) | set(edges.a) | set(edges.b))
    idx = {g: i for i, g in enumerate(genes)}

    size = go.groupby("go").gene.nunique()
    go = go[go.go.isin(size[(size >= GO_MIN) & (size <= GO_MAX)].index)]
    terms = {t: i for i, t in enumerate(sorted(go.go.unique()))}
    G = sparse.csr_matrix((np.ones(len(go)), (go.gene.map(idx), go.go.map(terms))), shape=(len(genes), len(terms)))
    print(f"GO matrix: {G.shape}, genes with at least one kept term: {int((G.getnnz(axis=1) > 0).sum())}")

    A = sparse.csr_matrix((edges.w, (edges.a.map(idx), edges.b.map(idx))), shape=(len(genes),) * 2)
    A = A.maximum(A.T)
    has_string = A.getnnz(axis=1) > 0
    A = A + sparse.diags(has_string.astype(float))          # a gene is its own neighbour, so two genes sharing partners look alike
    print(f"STRING matrix: {A.shape}, genes with an interaction: {int(has_string.sum())}")

    np.savez(OUT, genes=np.array(genes, dtype=object), go=reduce(G, N_DIM), string=reduce(A, N_DIM),
             has_go=G.getnnz(axis=1) > 0, has_string=has_string)
    print("wrote", OUT)


if __name__ == "__main__":
    main()