"""Turn each gene's protein sequence into 480 numbers with ESM-2 (35M parameters)."""
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROCESSED

MODEL = "facebook/esm2_t12_35M_UR50D"
MAX_RESIDUES = 1022          # ESM-2 can read at most 1022 amino acids
BATCH = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

table = pd.read_csv(PROCESSED / "gene_sequences.csv")
table["sequence"] = table["sequence"].str.slice(0, MAX_RESIDUES)    # cut very long proteins
table = table.sort_values("length").reset_index(drop=True)          # similar lengths batch together

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device).eval()
if device == "cuda":
    model = model.half()                                            # faster, half precision

vectors = []
with torch.no_grad():
    for start in range(0, len(table), BATCH):
        seqs = table["sequence"].iloc[start:start + BATCH].tolist()
        enc = tokenizer(seqs, return_tensors="pt", padding=True).to(device)
        out = model(**enc).last_hidden_state.float()                # (batch, tokens, 480)

        mask = enc["attention_mask"].clone()                        # 1 = real token, 0 = padding
        lengths = mask.sum(1)
        mask[:, 0] = 0                                              # ignore the start marker
        mask[torch.arange(len(seqs)), lengths - 1] = 0              # ignore the end marker
        m = mask.unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1)                        # average over the amino acids
        vectors.append(pooled.cpu().numpy())

        if (start // BATCH) % 50 == 0:
            print(f"{start}/{len(table)}", flush=True)

emb = np.concatenate(vectors).astype(np.float32)
genes = table["gene"].astype(str).to_numpy()
np.savez_compressed(PROCESSED / "esm2_35m.npz", genes=genes, emb=emb)
print("saved:", emb.shape, "| mean", float(emb.mean()), "| std", float(emb.std()))