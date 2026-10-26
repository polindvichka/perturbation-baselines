"""Score a prediction locally with the official cell-eval2 'vcc2026' scorer against a stand-in context.

The stand-in is chosen with VCC_PANEL (h1, k562; built by scripts/build_standin.py).
One-off setup per panel (about 10-30 minutes; builds the generic-response baseline and replicate anchors):
    VCC_PANEL=h1 python scripts/local_score.py prepare
Then, for each prediction h5ad (perturbed cells only; obs['target_gene']), e.g. from make_submission.py:
    VCC_PANEL=h1 python scripts/local_score.py score $VCC_DATA/processed/submissions/xxx.h5ad [LABEL]
or, for the mse metric alone (no DE tests; ~7 minutes instead of ~50 on H1):
    VCC_PANEL=h1 python scripts/local_score.py mse $VCC_DATA/processed/submissions/xxx.h5ad [LABEL]
  prints the raw value (1.0 = predicting the control) and the scaled score; appended to mse_scores.csv
  also prints the scorer's own notes on the sampling-noise credit (metrics/delta.py): how many knockdowns had
  the credit capped at the real side's noise, and how far it was scaled down because the predictions' spread
  across knockdowns was smaller than the credit claimed (no note = full credit)
The prediction is cut down to the genes the stand-in measured.

Prints the six metrics on the competition scale (0 = generic-response baseline, 1 = real replicate)
and their plain mean. Each run is appended to PROCESSED/standin/scores.csv.

Notes: the DE engine is chosen with the DE_BACKEND environment variable (auto | gpudge | pdex). Check new
setups against known leaderboard results before trusting the numbers.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import STANDIN
REAL = STANDIN / "real.h5ad"
WORK = STANDIN / "score_work"
BASELINE = WORK / "baseline"
BUNDLE = WORK / "bundle"
LOG = STANDIN / "scores.csv"
CELL_EVAL = shutil.which("cell-eval2") or str(Path(sys.executable).parent / "cell-eval2")
DE_BACKEND = os.environ.get("DE_BACKEND", "auto")      # auto = GPU (gpudge) if available, like the official scorer
COMMON = ["--preset", "vcc2026", "--pert-col", "target_gene", "--set", f"de.backend={DE_BACKEND}"]
SHORT = {"pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
         "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach",
         "de_wilcoxon_sig_jaccard": "jac"}


def sh(args):
    print("$", " ".join(map(str, args)), flush=True)
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"command failed: {args[1]}")
    return r.stdout + r.stderr


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    sh([CELL_EVAL, "baseline", "-ar", REAL, *COMMON, "-o", BASELINE, "--save-pred", BASELINE / "pred.h5ad"])
    sh([CELL_EVAL, "prep-real-bundle", "--real", REAL, "--baseline", BASELINE / "pred.h5ad",
        "-o", BUNDLE, "--force", *COMMON])
    print("ready:", BUNDLE)


# the scorer with its INFO notes shown (it logs the credit's caps and scaling only at INFO)
CELL_EVAL_NOTES = [sys.executable, "-c", "import logging, sys; logging.basicConfig(level=logging.INFO, "
                   "format='%(message)s'); from cell_eval2.cli import main; sys.exit(main())"]
MSE_METRICS = "metrics=[expr_mse_unbiased_capped,expr_distance_unbiased,expr_mse_unbiased_capped_norm]"


def score(pred_path, label=None, mse_only=False):
    pred_path = Path(pred_path)
    label = label or pred_path.stem
    real = ad.read_h5ad(REAL)
    ctrl = real[real.obs["target_gene"] == "non-targeting"]
    pred = ad.read_h5ad(pred_path)
    pred = ad.AnnData(pred[:, real.var_names].X, obs=pred.obs[["target_gene"]], var=real.var.copy())
    both = ad.concat([pred, ctrl])
    both.obs_names_make_unique()
    run_in = WORK / f"pred_with_controls_{label}.h5ad"
    both.write_h5ad(run_in)
    out = WORK / f"{'mse' if mse_only else 'run'}_{label}"
    if mse_only:
        notes = sh([*CELL_EVAL_NOTES, "run", "-ap", run_in, "-ar", REAL, "-o", out, *COMMON, "--set", MSE_METRICS])
        for line in notes.splitlines():
            if "sampling correction" in line or "across-perturbation spread" in line:
                print("  scorer:", line.strip())
    else:
        sh([CELL_EVAL, "run", "-ap", run_in, "-ar", REAL, "-o", out, *COMMON])
    run_in.unlink()                                    # several GB; only needed during the run
    if mse_only:
        raw = float(pd.read_csv(out / "agg_results.csv").set_index("statistic")
                    .loc["mean", "expr_mse_unbiased_capped_norm"])
        base = float(pd.read_csv(BASELINE / "baseline_agg.csv").set_index("statistic")
                     .loc["mean", "expr_mse_unbiased_capped_norm"])
        print(f"\n{label}: raw mse {raw:.4f} (baseline {base:.4f}; lower is better)")
        pd.DataFrame([{"label": label, "raw_mse": round(raw, 4), "baseline": round(base, 4)}]).to_csv(
            STANDIN / "mse_scores.csv", mode="a", header=not (STANDIN / "mse_scores.csv").exists(), index=False)
        return
    scored = out / "scored.csv"
    sh([CELL_EVAL, "score", "--user-agg", out / "agg_results.csv", "--real-bundle", BUNDLE, "-o", scored])
    df = pd.read_csv(scored).set_index("metric")
    col = df["from_replicate"]
    row = {SHORT[m]: round(float(col[m]), 4) for m in SHORT if m in col.index}
    row["overall"] = round(float(col["avg_score"]), 4)
    print(f"\n{label}")
    print(pd.Series(row).to_string())
    entry = pd.DataFrame([{"label": label, **row}])
    entry.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "prepare":
        prepare()
    elif len(sys.argv) >= 3 and sys.argv[1] in ("score", "mse"):
        score(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, mse_only=sys.argv[1] == "mse")
    else:
        raise SystemExit(__doc__)