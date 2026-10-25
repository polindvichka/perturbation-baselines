"""The tester's exam: re-make our past leaderboard submissions on a stand-in panel and check that the local
scores rank them the way the leaderboard did. A tester we can trust must pass this before it judges new ideas.

Each recipe is rebuilt with the stand-in's own data held out (src/paths.py HOLDOUT), rendered with
make_submission.py and scored with local_score.py (official scorer). Then, per metric, the Spearman rank
correlation between local and leaderboard scores over the recipes.

Usage: VCC_PANEL=h1 python scripts/exam.py [effects|build|score|report|all] [N_PARALLEL=4]
  effects  only the effect files (enough for scripts/mse_check.py; table-only panels have no cells to render)
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PANEL, PROCESSED, STANDIN, SUFFIX

STEP = sys.argv[1] if len(sys.argv) > 1 else "all"
N_PARALLEL = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PY = sys.executable

# effect files: name -> command that builds it (name gets the panel suffix)
EFFECTS = {
    "effects_v1": ["scripts/make_effects.py", "effects_v1", "k562", "hct116", "hek293t", "h1_train", "h1_val", "h1_test"],
    "effects_v2": ["scripts/make_effects.py", "effects_v2"],
    "effects_v1agree": ["scripts/agreement_scale.py", f"effects_v1{SUFFIX}", f"effects_v1agree{SUFFIX}", "2.25", "1.0"],
    "effects_ridge": ["scripts/transfer_model.py", "ridge_final"],
    "effects_ridgepk": ["scripts/transfer_model.py", "pk_final"],
}
# name: (effects, scale, render mode, leaderboard pds, mse, nmae, fid, reach, overall)
RECIPES = {
    "v6": ("effects_v1", 1, "resample", 0.4301, 0, 0.1374, -0.1550, 0.1651, 0.0939),
    "v7": ("effects_v1", 0.5, "resample", 0.3487, 0, 0.0822, -1.0602, 0.1249, -0.0883),
    "v8": ("effects_v1", 1.75, "resample", 0.5357, 0, 0.1249, -0.0106, 0.1803, 0.1376),
    "v9": ("effects_v1", 2.5, "exact", 0.6157, 0, 0.0467, 0.0022, 0.1963, 0.1436),
    "v10": ("effects_v1", 2.5, "exact15", 0.6158, 0, 0.0544, 0.0036, 0.1962, 0.1451),
    "v11": ("effects_v1", 1.75, "exact15", None, None, None, None, None, 0.1546),
    "v12": ("effects_v1agree", 1, "exact15", 0.634, 0, 0.105, -0.019, 0.190, 0.1500),
    "v13": ("effects_v2", 1.25, "exact15", 0.6491, 0, 0.1730, -0.1378, 0.2018, 0.1485),
    "v14": ("effects_ridge", 7, "exact15", 0.6476, 0.1831, 0.1235, -0.2681, 0.1694, 0.1434),
    "v15": ("effects_ridgepk", 7, "exact15", 0.6445, 0.2222, 0.1006, -0.4888, 0.1692, 0.1087),
    "v16": ("effects_ridgepk", 7, "exact15steer3000", 0.6284, 0.1352, 0.0817, 0.0197, 0.1980, 0.1784),
}
METRICS = ["pds", "mse", "nmae", "fid", "reach", "overall"]


def run(args, log):
    with open(log, "w") as f:
        r = subprocess.run([PY, *args], stdout=f, stderr=subprocess.STDOUT, env=os.environ)
    if r.returncode:
        raise SystemExit(f"failed: {' '.join(args)} (see {log})")


def submission(name):
    eff, scale, mode = RECIPES[name][:3]
    return PROCESSED / "submissions" / f"{eff}{SUFFIX}_x{scale:g}_{mode}.h5ad".replace("_resample.h5ad", ".h5ad")


def effects():
    STANDIN.mkdir(parents=True, exist_ok=True)
    for name, args in EFFECTS.items():                         # in order: v1agree needs v1
        if not (PROCESSED / f"{name}{SUFFIX}.npz").exists():
            print("building", name, flush=True)
            run(args, STANDIN / f"build_{name}.log")


def build():
    effects()
    todo = [n for n in RECIPES if not submission(n).exists()]
    with ThreadPoolExecutor(N_PARALLEL) as ex:
        list(ex.map(lambda n: run(["scripts/make_submission.py", RECIPES[n][0] + SUFFIX, str(RECIPES[n][1]), RECIPES[n][2]],
                                  STANDIN / f"render_{n}.log"), todo))
    print("rendered", todo, flush=True)


def score():
    with ThreadPoolExecutor(N_PARALLEL) as ex:
        list(ex.map(lambda n: run(["scripts/local_score.py", "score", str(submission(n)), f"exam_{n}"],
                                  STANDIN / f"score_{n}.log"), RECIPES))


def report():
    local = pd.read_csv(STANDIN / "scores.csv").drop_duplicates("label", keep="last").set_index("label")
    lb = pd.DataFrame({n: r[3:] for n, r in RECIPES.items()}, index=METRICS).T
    rows = []
    for n in RECIPES:
        if f"exam_{n}" in local.index:
            rows.append({"recipe": n, **{f"{m}": local.loc[f"exam_{n}", m] for m in METRICS},
                         **{f"lb_{m}": lb.loc[n, m] for m in METRICS}})
    t = pd.DataFrame(rows).set_index("recipe")
    pd.set_option("display.width", 200)
    print(f"panel {PANEL}: local score vs leaderboard (lb_)\n", t.round(3).to_string())
    print("\nrank agreement (Spearman) per metric:")
    for m in METRICS:
        ok = t[f"lb_{m}"].notna()
        if ok.sum() >= 3 and t.loc[ok, f"lb_{m}"].nunique() > 1:
            rho = spearmanr(t.loc[ok, m], t.loc[ok, f"lb_{m}"]).correlation
            print(f"  {m:8s} {rho:+.2f}  over {ok.sum()} recipes")
    t.to_csv(STANDIN / "exam.csv")


if __name__ == "__main__":
    if STEP == "effects":
        effects()
    if STEP in ("build", "all"):
        build()
    if STEP in ("score", "all"):
        score()
    if STEP in ("report", "all"):
        report()
