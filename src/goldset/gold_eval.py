"""Grade the shipped model against the gold labels - the honest version
of the base paper's openEuler deployment table (their accuracy 0.9857 /
precision 1.0 / C-index 0.8049 on 278 repos with 7 events; ours on a
random stratified 300-crate cohort with ~130 abandoned).

Design, fixed before looking at any result:
  - Scores = the SHIPPED model's risk scores at T (backtest_scores_
    <T>_surv.parquet - trained exactly as always, gold labels never
    touched training). Staleness bar = months_since_release, same as
    every other eval in the project.
  - y: abandoned=1, maintained=0; unclear EXCLUDED but reported (the
    judge only grades where the judge is sure).
  - Primary metric: ROC-AUC with bootstrap CI, overall + per stratum
    (the sample is stratified, not proportional - pooled numbers
    overweight dust vs the real population; per-stratum rows are the
    honest read).
  - Accuracy/precision/recall need an operating point: reported at the
    PREVALENCE-MATCHED threshold (flag exactly as many crates as are
    truly abandoned - the neutral budget) plus a small flag-budget sweep.
    No threshold is tuned on gold labels.

Tuning-era additions (11/08, after the shipped model lost to the
staleness bar 0.817 vs 0.909 on the full gold read):
  --scores=<v>      grade a different scores file: a backtest variant
                    suffix (surv | aft | surv_archpos | ...) or
                    sweep:<tag> for a sweep/refit output.
  --fold=dev|test|all   with the dev/test split (split_goldset.py) in
                    place, model selection reads DEV only; TEST is
                    sealed - ONE read for ONE chosen model. The
                    pre-split full-cohort read of the shipped model
                    (11/08) is the registered baseline.

Run:  uv run python src/goldset/gold_eval.py [YYYY-MM-DD] [--scores=surv] [--fold=dev]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
VARIANT = next((a.split("=", 1)[1] for a in sys.argv
                if a.startswith("--scores=")), "surv")
FOLD = next((a.split("=", 1)[1] for a in sys.argv
             if a.startswith("--fold=")), None)
SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"

labels = pd.read_parquet(DERIVED / f"goldset_labels_v2_{T}.parquet")
if VARIANT.startswith("sweep:"):
    scores_path = DERIVED / "sweep" / f"scores_{T}_{VARIANT[6:]}.parquet"
else:
    scores_path = DERIVED / f"backtest_scores_{T}_{VARIANT}.parquet"
scores = pd.read_parquet(scores_path)
feats = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")[
    ["name", "months_since_release"]]

df = labels.merge(scores[["name", "risk_score"]], on="name") \
           .merge(feats, on="name")
n_unclear = int((df["gold_label"] == "unclear").sum())
df = df[df["gold_label"] != "unclear"].copy()

split_path = DERIVED / f"goldset_split_{T}.parquet"
if split_path.exists():
    fold = FOLD or "dev"   # with a split in place, never default to all
    if fold != "all":
        keep = set(pd.read_parquet(split_path)
                   .query("fold == @fold")["name"])
        df = df[df["name"].isin(keep)].copy()
    if fold == "test":
        print("*** SEALED TEST READ - this is the one-shot number for the "
              "chosen model; do not iterate against it. ***")
elif FOLD:
    raise SystemExit("--fold given but no split file - run split_goldset.py")
else:
    fold = "all"
y = (df["gold_label"] == "abandoned").to_numpy().astype(int)
r = df["risk_score"].to_numpy()
stale = df["months_since_release"].to_numpy()

print(f"gold eval at T = {T}   scores: {VARIANT}   fold: {fold}   "
      f"cohort: {len(df)} labeled "
      f"({int(y.sum())} abandoned, {int((1 - y).sum())} maintained; "
      f"{n_unclear} unclear excluded)")


def auc_ci(y_, s_, n_boot=1000):
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y_ == 1), np.flatnonzero(y_ == 0)
    boots = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        boots.append(roc_auc_score(y_[idx], s_[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return roc_auc_score(y_, s_), lo, hi


print(f"\n{'cohort':<22}{'n':>5}{'abnd':>6}{'model AUC':>11}{'95% CI':>17}"
      f"{'staleness':>11}")
auc, lo, hi = auc_ci(y, r)
print(f"{'ALL labeled':<22}{len(df):>5}{int(y.sum()):>6}{auc:>11.3f}"
      f"   [{lo:.3f}, {hi:.3f}]{roc_auc_score(y, stale):>11.3f}")
for s in ["revdeps_0", "revdeps_1_9", "revdeps_10plus"]:
    m = (df["stratum"] == s).to_numpy()
    if len(set(y[m])) < 2:
        print(f"{s:<22}{int(m.sum()):>5}{int(y[m].sum()):>6}{'one-class':>11}")
        continue
    a, l_, h_ = auc_ci(y[m], r[m])
    print(f"{s:<22}{int(m.sum()):>5}{int(y[m].sum()):>6}{a:>11.3f}"
          f"   [{l_:.3f}, {h_:.3f}]{roc_auc_score(y[m], stale[m]):>11.3f}")

# confidence sensitivity: does the AUC survive dropping shaky labels?
hm = df["confidence"].isin(["high", "medium"]).to_numpy()
if len(set(y[hm])) == 2:
    print(f"{'high+med conf only':<22}{int(hm.sum()):>5}{int(y[hm].sum()):>6}"
          f"{roc_auc_score(y[hm], r[hm]):>11.3f}{'':>17}"
          f"{roc_auc_score(y[hm], stale[hm]):>11.3f}")

# operating points: flag the top-k by score. Prevalence-matched k first
# (flag as many as are truly abandoned), then a budget sweep. accuracy /
# precision / recall computed per point for the openEuler comparison.
print(f"\n{'flag budget':<22}{'k':>5}"
      f"{'model acc/prec/rec':>22}{'staleness acc/prec/rec':>26}"
      f"   (base paper: acc 0.9857, prec 1.0, rec 0.43; "
      f"majority-class acc here: {max(y.mean(), 1 - y.mean()):.3f})")
for tag, k in [("prevalence-matched", int(y.sum())),
               ("top 10%", max(1, len(df) // 10)),
               ("top 25%", len(df) // 4),
               ("top 50%", len(df) // 2)]:
    cells = []
    for s_ in (r, stale):
        flag = np.zeros(len(df), dtype=int)
        flag[np.argsort(-s_)[:k]] = 1
        tp = int(((flag == 1) & (y == 1)).sum())
        cells.append(f"{(flag == y).mean():.3f}/{tp / max(flag.sum(), 1):.3f}"
                     f"/{tp / y.sum():.3f}")
    print(f"{tag:<22}{k:>5}{cells[0]:>22}{cells[1]:>26}")

print("\nNote: pooled rows overweight revdeps_0 vs the real population "
      "(stratified sample); per-stratum rows are the honest read. Gold "
      "labels never touched training - this is a fully out-of-channel judge.")
