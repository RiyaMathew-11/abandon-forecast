"""Tier-2 bake-off: XGBoost as the PU bagging base - DETECTION evaluation.

Identical protocol to the logistic PU detection CV (same folds, same seed, same matched
bags, same eval: held-out RustSec positives vs security-only + strata) with
ONLY the base estimator swapped - so any movement is attributable to the
model class, not the pipeline.

The question: does nonlinearity extract more from the same columns than LR?
References (complete data, repo tier): LR full CV 0.885 +/- 0.013, nested
tuned 0.887; raw staleness 0.845; strata 0.858/0.913.
  - GBT ~ LR  -> the ceiling claim extends beyond linear models (airtight).
  - GBT > LR  -> nonlinear interactions matter; new headline, run the full
                 battery (sweep, strata, backtest) before believing it.

Ends with a 25-bag feature-importance audit (gain) - the story check: do
trees rank staleness/archived/commits the way LR's coefficients did?

Run:  uv run python src/gbt/gbt_detect.py [--no-repo] [--cuda]
      (~2-4 min CPU; --cuda not recommended at these bag sizes)
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from gbt_core import make_gbt
from pu_core import (MATCH_BINS, N_FOLDS, REPO_FEATURES, SEED, U_PER_P,
                     bag_scores, describe, load, matched_idx, progress,
                     strata_aucs)

cuda = "--cuda" in sys.argv
use_repo = REPO_FEATURES.exists() and "--no-repo" not in sys.argv
d = load(use_repo)
d.tag += " [XGBoost]"
describe(d)
est = lambda: make_gbt(cuda=cuda)

rng = np.random.default_rng(SEED)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
rows = []
print(f"{'fold':<6}{'AUC(bag)':>10}{'AUC lo-dl':>10}{'AUC hi-dl':>10}")
for k, (tr, te) in enumerate(kf.split(d.Xp)):
    progress(f"fold {k + 1}/{N_FOLDS}: GBT bagging on {len(tr)} positives "
             f"({len(te)} held out)")
    Xp_tr, Xp_te = d.Xp.iloc[tr], d.Xp.iloc[te]
    s_te, s_sec = bag_scores(d.Xu, Xp_tr, [Xp_te, d.Xsec], rng,
                             label=f"fold {k + 1}/{N_FOLDS}", est=est)
    auc = roc_auc_score(np.r_[np.ones(len(s_te)), np.zeros(len(s_sec))],
                        np.r_[s_te, s_sec])
    lo, hi = strata_aucs(s_te, s_sec, d.dl_p[te], d.dl_sec)
    rows.append((auc, lo, hi))
    print(f"{k:<6}{auc:>10.3f}{lo:>10.3f}{hi:>10.3f}")

r = np.array(rows)
mean, std = r.mean(axis=0), r.std(axis=0)
print(f"\n=== GBT detection results ({d.tag}) ===")
print(f"XGBoost PU bagging RustSec AUC: {mean[0]:.3f} +/- {std[0]:.3f}")
print(f"references: LR same protocol 0.885 +/- 0.013 (nested 0.887); "
      f"raw staleness 0.845")
print(f"strata: low {mean[1]:.3f}, high {mean[2]:.3f}   (LR: 0.858 / 0.913)")

# --- feature-importance audit (25 matched bags, gain) -------------------------
progress("importance audit: 25 matched bags on all positives")
dl = d.Xp["log_downloads_total"].to_numpy()
acc = np.zeros(d.Xu.shape[1])
for b in range(25):
    idx = matched_idx(d.Xu, dl, np.arange(len(d.Xu)), rng, U_PER_P, MATCH_BINS)
    Xb = pd.concat([d.Xp, d.Xu.iloc[idx]])
    yb = np.r_[np.ones(len(d.Xp)), np.zeros(len(idx))]
    acc += est().fit(Xb, yb).feature_importances_
imp = pd.Series(acc / 25, index=d.Xu.columns).sort_values(ascending=False)
print("\ntop feature importances (bag-averaged gain):")
print(imp.head(15).round(3).to_string())
