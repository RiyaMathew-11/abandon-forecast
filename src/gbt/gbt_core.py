"""Shared pieces for the Tier-2 GBT bake-off (src/gbt).

Library choice: XGBoost.
  - Handles NaN natively (learns a per-split default direction), so the GBT
    pipeline needs NO imputer/scaler - raw build_X output goes straight in.
    The missingness INFORMATION is still available to the trees via the NaNs
    themselves, replacing the LR pipeline's indicator columns.
  - device="cuda" is a one-flag switch (T4), though at ~260-row bags CPU is
    faster - the flag exists for full-population fits later.
  - The AFT/Cox objectives keep the door open for the survival axis.

Small-data defaults - 131-242 positives, ~260-row bags overfit trees fast:
shallow depth, heavy row/column subsampling, min_child_weight as a leaf
floor. GBT_GRID is the tuning grid for surrogate-selection runs.

Importing this module also makes pu_core importable from src/gbt scripts.
"""

import sys
from pathlib import Path

# make src/abandonment (pu_core and friends) importable from src/gbt scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "abandonment"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # snapshot.py

from xgboost import XGBClassifier  # noqa: E402  (path bootstrap must run first)

# (max_depth, learning_rate, n_estimators) - kept small on purpose
GBT_GRID = [(d, lr, n) for d in (2, 3) for lr in (0.05, 0.1) for n in (100, 300)]


def make_gbt(max_depth=3, learning_rate=0.05, n_estimators=300, cuda=False):
    """XGBoost base estimator for PU bagging (drop-in for make_lr via
    bag_scores(est=...)). Defaults are the pre-registered config; the grid
    varies only the three structural knobs."""
    return XGBClassifier(
        max_depth=max_depth, learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, tree_method="hist",
        device="cuda" if cuda else "cpu",
        n_jobs=4, eval_metric="logloss")
