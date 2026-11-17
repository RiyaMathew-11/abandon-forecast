"""SHAP analysis of the SHIPPED model (bagged PU Cox,
adv_d2_lr0.03_n200_m20_final) — how each feature moves the risk score.

Uses XGBoost's native exact TreeSHAP (booster.predict(pred_contribs=True)),
so no `shap` package is needed. SHAP values are in the model's log-hazard
units: positive = pushes the crate toward HIGHER abandonment risk. The
ensemble value is the mean SHAP across all 100 bags (same averaging as the
shipped score).

Two figures:
  diagrams/shap_bar.png       global importance = mean |SHAP| per feature
  diagrams/shap_beeswarm.png  per-crate SHAP, colored by feature value
                              (shows DIRECTION: does high/low push up/down)

Needs the refitted ensemble under data/models/<T>/<TAG>/ (`make reproduce`)
and the derived tables (`make fetch`).

Run:  uv run python exploration/shap_deployment_model.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "gbt"))
from cox_model import (BT_BOOL, BT_LIN, BT_LOG, BT_REPO_BOOL, BT_REPO_LIN,  # noqa: E402
                       BT_REPO_LOG, GH, load_ensemble)
from pu_core import build_X  # noqa: E402

T = "2024-07-16"
TAG = "adv_d2_lr0.03_n200_m20_final"
SEED = 42
N_SAMPLE = 8000   # crates to explain (beeswarm gets unreadable past ~10k)

DERIVED = ROOT / "data" / "derived-data"
OUT = ROOT / "diagrams"
OUT.mkdir(exist_ok=True)

# pretty names for the plots
NICE = {
    "months_since_release": "months since last release",
    "days_since_owner_change": "days since owner change",
    "log_reverse_deps": "dependents (log)",
    "release_cadence_decay": "release cadence slowdown",
    "log_n_versions": "number of releases (log)",
    "days_since_commit_at_T": "days since last commit",
    "n_owners": "number of owners",
    "is_pre_1_0": "pre-1.0 (0.x) crate",
    "log_commits_180d_at_T": "commits in last 180d (log)",
    "days_since_issue_close_at_T": "days since issue closed",
    "days_since_pr_merge_at_T": "days since PR merged",
    "issue_censored": "no closed issues",
    "pr_censored": "no merged PRs",
    "has_repo_at": "has GitHub repo",
}

# --- rebuild the exact feature matrix the model is served (as in export) ----
feats = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")
rs = pd.read_parquet(DERIVED / "repo_signals.parquet")
rs = rs[rs["repo_status"] == "found"].copy()
rs["_key"] = rs["repo"].str.lower()
snap_utc = pd.Timestamp(T, tz="UTC")
rs["days_since_commit_at_T"] = (snap_utc - rs["pushed_at"]).dt.days
rs["days_since_issue_close_at_T"] = (snap_utc - rs["last_issue_closed_at"]).dt.days
rs["days_since_pr_merge_at_T"] = (snap_utc - rs["last_pr_merged_at"]).dt.days
rs["issue_censored"] = rs["last_issue_closed_at"].isna()
rs["pr_censored"] = rs["last_pr_merged_at"].isna()
rs["commits_180d_at_T"] = rs["commits_180d"]
rs = rs.drop_duplicates("_key", keep="last")

crates = pd.read_parquet(DERIVED / "crate_repos.parquet")[["name", "repository"]]


def gh_key(url):
    m = GH.search(url) if isinstance(url, str) else None
    return f"{m.group(1)}/{m.group(2).replace('.git', '')}".lower() if m else None


df = feats.copy()
df["_key"] = df["name"].map(
    crates.drop_duplicates("name").set_index("name")["repository"]).map(gh_key)
df = df.merge(rs[["_key"] + BT_REPO_LOG + BT_REPO_LIN
                 + ["issue_censored", "pr_censored", "is_archived"]],
              on="_key", how="left")
df["has_repo_at"] = df["_key"].isin(set(rs["_key"]))

X = build_X(df, BT_LOG + BT_REPO_LOG, BT_LIN + BT_REPO_LIN,
            BT_BOOL + BT_REPO_BOOL, [])

boosters, meta = load_ensemble(T, TAG)
assert list(X.columns) == meta["features"]
feat_names = [NICE.get(f, f) for f in meta["features"]]

# --- exact TreeSHAP, averaged across the 100 bags --------------------------
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X), min(N_SAMPLE, len(X)), replace=False)
Xs = X.iloc[idx]
dm = xgb.DMatrix(Xs)
print(f"explaining {len(Xs):,} crates x {len(boosters)} bags "
      f"x {len(meta['features'])} features ...")

contribs = np.zeros((len(Xs), len(meta["features"]) + 1))
for b in boosters:
    contribs += b.predict(dm, pred_contribs=True)
contribs /= len(boosters)
shap = contribs[:, :-1]   # drop bias column

# --- global importance (mean |SHAP|) ---------------------------------------
imp = np.abs(shap).mean(axis=0)
order = np.argsort(imp)
print("\nmean |SHAP| (log-hazard units):")
for i in order[::-1]:
    print(f"  {feat_names[i]:28s} {imp[i]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(len(order)), imp[order], color="#c0392b")
ax.set_yticks(range(len(order)), [feat_names[i] for i in order])
ax.set_xlabel("mean |SHAP|  (average push on the risk score, log-hazard)")
ax.set_title("What drives the shipped model's risk score\n"
             "(bagged PU Cox, exact TreeSHAP over 100 bags)")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "shap_bar.png", dpi=200, bbox_inches="tight")
print(f"saved -> {(OUT / 'shap_bar.png').relative_to(ROOT)}")

# --- beeswarm: SHAP per crate, colored by (normalized) feature value -------
fig, ax = plt.subplots(figsize=(9, 6))
Xv = Xs.astype(float).to_numpy()  # bools -> 0/1, NaN preserved
for row, fi in enumerate(order):          # weakest at bottom -> strongest top
    sv = shap[:, fi]
    vals = Xv[:, fi]
    finite = np.isfinite(vals)
    lo, hi = (np.nanpercentile(vals[finite], [5, 95]) if finite.any() else (0, 1))
    norm = np.clip((vals - lo) / (hi - lo + 1e-12), 0, 1)
    jitter = row + (rng.random(len(sv)) - 0.5) * 0.7
    ax.scatter(sv, jitter, c=norm, cmap="coolwarm", s=6, alpha=0.5,
               linewidths=0)
ax.axvline(0, color="#888888", lw=0.8)
ax.set_yticks(range(len(order)), [feat_names[i] for i in order])
ax.set_xlabel("SHAP value  (left = lowers risk, right = raises risk)")
ax.set_title("Direction of each feature's effect\n"
             "(red = high feature value, blue = low)")
ax.spines[["top", "right"]].set_visible(False)
sm = plt.cm.ScalarMappable(cmap="coolwarm")
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.01)
cb.set_ticks([0, 1])
cb.set_ticklabels(["low", "high"])
cb.set_label("feature value")
fig.savefig(OUT / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
print(f"saved -> {(OUT / 'shap_beeswarm.png').relative_to(ROOT)}")
