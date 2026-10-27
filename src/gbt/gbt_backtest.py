"""Tier-2 backtest: XGBoost base on the as-of-T features.

The question: does the GBT detection gain (0.926 vs LR 0.885) transfer to
FORECASTING, or is it an in-time artifact? Same assembly and grading as
the logistic PU backtest - features_at + outcomes + repo@T tier - with only
the base estimator swapped (gbt_core defaults, pre-registered, evaluated once; no
tuning, so the number is honest as-is).

References at T=2024-07-16 (LR): advisory deaths 0.842 [0.780, 0.895],
archived deaths 0.711, prec@5000 4.2%; staleness bars 0.876 / 0.890.

Run:  uv run python src/gbt/gbt_backtest.py [YYYY-MM-DD] [--no-repo] [--cuda] [--tune]
      (~2 min CPU; --tune adds ~4-6 min of inner selection over GBT_GRID -
      selection never touches the outcomes)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from gbt_core import GBT_GRID, make_gbt
from pu_core import MATCH_BINS, SEED, bag_scores, build_X, matched_idx, progress

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
cuda = "--cuda" in sys.argv
ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"
CRATES = ROOT / "crates-data" / "2026-07-16-020017" / "data" / "crates.csv"
REPO_AT = DERIVED / f"repo_signals_at_{T}.parquet"
OUT = DERIVED / f"backtest_scores_{T}_gbt.parquet"
GH = re.compile(r"github\.com[/:]+([^/]+)/([^/#?]+)", re.I)

BT_LOG = ["n_versions", "reverse_deps"]
BT_LIN = ["n_owners", "crate_age_days", "release_cadence_decay",
          "days_since_owner_change", "months_since_release"]
BT_BOOL = ["is_pre_1_0"]
BT_REPO_LOG = ["commits_180d_at_T"]
BT_REPO_LIN = ["days_since_commit_at_T", "days_since_issue_close_at_T",
               "days_since_pr_merge_at_T"]
BT_REPO_BOOL = ["issue_censored", "pr_censored", "has_repo_at"]

feats = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")
outc = pd.read_parquet(DERIVED / f"outcomes_{T}.parquet")
df = feats.merge(outc[["name", "outcome", "declared_source"]], on="name")

use_repo = REPO_AT.exists() and "--no-repo" not in sys.argv
if use_repo:
    cut_utc = pd.Timestamp(T, tz="UTC")
    rs = pd.read_parquet(REPO_AT)
    rs = rs[rs["repo_status_at"] == "found"].copy()
    rs["_key"] = rs["repo"].str.lower()
    for col, ts in [("days_since_commit_at_T", "last_commit_at_T"),
                    ("days_since_issue_close_at_T", "last_issue_close_at_T"),
                    ("days_since_pr_merge_at_T", "last_pr_merge_at_T")]:
        rs[col] = (cut_utc - rs[ts]).dt.days
    rs = rs.drop_duplicates("_key", keep="last")
    crates = pd.read_csv(CRATES, usecols=["name", "repository"])
    def gh_key(url):
        m = GH.search(url) if isinstance(url, str) else None
        return f"{m.group(1)}/{m.group(2).replace('.git', '')}".lower() if m else None
    df["_key"] = df["name"].map(
        crates.drop_duplicates("name").set_index("name")["repository"]).map(gh_key)
    df = df.merge(rs[["_key"] + BT_REPO_LOG + BT_REPO_LIN
                     + ["issue_censored", "pr_censored"]], on="_key", how="left")
    df["has_repo_at"] = df["_key"].isin(set(rs["_key"]))
    df = df.drop(columns=["_key"])

log_f = BT_LOG + (BT_REPO_LOG if use_repo else [])
lin_f = BT_LIN + (BT_REPO_LIN if use_repo else [])
bool_f = BT_BOOL + (BT_REPO_BOOL if use_repo else [])
X = build_X(df, log_f, lin_f, bool_f, [])
train_pos = ((df["outcome"] == "declared_at_T")
             & df["declared_source"].isin(["advisory", "both"])).to_numpy()
undeclared = (df["outcome"] != "declared_at_T").to_numpy()
Xp, Xu = X[train_pos], X[undeclared]
sub = df[undeclared].reset_index(drop=True)

print(f"cutoff T = {T}  [XGBoost]  features: {X.shape[1]} cols")
if use_repo:
    print(f"repo tier @T: ON   coverage: {df['has_repo_at'].mean():.1%} of eligible crates")
print(f"training positives: {len(Xp)}   scoring population: {len(Xu):,}")
print("outcomes in window: "
      + ", ".join(f"{k}={v:,}" for k, v in sub["outcome"].value_counts().items()
                  if k != "unlabeled") + "\n")

rng = np.random.default_rng(SEED)

# --- honest tuning (--tune): GBT_GRID chosen WITHOUT touching outcomes -------
# Same protocol as the logistic backtest's tuning: inner 3-fold over training positives,
# surrogate criterion (held-out positives vs held-out matched-U, excluded
# from bag training); outcomes evaluated ONCE after the config is frozen.
params = (3, 0.05, 300)  # gbt_core defaults, pre-registered
if "--tune" in sys.argv:
    inner = KFold(n_splits=3, shuffle=True, random_state=SEED)
    inner_score = {g: [] for g in GBT_GRID}
    for j, (itr, ite) in enumerate(inner.split(Xp)):
        Xi_tr, Xi_te = Xp.iloc[itr], Xp.iloc[ite]
        neg = matched_idx(Xu, Xi_te["log_reverse_deps"].to_numpy(),
                          np.arange(len(Xu)), rng, 5, MATCH_BINS,
                          match_col="log_reverse_deps")
        pool = np.setdiff1d(np.arange(len(Xu)), neg)
        for g in GBT_GRID:
            s_p, s_n = bag_scores(Xu, Xi_tr, [Xi_te, Xu.iloc[neg]], rng,
                                  label=None, u_pool=pool,
                                  match_col="log_reverse_deps",
                                  est=lambda g=g: make_gbt(*g, cuda=cuda))
            inner_score[g].append(roc_auc_score(
                np.r_[np.ones(len(s_p)), np.zeros(len(s_n))], np.r_[s_p, s_n]))
        progress(f"tune: inner {j + 1}/3 done ({len(GBT_GRID)} configs)")
    params = max(inner_score, key=lambda g: np.mean(inner_score[g]))
    print("inner surrogate scores (selection never saw outcomes):")
    for g in GBT_GRID:
        mark = "  <- chosen" if g == params else ""
        print(f"  depth={g[0]} lr={g[1]:g} n={g[2]:<4} "
              f"{np.mean(inner_score[g]):.3f}{mark}")

(s_u,) = bag_scores(Xu, Xp, [Xu], rng, label="GBT backtest bagging",
                    match_col="log_reverse_deps",
                    est=lambda: make_gbt(*params, cuda=cuda))
sub["risk_score"] = s_u
stale = sub["months_since_release"].to_numpy()


def auc_ci(y, s, n_boot=1000):
    auc = roc_auc_score(y, s)
    rng_b = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = []
    for _ in range(n_boot):
        idx = np.r_[rng_b.choice(pos, len(pos)), rng_b.choice(neg, len(neg))]
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi


print(f"\n{'contrast':<32}{'AUC':>7}{'95% CI':>17}{'staleness bar':>15}")
for died, tag in (("died_advisory", "advisory deaths vs alive"),
                  ("died_archived", "archived deaths vs alive")):
    m = sub["outcome"].isin([died, "proved_alive"]).to_numpy()
    y = (sub.loc[m, "outcome"] == died).to_numpy().astype(int)
    auc, lo, hi = auc_ci(y, sub.loc[m, "risk_score"].to_numpy())
    bar = roc_auc_score(y, stale[m])
    print(f"{tag:<32}{auc:>7.3f}   [{lo:.3f}, {hi:.3f}]{bar:>15.3f}")

died_any = sub["outcome"].isin(["died_advisory", "died_archived"]).to_numpy()
order = np.argsort(-sub["risk_score"].to_numpy())
print(f"\nprecision@k (deaths in window; base rate {died_any.mean():.1%}):")
for k in (50, 100, 500, 1000, 5000):
    top = order[:k]
    stale_top = np.argsort(-stale)[:k]
    print(f"  top {k:>5}: model {died_any[top].mean():>6.1%}   "
          f"staleness-only {died_any[stale_top].mean():>6.1%}")

sub[["name", "outcome", "risk_score"]].to_parquet(OUT, index=False)
print(f"\nwrote {len(sub):,} scored crates -> {OUT.relative_to(ROOT)}")
