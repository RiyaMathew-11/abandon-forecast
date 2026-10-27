"""Importable training core for the bagged PU Cox model - the single
source of truth for data assembly (assemble), PU label construction
(cox_labels), bagged training (train_bagged), and model persistence.
Used by gbt_survival.py (the shipped-score producer), gbt_sweep.py,
gbt_refit.py, and the CLI export.

Model persistence: each bag is saved as XGBoost .ubj under
  data/models/<T>/<tag>/bag_NNN.ubj  +  meta.json
(config, positive set, feature list, n_bags, seed, n_pos), so a saved
ensemble can be reloaded with load_ensemble() and applied to any feature
table with score_ensemble() - no retraining needed.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

import gbt_core  # noqa: F401  (sys.path bootstrap: makes pu_core importable)
from pu_core import MATCH_BINS, SEED, U_PER_P, build_X, matched_idx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from snapshot import CRATE_REPOS  # noqa: E402

DERIVED = ROOT / "data" / "derived-data"
MODELS = ROOT / "data" / "models"
GH = re.compile(r"github\.com[/:]+([^/]+)/([^/#?]+)", re.I)

BT_LOG = ["n_versions", "reverse_deps"]
BT_LIN = ["n_owners", "release_cadence_decay",          # crate_age_days -> timescale
          "days_since_owner_change", "months_since_release"]
BT_BOOL = ["is_pre_1_0"]
BT_REPO_LOG = ["commits_180d_at_T"]
BT_REPO_LIN = ["days_since_commit_at_T", "days_since_issue_close_at_T",
               "days_since_pr_merge_at_T"]
BT_REPO_BOOL = ["issue_censored", "pr_censored", "has_repo_at"]

DEFAULT_PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=300,
                      min_child_weight=5)


def assemble(T, drop=(), use_repo=True):
    """Features + outcomes (+ repo tier at T) -> (df, X), row-aligned.

    The single assembly path shared by every Cox runner (gbt_survival.py,
    gbt_sweep.py, gbt_refit.py, ...).
    drop:     feature names (pre-transform) to exclude from X - e.g.
              ("days_since_owner_change",) for the dump-free serving variant.
    use_repo: True (default) merges the GitHub repo tier at T and requires
              repo_signals_at_<T>.parquet; False builds the registry-only
              feature set (the --no-repo variant)."""

    feats = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")
    outc = pd.read_parquet(DERIVED / f"outcomes_{T}.parquet")
    df = feats.merge(outc[["name", "outcome", "declared_source", "unm_at",
                           "archived_at", "outcome_date"]], on="name")

    if use_repo:
        repo_at = DERIVED / f"repo_signals_at_{T}.parquet"
        if not repo_at.exists():
            raise FileNotFoundError(f"{repo_at} missing - repo tier is required")
        cut_utc = pd.Timestamp(T, tz="UTC")
        rs = pd.read_parquet(repo_at)
        rs = rs[rs["repo_status_at"] == "found"].copy()
        rs["_key"] = rs["repo"].str.lower()
        for col, ts in [("days_since_commit_at_T", "last_commit_at_T"),
                        ("days_since_issue_close_at_T", "last_issue_close_at_T"),
                        ("days_since_pr_merge_at_T", "last_pr_merge_at_T")]:
            rs[col] = (cut_utc - rs[ts]).dt.days
        rs = rs.drop_duplicates("_key", keep="last")
        crates = pd.read_parquet(CRATE_REPOS)

        def gh_key(url):
            m = GH.search(url) if isinstance(url, str) else None
            return f"{m.group(1)}/{m.group(2).replace('.git', '')}".lower() if m else None

        df["_key"] = df["name"].map(
            crates.drop_duplicates("name").set_index("name")["repository"]).map(gh_key)
        df = df.merge(rs[["_key"] + BT_REPO_LOG + BT_REPO_LIN
                         + ["issue_censored", "pr_censored"]], on="_key", how="left")
        df["has_repo_at"] = df["_key"].isin(set(rs["_key"]))
        df = df.drop(columns=["_key"])

    keep = lambda cols: [c for c in cols if c not in drop]
    repo_log = BT_REPO_LOG if use_repo else []
    repo_lin = BT_REPO_LIN if use_repo else []
    repo_bool = BT_REPO_BOOL if use_repo else []
    X = build_X(df, keep(BT_LOG + repo_log), keep(BT_LIN + repo_lin),
                keep(BT_BOOL + repo_bool), [])
    return df, X


def cox_labels(df, T, pos_set="adv", seed=SEED, train_pos_mask=None):
    """Positive/undeclared masks + Cox age labels for a single-cutoff frame.

    Returns (train_pos, undeclared, event_age, cens_age):
      train_pos   boolean mask of training positives (see pos_set)
      undeclared  boolean mask of the undeclared scoring population
      event_age   age (days) at FIRST declaration for the positives (event)
      cens_age    age (days) at T for the undeclared (right-censored)
    pos_set semantics match train_bagged; train_pos_mask overrides pos_set.
    Shared by train_bagged and the survival/aft runner so label construction
    lives in one place."""
    cut = pd.Timestamp(T)
    declared = (df["outcome"] == "declared_at_T").to_numpy()
    adv_pos = declared & df["declared_source"].isin(["advisory", "both"]).to_numpy()
    arch_pool = declared & (df["declared_source"] == "archived").to_numpy()
    m_n = re.fullmatch(r"adv_arch(\d+)", pos_set)
    if train_pos_mask is not None:   # explicit mask overrides pos_set parsing
        train_pos = np.asarray(train_pos_mask, dtype=bool)
    elif pos_set == "adv":
        train_pos = adv_pos
    elif pos_set == "adv_arch":
        train_pos = adv_pos | arch_pool
    elif pos_set == "adv_archdust":
        dust = (df["n_versions"] <= 3).to_numpy() & (df["reverse_deps"] <= 2).to_numpy()
        train_pos = adv_pos | (arch_pool & dust)
    elif m_n:
        pool_idx = np.flatnonzero(arch_pool)
        pick = np.random.default_rng(seed).choice(
            pool_idx, min(int(m_n.group(1)), len(pool_idx)), replace=False)
        train_pos = adv_pos.copy()
        train_pos[pick] = True
    else:
        raise ValueError(f"unknown pos_set {pos_set!r}")
    undeclared = ~declared

    age_at_T = df["crate_age_days"].to_numpy(dtype=float)
    decl_at = df[["unm_at", "archived_at"]].min(axis=1)
    days_before_T = (cut - decl_at).dt.days.to_numpy(dtype=float)
    event_age = np.maximum(1.0, age_at_T - days_before_T)[train_pos]
    cens_age = np.maximum(1.0, age_at_T)[undeclared]
    return train_pos, undeclared, event_age, cens_age


def train_bagged(df, X, T, pos_set="adv", params=None, n_bags=100,
                 seed=SEED, save_tag=None, cuda=False, verbose=False,
                 train_pos_mask=None):
    """Bagged PU Cox training. Returns (scores, models).

    pos_set: which crates count as training positives (event age = age at
             FIRST declaration across channels):
               "adv"          advisory-declared only (the 131)
               "adv_arch"     + ALL archived-before-T
               "adv_arch<N>"  + a seeded random N of the archived pool
               "adv_archdust" + only dust-profile archived crates
                              (n_versions <= 3 and reverse_deps <= 2 -
                               the measured blind-spot profile)
    scores:  DataFrame name + risk_score over the undeclared scoring
             population (mean log-hazard over bags).
    save_tag: when given, persists every bag + meta.json under
             data/models/<T>/<save_tag>/.
    """
    train_pos, undeclared, event_age, cens_age = cox_labels(
        df, T, pos_set=pos_set, seed=seed, train_pos_mask=train_pos_mask)
    Xp, Xu = X[train_pos], X[undeclared]

    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    save_dir = None
    if save_tag:
        save_dir = MODELS / T / save_tag
        save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    sums = np.zeros(len(Xu))
    models = []
    for b in range(n_bags):
        idx = matched_idx(Xu, Xp["log_reverse_deps"].to_numpy(),
                          np.arange(len(Xu)), rng, U_PER_P, MATCH_BINS,
                          match_col="log_reverse_deps")
        Xb = pd.concat([Xp, Xu.iloc[idx]])
        yb = np.r_[event_age, -cens_age[idx]]
        fit = XGBRegressor(
            objective="survival:cox", subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, tree_method="hist",
            device="cuda" if cuda else "cpu", n_jobs=4, **p).fit(Xb, yb)
        sums += fit.predict(Xu, output_margin=True)
        models.append(fit)
        if save_dir is not None:
            fit.get_booster().save_model(str(save_dir / f"bag_{b:03d}.ubj"))
        if verbose and (b + 1) % 25 == 0:
            print(f"    {b + 1}/{n_bags} bags")
    if save_dir is not None:
        meta = dict(T=T, pos_set=pos_set, params=p, n_bags=n_bags, seed=seed,
                    n_pos=int(train_pos.sum()), features=list(X.columns))
        (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    scores = pd.DataFrame({"name": df.loc[undeclared, "name"].to_numpy(),
                           "risk_score": sums / n_bags})
    return scores, models


def load_ensemble(T, tag):
    """Reload a saved ensemble -> (boosters, meta)."""
    d = MODELS / T / tag
    meta = json.loads((d / "meta.json").read_text())
    boosters = []
    for f in sorted(d.glob("bag_*.ubj")):
        b = xgb.Booster()
        b.load_model(str(f))
        boosters.append(b)
    return boosters, meta


def score_ensemble(boosters, X_new):
    """Mean log-hazard of a reloaded ensemble on a feature table built
    with the same columns as meta['features']."""
    dm = xgb.DMatrix(X_new)
    return np.mean([b.predict(dm, output_margin=True) for b in boosters], axis=0)
