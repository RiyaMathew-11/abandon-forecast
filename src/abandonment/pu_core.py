"""Shared core for the PU (positive-unlabeled) abandonment models.

Framing: P = RustSec `is_unmaintained` crates (confirmed
abandoned), U = all no-advisory crates.
Security-only crates (patched CVE => confirmed maintained) stay out of
training and serve as hard negatives at evaluation time only - the "RustSec
AUC" every run reports is held-out P vs security-only.

The workhorse is PU bagging: N_BAGS base learners, each on P vs a
popularity-MATCHED U-subsample (same log-downloads profile as the positives),
scores averaged. Matching blocks the SCAR shortcut - RustSec favours popular
crates, so unmatched bags learn "popular", not "abandoned" (an unmatched
logistic baseline scored 0.608).

This module only defines things (feature lists, the matched-bag sampler,
build_X, load); work happens when a runner imports it:
  gbt/cox_model.py + gbt_survival.py / gbt_refit.py / gbt_sweep.py - the
                      shipped XGBoost-Cox survival model
  gbt/gbt_detect.py   - binary detection row of the results table
  gbt/gbt_backtest.py - binary forecasting backtest
  cli/export_artifacts.py - build_X for the exported model bundle
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"
TRAIN = DERIVED / "train.parquet"            # no-advisory crates = the U pool
VALID = DERIVED / "validation.parquet"       # RustSec crates: P + eval negatives
REPO_FEATURES = DERIVED / "repo_features.parquet"

# PU knobs - frozen: an early sweep showed the table is flat in these ----------
N_BAGS = 100      # bagging rounds; scores stabilise well before 100
U_PER_P = 1       # U-subsample size as a multiple of |P_train|
MATCH_BINS = 10   # popularity-match granularity (decile bins of the positives)
N_FOLDS = 5       # CV folds over the 242 positives (~48 held out per fold)
SEED = 42
LR_GRID = [(C, p) for p in ("l2", "l1") for C in (0.01, 0.1, 1.0, 10.0, 100.0)]

# progress updates -> stderr, so stdout stays a clean results log ---------------
T0 = time.monotonic()


def progress(msg):
    print(f"[{time.monotonic() - T0:6.1f}s] {msg}", file=sys.stderr, flush=True)


# feature groups - registry features; PU_LIN is PU-only (legal
# because the PU label is RustSec judgment, not a formula over the feature) ----
REG_LOG = ["downloads_total", "downloads_90d", "n_versions", "reverse_deps"]  # skew -> log1p
REG_LIN = ["n_owners", "crate_age_days", "yank_rate", "momentum_index",
           "release_cadence_decay", "days_since_owner_change"]
REG_BOOL = ["latest_yanked", "is_pre_1_0"]
PU_LIN = ["months_since_release"]
REPO_LOG = ["stars", "open_issues", "open_prs", "commits_180d"]
REPO_LIN = ["days_since_push", "days_since_pr_merge", "days_since_issue_close"]
REPO_BOOL = ["is_archived", "is_fork", "is_empty", "is_disabled",
             "unanswered_issues", "unanswered_prs",
             "has_repo_url", "has_repo_signal"]
REPO_CAT = ["repo_host", "repo_status"]


def build_X(df, log_f, lin_f, bool_f, cat_f):
    parts = []
    if log_f:
        parts.append(np.log1p(df[log_f]).add_prefix("log_"))
    if lin_f:
        parts.append(df[lin_f])
    if bool_f:
        parts.append(df[bool_f].astype("float64"))
    if cat_f:
        parts.append(pd.get_dummies(df[cat_f].astype("object"), prefix=cat_f, dummy_na=False))
    return pd.concat(parts, axis=1)


def make_lr(C=1.0, l1_ratio=0.0):
    """The baseline pipeline - the PU bagging base estimator. l1_ratio is
    sklearn>=1.8's penalty spelling (0 = l2, 1 = l1); l1 needs liblinear,
    lbfgs is l2-only."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=C,
                                   l1_ratio=l1_ratio,
                                   solver="liblinear" if l1_ratio == 1 else "lbfgs")),
    ])


def make_g():
    """Elkan-Noto's labeled-vs-unlabeled classifier: no class_weight, so
    predict_proba keeps the P(labeled|x) semantics that g/c relies on."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def load(use_repo, use_recency=True, drop=()):
    """Build the PU frames. Returns a namespace: Xu (unlabeled pool), Xp
    (positives), Xsec (eval negatives), Xv/dfv/dfu, unm/sec masks, downloads
    for the strata check, raw staleness, and a feature-set tag. `drop` removes
    named columns from the FEATURE lists only (they stay in dfu/dfv) - e.g.
    is_archived once it is promoted to label."""
    if use_repo and not REPO_FEATURES.exists():
        raise SystemExit(f"{REPO_FEATURES} missing - run fetch_github.py + "
                         "join_repo_signals.py first")
    dfu = pd.read_parquet(TRAIN)
    dfv = pd.read_parquet(VALID)
    if use_repo:
        repo = pd.read_parquet(REPO_FEATURES)
        dfu = dfu.merge(repo, on="name", how="left")
        dfv = dfv.merge(repo, on="name", how="left")

    present = lambda cols: [c for c in cols if c in dfu.columns and c not in drop]
    log_f = present(REG_LOG + (REPO_LOG if use_repo else []))
    lin_f = present(REG_LIN + (PU_LIN if use_recency else []) + (REPO_LIN if use_repo else []))
    bool_f = present(REG_BOOL + (REPO_BOOL if use_repo else []))
    cat_f = present(REPO_CAT if use_repo else [])

    Xu = build_X(dfu, log_f, lin_f, bool_f, cat_f)
    Xv = build_X(dfv, log_f, lin_f, bool_f, cat_f).reindex(columns=Xu.columns, fill_value=0)
    unm = dfv["is_unmaintained"].fillna(False).to_numpy()
    sec = (dfv["is_security"].fillna(False) & ~dfv["is_unmaintained"].fillna(False)).to_numpy()
    return SimpleNamespace(
        dfu=dfu, dfv=dfv, Xu=Xu, Xv=Xv, unm=unm, sec=sec,
        Xp=Xv[unm], Xsec=Xv[sec],
        dl_p=np.log1p(dfv.loc[unm, "downloads_total"].to_numpy()),
        dl_sec=np.log1p(dfv.loc[sec, "downloads_total"].to_numpy()),
        stale=dfv["months_since_release"].to_numpy(),
        tag="surface+derived" + ("+recency" if use_recency else "")
            + ("+repo" if use_repo else ""))


def describe(d, matched=True):
    """Header every runner prints: setup + the raw-staleness sanity bar."""
    print(f"feature set: {d.tag}   {d.Xu.shape[1]} columns")
    print(f"P (unmaintained): {len(d.Xp)}   U (no-advisory): {len(d.Xu):,}   "
          f"eval negatives (security-only): {len(d.Xsec)}")
    print(f"bags={N_BAGS}  U:P per bag={U_PER_P}:1  folds={N_FOLDS}  "
          f"matched pseudo-negatives={'ON' if matched else 'off'}\n")
    m = d.unm | d.sec
    print(f"raw months_since_release alone: RustSec AUC = "
          f"{roc_auc_score(d.unm[m], d.stale[m]):.3f}   (any model must beat this)\n")


def matched_idx(Xu, dl_pos, u_pool, rng, per_p, n_bins, match_col="log_downloads_total"):
    """Indices into Xu (drawn from u_pool) whose popularity profile matches
    dl_pos: bin dl_pos into n_bins quantile bins, sample per_p U crates per
    positive from the same bin. match_col picks the popularity proxy - the
    backtest uses log_reverse_deps (downloads are not reconstructible at T)."""
    edges = np.quantile(dl_pos, np.linspace(0, 1, n_bins + 1))[1:-1]
    need = np.bincount(np.digitize(dl_pos, edges), minlength=n_bins) * per_p
    u_bin = np.digitize(Xu[match_col].to_numpy()[u_pool], edges)
    parts = [rng.choice(u_pool[u_bin == i], size=n, replace=(u_bin == i).sum() < n)
             for i, n in enumerate(need) if n and (u_bin == i).any()]
    return np.concatenate(parts)


def bag_scores(Xu, Xpos, frames, rng, matched=True, label="bagging",
               u_per_p=None, n_bins=None, u_pool=None, lr_kw=None,
               match_col="log_downloads_total", est=None):
    """PU bagging: N_BAGS LRs on Xpos (y=1) vs a fresh U-subsample (y=0);
    return the bag-averaged positive-class score for each frame.

    matched draws each bag's U-subsample with Xpos's log-downloads profile
    (the SCAR mitigation); uniform otherwise. u_per_p / n_bins override the
    module knobs; lr_kw passes C/l1_ratio overrides to make_lr; u_pool
    restricts which U rows may be sampled (so sweep eval negatives can be held
    out of training). est overrides the base-estimator factory (a zero-arg
    callable, e.g. the GBT tier's make_gbt) - default is make_lr(**lr_kw).
    label=None mutes the per-25-bags progress line."""
    u_per_p = U_PER_P if u_per_p is None else u_per_p
    n_bins = MATCH_BINS if n_bins is None else n_bins
    u_pool = np.arange(len(Xu)) if u_pool is None else u_pool
    sums = [np.zeros(len(f)) for f in frames]
    dl = Xpos[match_col].to_numpy()
    for b in range(N_BAGS):
        if matched:
            idx = matched_idx(Xu, dl, u_pool, rng, u_per_p, n_bins, match_col)
        else:
            idx = rng.choice(u_pool, size=u_per_p * len(Xpos), replace=False)
        Xb = pd.concat([Xpos, Xu.iloc[idx]])
        yb = np.r_[np.ones(len(Xpos)), np.zeros(len(idx))]
        fit = (est() if est else make_lr(**(lr_kw or {}))).fit(Xb, yb)
        for s, f in zip(sums, frames):
            s += fit.predict_proba(f)[:, 1]
        if label and (b + 1) % 25 == 0:
            progress(f"{label}: {b + 1}/{N_BAGS} bags done")
    return [s / N_BAGS for s in sums]


def strata_aucs(s_pos, s_neg, d_pos, d_neg):
    """AUC within low/high halves of the eval pairs' downloads (SCAR check)."""
    y = np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))]
    s = np.r_[s_pos, s_neg]
    d = np.r_[d_pos, d_neg]
    hi = d >= np.median(d)
    out = []
    for mask in (~hi, hi):
        out.append(roc_auc_score(y[mask], s[mask]) if len(np.unique(y[mask])) > 1 else np.nan)
    return out


def nested_lr_sweep(d):
    """Nested, overfitting-safe LR regularisation sweep (C x penalty).

    Outer N_FOLDS folds report; inner 3 folds select. The inner criterion
    (inner-held-out positives vs a held-out matched-U sample, excluded from
    bag training) never sees security-only crates or the outer fold, so the
    chosen config's outer RustSec AUC is unbiased. The all-config outer table
    is printed for sensitivity/robustness ONLY - selection ignores it. The
    surrogate's price: it measures P-vs-matched-U, not the real target - the
    cost of never selecting on the benchmark."""
    kw = lambda g: {"lr_kw": {"C": g[0], "l1_ratio": 1.0 if g[1] == "l1" else 0.0}}
    gname = lambda g: f"C={g[0]:g},{g[1]}"
    default = (1.0, "l2")
    rng = np.random.default_rng(SEED)
    outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    sens = {g: [] for g in LR_GRID}   # outer AUC per config (sensitivity table)
    chosen = []                       # (config, outer AUC) per fold - the headline
    for k, (tr, te) in enumerate(outer.split(d.Xp)):
        Xp_tr, Xp_te = d.Xp.iloc[tr], d.Xp.iloc[te]
        # inner selection --------------------------------------------------
        inner_score = {g: [] for g in LR_GRID}
        inner = KFold(n_splits=3, shuffle=True, random_state=SEED + k)
        for j, (itr, ite) in enumerate(inner.split(Xp_tr)):
            Xi_tr, Xi_te = Xp_tr.iloc[itr], Xp_tr.iloc[ite]
            # held-out matched eval negatives, excluded from bag training
            neg = matched_idx(d.Xu, Xi_te["log_downloads_total"].to_numpy(),
                              np.arange(len(d.Xu)), rng, 5, MATCH_BINS)
            pool = np.setdiff1d(np.arange(len(d.Xu)), neg)
            for g in LR_GRID:
                s_p, s_n = bag_scores(d.Xu, Xi_tr, [Xi_te, d.Xu.iloc[neg]], rng,
                                      label=None, u_pool=pool, **kw(g))
                inner_score[g].append(roc_auc_score(
                    np.r_[np.ones(len(s_p)), np.zeros(len(s_n))], np.r_[s_p, s_n]))
            progress(f"sweep fold {k + 1}/{N_FOLDS}: inner {j + 1}/3 done "
                     f"({len(LR_GRID)} configs)")
        best = max(inner_score, key=lambda g: np.mean(inner_score[g]))
        # outer eval: every config for the sensitivity table; `best` is the pick
        for g in LR_GRID:
            s_te, s_sec = bag_scores(d.Xu, Xp_tr, [Xp_te, d.Xsec], rng,
                                     label=None, **kw(g))
            sens[g].append(roc_auc_score(
                np.r_[np.ones(len(s_te)), np.zeros(len(s_sec))], np.r_[s_te, s_sec]))
        chosen.append((best, sens[best][-1]))
        progress(f"sweep fold {k + 1}/{N_FOLDS}: picked {gname(best)} "
                 f"(inner {np.mean(inner_score[best]):.3f}, outer {sens[best][-1]:.3f})")

    print(f"=== LR sweep ({d.tag}; nested selection; default {gname(default)}) ===")
    print(f"{'config':<16}{'outer AUC':>12}{'+/-':>7}   (sensitivity only)")
    for g in LR_GRID:
        a = np.array(sens[g])
        flagd = "  <- default" if g == default else ""
        print(f"{gname(g):<16}{a.mean():>12.3f}{a.std():>7.3f}{flagd}")
    tuned = np.array([a for _, a in chosen])
    print(f"\nper-fold picks: {[gname(g) for g, _ in chosen]}")
    print(f"nested tuned RustSec AUC (the unbiased number): "
          f"{tuned.mean():.3f} +/- {tuned.std():.3f}")
    print("NB: report the sensitivity table for robustness; the headline is the "
          "nested number, selection never saw the benchmark.")
    return sens, chosen
