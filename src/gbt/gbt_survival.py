"""Strategy A1: survival reframing - XGBoost Cox on the single-cutoff frame.

Second response to the tuning law: use MORE of the data we already have.
The binary backtest collapses every training positive to an identical "1",
discarding WHEN each crate was declared. Survival keeps the single-cutoff
frame that works (A2 showed swapping the training population backfires) and
adds the time dimension instead:

  clock      = crate age (days since creation) - so crate_age_days LEAVES
               the covariates; it is now the timescale, not a feature
  positives  = the same 131 advisory-declared-before-T crates, labeled with
               their AGE AT DECLARATION (event)
  negatives  = matched unlabeled crates, right-censored at their age at T
  objective  = survival:cox (chosen over AFT: the sklearn API takes Cox's
               signed-label encoding directly, and the prediction is a
               hazard ranking - exactly what the eval grades; AFT would
               need the native DMatrix API for label bounds)

What the model can now learn that binary could not: which feature profiles
die YOUNG vs which grind on for years before declaration - 131 distinct
event ages instead of 131 identical labels, at zero new mining cost.

Caveats carried over unchanged from the binary backtest: positives'
features@T are post-declaration portraits, and declaration timing is
paperwork, not death (outcome-contamination argument stands).

Assembly and eval are IDENTICAL to gbt_backtest.py (repo@T tier auto-merged,
same matched bagging, same contrasts/precision@k), so rows compare directly:
  GBT binary + repo:      0.842 / 0.779, prec@500 7.0%
  GBT binary registry:    0.816 / 0.710, prec@500 6.2%

--aft swaps in the base paper's exact objective (Xu et al.: XGBoost AFT,
lognormal) via the native DMatrix API; risk = negative predicted lifetime.
Both objectives also report a FORWARD Harrell's C-index on the 2y eval
window (events = deaths at their outcome_date, everything else censored at
window end) - the comparable number to Xu et al.'s 0.846, except theirs is
a random-split in-time figure and ours is a temporal forecast.

--arch-pos (A1b): archived-before-T crates join the training positives,
event age = age at archiving (~25x more events). Registered referee: the
ADVISORY eval row only - the archived row becomes semi-circular (same
channel in training and grading). Detection-phase priors say this risks
learning owner conscientiousness (logistic PU trained on archival: 0.759 vs 0.885); the
survival frame + per-bag reverse-deps matching are what might save it.

--heartbeat (A1c): two composite staleness features on top of the same
table - days_since_heartbeat = min over all liveness channels (release,
commit, issue close, PR merge; censored channels stay NaN so no post-T
info enters), the exact analog of Xu et al.'s top-SHAP feature
latest_maintainer_activity_interval; heartbeat_ratio = that silence in
units of the crate's own mean release interval (finished != dead).
Registered prediction: raw min ~ no change (derived, trees compose it);
only the cadence-relative ratio has a shot (new per-crate rhythm info).

Run:  uv run python src/gbt/gbt_survival.py [YYYY-MM-DD] [--no-repo] [--cuda] [--aft] [--arch-pos] [--heartbeat]
      (~1 min CPU)
"""

import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import gbt_core  # noqa: F401  (sys.path bootstrap: makes cox_model / pu_core importable)
from cox_model import (BT_BOOL, BT_LIN, BT_LOG, BT_REPO_BOOL, BT_REPO_LIN,
                       BT_REPO_LOG, DERIVED, ROOT, assemble, cox_labels,
                       train_bagged)
from pu_core import MATCH_BINS, N_BAGS, SEED, U_PER_P, build_X, matched_idx, progress
import snapshot

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
CUT = pd.Timestamp(T)
SNAPSHOT = pd.Timestamp(snapshot.SNAPSHOT)
cuda = "--cuda" in sys.argv
aft = "--aft" in sys.argv
arch_pos = "--arch-pos" in sys.argv
heartbeat = "--heartbeat" in sys.argv
use_repo = (DERIVED / f"repo_signals_at_{T}.parquet").exists() and "--no-repo" not in sys.argv
OUT = DERIVED / (f"backtest_scores_{T}_{'aft' if aft else 'surv'}"
                 f"{'_archpos' if arch_pos else ''}"
                 f"{'_hb' if heartbeat else ''}.parquet")

AFT_PARAMS = dict(   # Xu et al.'s objective, our small-data knobs
    objective="survival:aft", aft_loss_distribution="normal",
    aft_loss_distribution_scale=1.0, eval_metric="aft-nloglik",
    max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_lambda=1.0, tree_method="hist",
    device="cuda" if cuda else "cpu", nthread=4)

# assembly (features + outcomes + repo tier @T) is shared with the sweep /
# refit runners via cox_model.assemble; --no-repo builds the registry-only set
df, X = assemble(T, use_repo=use_repo)

if heartbeat:
    # min over liveness channels, skipna: censored issue/PR channels are
    # already NaN (post-T info must not enter); the registry channel always
    # exists, so the min is never fully missing
    chans = [df["months_since_release"] * 30.44]
    if use_repo:
        chans += [df[c] for c in BT_REPO_LIN]
    df["days_since_heartbeat"] = pd.concat(chans, axis=1).min(axis=1)
    # the same silence in units of the crate's own mean release interval:
    # 90 quiet days alarms a weekly-release crate, not an annual one
    df["heartbeat_ratio"] = df["days_since_heartbeat"] / (
        df["crate_age_days"] / df["n_versions"].clip(lower=1))
    # rebuild X with the two heartbeat features appended (log_ / lin columns)
    X = build_X(df, BT_LOG + (BT_REPO_LOG if use_repo else []) + ["heartbeat_ratio"],
                BT_LIN + (BT_REPO_LIN if use_repo else []) + ["days_since_heartbeat"],
                BT_BOOL + (BT_REPO_BOOL if use_repo else []), [])

# positive set: --arch-pos folds archived-before-T deaths into the positives
pos_set = "adv_arch" if arch_pos else "adv"
train_pos, undeclared, event_age, cens_age = cox_labels(df, T, pos_set=pos_set)
Xp, Xu = X[train_pos], X[undeclared]
sub = df[undeclared].reset_index(drop=True)
stale = sub["months_since_release"].to_numpy()

print(f"cutoff T = {T}  [XGBoost survival:{'aft' if aft else 'cox'}, age clock]"
      f"  features: {X.shape[1]} cols")
if use_repo:
    print(f"repo tier @T: ON   coverage: {df['has_repo_at'].mean():.1%} of eligible crates")
if heartbeat:
    print(f"heartbeat tier: ON   days_since_heartbeat 25/50/75% = "
          f"{np.percentile(df['days_since_heartbeat'].dropna(), [25, 50, 75]).round(0)}"
          f"   ratio median = {df['heartbeat_ratio'].median():.2f}")
print(f"training positives: {len(Xp)}   scoring population: {len(Xu):,}")
if arch_pos:
    print("positives by source: "
          + df.loc[train_pos, "declared_source"].value_counts().to_string()
              .replace("\n", ", "))
print(f"age at declaration (positives, days): "
      f"25/50/75% = {np.percentile(event_age, [25, 50, 75]).round(0)}")
print(f"age at censoring (unlabeled, days):   "
      f"25/50/75% = {np.percentile(cens_age, [25, 50, 75]).round(0)}\n")

if aft:
    # AFT can't route through cox_model.train_bagged (Cox-only): keep its own
    # native-DMatrix loop, but on the shared assembly + labels above
    rng = np.random.default_rng(SEED)
    sums = np.zeros(len(Xu))
    d_eval = xgb.DMatrix(Xu)
    for b in range(N_BAGS):
        idx = matched_idx(Xu, Xp["log_reverse_deps"].to_numpy(), np.arange(len(Xu)),
                          rng, U_PER_P, MATCH_BINS, match_col="log_reverse_deps")
        Xb = pd.concat([Xp, Xu.iloc[idx]])
        dtr = xgb.DMatrix(Xb)
        dtr.set_float_info("label_lower_bound", np.r_[event_age, cens_age[idx]])
        dtr.set_float_info("label_upper_bound",
                           np.r_[event_age, np.full(len(idx), np.inf)])
        bst = xgb.train(AFT_PARAMS, dtr, num_boost_round=300)
        sums += -bst.predict(d_eval)              # shorter predicted life = higher risk
        if (b + 1) % 25 == 0:
            progress(f"survival bagging: {b + 1}/{N_BAGS} bags done")
    sub["risk_score"] = sums / N_BAGS
else:
    # Cox path (default + --arch-pos + --heartbeat): the shipped training core.
    # Identical seed/matching/params to the shared sweep+refit runners, so the
    # default run reproduces the shipped backtest_scores_<T>_surv.parquet.
    scores, _ = train_bagged(df, X, T, pos_set=pos_set, n_bags=N_BAGS, cuda=cuda)
    sub = sub.merge(scores, on="name", how="left")


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

# forward Harrell's C-index over the whole scoring population: events at
# their outcome_date, everything else censored at the snapshot (Xu et al.
# report 0.846 - but from a random in-time split, not a forward window)
t = np.full(len(sub), float((SNAPSHOT - CUT).days))
ev = died_any & sub["outcome_date"].notna().to_numpy()
t[ev] = (sub.loc[ev, "outcome_date"] - CUT).dt.days.to_numpy(dtype=float)
r = sub["risk_score"].to_numpy()
conc = ties = tot = 0
for i in np.flatnonzero(ev):
    later = t > t[i]
    tot += int(later.sum())
    conc += int((r[i] > r[later]).sum())
    ties += int((r[i] == r[later]).sum())
c_stale = 0.0
for i in np.flatnonzero(ev):   # same pairs, staleness as the risk score
    later = t > t[i]
    c_stale += (stale[i] > stale[later]).sum() + 0.5 * (stale[i] == stale[later]).sum()
print(f"\nforward Harrell's C-index ({int(ev.sum()):,} events, "
      f"{(SNAPSHOT - CUT).days}d window): "
      f"model {(conc + 0.5 * ties) / tot:.3f}   staleness-only {c_stale / tot:.3f}")

# labeled-only variant: pairs among deaths + proved_alive - the nearest
# analog of Xu et al.'s clean-censored setting (full-population C above is
# polluted by undeclared-dead "survivors", same blindness as precision@k)
lab = died_any | (sub["outcome"] == "proved_alive").to_numpy()
t_l, r_l, s_l, ev_l = t[lab], r[lab], stale[lab], ev[lab]
c_m = c_s = tot_l = 0
dd_m = dd_s = dd_tot = 0   # death-vs-death subset: paperwork-arrival ordering
for i in np.flatnonzero(ev_l):
    later = t_l > t_l[i]
    tot_l += int(later.sum())
    c_m += (r_l[i] > r_l[later]).sum() + 0.5 * (r_l[i] == r_l[later]).sum()
    c_s += (s_l[i] > s_l[later]).sum() + 0.5 * (s_l[i] == s_l[later]).sum()
    dl = later & ev_l
    dd_tot += int(dl.sum())
    dd_m += (r_l[i] > r_l[dl]).sum() + 0.5 * (r_l[i] == r_l[dl]).sum()
    dd_s += (s_l[i] > s_l[dl]).sum() + 0.5 * (s_l[i] == s_l[dl]).sum()
print(f"labeled-only C-index (deaths + proved-alive pairs): "
      f"model {c_m / tot_l:.3f}   staleness-only {c_s / tot_l:.3f}")
print(f"  decomposed - death-vs-alive pairs ({tot_l - dd_tot:,}): "
      f"model {roc_auc_score(ev_l, r_l):.3f}   staleness {roc_auc_score(ev_l, s_l):.3f}")
print(f"  decomposed - death-vs-death pairs ({dd_tot:,}, paperwork order): "
      f"model {dd_m / dd_tot:.3f}   staleness {dd_s / dd_tot:.3f}")

sub[["name", "outcome", "risk_score"]].to_parquet(OUT, index=False)
print(f"\nwrote {len(sub):,} scored crates -> {OUT.relative_to(ROOT)}")