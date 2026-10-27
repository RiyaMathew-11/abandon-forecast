"""Minimal, self-contained reproduction of the shipped bagged PU Cox model.

This is a faithful *registry-only* slice of the training core in
src/gbt/cox_model.py (the `--no-repo` variant), stripped of every dependency
on the full mining pipeline so it runs in seconds on the small shipped sample
(repro/data/sample.csv, ~5.5k crates). It reproduces the method end to end:

  clock       = crate age (days) -> crate_age_days is the timescale, NOT a feature
  positives   = the 131 advisory-declared-before-T crates, labelled with their
                AGE AT DECLARATION (the survival event)
  negatives   = popularity-matched undeclared crates, right-censored at their
                age at T (SCAR mitigation: match on log reverse-deps so bags
                learn "abandoned", not "popular")
  objective   = XGBoost survival:cox, bagged 100x, scores = mean log-hazard

It then grades the scores the way the dissertation does: AUC of model vs a raw
months-since-release "staleness" bar on the two death contrasts, plus a forward
Harrell's C-index. Numbers differ from the full-data run (this is a 6%-size
sample with the repo tier removed) but the *shape* of the result is the same:
the model ties staleness on the general population and separates the labelled
deaths, which a bare rule cannot calibrate.

Full method, evaluation and caveats: see the accompanying MSc dissertation and
the complete pipeline at https://github.com/RiyaMathew-11/abandon-forecast

Run:  uv run --no-project train_cox_mini.py           (~20s CPU, 100 bags)
      uv run --no-project train_cox_mini.py --bags 25 (faster, slightly noisier)

(--no-project makes uv ignore the surrounding repository and use this script's
inline PEP 723 dependencies, so no `uv sync` or project set-up is needed.)
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "pandas>=2.1",
#     "scikit-learn>=1.4",
#     "xgboost>=2.0",
#     "pyarrow>=15.0",
# ]
# ///

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor

# --- knobs, frozen to match the shipped model (src/abandonment/pu_core.py) ---
SEED = 42
U_PER_P = 1        # undeclared drawn per positive, per bag
MATCH_BINS = 10    # popularity-match granularity (deciles of the positives)
PARAMS = dict(max_depth=3, learning_rate=0.05, n_estimators=300,
              min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=1.0, tree_method="hist")

# registry feature set (crate_age_days is excluded: it is the survival clock)
LOG_FEATS = ["n_versions", "reverse_deps"]
LIN_FEATS = ["n_owners", "release_cadence_decay",
             "days_since_owner_change", "months_since_release"]
BOOL_FEATS = ["is_pre_1_0"]
SNAPSHOT = pd.Timestamp("2026-07-16")   # window end for the forward C-index
CUT = pd.Timestamp("2024-07-16")        # the as-of cutoff T of the sample


def build_X(df):
    """Feature matrix in a fixed column order; log1p the skewed counts."""
    log = np.log1p(df[LOG_FEATS]).add_prefix("log_")
    return pd.concat([log, df[LIN_FEATS], df[BOOL_FEATS].astype("float64")], axis=1)


def matched_idx(match_vals, pos_vals, rng):
    """Indices into the undeclared pool whose popularity profile matches the
    positives: bin the positives into deciles of log reverse-deps and draw
    U_PER_P undeclared crates per positive from the same bin."""
    edges = np.quantile(pos_vals, np.linspace(0, 1, MATCH_BINS + 1))[1:-1]
    need = np.bincount(np.digitize(pos_vals, edges), minlength=MATCH_BINS) * U_PER_P
    u_bin = np.digitize(match_vals, edges)
    pool = np.arange(len(match_vals))
    parts = [rng.choice(pool[u_bin == i], size=n, replace=(u_bin == i).sum() < n)
             for i, n in enumerate(need) if n and (u_bin == i).any()]
    return np.concatenate(parts)


def forward_c_index(risk, t, event):
    """Harrell's C over all comparable pairs: for each event, the share of
    longer-lived crates it correctly out-ranks (ties count half)."""
    conc = tot = 0.0
    for i in np.flatnonzero(event):
        later = t > t[i]
        tot += later.sum()
        conc += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return conc / tot if tot else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", type=int, default=100)
    ap.add_argument("--data", default=str(Path(__file__).parent / "data" / "sample.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["unm_at", "archived_at", "outcome_date"])
    X = build_X(df)

    # --- Cox labels on the single-cutoff frame -------------------------------
    declared = (df["outcome"] == "declared_at_T").to_numpy()
    adv_pos = declared & df["declared_source"].isin(["advisory", "both"]).to_numpy()
    undeclared = ~declared

    age_at_T = df["crate_age_days"].to_numpy(dtype=float)
    decl_at = df[["unm_at", "archived_at"]].min(axis=1)
    days_before_T = (CUT - decl_at).dt.days.to_numpy(dtype=float)
    event_age = np.maximum(1.0, age_at_T - days_before_T)[adv_pos]   # age at declaration
    cens_age = np.maximum(1.0, age_at_T)[undeclared]                 # age at T (censored)

    Xp, Xu = X[adv_pos], X[undeclared]
    sub = df[undeclared].reset_index(drop=True)
    stale = sub["months_since_release"].to_numpy()

    print(f"cutoff T = {CUT.date()}   [XGBoost survival:cox, age clock]   "
          f"features: {X.shape[1]} cols (registry-only)")
    print(f"training positives (advisory): {len(Xp)}   "
          f"scoring population (undeclared): {len(Xu):,}")
    print(f"age at declaration (days) 25/50/75% = "
          f"{np.percentile(event_age, [25, 50, 75]).round(0)}\n")

    # --- bagged PU Cox training ---------------------------------------------
    rng = np.random.default_rng(SEED)
    pos_pop = Xp["log_reverse_deps"].to_numpy()
    match_vals = Xu["log_reverse_deps"].to_numpy()
    sums = np.zeros(len(Xu))
    for b in range(args.bags):
        idx = matched_idx(match_vals, pos_pop, rng)
        Xb = pd.concat([Xp, Xu.iloc[idx]])
        yb = np.r_[event_age, -cens_age[idx]]   # Cox signed encoding: +event, -censored
        fit = XGBRegressor(objective="survival:cox", n_jobs=4, **PARAMS).fit(Xb, yb)
        sums += fit.predict(Xu, output_margin=True)
        if (b + 1) % 25 == 0:
            print(f"    {b + 1}/{args.bags} bags")
    sub["risk_score"] = sums / args.bags

    # --- evaluation: model vs the raw staleness bar --------------------------
    print(f"\n{'contrast':<30}{'model AUC':>11}{'staleness':>11}")
    for died, tag in (("died_advisory", "advisory deaths vs alive"),
                      ("died_archived", "archived deaths vs alive")):
        m = sub["outcome"].isin([died, "proved_alive"]).to_numpy()
        y = (sub.loc[m, "outcome"] == died).to_numpy().astype(int)
        auc = roc_auc_score(y, sub.loc[m, "risk_score"].to_numpy())
        bar = roc_auc_score(y, stale[m])
        print(f"{tag:<30}{auc:>11.3f}{bar:>11.3f}")

    # forward Harrell's C over the whole scoring population
    died_any = sub["outcome"].isin(["died_advisory", "died_archived"]).to_numpy()
    t = np.full(len(sub), float((SNAPSHOT - CUT).days))
    ev = died_any & sub["outcome_date"].notna().to_numpy()
    t[ev] = (sub.loc[ev, "outcome_date"] - CUT).dt.days.to_numpy(dtype=float)
    c_model = forward_c_index(sub["risk_score"].to_numpy(), t, ev)
    c_stale = forward_c_index(stale, t, ev)
    print(f"\nforward Harrell's C-index ({int(ev.sum())} events): "
          f"model {c_model:.3f}   staleness-only {c_stale:.3f}")
    print("\n(sample is ~6% of the full data with the repo tier removed, so "
          "absolute\nnumbers are lower and noisier than the reported run - the "
          "method is identical.)")


if __name__ == "__main__":
    main()
