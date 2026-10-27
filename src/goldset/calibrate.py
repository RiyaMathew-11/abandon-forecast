"""Gold-anchored calibration v4: score -> P(abandoned within ~2 years).

v4 adds C5 = beta calibration (Kull et al. 2017) on the (score, age)
index - 3 params, asymmetric/steeper-than-sigmoid, smooth (no cliff
step, unlike isotonic). ITERATION CAVEAT (logged): this is a second
look at the same 135 dev points after the v3 rule already selected C4,
so the v4 replacement rule was fixed before running: C5 ships iff it
(a) passes the age check, (b) is monotone (both beta coefs >= 0),
(c) worst-quintile miss <= C4's + 2pp. Smoothness is the declared
tie-breaker. True out-of-sample validation of any calibrator needs a
fresh gold draw regardless.


History (each step's tripwires registered before running):
  v1: logistic(score) - shape miss (Q3 16pp) AND age predicts residual.
  v2: three candidates showed the defects are INDEPENDENT - isotonic
      fixes shape (7.4% worst miss) but fails the age check (p=0.016);
      logistic(score,age) kills the age effect (rho +0.003) but keeps
      the stiff shape (14.6% worst miss).
  v3: adds C4 = isotonic ON TOP of the (score, age) logit - the linear
      index supplies age-aware ordering, isotonic bends it to observed
      frequencies. Same selection rule, now over C1-C4: must pass the
      age-residual check (p >= 0.05); among survivors, smallest
      worst-quintile |observed - predicted| wins; if none survive, C3.

Fitted ONLY on the gold dev fold (n=135), weighted to population
stratum shares. Winner + bootstrap bands -> calibration_<T>.json.

Run:  uv run python src/goldset/calibrate.py [YYYY-MM-DD]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
SEED = 42
# default = the 14-feature shipped model; --scores=<relative path under
# derived-data> recalibrates a different variant (e.g. the 13-feature
# no-owner model - calibration maps THIS model's scores, so any new
# variant refits it and re-passes the full candidate battery)
SCORES = next((a.split("=", 1)[1] for a in sys.argv
               if a.startswith("--scores=")),
              "sweep/scores_{}_adv_d2_lr0.03_n200_m20_final.parquet")

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"
OUT = DERIVED / f"calibration_{T}.json"

split = pd.read_parquet(DERIVED / f"goldset_split_{T}.parquet")
dev = split[split["fold"] == "dev"].copy()
scores = pd.read_parquet(DERIVED / SCORES.format(T))
feats_all = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")[
    ["name", "reverse_deps", "crate_age_days"]]
dev = dev.merge(scores, on="name").merge(feats_all, on="name")
y = (dev["gold_label"] == "abandoned").to_numpy().astype(int)
s = dev["risk_score"].to_numpy()
age = (dev["crate_age_days"] / 365.25).to_numpy()
X3 = np.c_[s, age]

pop = feats_all.merge(scores[["name"]], on="name")   # eligible scored population
bins, labs = [-1, 0, 9, np.inf], ["revdeps_0", "revdeps_1_9", "revdeps_10plus"]
pop_share = pd.cut(pop["reverse_deps"], bins, labels=labs).value_counts(normalize=True)
dev_share = dev["stratum"].value_counts(normalize=True)
w = dev["stratum"].map(pop_share / dev_share).astype(float).to_numpy()
print(f"T = {T}   dev n={len(dev)} ({int(y.sum())} abandoned)   "
      f"pop shares: " + ", ".join(f"{k} {v:.0%}" for k, v in pop_share.items()))

# --- candidates ------------------------------------------------------------
c1 = LogisticRegression(C=1e6).fit(s.reshape(-1, 1), y, sample_weight=w)
c2 = IsotonicRegression(out_of_bounds="clip").fit(s, y, sample_weight=w)
c3 = LogisticRegression(C=1e6).fit(X3, y, sample_weight=w)
idx3 = c3.decision_function(X3)
c4 = IsotonicRegression(out_of_bounds="clip").fit(idx3, y, sample_weight=w)


def beta_feats(index):
    pp = 1.0 / (1.0 + np.exp(-index))
    pp = np.clip(pp, 1e-6, 1 - 1e-6)
    return np.c_[np.log(pp), -np.log(1.0 - pp)]


# C5: beta calibration (Kull et al. 2017) on the (score, age) index -
# 3 params, allows asymmetric/steeper-than-sigmoid shapes, smooth.
# Registered v4 rule: replaces C4 only if it (a) passes the age check,
# (b) is monotone (both beta coefs >= 0), (c) worst-miss <= C4's + 2pp.
c5 = LogisticRegression(C=1e6).fit(beta_feats(idx3), y, sample_weight=w)
beta_a, beta_b = float(c5.coef_[0][0]), float(c5.coef_[0][1])
beta_monotone = beta_a >= 0 and beta_b >= 0

preds = {
    "C1 logistic(score)": c1.predict_proba(s.reshape(-1, 1))[:, 1],
    "C2 isotonic(score)": c2.predict(s),
    "C3 logistic(score,age)": c3.predict_proba(X3)[:, 1],
    "C4 iso(logit(score,age))": c4.predict(idx3),
    "C5 beta(logit(score,age))": c5.predict_proba(beta_feats(idx3))[:, 1],
}

q = np.quantile(s, [0, .2, .4, .6, .8, 1.0])
qm = [(s >= q[i]) & ((s <= q[i + 1]) if i == 4 else (s < q[i + 1]))
      for i in range(5)]
print(f"\n{'quintile obs':>14}" + "".join(f"{k.split()[0]:>8}" for k in preds))
stats = {}
for k, pr in preds.items():
    misses, rows = [], []
    for m in qm:
        obs = np.average(y[m], weights=w[m])
        pd_ = np.average(pr[m], weights=w[m])
        misses.append(abs(obs - pd_))
        rows.append(pd_)
    rho = spearmanr(age, y - pr)
    stats[k] = dict(maxmiss=max(misses), rho=rho.statistic, p=rho.pvalue,
                    rows=rows)
for i, m in enumerate(qm):
    obs = np.average(y[m], weights=w[m])
    print(f"q{i+1}  {obs:>10.1%}" + "".join(f"{stats[k]['rows'][i]:>8.1%}"
                                            for k in preds))
print("\ncandidate                   worst-quintile-miss   age-residual rho (p)")
for k, st in stats.items():
    print(f"{k:<28}{st['maxmiss']:>12.1%}         {st['rho']:+.3f} "
          f"(p={st['p']:.3f}){'  FAILS age check' if st['p'] < 0.05 else ''}")

# --- selection rule (v4, fixed pre-run) ------------------------------------
# v3 rule chose C4. v4 registered rule: C5 replaces C4 iff it passes the
# age check AND is monotone AND its worst-miss <= C4's + 2pp (smoothness
# is the declared tie-breaker - the cliff is a documented artifact).
k4, k5 = "C4 iso(logit(score,age))", "C5 beta(logit(score,age))"
c5_ok = (stats[k5]["p"] >= 0.05 and beta_monotone
         and stats[k5]["maxmiss"] <= stats[k4]["maxmiss"] + 0.02)
winner = k5 if c5_ok else k4
print(f"\nbeta coefs a={beta_a:+.3f} b={beta_b:+.3f} "
      f"({'monotone' if beta_monotone else 'NON-MONOTONE - disqualified'})")
print(f"winner by v4 rule: {winner}")

# --- bootstrap bands + save winner -----------------------------------------
rng = np.random.default_rng(SEED)
grid = np.linspace(s.min(), s.max(), 101)
# bands must be a function of (score, age): a score-only band at median
# age is badly wrong for old crates (the age term dominates their index)
ages_grid = np.array([1.5, 2.5, 3.5, 5.0, 7.0, 9.0, 11.0, 13.0])
med_age = float(np.median(pop["crate_age_days"] / 365.25))


def fit_predict(idx, gs, ga):
    if winner.startswith("C1"):
        m = LogisticRegression(C=1e6).fit(s[idx].reshape(-1, 1), y[idx],
                                          sample_weight=w[idx])
        return m.predict_proba(gs.reshape(-1, 1))[:, 1]
    if winner.startswith("C2"):
        m = IsotonicRegression(out_of_bounds="clip").fit(s[idx], y[idx],
                                                         sample_weight=w[idx])
        return m.predict(gs)
    m1 = LogisticRegression(C=1e6).fit(np.c_[s[idx], age[idx]], y[idx],
                                       sample_weight=w[idx])
    if winner.startswith("C3"):
        return m1.predict_proba(np.c_[gs, ga])[:, 1]
    if winner.startswith("C5"):
        m2 = LogisticRegression(C=1e6).fit(
            beta_feats(m1.decision_function(np.c_[s[idx], age[idx]])), y[idx],
            sample_weight=w[idx])
        return m2.predict_proba(beta_feats(m1.decision_function(np.c_[gs, ga])))[:, 1]
    m2 = IsotonicRegression(out_of_bounds="clip").fit(
        m1.decision_function(np.c_[s[idx], age[idx]]), y[idx],
        sample_weight=w[idx])
    return m2.predict(m1.decision_function(np.c_[gs, ga]))


def surface(idx):
    return np.array([fit_predict(idx, grid, np.full_like(grid, a))
                     for a in ages_grid])


boots = []
for _ in range(1000):
    idx = rng.choice(len(dev), len(dev))
    if len(set(y[idx])) < 2:
        continue
    boots.append(surface(idx))
boots = np.array(boots)                       # (B, n_age, n_score)
p2d = surface(np.arange(len(dev)))
lo2d, hi2d = np.percentile(boots, [2.5, 97.5], axis=0)
ai_med = int(np.argmin(np.abs(ages_grid - med_age)))
p_grid, lo, hi = p2d[ai_med], lo2d[ai_med], hi2d[ai_med]

all_s = scores["risk_score"].to_numpy()
note = "" if winner.startswith(("C1", "C2")) else f" (at median crate age {med_age:.1f}y)"
print(f"\n{'registry percentile':<20}{'score':>8}   P(abandoned in ~2y){note}")
for pct in [10, 25, 50, 75, 90, 99]:
    v = np.percentile(all_s, pct)
    gi = int(np.argmin(np.abs(grid - v)))
    print(f"p{pct:<19}{v:>8.2f}{p_grid[gi]:>14.0%}  [{lo[gi]:.0%}, {hi[gi]:.0%}]")

r4 = lambda arr: [[round(float(v), 4) for v in row] for row in arr]
payload = dict(T=T, model=SCORES.format(T), method=winner, n_dev=len(dev),
               median_pop_age_years=med_age,
               grid_scores=[round(float(g), 4) for g in grid],
               grid_ages=[float(a) for a in ages_grid],
               p2d=r4(p2d), lo2d=r4(lo2d), hi2d=r4(hi2d))
if not winner.startswith(("C1", "C2")):
    payload["coef_score"] = float(c3.coef_[0][0])
    payload["coef_age_years"] = float(c3.coef_[0][1])
    payload["intercept"] = float(c3.intercept_[0])
    payload["age_note"] = ("score-grid above is evaluated at median population "
                          "age; CLI should compute index = coef_score*score + "
                          "coef_age_years*age_y + intercept for the crate's "
                          "actual age")
    if winner.startswith("C4"):
        gx = np.linspace(idx3.min(), idx3.max(), 101)
        payload["iso_index_curve"] = [
            dict(index=round(float(g_), 4), p=round(float(pp), 4))
            for g_, pp in zip(gx, c4.predict(gx))]
    if winner.startswith("C5"):
        payload["beta_a"] = beta_a
        payload["beta_b"] = beta_b
        payload["beta_c"] = float(c5.intercept_[0])
        payload["beta_note"] = ("p = sigmoid(a*ln(p') - b*ln(1-p') + c) "
                               "where p' = sigmoid(index)")
OUT.write_text(json.dumps(payload, indent=1))
print(f"\ncurve -> {OUT.relative_to(ROOT)}")
print("Claim shipped with every number: horizon ~2y, judge-relative, "
      "stationarity assumed, n=135 -> report the BAND, never the point.")
