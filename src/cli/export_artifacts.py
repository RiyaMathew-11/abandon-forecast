"""M1: export the static artifacts cargo-pulse needs - everything a live
per-crate API call can never supply.

  model_trees.json.gz   the shipped ensemble (100 bags x 200 depth-2
                        trees) as compact node arrays + per-bag bias,
                        for the Rust tree-walker
  percentiles.json      1001 quantile cut-points of the undeclared
                        registry's score distribution at the snapshot
                        ("riskier than X%" needs everyone's score)
  owners.csv.gz         name -> last owner-added date. crate_owners.
                        created_at is dump-only (verified: /owners
                        returns no dates) and the feature is load-
                        bearing (12/08: dropping it cost -0.03 on both
                        channel contrasts)
  calibration.json      copy of the gold-fitted C4 curve
  meta.json             feature order, transforms, snapshot, model tag
  parity_sample.json    60 crates with dump-computed feature vectors +
                        ensemble scores - the Rust tree-walker must
                        reproduce these scores (train/serve harness)

Serving notes encoded in meta (from a manual pipeline test on
partiql-logical-planner): days_since_commit must come from the DEFAULT-
BRANCH commits endpoint, not pushed_at (212d vs 1d divergence there);
the offline percentile pass uses pushed_at as its approximation (the
detection mine has no default-branch commit date) - slight alive-ward
bias, documented.

Prereq:  uv run python src/abandonment/features_at.py 2026-07-16
Run:     uv run python src/cli/export_artifacts.py
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "gbt"))
from cox_model import (BT_BOOL, BT_LIN, BT_LOG, BT_REPO_BOOL, BT_REPO_LIN,  # noqa: E402
                       BT_REPO_LOG, GH, load_ensemble)
from pu_core import build_X  # noqa: E402
from snapshot import CRATE_OWNER_DATES, CRATE_REPOS, EXPORT_DIR, SNAPSHOT  # noqa: E402

SNAP = SNAPSHOT
MODEL_T = "2024-07-16"
TAG = "adv_d2_lr0.03_n200_m20_final"
SEED = 42

DERIVED = ROOT / "data" / "derived-data"
OUTDIR = EXPORT_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

feats = pd.read_parquet(DERIVED / f"features_at_{SNAP}.parquet")
print(f"eligible crates at snapshot {SNAP}: {len(feats):,}")

# --- repo tier at snapshot: adapt the detection-phase mine -----------------
snap_utc = pd.Timestamp(SNAP, tz="UTC")
rs = pd.read_parquet(DERIVED / "repo_signals.parquet")
rs = rs[rs["repo_status"] == "found"].copy()
rs["_key"] = rs["repo"].str.lower()
rs["days_since_commit_at_T"] = (snap_utc - rs["pushed_at"]).dt.days
rs["days_since_issue_close_at_T"] = (snap_utc - rs["last_issue_closed_at"]).dt.days
rs["days_since_pr_merge_at_T"] = (snap_utc - rs["last_pr_merged_at"]).dt.days
rs["issue_censored"] = rs["last_issue_closed_at"].isna()
rs["pr_censored"] = rs["last_pr_merged_at"].isna()
rs["commits_180d_at_T"] = rs["commits_180d"]
rs = rs.drop_duplicates("_key", keep="last")

crates = pd.read_parquet(CRATE_REPOS)


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
print(f"repo tier coverage at snapshot: {df['has_repo_at'].mean():.1%}")

X = build_X(df, BT_LOG + BT_REPO_LOG, BT_LIN + BT_REPO_LIN,
            BT_BOOL + BT_REPO_BOOL, [])

boosters, meta = load_ensemble(MODEL_T, TAG)
assert list(X.columns) == meta["features"], \
    f"feature mismatch:\n{list(X.columns)}\nvs\n{meta['features']}"

dm = xgb.DMatrix(X)
per_bag = np.array([b.predict(dm, output_margin=True) for b in boosters])
scores = per_bag.mean(axis=0)
print(f"scored {len(scores):,} crates with {len(boosters)} bags")

# --- percentile cut-points over the UNDECLARED population ------------------
lab = pd.read_parquet(DERIVED / "rustsec_labels.parquet")
unm = set(lab.loc[lab["is_unmaintained"].fillna(False), "crate_name"])
declared = (df["name"].isin(unm)
            | df["is_archived"].fillna(False).astype(bool)).to_numpy(dtype=bool)
und = scores[~declared]
print(f"undeclared for percentile distribution: {len(und):,} "
      f"(excluded {int(declared.sum()):,} declared)")
cutpoints = np.quantile(und, np.linspace(0, 1, 1001))
(OUTDIR / "percentiles.json").write_text(json.dumps(
    dict(n=int(len(und)), cutpoints=[round(float(c), 6) for c in cutpoints])))

# --- tree export: compact node arrays + per-bag empirical bias -------------
fidx = {f: i for i, f in enumerate(meta["features"])}


def compact(node, out):
    nid = node["nodeid"]
    while len(out) <= nid:
        out.append(None)
    if "leaf" in node:
        out[nid] = [-1, float(node["leaf"]), 0, 0, 0]
    else:
        out[nid] = [fidx[node["split"]], float(node["split_condition"]),
                    node["yes"], node["no"], node["missing"]]
        for ch in node["children"]:
            compact(ch, out)
    return out


def eval_tree(nodes, x):
    # booster compares in float32; f64 comparison flips cohorts whose
    # feature value IS the threshold (log1p of shared integer counts)
    i = 0
    while True:
        f, t, yes, no, miss = nodes[i]
        if f < 0:
            return t
        v = x[f]
        i = miss if np.isnan(v) else (yes if np.float32(v) < np.float32(t) else no)


probe_idx = np.random.default_rng(SEED).choice(len(X), 3, replace=False)
probes = X.iloc[probe_idx].to_numpy(dtype=float)
bags = []
for bi, b in enumerate(boosters):
    trees = [compact(json.loads(t), []) for t in b.get_dump(dump_format="json")]
    sums = np.array([[eval_tree(t, p) for t in trees] for p in probes]).sum(axis=1)
    bias = per_bag[bi][probe_idx] - sums
    assert bias.std() < 1e-4, f"bag {bi}: non-constant bias {bias}"
    bags.append(dict(bias=float(bias.mean()), trees=trees))
with gzip.open(OUTDIR / "model_trees.json.gz", "wt") as fh:
    json.dump(bags, fh)
print(f"exported {len(bags)} bags x {len(bags[0]['trees'])} trees "
      f"(bias check passed on 3 probes)")

# --- owner sidecar (dump-only data: crate_owners.created_at) ---------------
own = pd.read_parquet(CRATE_OWNER_DATES)
with gzip.open(OUTDIR / "owners.csv.gz", "wt") as fh:
    own.to_csv(fh, index=False)
print(f"owner sidecar: {len(own):,} crates")

# --- calibration copy + meta -----------------------------------------------
calib = json.loads((DERIVED / f"calibration_{MODEL_T}.json").read_text())
(OUTDIR / "calibration.json").write_text(json.dumps(calib, indent=1))
(OUTDIR / "meta.json").write_text(json.dumps(dict(
    snapshot=SNAP, model_T=MODEL_T, model_tag=TAG, n_bags=len(bags),
    features=meta["features"], eligibility_months=18, horizon="~2 years",
    log1p_features=[f for f in meta["features"] if f.startswith("log_")],
    notes="months_since_release = calendar-month boundary count; cadence "
          "gaps over ALL versions incl yanked; is_pre_1_0 = max leading "
          "major over all versions == 0; days_since_commit = DEFAULT-"
          "BRANCH latest commit (not pushed_at)"), indent=1))

# --- parity sample ----------------------------------------------------------
rng = np.random.default_rng(SEED)
pick = rng.choice(len(X), 60, replace=False)
sample = []
for i in pick:
    row = X.iloc[int(i)]
    sample.append(dict(
        name=str(df["name"].iloc[int(i)]),
        features=[None if pd.isna(v) else float(v) for v in row],
        score=float(scores[int(i)])))
(OUTDIR / "parity_sample.json").write_text(json.dumps(sample, indent=1))
print(f"parity sample: {len(sample)} crates")

# internal full-parity assertion: compact trees + f32 semantics must
# reproduce the booster on all 60 sample crates before anything ships
worst = 0.0
for s_row in sample:
    xv = np.array([np.nan if v is None else v for v in s_row["features"]])
    mine = float(np.mean([b["bias"] + sum(eval_tree(t, xv) for t in b["trees"])
                          for b in bags]))
    worst = max(worst, abs(mine - s_row["score"]))
assert worst < 1e-3, f"internal parity FAIL: max diff {worst:.2e}"
print(f"internal parity: max |compact - booster| = {worst:.2e}  PASS")
print(f"\nartifacts -> {OUTDIR.relative_to(ROOT) if OUTDIR.is_relative_to(ROOT) else OUTDIR}/")
