"""Split features.parquet into a training set and a RustSec validation set.

The validation set = every crate that carries a RustSec advisory (join by name),
with the ground-truth label columns attached. The training set = everything else.

Why hold the advisory crates out: the training LABEL is the
release-inactivity proxy (`release_stale`), while the validation LABEL is RustSec truth -
different labels, so this is not leakage-prevention but hygiene: the RustSec metric
is then computed on crates the model's fitting never saw. We hold out ALL advisory
crates (not just `unmaintained`) so the security-only-but-patched ones can serve as
hard negatives (a fixed CVE must NOT read as abandoned).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"
FEATURES = ROOT / "data" / "features.parquet"

feat = pd.read_parquet(FEATURES)
labels = pd.read_parquet(DERIVED / "rustsec_labels.parquet")

# left-join RustSec truth onto features by crate name
merged = feat.merge(labels, left_on="name", right_on="crate_name", how="left")
has_adv = merged["has_advisory"].fillna(False).astype(bool)

validation = merged[has_adv].copy()
train = merged[~has_adv].drop(columns=labels.columns.difference(["crate_name"])).copy()

validation.to_parquet(DERIVED / "validation.parquet", index=False)
train.to_parquet(DERIVED / "train.parquet", index=False)

# --- coverage report ----------------------------------------------------------
n_unm_total = int(labels["is_unmaintained"].sum())
n_unm_matched = int(validation["is_unmaintained"].fillna(False).sum())
print(f"features rows: {len(feat):,}")
print(f"  train (no advisory):      {len(train):,}  ->  {DERIVED / 'train.parquet'}")
print(f"  validation (advisory):    {len(validation):,}  ->  {DERIVED / 'validation.parquet'}")
print(
    f"unmaintained crates: {n_unm_matched}/{n_unm_total} matched into features "
    f"({n_unm_total - n_unm_matched} absent - newer than 24mo, renamed, or non-crate)"
)
# which crates fell out of the join (absent from features.parquet)?
matched = set(feat["name"])
unmatched = labels[~labels["crate_name"].isin(matched)]
unmatched_unm = sorted(unmatched.loc[unmatched["is_unmaintained"].fillna(False), "crate_name"])
print(f"  unmatched UNMAINTAINED ({len(unmatched_unm)}): {', '.join(unmatched_unm)}")
print(f"  unmatched any-advisory ({len(unmatched)}): {', '.join(sorted(unmatched['crate_name']))}")
print("validation label counts:")
print(
    validation[["is_unmaintained", "is_unsound", "is_security", "has_unpatched_security"]]
    .fillna(False).sum().astype(int).rename("crates").to_string()
)
print("\nproxy `release_stale` rate within each RustSec class (sanity - expect "
      "unmaintained highest):")
for col in ["is_unmaintained", "is_security"]:
    sub = validation[validation[col].fillna(False)]
    if len(sub):
        print(f"  {col:<24} n={len(sub):<5} release_stale={sub['release_stale'].mean():.1%}")
