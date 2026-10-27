"""Split the decided gold labels into DEV / TEST halves - required before
any model tuning against the gold set.

DEV (half) = model selection: the sweep grades every config against it.
TEST (half) = sealed: read exactly once, for the single chosen model.
Tuning on all 268 and reporting on the same 268 would make the final
number circular - this split is what keeps the eventual headline honest.

Stratified by (stratum x gold_label) so both halves carry the same mix,
seeded, and REFUSES to re-run if the split already exists (re-splitting
would unseal the test half).

Run:  uv run python src/goldset/split_goldset.py [YYYY-MM-DD]
"""

import sys
from pathlib import Path

import pandas as pd

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"
LABELS = DERIVED / f"goldset_labels_v2_{T}.parquet"
OUT = DERIVED / f"goldset_split_{T}.parquet"

if OUT.exists():
    raise SystemExit(f"{OUT.name} already exists - never re-split; that would "
                     "unseal the test half.")

labels = pd.read_parquet(LABELS)
dec = labels[labels["gold_label"] != "unclear"].copy()

# shuffle once (seeded), then alternate dev/test within each cell so both
# halves get the same stratum x label composition
dec = dec.sample(frac=1, random_state=SEED).reset_index(drop=True)
alt = dec.groupby(["stratum", "gold_label"], observed=True).cumcount() % 2
dec["fold"] = alt.map({0: "dev", 1: "test"})

dec[["name", "stratum", "gold_label", "fold"]].to_parquet(OUT, index=False)
print(f"split -> {OUT.relative_to(ROOT)}")
print(pd.crosstab([dec["fold"], dec["gold_label"]], dec["stratum"]).to_string())
print(f"\ndev: {int((dec['fold'] == 'dev').sum())}   "
      f"test: {int((dec['fold'] == 'test').sum())}  (test is SEALED until the "
      "final single read)")
