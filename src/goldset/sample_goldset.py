"""Draw the gold-set sample: a random, stratified slice of the scoring
population to be labeled maintained/abandoned by GitHub mining + LLM + manual
verification (the Xu et al. labeling pipeline, pointed at our judge gap).

Design, registered BEFORE any label is looked at:
  - Population = the scoring population at T (undeclared at T) - the same
    crates every model is graded on. No popularity filter.
  - Stratified by reverse_deps at T: 0 / 1-9 / 10+, N_PER_STRATUM each, so
    the sample contains dust and celebrities alike (the 73-alive judge is
    all-celebrity; this is the corrective).
  - Fixed seed, drawn BLIND to model scores: this script never reads a
    scores parquet. Manual/LLM effort is verification, never selection.
  - The declaration-channel outcome is carried along for a later agreement
    check (gold label vs channel label), not used in sampling.

Run:  uv run python src/goldset/sample_goldset.py [YYYY-MM-DD]
      (default 2024-07-16; writes data/derived-data/goldset_sample_<T>.parquet)
"""

import re
import sys
from pathlib import Path

import pandas as pd

def gh_key(url):
    m = GH.search(url) if isinstance(url, str) else None
    return f"{m.group(1)}/{m.group(2).replace('.git', '')}" if m else None

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
SEED = 42
N_PER_STRATUM = 100

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from snapshot import CRATE_REPOS  # noqa: E402

DERIVED = ROOT / "data" / "derived-data"
OUT = DERIVED / f"goldset_sample_{T}.parquet"
GH = re.compile(r"github\.com[/:]+([^/]+)/([^/#?]+)", re.I)

feats = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")
outc = pd.read_parquet(DERIVED / f"outcomes_{T}.parquet")[["name", "outcome"]]
df = feats.merge(outc, on="name")
df = df[df["outcome"] != "declared_at_T"].copy()  # scoring population only

crates = pd.read_parquet(CRATE_REPOS)

df["repo"] = df["name"].map(
    crates.drop_duplicates("name").set_index("name")["repository"]).map(gh_key)

df["stratum"] = pd.cut(df["reverse_deps"], bins=[-1, 0, 9, float("inf")],
                       labels=["revdeps_0", "revdeps_1_9", "revdeps_10plus"])

sample = (df.sample(frac=1, random_state=SEED)
            .groupby("stratum", observed=True)
            .head(N_PER_STRATUM))
sample = sample[["name", "repo", "reverse_deps", "n_versions", "crate_age_days",
                 "months_since_release", "stratum", "outcome"]].reset_index(drop=True)
sample.to_parquet(OUT, index=False)

print(f"cutoff T = {T}  seed = {SEED}  ->  {OUT.relative_to(ROOT)}")
print(f"population (undeclared at T): {len(df):,}   sampled: {len(sample)}")
print("\nby stratum:\n" + sample["stratum"].value_counts().to_string())
print("\ndeclaration-channel outcome inside the sample (for the later "
      "agreement check):\n" + sample["outcome"].value_counts().to_string())
print(f"\nwith a GitHub repo: {int(sample['repo'].notna().sum())} / {len(sample)}"
      "  (no-repo crates STAY in the sample - registry evidence only)")
