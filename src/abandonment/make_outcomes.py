"""Assemble per-crate backtest outcomes for a cutoff T.

Joins the three dated sources onto the eligible-at-T population
(features_at_<T>.parquet):
  - RustSec advisories (rustsec_advisories.parquet, all dated)
  - repo archiving (archived_dates.parquet via the crate -> GitHub-repo key)
  - the snapshot end of observation (2026-07-16)

Per crate, in precedence order:
  declared_at_T  : unmaintained advisory OR repo archived BEFORE T ->
                   excluded from the scoring population (the CLI lookup layer
                   would have answered; nothing to predict)
  died_advisory  : unmaintained advisory filed in (T, snapshot]
  died_archived  : repo archived in (T, snapshot] (still archived now;
                   un-archived repos = retracted declarations, not outcomes)
  proved_alive   : PATCHED security advisory dated in the window and did not
                   die in it (the patch is the proof of maintenance)
  unlabeled      : everything else (PU treatment, as ever)

Monorepo caveat carries over: one archived repo marks ALL its crates
died_archived. archived_at is the DECLARATION date, not the death date
(cleanup-sweep lag) - the tool's claim is about predicting declarations.

Backtest evals must SPLIT by outcome source (died_advisory skews popular,
died_archived skews obscure - pooling would let the obscure mass dominate).

Run:  uv run python src/abandonment/make_outcomes.py [YYYY-MM-DD]
      (default 2024-07-16; writes data/derived-data/outcomes_<T>.parquet)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
CUT = pd.Timestamp(T)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import snapshot  # noqa: E402
from snapshot import CRATE_REPOS  # noqa: E402

SNAPSHOT = pd.Timestamp(snapshot.SNAPSHOT)
DERIVED = ROOT / "data" / "derived-data"
OUT = DERIVED / f"outcomes_{T}.parquet"
GH = re.compile(r"github\.com[/:]+([^/]+)/([^/#?]+)", re.I)

elig = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")[["name"]]

# crate -> github repo key (same extraction as join_repo_signals.py)
crates = pd.read_parquet(CRATE_REPOS)
def gh_key(url):
    m = GH.search(url) if isinstance(url, str) else None
    return f"{m.group(1)}/{m.group(2).replace('.git', '')}".lower() if m else None
crates["_key"] = crates["repository"].map(gh_key)
df = elig.merge(crates[["name", "_key"]], on="name", how="left")

# archiving events: valid = dated AND still archived (un-archived = retracted)
ad = pd.read_parquet(DERIVED / "archived_dates.parquet")
ad = ad[ad["archived_at"].notna() & (ad["still_archived"] != False)].copy()
ad["_key"] = ad["repo"].str.lower()
ad["archived_at"] = ad["archived_at"].dt.tz_localize(None)
arch_at = ad.groupby("_key")["archived_at"].max()
df["archived_at"] = df["_key"].map(arch_at)

# advisory events (non-withdrawn), per crate
adv = pd.read_parquet(DERIVED / "rustsec_advisories.parquet")
adv = adv[~adv["withdrawn"]]
unm = adv[adv["informational"] == "unmaintained"].groupby("crate_name")["date"].min()
df["unm_at"] = df["name"].map(unm)
sec = adv[adv["informational"].isna() & ~adv["unpatched"]]
df["patched_sec_in_window"] = df["name"].map(
    sec[(sec["date"] > CUT) & (sec["date"] <= SNAPSHOT)].groupby("crate_name").size()
).notna()

# precedence: declared@T > died_advisory > died_archived > proved_alive > unlabeled
declared_unm = df["unm_at"] <= CUT
declared_arch = df["archived_at"] <= CUT
died_adv = df["unm_at"].between(CUT, SNAPSHOT, inclusive="right")
died_arch = df["archived_at"].between(CUT, SNAPSHOT, inclusive="right")
df["outcome"] = np.select(
    [declared_unm | declared_arch, died_adv, died_arch, df["patched_sec_in_window"]],
    ["declared_at_T", "died_advisory", "died_archived", "proved_alive"],
    default="unlabeled")
df["declared_source"] = np.select(
    [declared_unm & declared_arch, declared_unm, declared_arch],
    ["both", "advisory", "archived"], default="")
df["outcome_date"] = df["unm_at"].where(died_adv, df["archived_at"].where(died_arch))

df = df.drop(columns=["_key"])
df.to_parquet(OUT, index=False)

print(f"cutoff T = {T}   eligible crates: {len(df):,}  ->  {OUT.relative_to(ROOT)}")
print(df["outcome"].value_counts().to_string())
print("\ndeclared_at_T by source:\n"
      + df.loc[df["outcome"] == "declared_at_T", "declared_source"].value_counts().to_string())
both = int((died_adv & died_arch).sum())
print(f"\ndied by BOTH channels in window (counted as died_advisory): {both}")
print(f"scoring population (undeclared at T): {int((df['outcome'] != 'declared_at_T').sum()):,}")
