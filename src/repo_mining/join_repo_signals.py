"""Attach mined GitHub repo signals to each crate + encode missingness as features.

Repo signals are per-REPO (repo_signals.parquet); crates are the unit of modelling,
so we map each crate -> its repository URL -> repo signals. Non-GitHub and no-URL
crates get NaN metrics, but we do NOT drop them - the absence is itself signal:

  has_repo_url    : crate declared a repository at all (17% don't)
  repo_host       : github / gitlab / codeberg / srht / gitee / other / none
  repo_status     : found | not_found | non_github | no_url | not_mined
                    (not_found = GitHub 404 = deleted/private/renamed = strong distress)
  has_repo_signal : mask - True only when real metrics are present, so linear models
                    don't misread imputed NaNs as real zeros.

Metrics (is_archived, days_since_push, commits_180d, ...) are NaN unless repo_status
== found. Output aligns 1:1 with features.parquet (eligible crates) by name.

NB monorepo caveat: many crates share one repo, so repo signals are duplicated
across them - a live repo does not prove a given crate within it is alive (logged).

Run (after fetch_github.py finishes):  uv run python src/repo_mining/join_repo_signals.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from snapshot import CRATE_REPOS  # noqa: E402
DERIVED = ROOT / "data" / "derived-data"
FEATURES = ROOT / "data" / "features.parquet"
OUT = DERIVED / "repo_features.parquet"

GH = re.compile(r"github\.com[/:]+([^/]+)/([^/#?]+)", re.I)
METRIC_COLS = ["is_archived", "is_fork", "is_empty", "is_disabled",
               "pushed_at", "days_since_push", "stars", "open_issues",
               "open_prs", "commits_180d",
               "last_pr_merged_at", "days_since_pr_merge",
               "last_issue_closed_at", "days_since_issue_close"]
# "no response" window for the unanswered-demand conjunctions below. A prior,
# not calibrated (Tier-2 GBTs get the raw days-since columns to learn their own).
UNANSWERED_DAYS = 365


def host_of(url: str | float) -> str:
    if not isinstance(url, str):
        return "none"
    m = re.search(r"https?://([^/]+)", url) or re.search(r"@([^:/]+)", url)
    h = (m.group(1).lower() if m else "")
    if "github.com" in h: return "github"
    if "gitlab" in h:     return "gitlab"
    if "codeberg" in h:   return "codeberg"
    if "sr.ht" in h:      return "srht"
    if "gitee" in h:      return "gitee"
    return "other" if h else "none"


feat = pd.read_parquet(FEATURES)[["name"]]
crates = pd.read_parquet(CRATE_REPOS)
df = feat.merge(crates, on="name", how="left")

df["has_repo_url"] = df["repository"].notna()
df["repo_host"] = df["repository"].map(host_of)

# extract github owner/name key for the lookup
def gh_key(url):
    if not isinstance(url, str):
        return None
    m = GH.search(url)
    return f"{m.group(1)}/{m.group(2).replace('.git', '')}".lower() if m else None

df["_key"] = df["repository"].map(gh_key)

# repo signals keyed by lowercased owner/name
sig = pd.read_parquet(OUT.parent / "repo_signals.parquet") if (DERIVED / "repo_signals.parquet").exists() else None
if sig is not None:
    sig = sig.copy()
    sig["_key"] = sig["repo"].str.lower()
    sig_status = sig.set_index("_key")["repo_status"].to_dict()
    metrics = {c: sig.set_index("_key")[c].to_dict() for c in METRIC_COLS if c in sig.columns}
else:
    print("WARNING: repo_signals.parquet not found - run fetch_github.py first. "
          "Emitting missingness features only.")
    sig_status, metrics = {}, {}


def status(row):
    if row["repo_host"] == "none":
        return "no_url"
    if row["repo_host"] != "github":
        return "non_github"
    s = sig_status.get(row["_key"])
    return s if s else "not_mined"  # github URL but absent from mine


df["repo_status"] = df.apply(status, axis=1)
df["has_repo_signal"] = df["repo_status"] == "found"

for c in METRIC_COLS:
    src = metrics.get(c, {})
    df[c] = df["_key"].map(src).where(df["has_repo_signal"])

# unanswered-demand conjunctions: demand exists (open
# items) AND no response in a year. LR can't learn interactions, so they are
# hand-engineered here. NaN days-since with open items = NEVER answered = the
# strongest form of the signal, hence the isna() arm - but only where the repo
# was actually mined (has_repo_signal), so unmined crates stay False, covered
# by the missingness indicators.
for flag, n_open, days in [("unanswered_issues", "open_issues", "days_since_issue_close"),
                           ("unanswered_prs", "open_prs", "days_since_pr_merge")]:
    df[flag] = (df["has_repo_signal"] & (df[n_open] > 0)
                & ((df[days] > UNANSWERED_DAYS) | df[days].isna()))

df = df.drop(columns=["repository", "_key"])
df.to_parquet(OUT, index=False)

# --- report -------------------------------------------------------------------
print(f"crates: {len(df):,}  ->  {OUT}")
print("repo_host:\n" + df["repo_host"].value_counts().to_string())
print("repo_status:\n" + df["repo_status"].value_counts().to_string())
print(f"has_repo_signal (real metrics): {int(df['has_repo_signal'].sum()):,} "
      f"({df['has_repo_signal'].mean():.1%})")
if df["has_repo_signal"].any():
    got = df[df["has_repo_signal"]]
    print(f"  archived={int(got['is_archived'].fillna(False).sum()):,}  "
          f"median days_since_push={got['days_since_push'].median():.0f}  "
          f"not_found(404)={int((df['repo_status'] == 'not_found').sum()):,}")
    print(f"  unanswered_issues={int(df['unanswered_issues'].sum()):,}  "
          f"unanswered_prs={int(df['unanswered_prs'].sum()):,}  "
          f"(open items, no close/merge in {UNANSWERED_DAYS}d)")
