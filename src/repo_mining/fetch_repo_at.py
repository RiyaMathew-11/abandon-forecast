"""Mine repo signals AS-OF a cutoff date T - the backtest's repo tier.

Reconstructs, per repo, what a 2024 observer could have seen (nothing after
T may leak):

  - last_commit_at_T   : newest default-branch commit <= T  (history until:T)
  - commits_180d_at_T  : commits in [T-180d, T]             (history since/until)
  - last_issue_close_at_T / last_pr_merge_at_T : maintainer responsiveness.
    No closed/merged-date sort or until-filter exists, so: fetch the 10 most
    recently UPDATED closed issues / merged PRs, keep max(date <= T). Repos
    whose 10 are ALL post-T are CENSORED (issue_censored / pr_censored=True,
    date=NaT): too active after T to see the past - a missingness indicator
    downstream, and itself weak evidence of liveliness. The asymmetry works
    for us: dead-by-T repos (the ones that matter) answer on page one.

Skipped by decision (30/07): stars@T, open counts@T - crowd signals with no
history; closing/merging = maintainer heartbeat, which is what we keep.

Population: repos of the backtest-eligible crates (features_at_<T>.parquet).
Resumable checkpoint like the main miner. ~50-70k repos, roughly 1-3 h.

Requires:  export GITHUB_TOKEN=ghp_xxx
Run:       uv run python src/repo_mining/fetch_repo_at.py [YYYY-MM-DD]
"""

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
CUT = pd.Timestamp(T, tz="UTC")
UNTIL = CUT.strftime("%Y-%m-%dT%H:%M:%SZ")
SINCE = (CUT - pd.Timedelta(days=180)).strftime("%Y-%m-%dT%H:%M:%SZ")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from snapshot import CRATE_REPOS  # noqa: E402
FEATURES_AT = ROOT / "data" / "derived-data" / f"features_at_{T}.parquet"
OUT = ROOT / "data" / "derived-data" / f"repo_signals_at_{T}.parquet"
API = "https://api.github.com/graphql"
TOKEN = os.environ.get("GITHUB_TOKEN")
BATCH = 50
CHECKPOINT_EVERY = 20
GH = re.compile(r"github\.com[/:]+([^/]+)/([^/#?]+)", re.I)

FRAGMENT = """
fragment R on Repository {
  nameWithOwner
  defaultBranchRef { target { ... on Commit {
    lastCommit: history(until: $until, first: 1) { nodes { committedDate } }
    window: history(since: $since, until: $until) { totalCount }
  } } }
  closedIssues: issues(states: CLOSED, first: 10, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes { closedAt } }
  mergedPRs: pullRequests(states: MERGED, first: 10, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes { mergedAt } }
}
"""


def run_query(batch):
    aliases = "\n".join(
        f'r{i}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) {{ ...R }}'
        for i, (o, n) in enumerate(batch)
    )
    query = (f"query($since: GitTimestamp!, $until: GitTimestamp!) {{\n{aliases}\n"
             f"  rateLimit {{ remaining resetAt }}\n}}\n{FRAGMENT}")
    payload = json.dumps({"query": query,
                          "variables": {"since": SINCE, "until": UNTIL}}).encode()
    req = urllib.request.Request(
        API, data=payload, method="POST",
        headers={"Authorization": f"bearer {TOKEN}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # 5xx = GitHub-side load (history queries are heavy; 504s happen),
            # 403 = secondary rate limit - both transient, back off and retry
            if e.code in (500, 502, 503, 504, 403) and attempt < 4:
                time.sleep(2 ** attempt * 5)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
            if attempt < 4:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    return {}


def pre_T_max(conn, field):
    """(max date <= T, censored). Censored = a FULL page, all post-T."""
    dates = [pd.Timestamp(n[field]) for n in (conn or {}).get("nodes") or []
             if n and n.get(field)]
    pre = [d for d in dates if d <= CUT]
    if pre:
        return max(pre), False
    return pd.NaT, len(dates) == 10


if not TOKEN:
    raise SystemExit("Set GITHUB_TOKEN (read-only PAT) in the environment first.")

elig = set(pd.read_parquet(FEATURES_AT)["name"])
crates = pd.read_parquet(CRATE_REPOS)
crates = crates[crates["name"].isin(elig)]
repos = set()
for u in crates["repository"].dropna().astype(str):
    m = GH.search(u)
    if m:
        repos.add((m.group(1), m.group(2).replace(".git", "")))
repos = sorted(repos)

done, results = set(), []
if OUT.exists():
    prev = pd.read_parquet(OUT)
    results = prev.to_dict("records")
    done = set(prev["repo"].str.lower())
todo = [(o, n) for (o, n) in repos if f"{o}/{n}".lower() not in done]
n_batches = (len(todo) + BATCH - 1) // BATCH
print(f"cutoff T = {T}   repos of eligible crates: {len(repos):,}   "
      f"already done: {len(done):,}   to fetch: {len(todo):,} ({n_batches:,} batches)")

t0 = time.monotonic()
tally = {"found": 0, "gone": 0, "stale1yr": 0, "i_cens": 0, "pr_cens": 0}
STALE_T = CUT - pd.Timedelta(days=365)
for b, start in enumerate(range(0, len(todo), BATCH)):
    batch = todo[start:start + BATCH]
    data = run_query(batch)
    body = data.get("data")
    if not body:
        raise SystemExit(f"GraphQL error, no data returned: {data.get('errors')}")
    for i, (o, n) in enumerate(batch):
        r = body.get(f"r{i}")
        if r is None:  # 404 now: unknowable at T -> missing (NOT a 2024 fact)
            results.append({"repo": f"{o}/{n}", "repo_status_at": "not_found_now"})
            tally["gone"] += 1
            continue
        target = (r.get("defaultBranchRef") or {}).get("target") or {}
        last = ((target.get("lastCommit") or {}).get("nodes") or [{}])
        lc = (pd.Timestamp(last[0]["committedDate"])
              if last and last[0].get("committedDate") else pd.NaT)
        ic, ic_cens = pre_T_max(r.get("closedIssues"), "closedAt")
        pm, pm_cens = pre_T_max(r.get("mergedPRs"), "mergedAt")
        results.append({
            "repo": f"{o}/{n}",
            "repo_status_at": "found",
            "last_commit_at_T": lc,
            "commits_180d_at_T": (target.get("window") or {}).get("totalCount"),
            "last_issue_close_at_T": ic, "issue_censored": ic_cens,
            "last_pr_merge_at_T": pm, "pr_censored": pm_cens,
        })
        tally["found"] += 1
        tally["stale1yr"] += bool(pd.notna(lc) and lc < STALE_T)
        tally["i_cens"] += ic_cens
        tally["pr_cens"] += pm_cens
    mined = start + len(batch)
    rl = body.get("rateLimit") or {}
    rate = mined / max(time.monotonic() - t0, 1e-9)
    print(f"batch {b + 1:,}/{n_batches:,}  {mined:,}/{len(todo):,} "
          f"({mined / len(todo):.1%})  last={batch[-1][0]}/{batch[-1][1]}  "
          f"run totals: found={tally['found']:,} gone={tally['gone']:,} "
          f"commit-stale1yr@T={tally['stale1yr']:,} "
          f"cens(i/pr)={tally['i_cens']:,}/{tally['pr_cens']:,}  "
          f"rl={rl.get('remaining')}  eta={(len(todo) - mined) / rate / 60:.0f}m")
    if (b + 1) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_parquet(OUT, index=False)
        print(f"  -- checkpoint: {len(results):,} rows -> {OUT.name}")
    rem = rl.get("remaining")
    if rem is not None and rem < 100:
        reset = pd.to_datetime(rl["resetAt"])
        wait = max(0, (reset - pd.Timestamp.now(tz="UTC")).total_seconds()) + 5
        print(f"  rate limit low ({rem}); sleeping {wait:.0f}s until reset")
        time.sleep(wait)

df = pd.DataFrame(results)
df.to_parquet(OUT, index=False)
found = df["repo_status_at"] == "found"
print(f"\ndone: {len(df):,} repos -> {OUT}")
print(f"  found={int(found.sum()):,}  gone-now={int((~found).sum()):,}")
got = df[found]
print(f"  median days since last commit at T: "
      f"{(CUT.tz_localize(None) - got['last_commit_at_T'].dt.tz_localize(None)).dt.days.median():.0f}")
print(f"  issue-censored: {int(got['issue_censored'].sum()):,}  "
      f"pr-censored: {int(got['pr_censored'].sum()):,}  (active post-T, past unseen)")
