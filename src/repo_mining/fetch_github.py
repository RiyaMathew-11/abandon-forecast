"""Mine GitHub repo signals for crates via the GraphQL API (cheap + medium tier).

Signals (all from one aliased GraphQL query, ~1 point per batch of BATCH repos):
  - pushed_at / days_since_push  -> RustSec "stale repo, 1yr+ inactivity" criterion
  - is_archived                  -> owner explicitly archived it (clean abandonment)
  - commits_180d                 -> recent maintenance activity (medium tier)
  - last PR merge / issue close  -> responsiveness: raw
    ingredients for the "unanswered demand" conjunctions built in the join step.
    No MERGED_AT/CLOSED_AT sort exists, so: UPDATED_AT desc, first 5, max(date).
  - open_issues / open_prs, stars, is_fork/is_empty/is_disabled  -> context
  - repo_status: found | not_found -> a 404 (deleted/private/renamed) is itself a
    strong abandonment signal; recorded distinctly from "not mined".

Only GitHub is mined (94% of repo URLs). Non-GitHub / no-URL crates are handled
downstream as structured missingness (has_repo_url, repo_host) - NOT backfilled.

Resumable: results checkpoint to repo_signals.parquet; a re-run skips done repos.
Rate-limit aware: watches rateLimit.remaining and sleeps until reset.

Requires a token:  export GITHUB_TOKEN=ghp_xxx   (read-only / public_repo scope)
Run:               uv run python src/repo_mining/fetch_github.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import snapshot  # noqa: E402
from snapshot import CRATE_REPOS  # noqa: E402
OUT = ROOT / "data" / "derived-data" / "repo_signals.parquet"
SNAPSHOT = pd.Timestamp(snapshot.SNAPSHOT, tz="UTC")
SINCE = (SNAPSHOT - pd.Timedelta(days=180)).strftime("%Y-%m-%dT%H:%M:%SZ")  # commits_180d window

TOKEN = os.environ.get("GITHUB_TOKEN")
API = "https://api.github.com/graphql"
BATCH = 50            # repos aliased per query
CHECKPOINT_EVERY = 20  # write parquet every N batches (= 1000 repos)

GH = re.compile(r"github\.com[/:]+([^/]+)/([^/#?]+)", re.I)

FRAGMENT = """
fragment R on Repository {
  nameWithOwner
  isArchived
  isFork
  isEmpty
  isDisabled
  pushedAt
  updatedAt
  stargazerCount
  openIssues: issues(states: OPEN) { totalCount }
  openPRs: pullRequests(states: OPEN) { totalCount }
  lastMergedPRs: pullRequests(states: MERGED, first: 5, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes { mergedAt } }
  lastClosedIssues: issues(states: CLOSED, first: 5, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes { closedAt } }
  defaultBranchRef { target { ... on Commit { history(since: $since) { totalCount } } } }
}
"""


def extract_repos() -> list[tuple[str, str]]:
    """Distinct (owner, name) GitHub repos across ELIGIBLE crates (dedups
    monorepos). Restricting to features.parquet halves the mine vs raw
    crates.csv (76,768 repos vs 139,929); falls back to all crates if absent."""
    c = pd.read_parquet(CRATE_REPOS)
    eligible = ROOT / "data" / "features.parquet"
    if eligible.exists():
        c = c[c["name"].isin(set(pd.read_parquet(eligible)["name"]))]
    repos = set()
    for u in c["repository"].dropna().astype(str):
        m = GH.search(u)
        if m:
            repos.add((m.group(1), m.group(2).replace(".git", "")))
    return sorted(repos)


def run_query(batch: list[tuple[str, str]]) -> dict:
    aliases = "\n".join(
        f'r{i}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) {{ ...R }}'
        for i, (o, n) in enumerate(batch)
    )
    query = f"query($since: GitTimestamp!) {{\n{aliases}\n  rateLimit {{ remaining resetAt cost }}\n}}\n{FRAGMENT}"
    payload = json.dumps({"query": query, "variables": {"since": SINCE}}).encode()
    req = urllib.request.Request(
        API, data=payload, method="POST",
        headers={"Authorization": f"bearer {TOKEN}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (500, 502, 503, 504, 403) and attempt < 4:
                time.sleep(2 ** attempt * 5)  # backoff on transient / secondary limit
                continue
            raise
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
            # HTTPException covers IncompleteRead / RemoteDisconnected -
            # connection dropped mid-response (seen 30/07); retry like a timeout
            if attempt < 4:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    return {}


def max_date(conn, field) -> pd.Timestamp:
    """Latest `field` timestamp across a connection's nodes (NaT if none).
    UPDATED_AT-desc ordering is a proxy (an old PR commented on later bubbles
    up), so we fetch 5 and take the max - nearly always the true latest."""
    dates = [n[field] for n in (conn or {}).get("nodes") or [] if n and n.get(field)]
    return pd.to_datetime(max(dates), utc=True) if dates else pd.NaT


def parse(batch, data) -> list[dict]:
    rows = []
    for i, (owner, name) in enumerate(batch):
        r = data.get(f"r{i}")
        key = f"{owner}/{name}"
        if r is None:  # null alias == repo not found (deleted / private / renamed)
            rows.append({"repo": key, "repo_status": "not_found"})
            continue
        pushed = pd.to_datetime(r["pushedAt"], utc=True) if r.get("pushedAt") else pd.NaT
        merged = max_date(r.get("lastMergedPRs"), "mergedAt")
        closed = max_date(r.get("lastClosedIssues"), "closedAt")
        target = (r.get("defaultBranchRef") or {}).get("target") or {}
        commits = (target.get("history") or {}).get("totalCount")
        rows.append({
            # keyed by the QUERIED owner/name (what crate URLs point at), not
            # nameWithOwner: renamed/transferred repos redirect, and storing the
            # canonical name orphaned 7k repos at join time (30/07). The
            # canonical name is kept alongside - a rename is itself a signal.
            "repo": key,
            "repo_canonical": r["nameWithOwner"],
            "repo_status": "found",
            "is_archived": r["isArchived"],
            "is_fork": r["isFork"],
            "is_empty": r["isEmpty"],
            "is_disabled": r["isDisabled"],
            "pushed_at": pushed,
            "days_since_push": (SNAPSHOT - pushed).days if pd.notna(pushed) else None,
            "stars": r["stargazerCount"],
            "open_issues": r["openIssues"]["totalCount"],
            "open_prs": r["openPRs"]["totalCount"],
            "commits_180d": commits,
            # responsiveness: NaT = never merged a PR / never closed an issue -
            # meaningful, not missing
            "last_pr_merged_at": merged,
            "days_since_pr_merge": (SNAPSHOT - merged).days if pd.notna(merged) else None,
            "last_issue_closed_at": closed,
            "days_since_issue_close": (SNAPSHOT - closed).days if pd.notna(closed) else None,
        })
    return rows


if not TOKEN:
    raise SystemExit("Set GITHUB_TOKEN (read-only PAT) in the environment first.")
repos = extract_repos()

done, results = set(), []
if OUT.exists():
    prev = pd.read_parquet(OUT)
    results = prev.to_dict("records")
    done = set(prev["repo"].str.lower())
todo = [(o, n) for (o, n) in repos if f"{o}/{n}".lower() not in done]
n_batches = (len(todo) + BATCH - 1) // BATCH
print(f"distinct github repos: {len(repos):,}   already done: {len(done):,}   "
      f"to fetch: {len(todo):,} ({n_batches:,} batches of {BATCH})")

t0 = time.monotonic()
tally = {"found": 0, "not_found": 0, "archived": 0, "stale_1yr": 0}

for b, start in enumerate(range(0, len(todo), BATCH)):
    batch = todo[start:start + BATCH]
    data = run_query(batch)
    body = data.get("data")
    if not body:  # query-level failure (bad token/syntax) - do NOT mislabel as 404
        raise SystemExit(f"GraphQL error, no data returned: {data.get('errors')}")
    rows = parse(batch, body)
    results.extend(rows)

    # per-batch progress line: position, last repo mined, what this batch yielded,
    # running tallies (this run only), rate limit and ETA
    found = sum(r["repo_status"] == "found" for r in rows)
    tally["found"] += found
    tally["not_found"] += len(rows) - found
    tally["archived"] += sum(bool(r.get("is_archived")) for r in rows)
    tally["stale_1yr"] += sum((r.get("days_since_push") or 0) > 365 for r in rows)
    mined = start + len(batch)
    rl = body.get("rateLimit") or {}
    rem = rl.get("remaining")
    rate = mined / max(time.monotonic() - t0, 1e-9)  # repos/s this run
    eta_min = (len(todo) - mined) / rate / 60
    print(f"batch {b + 1:,}/{n_batches:,}  {mined:,}/{len(todo):,} "
          f"({mined / len(todo):.1%})  last={batch[-1][0]}/{batch[-1][1]}  "
          f"+found={found} +404={len(rows) - found}  "
          f"run totals: found={tally['found']:,} 404={tally['not_found']:,} "
          f"archived={tally['archived']:,} stale1yr={tally['stale_1yr']:,}  "
          f"rl={rem}  eta={eta_min:.0f}m")

    if (b + 1) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_parquet(OUT, index=False)
        print(f"  -- checkpoint: {len(results):,} rows -> {OUT.name}")
    if rem is not None and rem < 50:  # near the points cap - wait for reset
        reset = pd.to_datetime(rl["resetAt"])
        wait = max(0, (reset - pd.Timestamp.now(tz="UTC")).total_seconds()) + 5
        print(f"  rate limit low ({rem}); sleeping {wait:.0f}s until reset")
        time.sleep(wait)

df = pd.DataFrame(results)
df.to_parquet(OUT, index=False)
found = (df["repo_status"] == "found").sum()
print(f"\ndone: {len(df):,} repos -> {OUT}")
print(f"  found={found:,}  not_found(404/private)={len(df) - found:,}")
if "is_archived" in df:
    print(f"  archived={int(df['is_archived'].fillna(False).sum()):,}  "
          f"median days_since_push={df['days_since_push'].median():.0f}")
