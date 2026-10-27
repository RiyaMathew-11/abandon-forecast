"""Mine the evidence bundle for each gold-set crate - the raw material the
LLM (and the human verifier) will judge maintained/abandoned from.

Two sides, deliberately including evidence OUTSIDE the model's feature set
(otherwise the gold label would just re-implement the staleness rule and the
eval would be circular):

  Registry (local, no API): releases in (T, snapshot] and the last release
    date - post-cutoff activity is legitimate outcome information.
  GitHub (GraphQL, current state): archived flag + date, repo description,
    README text (deprecation/moved notices live here), pushed_at, commits
    since T, open-issue count and the 5 most recently updated open issues
    (title + age + reply count - unanswered demand is the semantic signal),
    and (v2) the PR side of demand: open-PR count, 5 most recent open PRs
    with author + bot flag, and the most recent merged-PR date - human PRs
    ignored for years are rule-3 evidence the issue list alone misses.

Crates without a resolvable GitHub repo keep registry evidence only (gh_found
= False) - dropping them would bias the sample toward well-kept crates.

Requires:  export GITHUB_TOKEN=ghp_xxx   (read-only PAT)
Run:       uv run python src/goldset/fetch_goldset_evidence.py [YYYY-MM-DD]
           (writes data/derived-data/goldset_evidence_<T>.parquet; resumable)
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from snapshot import DATA, SNAPSHOT  # noqa: E402

DERIVED = ROOT / "data" / "derived-data"
SAMPLE = DERIVED / f"goldset_sample_{T}.parquet"
OUT = DERIVED / f"goldset_evidence_{T}.parquet"
API = "https://api.github.com/graphql"
TOKEN = os.environ.get("GITHUB_TOKEN")
BATCH = 20
README_CHARS = 3000

FRAGMENT = """
fragment R on Repository {
  nameWithOwner
  isArchived
  archivedAt
  pushedAt
  description
  readme: object(expression: "HEAD:README.md") { ... on Blob { text } }
  openIssues: issues(states: OPEN) { totalCount }
  recentIssues: issues(states: OPEN, first: 5,
                       orderBy: {field: UPDATED_AT, direction: DESC}) {
    nodes { title createdAt comments { totalCount } }
  }
  openPRs: pullRequests(states: OPEN) { totalCount }
  recentPRs: pullRequests(states: OPEN, first: 5,
                          orderBy: {field: UPDATED_AT, direction: DESC}) {
    nodes { title createdAt comments { totalCount }
            author { login __typename } }
  }
  mergedPRs: pullRequests(states: MERGED, first: 5,
                          orderBy: {field: UPDATED_AT, direction: DESC}) {
    nodes { mergedAt }
  }
  defaultBranchRef { target { ... on Commit {
    commitsSinceT: history(since: $since) { totalCount }
  } } }
}
"""


def run_query(batch):
    aliases = "\n".join(
        f'r{i}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) {{ ...R }}'
        for i, (o, n) in enumerate(batch))
    query = (f"query($since: GitTimestamp!) {{\n{aliases}\n"
             f"  rateLimit {{ remaining }}\n}}\n{FRAGMENT}")
    payload = json.dumps({"query": query,
                          "variables": {"since": f"{T}T00:00:00Z"}}).encode()
    req = urllib.request.Request(
        API, data=payload, method="POST",
        headers={"Authorization": f"bearer {TOKEN}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt < 4:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    return {}


if not TOKEN:
    raise SystemExit("Set GITHUB_TOKEN (read-only PAT) in the environment first.")

sample = pd.read_parquet(SAMPLE)

# --- registry side: releases after T, from the local dump ------------------
con = duckdb.connect()
con.execute(f"""CREATE VIEW versions AS SELECT * FROM read_csv(
    '{DATA}/versions.csv', sample_size=-1, max_line_size=20000000)""")
con.execute(f"""CREATE VIEW crates AS SELECT * FROM read_csv(
    '{DATA}/crates.csv', sample_size=-1, max_line_size=20000000)""")
con.register("sample_names", sample[["name"]])
rel = con.execute(f"""
    SELECT c.name,
           count(*) FILTER (WHERE v.created_at > DATE '{T}'
                            AND NOT v.yanked)          AS releases_post_T,
           max(v.created_at) FILTER (WHERE NOT v.yanked) AS last_release
    FROM versions v
    JOIN crates c ON c.id = v.crate_id
    JOIN sample_names s ON s.name = c.name
    GROUP BY c.name
""").df()
con.close()
df = sample.merge(rel, on="name", how="left")
df["releases_post_T"] = df["releases_post_T"].fillna(0).astype(int)

# --- github side: current state (post-window is legit outcome info) --------
done = set()
gh_rows = []
if OUT.exists():
    prev = pd.read_parquet(OUT)
    # a file without the v2 PR columns is stale - refetch everything
    if "gh_found" in prev.columns and "gh_open_prs" in prev.columns:
        gh_cols = [c for c in prev.columns if c.startswith("gh_") or c == "repo"]
        gh_rows = prev[prev["gh_found"].notna()][["name"] + gh_cols].to_dict("records")
        done = {r["name"] for r in gh_rows}

todo = df[df["repo"].notna() & ~df["name"].isin(done)][["name", "repo"]]
repos = list(todo.itertuples(index=False))
print(f"T = {T}  sample: {len(df)}  with repo: {int(df['repo'].notna().sum())}  "
      f"already mined: {len(done)}  to fetch: {len(repos)}")

for start in range(0, len(repos), BATCH):
    batch = repos[start:start + BATCH]
    parts = [r.repo.split("/", 1) for r in batch]
    data = run_query([(o, n) for o, n in parts])
    body = data.get("data") or {}
    for i, row in enumerate(batch):
        r = body.get(f"r{i}")
        if r is None:
            gh_rows.append({"name": row.name, "repo": row.repo, "gh_found": False})
            continue
        target = (r.get("defaultBranchRef") or {}).get("target") or {}
        issues = [{"title": n["title"][:120], "created": n["createdAt"][:10],
                   "replies": n["comments"]["totalCount"]}
                  for n in (r.get("recentIssues") or {}).get("nodes") or [] if n]
        prs = [{"title": n["title"][:120], "created": n["createdAt"][:10],
                "replies": n["comments"]["totalCount"],
                "author": (n.get("author") or {}).get("login") or "",
                "bot": (n.get("author") or {}).get("__typename") == "Bot"
                       or "bot" in ((n.get("author") or {}).get("login") or "").lower()}
               for n in (r.get("recentPRs") or {}).get("nodes") or [] if n]
        merged = [(n.get("mergedAt") or "")[:10]
                  for n in (r.get("mergedPRs") or {}).get("nodes") or [] if n]
        readme = ((r.get("readme") or {}).get("text") or "")[:README_CHARS]
        gh_rows.append({
            "name": row.name, "repo": row.repo, "gh_found": True,
            "gh_archived": bool(r.get("isArchived")),
            "gh_archived_at": (r.get("archivedAt") or "")[:10],
            "gh_pushed_at": (r.get("pushedAt") or "")[:10],
            "gh_description": r.get("description") or "",
            "gh_readme": readme,
            "gh_open_issues": (r.get("openIssues") or {}).get("totalCount"),
            "gh_recent_issues": json.dumps(issues),
            "gh_open_prs": (r.get("openPRs") or {}).get("totalCount"),
            "gh_recent_prs": json.dumps(prs),
            "gh_last_merged_pr": max(merged) if merged else "",
            "gh_commits_since_T": (target.get("commitsSinceT") or {}).get("totalCount"),
        })
    print(f"  {min(start + BATCH, len(repos))}/{len(repos)}  "
          f"rl={((data.get('data') or {}).get('rateLimit') or {}).get('remaining')}")

gh = pd.DataFrame(gh_rows) if gh_rows else pd.DataFrame(columns=["name", "gh_found"])
out = df.merge(gh.drop(columns=["repo"], errors="ignore"), on="name", how="left")
out["gh_found"] = out["gh_found"].fillna(False)
out.to_parquet(OUT, index=False)

print(f"\ndone -> {OUT.relative_to(ROOT)}")
print(f"  gh found: {int(out['gh_found'].sum())}   archived now: "
      f"{int(out.get('gh_archived', pd.Series(dtype=bool)).fillna(False).sum())}")
print(f"  released after T (non-yanked): {int((out['releases_post_T'] > 0).sum())}"
      f" / {len(out)}")
