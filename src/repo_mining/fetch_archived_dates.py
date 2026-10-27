"""Fetch archivedAt for every repo the main mine found archived (one-off).

The main mine (fetch_github.py) stored is_archived as a snapshot boolean -
no date. The backtest needs the DATE: archivedAt tells us (a)
which repos were already archived at cutoff T (declared -> excluded from the
scoring population) and (b) which were archived AFTER T (dated abandonment
outcomes for the backtest window).

Queries only the repos with is_archived=True in repo_signals.parquet
(~6-7k -> ~70 batches, minutes). Also records isArchived as of this fetch
(a repo can be UN-archived; archivedAt stays set - such repos are rare and
get archived_at as their last archiving). Repos that 404 since the main
mine get archived_at = NaT.

Resumable like the main miner; safe to re-run.

Requires:  export GITHUB_TOKEN=ghp_xxx
Run:       uv run python src/repo_mining/fetch_archived_dates.py
"""

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIG = ROOT / "data" / "derived-data" / "repo_signals.parquet"
OUT = ROOT / "data" / "derived-data" / "archived_dates.parquet"
API = "https://api.github.com/graphql"
TOKEN = os.environ.get("GITHUB_TOKEN")
BATCH = 100  # two scalar fields per repo - far cheaper than the main mine
CHECKPOINT_EVERY = 20


def run_query(batch: list[tuple[str, str]]) -> dict:
    aliases = "\n".join(
        f'r{i}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) '
        "{ nameWithOwner isArchived archivedAt }"
        for i, (o, n) in enumerate(batch)
    )
    payload = json.dumps({"query": f"query {{\n{aliases}\n  rateLimit {{ remaining }}\n}}"}).encode()
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
                time.sleep(2 ** attempt * 5)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException):
            if attempt < 4:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    return {}


if not TOKEN:
    raise SystemExit("Set GITHUB_TOKEN (read-only PAT) in the environment first.")

sig = pd.read_parquet(SIG)
repos = sorted({r for r in sig.loc[sig["is_archived"].fillna(False), "repo"]
                if isinstance(r, str) and "/" in r})

done, results = set(), []
if OUT.exists():
    prev = pd.read_parquet(OUT)
    results = prev.to_dict("records")
    done = set(prev["repo"].str.lower())
todo = [r for r in repos if r.lower() not in done]
print(f"archived repos: {len(repos):,}   already done: {len(done):,}   to fetch: {len(todo):,}")

for b, start in enumerate(range(0, len(todo), BATCH)):
    batch = [tuple(r.split("/", 1)) for r in todo[start:start + BATCH]]
    data = run_query(batch)
    body = data.get("data")
    if not body:
        raise SystemExit(f"GraphQL error, no data returned: {data.get('errors')}")
    for i, (o, n) in enumerate(batch):
        r = body.get(f"r{i}")
        results.append({
            "repo": f"{o}/{n}",
            "archived_at": pd.to_datetime(r["archivedAt"], utc=True)
                           if r and r.get("archivedAt") else pd.NaT,
            "still_archived": bool(r["isArchived"]) if r else None,
        })
    if (b + 1) % 5 == 0 or start + BATCH >= len(todo):
        print(f"batch {b + 1}/{(len(todo) + BATCH - 1) // BATCH}  "
              f"rl={(body.get('rateLimit') or {}).get('remaining')}")
    if (b + 1) % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_parquet(OUT, index=False)

df = pd.DataFrame(results)
df.to_parquet(OUT, index=False)
dated = df["archived_at"].notna()
print(f"\ndone: {len(df):,} repos -> {OUT}")
print(f"  archivedAt present: {int(dated.sum()):,}   gone/404 since mine: {int((~dated).sum()):,}")
if dated.any():
    yr = df.loc[dated, "archived_at"].dt.year.value_counts().sort_index()
    print("archived per year:\n" + yr.to_string())
    for T in ("2023-07-16", "2024-07-16"):
        after = (df["archived_at"] > pd.Timestamp(T, tz="UTC")).sum()
        print(f"archived AFTER {T}: {int(after):,}  (candidate backtest outcomes)")
