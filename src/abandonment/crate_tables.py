"""Two small registry tables lifted out of the raw dump so that no later
stage needs the dump itself:

  crate_repos.parquet        name, repository      (the crate -> repo URL map
                                                    behind every GitHub join)
  crate_owner_dates.parquet  name, last_owner_added (date the most recent
                                                    owner was added; cargo-pulse's
                                                    owner sidecar)

Both ship in the Zenodo bundle. This stage (`make crates`) is the only one
besides features_at.py that reads the dump CSVs.

Run:  uv run python src/abandonment/crate_tables.py
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapshot import CRATE_OWNER_DATES, CRATE_REPOS, DATA, DERIVED, ROOT  # noqa: E402

BIG = "max_line_size=20000000"
DERIVED.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
repos = con.execute(f"""
    SELECT name, repository
    FROM read_csv('{DATA}/crates.csv', sample_size=-1, {BIG})
    ORDER BY name
""").df()
owners = con.execute(f"""
    SELECT c.name, max(o.created_at)::date AS last_owner_added
    FROM read_csv('{DATA}/crate_owners.csv', sample_size=-1) o
    JOIN read_csv('{DATA}/crates.csv', sample_size=-1, {BIG}) c ON c.id = o.crate_id
    GROUP BY c.name
    ORDER BY c.name
""").df()
con.close()

repos.to_parquet(CRATE_REPOS, index=False)
owners.to_parquet(CRATE_OWNER_DATES, index=False)
print(f"{len(repos):,} crates ({repos['repository'].notna().sum():,} with a repository URL) "
      f"-> {CRATE_REPOS.relative_to(ROOT)}")
print(f"{len(owners):,} crates with owner dates -> {CRATE_OWNER_DATES.relative_to(ROOT)}")
