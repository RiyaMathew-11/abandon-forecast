"""Registry features AS-OF a cutoff date T - the backtest feature builder.

Prognosis design: features computed only
from information that existed at T; outcomes (advisories / archivedAt) come
from AFTER T. This script builds the registry side by replaying the versions
table up to T.

Kept (fully dated, from versions/crates/crate_owners created_at):
  months_since_release@T, n_versions@T, crate_age_days@T, is_pre_1_0@T,
  release_cadence_decay@T, n_owners@T, days_since_owner_change@T,
  reverse_deps@T (approximation: dependencies declared by versions published
  before T; crates deleted since still count - acceptable).

DROPPED vs the snapshot-time feature set - state-only, no history:
  downloads_total/90d, momentum_index (download-based), yank_rate /
  latest_yanked (the yanked flag has no yank DATE), stars/404/anything repo
  (repo history@T is mined separately). The dropped set is mostly what the
  coefficient audit found to be dead weight or popularity residue; the
  staleness backbone survives intact.

Known impurity, accepted + logged: crate_owners has no removal history, so
n_owners@T counts current owners added before T (removed-then owners lost).

Eligibility at T: created >= ELIGIBLE_MONTHS before T AND at least one
release by T. Set to 18, NOT the snapshot feature set's 24:
the 24 existed to stay longer than the proxy's 18-month stale window, and
the backtest has no proxy label - outcomes are dated events. 18 months is
enough for a lasting crate to show its pattern (major activity concentrates
in the first 1-2 years). Age-based gate
only: >= 18mo old + >= 1 release EVER; steady activity is measured by the
features, never required for eligibility.

Run:  uv run python src/abandonment/features_at.py [YYYY-MM-DD]
      (default cutoff 2024-07-16; writes data/derived-data/features_at_<T>.parquet)
"""

import sys
from pathlib import Path

import duckdb

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
CUTOFF = f"DATE '{T}'"
ELIGIBLE_MONTHS = 18  # eligibility gate at T (snapshot feature set keeps 24)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from snapshot import DATA  # noqa: E402
OUT = ROOT / "data" / "derived-data" / f"features_at_{T}.parquet"

BIG = "max_line_size=20000000"
VIEWS = {
    "crates":       f"read_csv('{DATA}/crates.csv', sample_size=-1, {BIG})",
    "versions":     f"read_csv('{DATA}/versions.csv', sample_size=-1, {BIG})",
    "dependencies": f"read_csv('{DATA}/dependencies.csv', sample_size=-1, {BIG})",
    "crate_owners": f"read_csv('{DATA}/crate_owners.csv', sample_size=-1)",
}

QUERY = f"""
WITH v AS (  -- the versions table as it stood at T
    SELECT * FROM versions WHERE created_at <= {CUTOFF}
),
last_rel AS (
    SELECT crate_id,
           max(created_at) AS last_release,
           count(*)        AS n_versions,
           (max(try_cast(split_part(num, '.', 1) AS INTEGER)) = 0)::int AS is_pre_1_0
    FROM v GROUP BY crate_id
),
cadence AS (  -- release cadence over pre-T releases only
    SELECT crate_id,
           median(gap)              AS median_gap,
           arg_max(gap, created_at) AS last_gap
    FROM (
        SELECT crate_id, created_at,
               date_diff('day',
                   lag(created_at) OVER (PARTITION BY crate_id ORDER BY created_at),
                   created_at) AS gap
        FROM v
    ) g
    WHERE gap IS NOT NULL
    GROUP BY crate_id
),
owners AS (
    SELECT crate_id, count(*) AS n_owners,
           max(created_at) AS last_owner_change
    FROM crate_owners WHERE created_at <= {CUTOFF}
    GROUP BY crate_id
),
revdeps AS (
    SELECT d.crate_id, count(DISTINCT dep.crate_id) AS reverse_deps
    FROM dependencies d
    JOIN v dep ON dep.id = d.version_id
    GROUP BY d.crate_id
)
SELECT
    c.id,
    c.name,
    lr.n_versions                                    AS n_versions,
    coalesce(o.n_owners, 0)                          AS n_owners,
    coalesce(rd.reverse_deps, 0)                     AS reverse_deps,
    date_diff('day', c.created_at, {CUTOFF})         AS crate_age_days,
    lr.is_pre_1_0                                    AS is_pre_1_0,
    round(cad.last_gap::double / nullif(cad.median_gap, 0), 3) AS release_cadence_decay,
    date_diff('day', o.last_owner_change, {CUTOFF})  AS days_since_owner_change,
    date_diff('month', lr.last_release, {CUTOFF})    AS months_since_release
FROM crates c
JOIN last_rel lr        ON lr.crate_id = c.id
LEFT JOIN owners o      ON o.crate_id = c.id
LEFT JOIN revdeps rd    ON rd.crate_id = c.id
LEFT JOIN cadence cad   ON cad.crate_id = c.id
WHERE c.created_at < {CUTOFF} - INTERVAL {ELIGIBLE_MONTHS} MONTH
"""

con = duckdb.connect()
for name, src in VIEWS.items():
    con.execute(f"CREATE VIEW {name} AS SELECT * FROM {src}")
df = con.execute(QUERY).df()
con.close()

df.to_parquet(OUT, index=False)
print(f"cutoff T = {T}   eligible crates at T: {len(df):,}  ->  {OUT.relative_to(ROOT)}")
print(df.drop(columns=["id", "name"]).describe().round(2).T)
