"""Build a per-crate SURFACE feature table from the crates.io DuckDB dump.

Surface activity = raw counts (downloads, versions/tags, owners, reverse-deps).
These are the weakest predictors per the base paper, so this is a baseline
to beat, not the final model.

Proxy target (dump-only): `release_stale` = 1 if a crate's last release is older
than STALE_MONTHS. This measures RELEASE INACTIVITY, not abandonment - a stale
crate may be mature/feature-complete (`lazy_static`, `hex`), so we deliberately
do NOT call it "abandoned". True abandonment ground truth is external (RustSec
`is_unmaintained`), used for validation only. `release_stale` is just the dense
proxy the model trains on; the RustSec check tests whether the learned score
recovers real abandonment. Eligibility: only crates old enough to *possibly* be
stale (created before the cutoff) - else a brand-new crate is trivially "active".

Caveat: the proxy is derived from release recency, so features that also depend
on release timing would leak. Keep features to counts/usage that are NOT the
last-release date itself.
"""

import sys
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from snapshot import DATA, SNAPSHOT  # noqa: E402

# The snapshot date is "now" (the dump does not update).
NOW = f"DATE '{SNAPSHOT}'"
# Two decoupled thresholds. ELIGIBLE_MONTHS = population: a crate must be old
# enough to have shown a maintenance pattern. STALE_MONTHS = the
# release-silence threshold for the `release_stale` label, calibrated against the
# RustSec unmaintained set (max Youden's J at 18 months). Eligibility (24) is
# kept LONGER than the label window (18) so "old enough to judge" is not the
# same number as "counts as silent".
ELIGIBLE_MONTHS = 24
STALE_MONTHS = 18

OUT = ROOT / "data" / "features.parquet"

# Read the CSVs directly (in-memory) rather than crates.duckdb, so this never
# fights the DuckDB UI for the single-writer lock. Text-heavy files need the
# larger max_line_size (a versions row has a ~2 MB features blob).
BIG = "max_line_size=20000000"
VIEWS = {
    "crates":            f"read_csv('{DATA}/crates.csv', sample_size=-1, {BIG})",
    "versions":          f"read_csv('{DATA}/versions.csv', sample_size=-1, {BIG})",
    "dependencies":      f"read_csv('{DATA}/dependencies.csv', sample_size=-1, {BIG})",
    "crate_owners":      f"read_csv('{DATA}/crate_owners.csv', sample_size=-1)",
    "crate_downloads":   f"read_csv('{DATA}/crate_downloads.csv', sample_size=-1)",
    "version_downloads": f"read_csv('{DATA}/version_downloads.csv', sample_size=-1)",
}

QUERY = f"""
WITH last_rel AS (
    SELECT crate_id,
           max(created_at) AS last_release,
           count(*)        AS n_versions,
           count(*) FILTER (WHERE yanked) AS n_yanked,
           -- latest_yanked: was the NEWEST version retracted? Stronger distress
           -- signal than the lifetime yank_rate.
           arg_max(yanked, created_at)::int AS latest_yanked,
           -- is_pre_1_0: crate never shipped a stable >=1.0 release (abandoned
           -- in beta). max major version across all releases == 0.
           (max(try_cast(split_part(num, '.', 1) AS INTEGER)) = 0)::int AS is_pre_1_0
    FROM versions GROUP BY crate_id
),
cadence AS (
    -- release-cadence decay: gap between the last two releases vs the crate's
    -- historical median gap. >1 => releases slowing (development winding down).
    -- NULL for crates with <2 gaps (too few releases to have a trend).
    SELECT crate_id,
           median(gap)              AS median_gap,
           arg_max(gap, created_at) AS last_gap
    FROM (
        SELECT crate_id, created_at,
               date_diff('day',
                   lag(created_at) OVER (PARTITION BY crate_id ORDER BY created_at),
                   created_at) AS gap
        FROM versions
    ) g
    WHERE gap IS NOT NULL
    GROUP BY crate_id
),
owners AS (
    -- dump's crate_owners CSV has no `deleted` column, so all rows are counted.
    SELECT crate_id, count(*) AS n_owners,
           max(created_at) AS last_owner_change
    FROM crate_owners GROUP BY crate_id
),
revdeps AS (
    SELECT d.crate_id, count(DISTINCT dep.crate_id) AS reverse_deps
    FROM dependencies d
    JOIN versions dep ON dep.id = d.version_id
    GROUP BY d.crate_id
),
recent AS (
    SELECT v.crate_id, sum(vd.downloads) AS dl_90d
    FROM version_downloads vd JOIN versions v ON v.id = vd.version_id
    GROUP BY v.crate_id
)
SELECT
    c.id,
    c.name,
    -- features (surface counts / usage) --------------------------------------
    coalesce(cd.downloads, 0)                       AS downloads_total,
    coalesce(r.dl_90d, 0)                           AS downloads_90d,
    lr.n_versions                                   AS n_versions,
    coalesce(o.n_owners, 0)                         AS n_owners,
    coalesce(rd.reverse_deps, 0)                    AS reverse_deps,
    date_diff('day', c.created_at, {NOW})           AS crate_age_days,
    lr.n_yanked * 1.0 / lr.n_versions               AS yank_rate,
    -- momentum_index: recent 90d download share vs a flat lifetime rate.
    -- ~1 steady/mature, >1 accelerating, <1 declining (users migrating off).
    -- Raw per-crate ratio ONLY (leak-free); logs/percentiles/age-bands are
    -- left to the modelling pipeline. NB no download floor here, so momentum
    -- on low-download crates is floor-level noise (80/0.3 skew).
    -- Age basis = crate_age_days (created_at), ~= first_release; the exploratory
    -- momentum_index_test parquet used first_release - near-identical.
    round(
        (coalesce(r.dl_90d, 0)::double / nullif(cd.downloads, 0))
        / (90.0 / nullif(date_diff('day', c.created_at, {NOW}), 0)),
    3)                                              AS momentum_index,
    -- derived maintenance signals --------------------------------------------
    lr.latest_yanked                                AS latest_yanked,
    lr.is_pre_1_0                                   AS is_pre_1_0,
    round(cad.last_gap::double / nullif(cad.median_gap, 0), 3) AS release_cadence_decay,
    date_diff('day', o.last_owner_change, {NOW})    AS days_since_owner_change,
    -- proxy target ------------------------------------------------------------
    date_diff('month', lr.last_release, {NOW})      AS months_since_release,
    -- release_stale: last release older than STALE_MONTHS. RELEASE INACTIVITY,
    -- not confirmed abandonment (see module docstring). Dense proxy for training.
    (lr.last_release < {NOW} - INTERVAL {STALE_MONTHS} MONTH)::int AS release_stale
FROM crates c
JOIN last_rel lr        ON lr.crate_id = c.id
LEFT JOIN owners o      ON o.crate_id = c.id
LEFT JOIN revdeps rd    ON rd.crate_id = c.id
LEFT JOIN crate_downloads cd ON cd.crate_id = c.id
LEFT JOIN recent r      ON r.crate_id = c.id
LEFT JOIN cadence cad   ON cad.crate_id = c.id
-- eligibility: crate must be old enough to have shown a pattern (ELIGIBLE_MONTHS),
-- kept longer than the STALE_MONTHS label window above.
WHERE c.created_at < {NOW} - INTERVAL {ELIGIBLE_MONTHS} MONTH
"""



OUT.parent.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()  # in-memory: reads CSVs, no lock on crates.duckdb
for name, src in VIEWS.items():
    con.execute(f"CREATE VIEW {name} AS SELECT * FROM {src}")
df = con.execute(QUERY).df()
con.close()

df.to_parquet(OUT, index=False)
rate = df["release_stale"].mean()
print(f"rows: {len(df):,}  ->  {OUT.relative_to(ROOT)}")
print(f"release_stale (>{STALE_MONTHS}mo): {df['release_stale'].sum():,}  ({rate:.1%})")
print(df.drop(columns=["id", "name"]).describe().round(2).T)



