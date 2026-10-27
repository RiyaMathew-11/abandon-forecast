# Data dictionary — abandon-forecast derived tables

All derived tables are Apache Parquet in `data/derived-data/` (~200 MB,
gitignored), built by the pipeline stages in the root `Makefile`. Feature and
outcome tables exist per cutoff `T` (yearly 2019–2024, plus the 2026 snapshot);
rows below are at `T = 2024-07-16` unless noted.

## Provenance pins

| Source | Pin |
|---|---|
| crates.io database dump | snapshot `2026-07-16T02:00:17Z` (`crates-data/2026-07-16-020017/metadata.json`) |
| RustSec advisory-db | commit `7003e463`, shallow-cloned 2026-07-28 (`crates-data/advisory-db/`) |
| GitHub GraphQL mining | mined July–August 2026; as-of-`T` reconstruction via history filters |
| Gold-set LLM judge | `claude-sonnet-5` via the Anthropic API; human-verified |
| Random seeds | 42 everywhere |

Crates join to repositories through `crates.csv.repository` normalised to an
`owner/name` GitHub key (`gh_key` in `src/gbt/cox_model.py`).

## Tables

### `features_at_<T>.parquet` — 94,882 rows (163,396 at the 2026 snapshot)
One row per eligible crate (≥ 18 months old at `T`); registry tier, replayed
from dated dump records so nothing after `T` leaks.

| Column | Meaning |
|---|---|
| `id`, `name` | crate id / name |
| `n_versions` | versions published up to `T` |
| `n_owners` | owners at `T` |
| `reverse_deps` | crates depending on it at `T` (popularity proxy) |
| `crate_age_days` | days since creation at `T` — survival clock, not a feature |
| `is_pre_1_0` | never reached 1.0 by `T` |
| `release_cadence_decay` | slowing of release cadence over the crate's life |
| `days_since_owner_change` | days since ownership last changed |
| `months_since_release` | staleness signal (the baseline rule) |

### `repo_signals_at_<T>.parquet` — 52,867 rows
Repository tier as a 2024 observer could have seen it (GraphQL
`until`/`since` history filters).

| Column | Meaning |
|---|---|
| `repo` | `owner/name` key |
| `repo_status_at` | found / not_found at mining time |
| `last_commit_at_T` | newest default-branch commit ≤ `T` |
| `commits_180d_at_T` | commits in `[T−180d, T]` |
| `last_issue_close_at_T`, `issue_censored` | last issue close ≤ `T`; censored if all visible activity is post-`T` |
| `last_pr_merge_at_T`, `pr_censored` | same for merged PRs |

### `repo_signals.parquet` — 147,117 rows
Snapshot-time repository state: `is_archived`, `is_fork`, `pushed_at`,
`stars`, `commits_180d`, open issue/PR counts, `repo_status`.

### `archived_dates.parquet` — 6,058 rows
`repo`, `archived_at`, `still_archived`. Archival emits no public GitHub
event, so these dates exist only because they were fetched directly; deleted
repos get `archived_at = NaT`.

### `outcomes_<T>.parquet` — 94,882 rows
Forward outcome per eligible crate from the window `(T, snapshot]`, in
precedence order: `declared_at_T` → `died_advisory` → `died_archived` →
`proved_alive` → `unlabeled`. Columns: `name`, `archived_at`, `unm_at`,
`patched_sec_in_window`, `outcome`, `declared_source`, `outcome_date`.
At `T = 2024-07-16`: scoring population 91,437; 1,950 died-archived,
100 died-advisory, 73 proved-alive by the 2026 snapshot.

### `rustsec_advisories.parquet` / `rustsec_labels.parquet` — 1,152 / 873 rows
One row per advisory / per crate. `unmaintained` advisories are abandonment
ground truth; patched security advisories are confirmed-alive evidence;
withdrawn advisories excluded.

### `crate_repos.parquet` / `crate_owner_dates.parquet` — one row per crate
The two registry facts later stages need from the raw dump, lifted out by
`make crates` so training, outcomes, calibration and export run from this
directory alone: `name, repository` (the URL behind every GitHub join) and
`name, last_owner_added` (date the most recent owner was added; cargo-pulse's
owner sidecar).

### Gold set — `goldset_evidence` / `goldset_labels_v2` / `goldset_split`
300 crates sampled in three reverse-dependency strata (seed 42, blind to any
model); 268 confidently decided (human/LLM, Cohen's κ = 0.438). Evidence
bundles, per-crate label + confidence + reasoning, and a dev/sealed-test
split (the test fold was read exactly once).

### `train.parquet` / `validation.parquet` — 143,914 / 778 rows
Snapshot-time (not point-in-time) PU frames behind the *detection* row of the
results table: surface registry features at the snapshot for every eligible
crate (`features.py`), split into the no-advisory pool (`train`) and the
RustSec crates with their advisory columns (`validation`, `split.py`), with
`is_abandoned` / `label_source` stamped by `make_labels.py` (advisory
`unmaintained` OR owner-archived repository). Not used by the forecasting
model.

### `backtest_scores_*.parquet` / `sweep/scores_*.parquet` — 91,437 rows each
`name`, `risk_score` per model variant; the shipped variant is tag
`adv_d2_lr0.03_n200_m20` (XGBoost `survival:cox`, 100 PU bags, depth 2,
lr 0.03, 200 trees, min_child_weight 20, isotonic calibration on the gold
dev fold).

## Committed samples (run without the full data)

- `repro/data/sample.csv` — 5,455 crates: all 131 advisory positives +
  labelled eval crates + a 3k undeclared sample; drives `make demo`.
- `cargo-pulse/data/` — the exported model bundle (trees, calibration,
  percentiles, owner dates, 60-crate parity sample).
