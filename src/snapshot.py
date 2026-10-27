"""Which crates.io dump the pipeline reads, and the date it was taken.

Every stage imports its paths from here instead of hard-coding them, so the
whole pipeline can be pointed at a different snapshot with two environment
variables (see README, "Bring your own snapshot"):

    CRATES_DUMP=crates-data/2027-01-05-020014   # extracted db-dump directory
    CRATES_SNAPSHOT=2027-01-05                  # its date == "now" for outcomes

The defaults are the pinned snapshot the paper was built from.

Two small derived tables stand in for the raw dump downstream of `make crates`
(crate -> repository URL, crate -> last owner-added date), so training,
outcomes, calibration and export need only data/derived-data/ -- the Zenodo
bundle -- and never the 11 GB dump.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT = os.environ.get("CRATES_SNAPSHOT", "2026-07-16")
DUMP = Path(os.environ.get("CRATES_DUMP", ROOT / "crates-data" / "2026-07-16-020017"))
DATA = DUMP / "data"                      # the dump's CSVs
ADVISORY_DB = ROOT / "crates-data" / "advisory-db"

DERIVED = ROOT / "data" / "derived-data"
CRATE_REPOS = DERIVED / "crate_repos.parquet"              # name, repository
CRATE_OWNER_DATES = DERIVED / "crate_owner_dates.parquet"  # name, last_owner_added

# Where export_artifacts.py writes the model bundle. cargo-pulse reads it via
# CARGO_PULSE_DATA; with a cargo-pulse checkout alongside it lands straight
# in the tool's data/, elsewhere in data/export/.
EXPORT_DIR = Path(os.environ.get("CARGO_PULSE_DATA",
    ROOT / "cargo-pulse" / "data" if (ROOT / "cargo-pulse" / "Cargo.toml").exists()
    else ROOT / "data" / "export")).resolve()
