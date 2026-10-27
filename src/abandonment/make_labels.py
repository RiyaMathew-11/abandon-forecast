"""Unify the abandonment label: RustSec `unmaintained` OR owner-archived repo.

RustSec's own unmaintained criteria list "repository
archived" - an archived repo is an owner DECLARATION of abandonment, the same
definition the advisories are filed under. So is_archived is promoted from
feature to LABEL. This script stamps one `is_abandoned` column onto
train.parquet and validation.parquet, plus `label_source` for provenance
(advisory / archived / advisory+archived) - needed for eval design (archived
security-only crates cannot serve as eval negatives), E-N calibration and the
moved-not-dead diagnostics.

is_archived itself is NOT kept on train/validation (one label, not two
columns); the raw mined value stays in repo_features.parquet so labels can be
rebuilt, but every model must drop it from the feature set - it is the label
now (pu_core.load(drop=("is_archived",))).

Pipeline order: features -> split -> fetch_github -> join_repo_signals ->
THIS (rerun after any re-join). Idempotent - recomputes from
repo_features.parquet each run.

Run:  uv run python src/abandonment/make_labels.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"

arch = pd.read_parquet(DERIVED / "repo_features.parquet")[["name", "is_archived"]]

for path in (DERIVED / "train.parquet", DERIVED / "validation.parquet"):
    df = pd.read_parquet(path)
    df = df.drop(columns=[c for c in ("is_archived", "is_abandoned", "label_source")
                          if c in df.columns])
    df = df.merge(arch, on="name", how="left")
    archived = df["is_archived"].fillna(False).astype(bool)
    advisory = (df["is_unmaintained"].fillna(False).astype(bool)
                if "is_unmaintained" in df.columns else pd.Series(False, index=df.index))
    df["is_abandoned"] = advisory | archived
    df["label_source"] = np.select(
        [advisory & archived, advisory, archived],
        ["advisory+archived", "advisory", "archived"], default="none")
    df = df.drop(columns=["is_archived"])
    df.to_parquet(path, index=False)
    print(f"{path.name}: {len(df):,} rows   is_abandoned={int(df['is_abandoned'].sum()):,}")
    print(df["label_source"].value_counts().to_string(), "\n")
