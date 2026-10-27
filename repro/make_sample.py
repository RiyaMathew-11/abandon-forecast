"""Provenance: how repro/data/sample.csv was drawn from the full dataset.

This documents (and regenerates) the shipped sample. It requires the full
derived-data tables, which are NOT in this repository (they are ~100 MB of
mined registry + GitHub signals); it will not run from a fresh clone. It is
here so the sampling is transparent and auditable, not so it can be re-run
without the mining pipeline.

The sample keeps EVERY crate the registry-only Cox frame needs -
  - all 131 advisory-declared-before-T positives (the training events)
  - all labelled evaluation crates (died_advisory / died_archived / proved_alive)
- plus a fixed random 3,000 of the undeclared scoring population and 200
archived declarations for completeness. Seed 42, cutoff T = 2024-07-16.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived-data"
T = "2024-07-16"


def main():
    feats = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")
    outc = pd.read_parquet(DERIVED / f"outcomes_{T}.parquet")
    df = feats.merge(outc[["name", "outcome", "declared_source",
                           "unm_at", "archived_at", "outcome_date"]], on="name")
    rng = np.random.default_rng(42)

    adv_pos = (df["outcome"] == "declared_at_T") & df["declared_source"].isin(["advisory", "both"])
    eval_lab = df["outcome"].isin(["died_advisory", "died_archived", "proved_alive"])
    keep = adv_pos | eval_lab

    arch_decl = df.index[(df["outcome"] == "declared_at_T")
                         & (df["declared_source"] == "archived")].to_numpy()
    keep.iloc[rng.choice(arch_decl, 200, replace=False)] = True
    und = df.index[df["outcome"] == "unlabeled"].to_numpy()
    keep.iloc[rng.choice(und, 3000, replace=False)] = True

    small = df[keep].sample(frac=1, random_state=1).reset_index(drop=True)
    for c in ("unm_at", "archived_at", "outcome_date"):
        small[c] = pd.to_datetime(small[c]).dt.strftime("%Y-%m-%d")
    out = Path(__file__).parent / "data" / "sample.csv"
    small.to_csv(out, index=False)
    print(f"wrote {len(small):,} rows -> {out}")


if __name__ == "__main__":
    main()
