"""Parse the RustSec advisory-db into datapoints + per-crate ground-truth labels.

Source: crates-data/advisory-db (shallow clone of rustsec/advisory-db). Each
advisory is crates/<crate>/RUSTSEC-YYYY-NNNN.md, whose head is a ```toml fenced
front-matter block. We extract that block, parse it, and emit:

  - rustsec_advisories.parquet : one row per advisory (the raw datapoints)
  - rustsec_labels.parquet     : one row per crate (the validation set)

Label discipline:
  informational = "unmaintained" -> abandonment ground truth (primary positive)
  informational = "unsound"      -> soundness hole
  informational absent           -> real security vuln; its PATCH STATUS is a
                                    feature, its mere presence is NOT abandonment
  withdrawn advisories are excluded from the labels.
"""

import re
import sys
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import snapshot  # noqa: E402

ADV_DIR = snapshot.ADVISORY_DB / "crates"
OUT = ROOT / "data" / "derived-data"
SNAPSHOT = pd.Timestamp(snapshot.SNAPSHOT)  # crates.io dump date == "now"

# Front matter is a ```toml ... ``` fence at the top of each .md advisory.
FENCE = re.compile(r"```toml\s*(.*?)```", re.DOTALL)


def parse_advisory(path: Path) -> dict | None:
    m = FENCE.search(path.read_text(encoding="utf-8"))
    if not m:
        return None
    data = tomllib.loads(m.group(1))
    adv = data.get("advisory", {})
    if not adv.get("package"):
        return None
    patched = data.get("versions", {}).get("patched", []) or []
    return {
        "crate_name": adv.get("package"),
        "advisory_id": adv.get("id"),
        "date": str(adv["date"]) if adv.get("date") else None,
        "informational": adv.get("informational"),   # None => security vuln
        "withdrawn": adv.get("withdrawn") is not None,
        "unpatched": len(patched) == 0,              # no fixed version shipped
        "categories": ",".join(adv.get("categories", []) or []),
        "aliases": ",".join(adv.get("aliases", []) or []),
    }


rows, skipped = [], 0
for md in sorted(ADV_DIR.glob("*/*.md")):
    row = parse_advisory(md)
    if row:
        rows.append(row)
    else:
        skipped += 1

adv = pd.DataFrame(rows)
adv["date"] = pd.to_datetime(adv["date"], errors="coerce")

OUT.mkdir(parents=True, exist_ok=True)
adv.to_parquet(OUT / "rustsec_advisories.parquet", index=False)

# --- per-crate rollup = the validation set ------------------------------------
live = adv[~adv["withdrawn"]].copy()
live["is_unmaintained"] = live["informational"] == "unmaintained"
live["is_unsound"] = live["informational"] == "unsound"
live["is_notice"] = live["informational"] == "notice"
live["is_security"] = live["informational"].isna()
live["is_unpatched_security"] = live["is_security"] & live["unpatched"]

g = live.groupby("crate_name")
labels = pd.DataFrame(
    {
        "has_advisory": True,
        "is_unmaintained": g["is_unmaintained"].any(),
        "is_unsound": g["is_unsound"].any(),
        "is_security": g["is_security"].any(),
        "has_unpatched_security": g["is_unpatched_security"].any(),
        "n_advisories": g.size(),
        "n_security": g["is_security"].sum(),
        "n_unpatched_security": g["is_unpatched_security"].sum(),
        "first_advisory_date": g["date"].min(),
    }
)
# days since the oldest still-unpatched security advisory (RustSec's "unresponsive
# after a vuln notice" criterion); NaN for crates with none.
oldest = live[live["is_unpatched_security"]].groupby("crate_name")["date"].min()
labels = labels.reset_index()  # crate_name back to a column
labels["days_since_oldest_unpatched"] = (
    SNAPSHOT - labels["crate_name"].map(oldest)
).dt.days

labels.to_parquet(OUT / "rustsec_labels.parquet", index=False)

# --- report -------------------------------------------------------------------
print(f"advisory files parsed: {len(adv):,}   (skipped, no toml/package: {skipped})")
print(f"non-withdrawn advisories: {len(live):,}")
print(f"crates with >=1 advisory: {len(labels):,}  ->  {OUT / 'rustsec_labels.parquet'}")
print(labels[["is_unmaintained", "is_unsound", "is_security",
              "has_unpatched_security"]].sum().rename("crates").to_string())
