"""Pack the derived tables for the Zenodo deposit.

Writes to publication/zenodo/:
  abandon-forecast-derived-data-v1.tar.gz   everything `make reproduce` needs (~55 MB)
  abandon-forecast-sweep-v1.tar.gz          the 800-config sweep scores (~144 MB, optional)
  SHA256SUMS                      checksums `make fetch` verifies

Both tarballs unpack into data/ (entries are derived-data/...), with the data
dictionary and the data licence at the top level.

Run:  uv run python publication/make_zenodo_bundle.py
"""
import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = DATA / "derived-data"
OUT = ROOT / "publication" / "zenodo"
VERSION = "v1"

# What ships, as globs under data/derived-data/. Scratch outputs
# (momentum_index_test*, examiner_*, goldset_labels v1 - exploratory runs) stay out.
MAIN = [
    "features_at_*.parquet", "outcomes_*.parquet",
    "repo_signals*.parquet", "repo_features.parquet", "archived_dates.parquet",
    "rustsec_advisories.parquet", "rustsec_labels.parquet",
    "crate_repos.parquet", "crate_owner_dates.parquet",
    "goldset_evidence_*.parquet", "goldset_labels_v2_*.parquet",
    "goldset_split_*.parquet", "goldset_sample_*.parquet", "goldset_verify_*.csv",
    "backtest_scores_*.parquet", "pu_scores.parquet",
    "train.parquet", "validation.parquet",
    "calibration_*.json",
    "sweep/scores_*_adv_d2_lr0.03_n200_m20_final.parquet",   # the shipped model's scores
    "arch_holdout_results_*.csv", "posmix_results_*.csv", "sweep_results_*.csv",
]
SWEEP = ["sweep/*.parquet"]
TOP = {  # top-level files inside every tarball
    "DATA.md": ROOT / "publication" / "DATA.md",
    "LICENSE": ROOT / "publication" / "LICENSE-DATA",
}


def collect(globs: list[str]) -> list[Path]:
    files = {p for g in globs for p in DERIVED.glob(g) if p.is_file()}
    return sorted(files)


def pack(name: str, files: list[Path]) -> Path:
    out = OUT / f"abandon-forecast-{name}-{VERSION}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for arc, src in TOP.items():
            tar.add(src, arcname=arc)
        for f in files:
            tar.add(f, arcname=str(f.relative_to(DATA)))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(name: str, files: list[Path], tarball: Path) -> None:
    raw = sum(f.stat().st_size for f in files) / 1e6
    print(f"{tarball.name}: {len(files)} files, {raw:.0f} MB raw -> "
          f"{tarball.stat().st_size / 1e6:.0f} MB packed")


OUT.mkdir(exist_ok=True)
for arc, src in TOP.items():
    assert src.exists(), f"missing {src} (the bundle must carry {arc})"

main_files = collect(MAIN)
sweep_files = [f for f in collect(SWEEP) if f not in set(main_files)]
assert main_files, "no derived tables found - run the pipeline or `make fetch` first"

tarballs = [pack("derived-data", main_files)]
report("derived-data", main_files, tarballs[0])
if sweep_files:
    tarballs.append(pack("sweep", sweep_files))
    report("sweep", sweep_files, tarballs[1])

sums = OUT / "SHA256SUMS"
sums.write_text("".join(f"{sha256(t)}  {t.name}\n" for t in tarballs))
print(f"checksums -> {sums.relative_to(ROOT)}")
