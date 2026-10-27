#!/usr/bin/env bash
# Runtime benchmark: cargo-pulse audit vs existing dependency-maintenance tools.
#
# Comparators (2026-10 survey, see publication notes):
#   cargo-unmaintained  heuristics: archived/missing repo, 1yr staleness
#   cargo-audit         RustSec advisories (incl. unmaintained)
#   cargo-deny          same advisories as a CI gate
#   cargo-diagnose      heuristic per-dep score (OSV + GitHub + crates.io)
#   pkgrisk             single-crate heuristic score (vs `cargo pulse <crate>`)
#
# All timings are WARM (advisory DBs cached, lockfiles present). Network
# latency dominates every tool here; report medians, not means, in the paper.
#
# Usage:
#   ./bench_tools.sh check     # verify prerequisites, print install commands
#   ./bench_tools.sh fixtures  # scaffold fixture projects + lockfiles
#   ./bench_tools.sh audit     # project-audit benchmark (small, then medium)
#   ./bench_tools.sh single    # single-crate benchmark (cargo-pulse vs pkgrisk)
#
# Results land in exploration/bench_results/ as markdown + JSON.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
FIX="$HERE/bench_fixtures"
OUT="$HERE/bench_results"
PULSE="$ROOT/cargo-pulse/target/release/cargo-pulse"
RUNS=3

SMALL_DEPS=(serde rand log regex chrono)
MEDIUM_DEPS=("${SMALL_DEPS[@]}" serde_json anyhow thiserror clap itertools
             once_cell libc bytes url uuid tempfile toml flate2 memchr bitflags)
SINGLE_CRATES=(serde chrono lazy_static)

check() {
  local ok=1
  for t in hyperfine cargo-unmaintained cargo-audit cargo-deny cargo-diagnose pkgrisk; do
    if command -v "$t" >/dev/null; then echo "  ok      $t"; else echo "  MISSING $t"; ok=0; fi
  done
  [ -x "$PULSE" ] && echo "  ok      cargo-pulse ($PULSE)" || { echo "  MISSING cargo-pulse (cd cargo-pulse && cargo build --release)"; ok=0; }
  [ -n "${GITHUB_TOKEN:-}" ] && echo "  ok      GITHUB_TOKEN set" || echo "  note    GITHUB_TOKEN not set (cargo-unmaintained + forge checks will be rate-limited)"
  if [ "$ok" = 0 ]; then
    echo
    echo "Install:  brew install hyperfine"
    echo "          cargo install cargo-unmaintained cargo-audit cargo-deny cargo-diagnose pkgrisk"
    exit 1
  fi
}

fixture() { # name deps...
  local name="$1"; shift
  local dir="$FIX/$name"
  [ -f "$dir/Cargo.lock" ] && { echo "fixture $name exists"; return; }
  mkdir -p "$dir/src"
  printf 'fn main() {}\n' > "$dir/src/main.rs"
  {
    printf '[package]\nname = "%s"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\n' "bench-$name"
    for d in "$@"; do printf '%s = "*"\n' "$d"; done
  } > "$dir/Cargo.toml"
  (cd "$dir" && cargo generate-lockfile)
  echo "fixture $name: $# direct deps"
}

fixtures() {
  fixture small "${SMALL_DEPS[@]}"
  fixture medium "${MEDIUM_DEPS[@]}"
}

audit_bench() { # fixture-name
  local name="$1" dir="$FIX/$1"
  [ -d "$dir" ] || { echo "run '$0 fixtures' first"; exit 1; }
  mkdir -p "$OUT"
  # one unmeasured warm pass per tool fills advisory-DB caches
  hyperfine --shell=default --runs "$RUNS" --warmup 1 --ignore-failure \
    --export-markdown "$OUT/audit_$name.md" --export-json "$OUT/audit_$name.json" \
    -n cargo-pulse        "cd $dir && $PULSE audit ." \
    -n cargo-unmaintained "cd $dir && cargo unmaintained" \
    -n cargo-audit        "cd $dir && cargo audit" \
    -n cargo-deny         "cd $dir && cargo deny check advisories" \
    -n cargo-diagnose     "cd $dir && cargo diagnose"
  echo "-> $OUT/audit_$name.md"
}

single_bench() {
  mkdir -p "$OUT"
  for c in "${SINGLE_CRATES[@]}"; do
    hyperfine --shell=default --runs "$RUNS" --warmup 1 --ignore-failure \
      --export-markdown "$OUT/single_$c.md" --export-json "$OUT/single_$c.json" \
      -n "cargo-pulse $c" "$PULSE $c" \
      -n "pkgrisk $c"     "pkgrisk check $c"
    echo "-> $OUT/single_$c.md"
  done
}

case "${1:-}" in
  check)    check ;;
  fixtures) fixtures ;;
  audit)    check; audit_bench small; audit_bench medium ;;
  single)   check; single_bench ;;
  *)        echo "usage: $0 {check|fixtures|audit|single}"; exit 1 ;;
esac
