# abandon-forecast — pipeline runner
#
# One target per pipeline stage, in build order. `make help` prints the map.
# Data dictionary and schemas: publication/DATA.md
#
# Three ways in (see README):
#   make demo                        offline, 20 s, committed sample
#   make fetch ZENODO=<id> reproduce  Tier 1: retrain + grade + export from the archived tables
#   make db crates rustsec features … Tier 2: rebuild everything from a crates.io dump
#
# Provenance pins (the paper's snapshot; override DUMP/SNAP for your own):
#   crates.io dump      2026-07-16T02:00:17Z  (crates-data/2026-07-16-020017/)
#   RustSec advisory-db commit 7003e463, cloned 2026-07-28 (crates-data/advisory-db/)
#   seeds               42 everywhere
#
# Long or credentialed stages (db, mine, goldset) only PRINT their commands —
# run them yourself after checking the prerequisites they list.

T    ?= 2024-07-16                          # training / backtest cutoff
SNAP ?= 2026-07-16                          # dump date == "now" for outcomes
DUMP ?= crates-data/2026-07-16-020017       # extracted db-dump directory
TAG  ?= adv_d2_lr0.03_n200_m20              # the shipped model's sweep config
ZENODO ?=                                   # Zenodo record id for `make fetch`
EXPORT ?= $(if $(wildcard cargo-pulse/Cargo.toml),cargo-pulse/data,data/export)  # model bundle out dir
BUNDLE := abandon-forecast-derived-data-v1.tar.gz

export CRATES_SNAPSHOT := $(strip $(SNAP))
export CRATES_DUMP     := $(strip $(DUMP))
export CARGO_PULSE_DATA := $(strip $(EXPORT))
PY := uv run python

.DEFAULT_GOAL := help
.PHONY: help demo fetch fetch-sweep reproduce bundle db crates rustsec features mine outcomes train refit goldset calibrate export

help: ## Show the stage map
	@echo "abandon-forecast pipeline — stages in build order (T=$(strip $(T)), snapshot $(strip $(SNAP)), dump $(strip $(DUMP)))"
	@echo
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'
	@echo
	@echo "Offline demo (~20 s):     make demo"
	@echo "Reproduce the paper:      make fetch ZENODO=<record id> && make reproduce"
	@echo "Full rebuild order:       db -> crates -> rustsec -> features -> mine -> outcomes -> refit -> goldset -> calibrate -> export"

# ---------------------------------------------------------------- tier 0 / 1

demo: ## Offline 20 s reproduction of the training method on the committed sample
	cd repro && uv run --no-project train_cox_mini.py

fetch: ## Download the Zenodo bundle into data/derived-data/ and verify checksums (ZENODO=<record id>)
	@test -n "$(strip $(ZENODO))" || { echo "usage: make fetch ZENODO=<zenodo record id>"; exit 1; }
	mkdir -p data
	curl -fL -o data/$(BUNDLE)  "https://zenodo.org/records/$(strip $(ZENODO))/files/$(BUNDLE)?download=1"
	curl -fL -o data/SHA256SUMS "https://zenodo.org/records/$(strip $(ZENODO))/files/SHA256SUMS?download=1"
	cd data && shasum -a 256 --ignore-missing -c SHA256SUMS
	tar xzf data/$(BUNDLE) -C data
	@echo "-> data/derived-data/  (dictionary: data/DATA.md)"

fetch-sweep: ## Also download the 800-config hyper-parameter sweep scores (144 MB, optional)
	@test -n "$(strip $(ZENODO))" || { echo "usage: make fetch-sweep ZENODO=<zenodo record id>"; exit 1; }
	curl -fL -o data/abandon-forecast-sweep-v1.tar.gz "https://zenodo.org/records/$(strip $(ZENODO))/files/abandon-forecast-sweep-v1.tar.gz?download=1"
	cd data && shasum -a 256 --ignore-missing -c SHA256SUMS
	tar xzf data/abandon-forecast-sweep-v1.tar.gz -C data

reproduce: ## Refit the shipped model from the bundle, grade it on the sealed gold test, calibrate, export, parity-check
	$(PY) src/gbt/gbt_refit.py $(strip $(T)) $(strip $(TAG))
	$(PY) src/goldset/gold_eval.py $(strip $(T)) --scores=sweep:$(strip $(TAG))_final --fold=test
	$(PY) src/goldset/calibrate.py $(strip $(T))
	$(PY) src/cli/export_artifacts.py
	@command -v cargo-pulse >/dev/null || { echo "parity check needs the tool: cargo install cargo-pulse"; exit 1; }
	cargo-pulse --parity

bundle: ## Pack data/derived-data/ into the Zenodo tarballs + SHA256SUMS (publication/make_zenodo_bundle.py)
	$(PY) publication/make_zenodo_bundle.py

# ---------------------------------------------------------------- tier 2

db: ## [prints only] Load the crates.io dump CSVs into DuckDB
	@echo "Builds $(strip $(DUMP))/crates.duckdb from the dump CSVs (crates-data/load.sql)."
	@echo "Needs: db-dump.tar.gz extracted into $(strip $(DUMP)); duckdb CLI installed."
	@echo
	@echo "  cd $(strip $(DUMP))"
	@echo "  duckdb crates.duckdb < ../load.sql"

crates: ## Lift crate -> repository URL and owner dates out of the dump (crate_repos / crate_owner_dates)
	$(PY) src/abandonment/crate_tables.py

rustsec: ## Parse advisory-db -> rustsec_advisories.parquet + rustsec_labels.parquet
	$(PY) src/rustsec/extract.py

features: ## Registry features as of T -> features_at_<T>.parquet
	$(PY) src/abandonment/features_at.py $(strip $(T))

mine: ## [prints only] GitHub mining (needs GITHUB_TOKEN; hours; resumable)
	@echo "Three miners, in order (export GITHUB_TOKEN=... first; each checkpoints and resumes):"
	@echo
	@echo "  $(PY) src/repo_mining/fetch_github.py            # snapshot signals, ~147k repos"
	@echo "  $(PY) src/repo_mining/fetch_archived_dates.py    # archivedAt for archived repos, minutes"
	@echo "  $(PY) src/repo_mining/fetch_repo_at.py $(strip $(T))      # as-of-T signals, ~50-70k repos, 1-3 h"
	@echo "  $(PY) src/repo_mining/join_repo_signals.py       # -> repo_signals*.parquet"
	@echo
	@echo "Note: re-mining later cannot reproduce the archived tables exactly (deleted repos 404)."

outcomes: ## Forward outcomes in (T, snapshot] -> outcomes_<T>.parquet
	$(PY) src/abandonment/make_outcomes.py $(strip $(T))

train: ## Default-parameter bagged PU Cox ensemble at T -> backtest scores (ARGS=--cuda / --no-repo / --aft)
	$(PY) src/gbt/gbt_survival.py $(strip $(T)) $(ARGS)

refit: ## Refit one sweep config (TAG) at 100 bags and save the ensemble under data/models/<T>/
	$(PY) src/gbt/gbt_refit.py $(strip $(T)) $(strip $(TAG)) $(ARGS)

goldset: ## [prints only] Gold-set build (needs ANTHROPIC_API_KEY + network)
	@echo "Gold-set chain (300 crates; split refuses to overwrite an existing split):"
	@echo
	@echo "  $(PY) src/goldset/sample_goldset.py $(strip $(T))"
	@echo "  $(PY) src/goldset/fetch_goldset_evidence.py $(strip $(T))"
	@echo "  $(PY) src/goldset/llm_label_goldset.py $(strip $(T))      # ANTHROPIC_API_KEY"
	@echo "  $(PY) src/goldset/verify_goldset.py $(strip $(T))"
	@echo "  $(PY) src/goldset/split_goldset.py $(strip $(T))"
	@echo "  $(PY) src/goldset/gold_eval.py $(strip $(T)) --scores=sweep:$(strip $(TAG))_final --fold=test"

calibrate: ## Gold-anchored calibration: score -> P(abandoned within ~2y)
	$(PY) src/goldset/calibrate.py $(strip $(T))

export: ## Freeze the shipped ensemble into $(strip $(EXPORT)) (builds features at the snapshot first)
	$(PY) src/abandonment/features_at.py $(strip $(SNAP))
	$(PY) src/cli/export_artifacts.py
