# abandon-forecast — forecasting package abandonment on crates.io

The dataset, mining pipeline and model training behind **cargo-pulse**, a
cargo subcommand that reports recorded abandonment facts first and, for live
crates, a calibrated two-year abandonment probability.

- **Dataset** (Zenodo, DOI *to be added*): point-in-time feature tables for
  every mature crate at yearly cutoffs 2019–2024 plus the 2026 snapshot,
  GitHub activity reconstructed *as of* each cutoff, outcome labels from
  RustSec advisories and repository archival, and a 300-crate human/LLM gold
  set. Dictionary: [`publication/DATA.md`](publication/DATA.md).
- **Tool**: [cargo-pulse](https://github.com/RiyaMathew-11/cargo-pulse)
  (`cargo install cargo-pulse`). Its model bundle is exported by this
  pipeline (`cargo-pulse/data/`).

## Three ways to run it

| | Command | Needs | Time |
|---|---|---|---|
| **Demo** | `make demo` | nothing (committed 5.5k-crate sample) | ~20 s |
| **Reproduce the paper** | `make fetch ZENODO=<id>` then `make reproduce` | the 55 MB Zenodo bundle | ~1 min on a laptop CPU |
| **Rebuild from a dump** | `make db crates rustsec features mine outcomes refit goldset calibrate export` | a crates.io dump, `GITHUB_TOKEN`, `ANTHROPIC_API_KEY` | hours |

Everything runs through [uv](https://docs.astral.sh/uv/) (`curl -LsSf
https://astral.sh/uv/install.sh | sh`); the final parity step needs the tool
installed (`cargo install cargo-pulse`). `uv` provisions the pinned Python 3.12
environment from `uv.lock` on first use. `make help` lists every stage.

### Demo

`repro/` is a self-contained copy of the training core (bagged
positive-unlabelled XGBoost–Cox on the crate-age clock) that runs offline on
a committed sample. It reproduces the *method*, not the mining. See
[`repro/README.md`](repro/README.md).

### Reproduce the paper

```sh
make fetch ZENODO=<record id>   # downloads + checksums the bundle into data/derived-data/ (TBD)
make reproduce                  # refit -> gold test -> calibrate -> export -> parity
```

`reproduce` runs the five steps that produce the paper's numbers and the
shipped tool:

1. `gbt_refit.py` — refits the shipped configuration (`adv_d2_lr0.03_n200_m20`:
   XGBoost `survival:cox`, 100 PU bags, depth 2, lr 0.03, 200 trees) at the
   backtest cutoff T = 2024-07-16 and prints the channel contrasts (archived
   vs proved-alive AUC, with the staleness bar beside it).
2. `gold_eval.py --fold=test` — grades those scores on the sealed gold test fold.
3. `calibrate.py` — fits the isotonic map from (score, age) to
   P(abandoned within ~2 y) on the gold dev fold.
4. `export_artifacts.py` — freezes the ensemble, calibration, percentiles and
   owner sidecar into `data/export/` (the bundle cargo-pulse ships).
5. `cargo-pulse --parity`, with `CARGO_PULSE_DATA=data/export` — the installed
   Rust scorer re-scores a 60-crate reference sample from that bundle and must
   match the Python model to 1e-3.

Seeds are fixed (42) so a CPU run reproduces the archived scores; `ARGS=--cuda`
on `make refit` trains on a GPU and may differ in the last decimal.
Optional: `make fetch-sweep` adds the 800-configuration hyper-parameter
sweep behind the paper's negative result (144 MB).

### Rebuild from a dump — bring your own snapshot

The pipeline is not tied to our snapshot. To build the dataset from a fresh
crates.io dump and run your own training/backtest experiment:

1. **Get a dump.** crates.io publishes a daily database dump at
   <https://static.crates.io/db-dump.tar.gz> (only the latest is served, so
   record the date). Extract it under `crates-data/`; the archive unpacks into
   a directory named by its timestamp, e.g. `crates-data/2027-01-05-020014/`.
2. **Get the advisories.** `git clone --depth 1
   https://github.com/rustsec/advisory-db crates-data/advisory-db` and note
   the commit.
3. **Point the pipeline at them.** Every stage reads its paths from
   [`src/snapshot.py`](src/snapshot.py), overridden by two variables:
   ```sh
   export CRATES_DUMP=crates-data/2027-01-05-020014   # or: make … DUMP=…
   export CRATES_SNAPSHOT=2027-01-05                  # or: make … SNAP=…
   ```
   The snapshot date is "now" for outcomes: a crate counts as abandoned in a
   backtest if its advisory or archival falls in (T, snapshot].
4. **Pick a cutoff T** at least two years before the snapshot so the outcome
   window is long enough (we use T = snapshot − 2 y), then run the stages:
   ```sh
   make db                 # prints the DuckDB load command
   make crates rustsec     # crate -> repo map, owner dates, advisory labels
   make features T=2025-01-05
   make mine               # prints the three GitHub miners (GITHUB_TOKEN; hours)
   make outcomes T=2025-01-05
   make refit T=2025-01-05 # or `make train` for the default parameters
   ```
   `gbt_refit.py` prints the backtest contrasts; `gold_eval.py` needs a gold
   set, which `make goldset` prints the chain for (LLM labelling needs an
   `ANTHROPIC_API_KEY`; you can also hand-label the sampled 300).
5. **Ship it.** `make calibrate export T=…` writes a new model bundle into
   `data/export/`; `CARGO_PULSE_DATA=data/export cargo-pulse --parity` checks
   the installed tool against it, and `CARGO_PULSE_DATA=data/export cargo pulse
   <crate>` serves forecasts from your model.

Caveats that also apply to our own snapshot: GitHub mining is not
reproducible for repositories deleted since (they 404), and advisory-channel
labels lag real disuse. Both are discussed in the paper.

## Layout

```
Makefile              stage runner (make help)
src/snapshot.py       which dump / snapshot date the pipeline reads
src/abandonment/      registry features at T, outcomes, PU core, crate tables
src/repo_mining/      GitHub GraphQL miners (snapshot and as-of-T)
src/rustsec/          advisory-db parsing -> labels
src/gbt/              XGBoost Cox/AFT training, refit, sweep, holdouts
src/goldset/          gold-set sampling, evidence, LLM labelling, calibration
src/cli/              export of the model bundle for cargo-pulse
repro/                offline 20 s reproduction of the training method
publication/          data dictionary, data licence, Zenodo bundling
data/derived-data/    the tables (gitignored; `make fetch` or the pipeline)
data/export/          the exported model bundle for cargo-pulse
crates-data/          load.sql (DuckDB loader); the dump and advisory-db live here, gitignored
```

## Citing

See [`CITATION.cff`](CITATION.cff). The dataset and the tool have separate
DOIs/versions; cite the one you used.

## Licence

Code: see `LICENSE`. Data (the Zenodo bundle): CC BY 4.0, see
[`publication/LICENSE-DATA`](publication/LICENSE-DATA) (shipped inside the bundle as `LICENSE`). The crates.io dump and RustSec advisory-db are themselves
public (CC0 / dual MIT–Apache-2.0); GitHub-derived columns are public
repository metadata.
