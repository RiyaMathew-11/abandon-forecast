# repro — minimal reproduction of the shipped Cox model

A self-contained, runnable slice of the abandonment-forecasting pipeline from
the MSc dissertation *Predicting Package Abandonment in the Rust Ecosystem*.
The full study mines ~95k crates and their GitHub repositories (~100 MB of
derived tables, too large to ship); this directory reproduces the **training
method** on a small, shipped sample so the core result can be re-run in seconds.

Full pipeline and data: <https://github.com/RiyaMathew-11/abandon-forecast>

## What it reproduces

The shipped model is a **bagged positive-unlabelled (PU) survival model**: an
XGBoost Cox ensemble on the crate-age clock, trained on the 131 advisory-declared
crates as survival events against popularity-matched, right-censored undeclared
crates. `train_cox_mini.py` is a faithful *registry-only* copy of that training
core (`src/gbt/cox_model.py`, the `--no-repo` variant), with every dependency on
the mining pipeline removed.

It trains, scores the undeclared population, and grades the scores against a raw
"months-since-release" staleness bar — the same evaluation shape as the full run.

**No data download needed.** Every feature is already engineered and frozen into
the committed `data/sample.csv`, so this runs fully offline: no crates.io or
GitHub access, no DuckDB, none of the mined `derived-data/` tables. What is
reproduced here is the training and evaluation *method*, not the mining that
produced the features.

## Contents

| File | Purpose |
|------|---------|
| `train_cox_mini.py` | standalone trainer + evaluation (no `src/` imports) |
| `data/sample.csv` | ~5.5k crates: all 131 advisory positives + labelled eval crates + a 3k undeclared sample (seed 42, cutoff T = 2024-07-16) |
| `make_sample.py` | provenance — how `sample.csv` was drawn (needs the full data; not runnable from a clone) |
| `requirements.txt` | dependencies (minimum versions) |

## Run it

The project uses [uv](https://docs.astral.sh/uv/). The trainer carries inline
(PEP 723) dependency metadata, so uv provisions an isolated environment on first
run — nothing to install by hand and no `uv sync` needed. The `--no-project`
flag tells uv to ignore the surrounding repository and use the script's own
lightweight dependencies (rather than building the full parent project):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh    # install uv (once)
cd repro
uv run --no-project train_cox_mini.py            # 100 bags, ~20s on a laptop CPU
uv run --no-project train_cox_mini.py --bags 25  # faster, slightly noisier
```

<details><summary>Without uv (plain venv + pip)</summary>

```sh
cd repro
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_cox_mini.py
```
</details>

## Expected output

```
cutoff T = 2024-07-16   [XGBoost survival:cox, age clock]   features: 7 cols (registry-only)
training positives (advisory): 131   scoring population (undeclared): 5,123
...
contrast                        model AUC  staleness
advisory deaths vs alive            0.827      0.876
archived deaths vs alive            0.893      0.890

forward Harrell's C-index (2050 events): model 0.529   staleness-only 0.501
```

Absolute numbers are lower and noisier than the reported full-data run — this is
~6% of the data with the GitHub repo tier removed — but the **shape is identical**:
the model matches the staleness rule on the general population while separating
the labelled deaths, and the forward C-index sits near 0.5 because most "deaths"
are undeclared survivors the window cannot see (the C-index degeneration analysed
in the dissertation). The full model with the repo tier reaches 0.879 / 0.944 on
the two channels.

## License

Same as the parent repository.
