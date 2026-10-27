"""Hyperparameter x positive-set sweep for the Cox model, graded on the
DEV half of the gold set (test half stays sealed).

Motivation (first gold-set evaluation): the shipped model loses to the staleness
bar on the out-of-channel gold cohort (0.817 vs 0.909), failure
localised to silent single-release dust - a death profile the 131
advisory positives never contained. Two axes swept:
  - positive set: adv (advisory-only) vs adv_arch (+archived-before-T;
    archived deaths include the quiet dust deaths - the likelier fix)
  - hyperparameters: depth / lr / n_estimators / min_child_weight
    (never systematically searched before)

Protocol (fixed before running):
  - Each config trains via cox_model.train_bagged at N_BAGS_SWEEP=25
    (speed), scores the full population, and is graded on the gold DEV
    fold only. TEST is sealed: one read, one winner, via
    gold_eval.py --scores=sweep:<tag>_final --fold=test after refit.
  - Referees per config: DEV-gold AUC (primary), forward advisory
    contrast AUC (context; archived contrast is semi-circular for
    adv_arch configs and deliberately absent from the table).
  - Every config's model ensemble is saved to data/models/<T>/<tag>/
    (bag_NNN.ubj + meta.json; the 72-config grid totals ~1-2 GB - losers
    can be deleted after selection). Scores go to
    data/derived-data/sweep/. Resumable: done configs are skipped.

Run:  uv run python src/gbt/gbt_sweep.py [YYYY-MM-DD] [--cuda]
      (~30-60 min CPU for the 72-config grid)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cox_model import DERIVED, ROOT, assemble, train_bagged

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
cuda = "--cuda" in sys.argv
SWEEP_DIR = DERIVED / "sweep"
RESULTS = DERIVED / f"sweep_results_{T}.csv"
N_BAGS_SWEEP = 25

GRID = [dict(pos=pos, depth=d, lr=lr, n_est=n, mcw=m)
        for pos in ("adv", "adv_arch")
        for d in (2, 3, 4)
        for lr in (0.03, 0.05, 0.1)
        for n in (200, 400)
        for m in (5, 20)]

df, X = assemble(T)
undeclared = (df["outcome"] != "declared_at_T").to_numpy()
sub = df[undeclared].reset_index(drop=True)
stale = sub["months_since_release"].to_numpy()

# --- referees, precomputed --------------------------------------------------
split = pd.read_parquet(DERIVED / f"goldset_split_{T}.parquet")
dev = split[split["fold"] == "dev"]
name_pos = pd.Series(np.arange(len(sub)), index=sub["name"])
dev_idx = name_pos.loc[dev["name"]].to_numpy()
y_dev = (dev["gold_label"] == "abandoned").to_numpy().astype(int)

adv_m = sub["outcome"].isin(["died_advisory", "proved_alive"]).to_numpy()
y_adv = (sub.loc[adv_m, "outcome"] == "died_advisory").to_numpy().astype(int)

print(f"T = {T}  sweep: {len(GRID)} configs x {N_BAGS_SWEEP} bags   "
      f"dev fold: {len(dev)} crates ({int(y_dev.sum())} abandoned)")
print(f"bars - DEV staleness AUC: {roc_auc_score(y_dev, stale[dev_idx]):.3f}   "
      f"advisory-contrast staleness: {roc_auc_score(y_adv, stale[adv_m]):.3f}")
shipped = pd.read_parquet(DERIVED / f"backtest_scores_{T}_surv.parquet")
ship = sub[["name"]].merge(shipped[["name", "risk_score"]], on="name")
print(f"shipped model (adv d3 lr0.05 n300 m5, 100 bags) - DEV gold AUC: "
      f"{roc_auc_score(y_dev, ship['risk_score'].to_numpy()[dev_idx]):.3f}\n")

SWEEP_DIR.mkdir(exist_ok=True)
rows = pd.read_csv(RESULTS).to_dict("records") if RESULTS.exists() else []
done = {r["tag"] for r in rows}

for cfg in GRID:
    tag = f"{cfg['pos']}_d{cfg['depth']}_lr{cfg['lr']}_n{cfg['n_est']}_m{cfg['mcw']}"
    if tag in done:
        continue
    t0 = time.time()
    params = dict(max_depth=cfg["depth"], learning_rate=cfg["lr"],
                  n_estimators=cfg["n_est"], min_child_weight=cfg["mcw"])
    scores, _ = train_bagged(df, X, T, pos_set=cfg["pos"], params=params,
                             n_bags=N_BAGS_SWEEP, save_tag=tag, cuda=cuda)
    scores.to_parquet(SWEEP_DIR / f"scores_{T}_{tag}.parquet", index=False)
    r = scores["risk_score"].to_numpy()
    row = dict(tag=tag, **cfg,
               dev_gold_auc=round(roc_auc_score(y_dev, r[dev_idx]), 4),
               adv_auc=round(roc_auc_score(y_adv, r[adv_m]), 4),
               secs=round(time.time() - t0, 1))
    rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS, index=False)
    print(f"{tag:<30} dev_gold {row['dev_gold_auc']:.3f}  "
          f"advisory {row['adv_auc']:.3f}  ({row['secs']}s)")

res = pd.DataFrame(rows).sort_values("dev_gold_auc", ascending=False)
print("\ntop 10 by DEV gold AUC (bars + shipped baseline printed above):")
print(res.head(10).to_string(index=False))
print(f"\nfull table -> {RESULTS.relative_to(ROOT)}   scores -> "
      f"{SWEEP_DIR.relative_to(ROOT)}/   models -> data/models/{T}/<tag>/")
print("\nNext: pick ONE config (dev AUC + judgment, not blindly the max), "
      f"refit at 100 bags: uv run python src/gbt/gbt_refit.py {T} <tag>, "
      "then the single sealed read: gold_eval.py --scores=sweep:<tag>_final "
      "--fold=test")
