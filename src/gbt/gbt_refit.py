"""Refit ONE chosen sweep config at the full 100 bags, save the ensemble,
and write its scores - the model that then gets the single sealed test
read (gold_eval.py --scores=sweep:<tag>_final --fold=test).

Run:  uv run python src/gbt/gbt_refit.py [YYYY-MM-DD] <tag> [--cuda]
      e.g. uv run python src/gbt/gbt_refit.py 2024-07-16 adv_arch_d3_lr0.05_n400_m5
"""

import re
import sys

import pandas as pd

from cox_model import DERIVED, ROOT, assemble, train_bagged

args = [a for a in sys.argv[1:] if not a.startswith("-")]
T = args[0] if args and args[0][:2] == "20" else "2024-07-16"
tag = args[-1] if args else ""
cuda = "--cuda" in sys.argv

m = re.fullmatch(r"(adv(?:_arch)?)_d(\d+)_lr([\d.]+)_n(\d+)_m(\d+)", tag)
if not m:
    raise SystemExit("usage: gbt_refit.py [YYYY-MM-DD] <tag>  (tag like "
                     "adv_arch_d3_lr0.05_n400_m5)")
pos_set, depth, lr, n_est, mcw = (m.group(1), int(m.group(2)),
                                  float(m.group(3)), int(m.group(4)),
                                  int(m.group(5)))

print(f"T = {T}  refitting {tag} at 100 bags "
      f"(pos={pos_set} depth={depth} lr={lr} n={n_est} mcw={mcw})")
df, X = assemble(T)
scores, _ = train_bagged(
    df, X, T, pos_set=pos_set,
    params=dict(max_depth=depth, learning_rate=lr, n_estimators=n_est,
                min_child_weight=mcw),
    n_bags=100, save_tag=f"{tag}_final", cuda=cuda, verbose=True)

out = DERIVED / "sweep" / f"scores_{T}_{tag}_final.parquet"
scores.to_parquet(out, index=False)
print(f"\nmodel -> data/models/{T}/{tag}_final/   scores -> "
      f"{out.relative_to(ROOT)}")

# channel contrasts + staleness bars, so a refit at any cutoff is a
# complete temporal-robustness read in one command
from sklearn.metrics import roc_auc_score  # noqa: E402

undeclared = (df["outcome"] != "declared_at_T").to_numpy()
sub = df[undeclared].reset_index(drop=True).merge(scores, on="name")
stale = sub["months_since_release"].to_numpy()
print(f"\n{'contrast':<28}{'model':>8}{'staleness bar':>15}")
for died, lab in (("died_advisory", "advisory vs proved-alive"),
                  ("died_archived", "archived vs proved-alive")):
    m = sub["outcome"].isin([died, "proved_alive"]).to_numpy()
    yy = (sub.loc[m, "outcome"] == died).to_numpy().astype(int)
    print(f"{lab:<28}{roc_auc_score(yy, sub.loc[m, 'risk_score']):>8.3f}"
          f"{roc_auc_score(yy, stale[m]):>15.3f}")
if T == "2024-07-16":
    print(f"\nSealed read: uv run python src/goldset/gold_eval.py {T} "
          f"--scores=sweep:{tag}_final --fold=test")
