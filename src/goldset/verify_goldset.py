"""Manual verification of the LLM gold labels - the human half of the
labeling pipeline. Comparator from Xu et al.'s labeling stage: their two
human annotators agreed 95% (kappa 0.814) on a keyword-prefiltered,
binary task; the oft-quoted 95.91% is GPT-4o agreeing with their SetFit
classifier - machine-vs-machine, both tuned on the same human benchmark.

Two modes, one flat script:
  1. No verification CSV yet -> draw ~50 crates (balanced across gold
     labels, spread over strata, seed fixed) and write a BLIND checklist:
     name + links + empty human_label column. The LLM's label is NOT in
     the file - seeing it would anchor your judgment and inflate
     agreement. Judge from the links (and the same evidence the LLM saw),
     fill human_label with maintained/abandoned/unclear, save.
  2. CSV exists and human_label is filled -> score it: raw agreement,
     Cohen's kappa, and the disagreement list (each one worth reading -
     either the LLM erred, or you found an evidence gap).

Run:  uv run python src/goldset/verify_goldset.py [YYYY-MM-DD]
      (round 1 writes data/derived-data/goldset_verify_<T>.csv,
       round 2 scores it once you have filled the human_label column)
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
SEED = 42
N_PER_LABEL = {"abandoned": 20, "maintained": 20, "unclear": 10}

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"
# v2 labels (post-amendment). Caveat when re-scoring the round-1 CSV: the
# v2 amendments were informed by this same CSV's disagreements, so the v2
# agreement number is partly in-sample; the v1 number (64%, kappa 0.438,
# zero contradictions) stays the registered blind result.
LABELS = DERIVED / f"goldset_labels_v2_{T}.parquet"
EVIDENCE = DERIVED / f"goldset_evidence_{T}.parquet"
CSV = DERIVED / f"goldset_verify_{T}.csv"

labels = pd.read_parquet(LABELS)

if not CSV.exists():
    ev = pd.read_parquet(EVIDENCE)[["name", "repo", "releases_post_T"]]
    df = labels.merge(ev, on="name", how="left")
    # balanced across gold labels so agreement is measured on all three
    # verdict types, not just the majority class; shuffle-then-head per
    # label keeps the draw seeded and spreads strata naturally
    picks = []
    for lab, n in N_PER_LABEL.items():
        pool = df[df["gold_label"] == lab].sample(frac=1, random_state=SEED)
        picks.append(pool.head(n))
    out = pd.concat(picks).sample(frac=1, random_state=SEED)  # shuffle order too
    out["crates_io"] = "https://crates.io/crates/" + out["name"]
    out["github"] = out["repo"].map(
        lambda r: f"https://github.com/{r}" if isinstance(r, str) else "")
    out = out[["name", "stratum", "crates_io", "github"]]
    out["human_label"] = ""
    out["notes"] = ""
    out.to_csv(CSV, index=False)
    print(f"blind checklist ({len(out)} crates) -> {CSV.relative_to(ROOT)}")
    print("Judge each crate (maintained/abandoned/unclear) WITHOUT looking at "
          "the LLM labels,\nfill the human_label column, save, then rerun this "
          "script to score agreement.")
    raise SystemExit

v = pd.read_csv(CSV)
filled = v[v["human_label"].astype(str).str.strip()
           .isin(["maintained", "abandoned", "unclear"])]
if filled.empty:
    raise SystemExit(f"{CSV.name} exists but human_label is empty - fill it in "
                     "first (maintained/abandoned/unclear).")
m = filled.merge(labels[["name", "gold_label", "confidence", "key_evidence"]],
                 on="name")
print(f"scored {len(m)} / {len(v)} verified crates")

agree = (m["human_label"] == m["gold_label"])
print(f"\nraw agreement: {agree.mean():.1%}  ({int(agree.sum())}/{len(m)})"
      f"   [Xu et al. human-human: 95%, kappa 0.814; their 95.91% is "
      f"GPT-4o-vs-SetFit, machine-vs-machine]")
print(f"Cohen's kappa: {cohen_kappa_score(m['human_label'], m['gold_label']):.3f}")
print("\nhuman vs LLM:\n"
      + pd.crosstab(m["human_label"], m["gold_label"]).to_string())

dis = m[~agree]
if len(dis):
    print(f"\ndisagreements ({len(dis)}) - read each; either the LLM erred or "
          "the evidence bundle has a gap:")
    for _, r in dis.iterrows():
        print(f"  {r['name']}: you={r['human_label']}  llm={r['gold_label']} "
              f"({r['confidence']})  llm evidence: {r['key_evidence'][:100]}")
        if isinstance(r.get("notes"), str) and r["notes"].strip():
            print(f"    your note: {r['notes']}")
else:
    print("\nno disagreements.")
