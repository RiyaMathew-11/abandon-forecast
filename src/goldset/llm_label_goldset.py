"""LLM-label the gold-set crates maintained/abandoned from the mined
evidence bundle - the scaling half of the Xu et al. labeling pipeline
(they: keyword filter -> manual annotation -> SetFit/GPT-4o; we: random
stratified sample -> GitHub+registry evidence -> Claude -> manual
verification of a subsample).

The one design rule (anti-circularity): the label must NOT be "has it
committed lately?" - that would appoint the staleness rule as ground truth
and the eval would grade models against staleness by construction. The
prompt therefore forces the judgment onto semantic evidence (deprecation
notices, archived status, unanswered demand, maintainer statements) plus
post-cutoff activity, and explicitly allows "complete and stable" quiet
crates to be maintained. Low-signal crates get label "unclear" and are
excluded from the judge (reported, never silently dropped).

Verdict is about the crate AS OF the snapshot (2026-07-16): would a
security-relevant bug reported today plausibly get fixed?

v2 (registered after blind verification round 1, before any gold eval ran):
evidence bundle gains the PR side of demand (open PRs + bot flags + last
merged date); rubric gains three clarifications - demand weighed by
substance/freshness, bot activity is not maintainer life, and the
finished-library allowance only covers crates that plausibly ARE finished.
The amendments were informed by the 50-crate verification disagreements, so
agreement re-scored on that same CSV is partly in-sample - noted in the log.
v1 labels are kept; the run ends with a v1-vs-v2 churn crosstab.

Requires:  ANTHROPIC_API_KEY in the environment (or `ant auth login`),
           uv add anthropic
Run:       uv run python src/goldset/llm_label_goldset.py [YYYY-MM-DD]
           (writes data/derived-data/goldset_labels_<T>.parquet; resumable)
"""

import json
import sys
import time
from pathlib import Path
from typing import Literal

import anthropic
import pandas as pd
import pydantic
from pydantic import BaseModel

T = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "2024-07-16"
MODEL = "claude-sonnet-5"

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived-data"
EVIDENCE = DERIVED / f"goldset_evidence_{T}.parquet"
OUT = DERIVED / f"goldset_labels_v2_{T}.parquet"
V1 = DERIVED / f"goldset_labels_{T}.parquet"
CHECKPOINT_EVERY = 20


class Verdict(BaseModel):
    label: Literal["maintained", "abandoned", "unclear"]
    confidence: Literal["high", "medium", "low"]
    key_evidence: str  # the single decisive fact, one sentence
    reasoning: str     # 2-3 sentences


SYSTEM = """You judge whether a Rust crate is MAINTAINED or ABANDONED as of \
2026-07-16, from a registry + GitHub evidence bundle.

Definitions:
- maintained: a maintainer is still behind this crate - if a \
security-relevant bug were reported today, it would plausibly get a fix or \
a response.
- abandoned: no maintainer is behind it anymore - issues would go \
unanswered, bugs unfixed.
- unclear: the evidence genuinely does not decide it.

Judging rules, in order of weight:
1. Explicit statements win: repository archived, README/description saying \
deprecated / unmaintained / moved to X / no longer developed -> abandoned. \
Statements of stability ("complete", "stable, no further features planned") \
-> lean maintained.
2. Post-2024 activity is strong evidence of life: releases after the cutoff, \
recent commits, maintainer replying to issues.
3. Unanswered demand is strong evidence of death: open issues or PRs with \
real questions/bugs sitting without any maintainer reply for a long time. \
Weigh demand by substance and freshness - a bug report, a question needing \
action, or a human-authored PR left hanging counts heavily; years-old \
feature wishes against an otherwise complete library are weak evidence.
4. Bot activity is not maintainer life. Commits or PRs from dependabot, \
renovate, or CI bots do not show a maintainer present. Bot update PRs \
ignored for years lean only mildly toward abandonment; HUMAN demand ignored \
(rule 3) counts far more.
5. SILENCE ALONE IS NOT ABANDONMENT. A small, finished library with no open \
issues and no commits may simply be done - that is "unclear" or even \
"maintained" if it is complete and nothing has been asked of the maintainer. \
Do not label abandoned purely from inactivity; you need a semantic signal \
(rule 1 or 3) or a dead repo (deleted/archived). This allowance covers only \
crates that plausibly ARE finished: a README describing work-in-progress \
plans or unfinished features, followed by years of silence, leans abandoned \
- going quiet mid-construction is itself a semantic signal.
6. Repository not found on GitHub anymore (gh_found false but a repo URL \
existed) suggests deletion -> lean abandoned; a crate that never had a repo \
is judged on registry evidence alone.

Be decisive when the evidence is decisive; use "unclear" honestly when it is \
not."""


def bundle(row) -> str:
    parts = [
        f"crate: {row['name']}",
        f"reverse dependencies at cutoff: {row['reverse_deps']}",
        f"total releases ever: {row['n_versions']}",
        f"months since last release at cutoff (2024-07-16): {row['months_since_release']}",
        f"non-yanked releases AFTER the cutoff (2024-07 .. 2026-07): {row['releases_post_T']}",
        f"last non-yanked release date: {str(row.get('last_release', ''))[:10]}",
        f"github repo: {row['repo'] or 'none listed'}",
    ]
    if row.get("gh_found"):
        parts += [
            f"repo archived: {row['gh_archived']}"
            + (f" (on {row['gh_archived_at']})" if row.get("gh_archived_at") else ""),
            f"last push to repo: {row.get('gh_pushed_at', '')}",
            f"commits since cutoff: {row.get('gh_commits_since_T', 'unknown')}",
            f"open issues: {row.get('gh_open_issues', 'unknown')}",
            f"5 most recently updated open issues (title, opened, reply count): "
            f"{row.get('gh_recent_issues', '[]')}",
            f"open PRs: {row.get('gh_open_prs', 'unknown')}",
            f"5 most recently updated open PRs (title, opened, replies, author, "
            f"bot?): {row.get('gh_recent_prs', '[]')}",
            f"most recent merged PR: {row.get('gh_last_merged_pr') or 'none found'}",
            f"repo description: {row.get('gh_description', '')!r}",
            f"README (first 3000 chars):\n{row.get('gh_readme', '')}",
        ]
    elif row["repo"]:
        parts.append("repo listed in crate metadata but NOT FOUND on GitHub now "
                     "(deleted, renamed, or made private)")
    return "\n".join(parts)


client = anthropic.Anthropic()
df = pd.read_parquet(EVIDENCE)

results = []
if OUT.exists():
    results = pd.read_parquet(OUT).to_dict("records")
done = {r["name"] for r in results}
todo = df[~df["name"].isin(done)]
print(f"T = {T}  model = {MODEL}  crates: {len(df)}  labeled: {len(done)}  "
      f"to label: {len(todo)}")

for i, (_, row) in enumerate(todo.iterrows(), 1):
    # thinking tokens count against max_tokens; 2000 truncated the JSON
    # mid-string on long deliberations. Retry transient/parse failures,
    # checkpoint before giving up so no progress is ever lost.
    for attempt in range(3):
        try:
            resp = client.messages.parse(
                model=MODEL,
                max_tokens=8000,
                thinking={"type": "adaptive"},
                system=SYSTEM,
                messages=[{"role": "user", "content": bundle(row)}],
                output_format=Verdict,
            )
            v = resp.parsed_output
            break
        except (pydantic.ValidationError, anthropic.APIError) as e:
            if attempt == 2:
                pd.DataFrame(results).to_parquet(OUT, index=False)
                print(f"  giving up on {row['name']} after 3 attempts; "
                      f"{len(results)} labels checkpointed")
                raise
            print(f"  {row['name']}: {type(e).__name__}, retrying")
            time.sleep(5)
    results.append({"name": row["name"], "stratum": str(row["stratum"]),
                    "channel_outcome": row["outcome"], "gold_label": v.label,
                    "confidence": v.confidence, "key_evidence": v.key_evidence,
                    "reasoning": v.reasoning})
    print(f"  {len(done) + i}/{len(df)}  {row['name']}: {v.label} ({v.confidence})")
    if i % CHECKPOINT_EVERY == 0:
        pd.DataFrame(results).to_parquet(OUT, index=False)

out = pd.DataFrame(results)
out.to_parquet(OUT, index=False)

print(f"\ndone -> {OUT.relative_to(ROOT)}")
print("\nlabel distribution:\n" + out["gold_label"].value_counts().to_string())
print("\nby confidence:\n"
      + out.groupby(["gold_label", "confidence"]).size().to_string())
print("\nsanity check - gold label vs declaration channel (died_* crates "
      "should be mostly abandoned):\n"
      + pd.crosstab(out["channel_outcome"], out["gold_label"]).to_string())

if V1.exists():
    v1 = pd.read_parquet(V1)[["name", "gold_label"]] \
           .rename(columns={"gold_label": "v1_label"})
    ch = out.merge(v1, on="name")
    moved = int((ch["gold_label"] != ch["v1_label"]).sum())
    print(f"\nv1 -> v2 churn: {moved}/{len(ch)} labels changed  (rows=v1, cols=v2)")
    print(pd.crosstab(ch["v1_label"], ch["gold_label"]).to_string())

print("\nNext: rerun verify_goldset.py to re-score the human 50 against v2 "
      "(partly in-sample - amendments were informed by that CSV), then "
      "gold_eval.py.")
