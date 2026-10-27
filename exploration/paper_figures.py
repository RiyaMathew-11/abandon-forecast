"""Figures for the MSR 2027 showcase paper (publication/paper/figures/).

  1. dataset_funnel.pdf  - collection pipeline with counts at every stage
  2. calibration.pdf     - reliability diagram of the shipped C4 calibration
                           on the gold dev fold (reproduces calibrate.py's
                           index -> isotonic mapping from the saved JSON)

Run:  uv run python exploration/paper_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived-data"
FIGS = ROOT / "publication" / "paper" / "figures"
T = "2024-07-16"

BLUE = "#31679B"
INK = "#222222"
MUTED = "#666666"

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                     "text.color": INK, "axes.edgecolor": MUTED})


def box(ax, y, text, sub, h=0.115, w=0.92, x=0.04):
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#EDF2F7",
                               edgecolor=BLUE, linewidth=1.0))
    ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center",
            fontsize=8.5, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=7.5, color=MUTED)


def arrow(ax, y_from, y_to, note=None):
    ax.annotate("", xy=(0.5, y_to), xytext=(0.5, y_from),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    if note:
        ax.text(0.54, (y_from + y_to) / 2, note, fontsize=7.2,
                color=MUTED, va="center")


def funnel_figure() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 3.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0.865, "crates.io dump  2026-07-16", "300,281 crates")
    arrow(ax, 0.865, 0.795, "age ≥ 18 months at T")
    box(ax, 0.68, "eligible at T = 2024-07-16", "94,882 crates · 14 features as of T")
    arrow(ax, 0.68, 0.61, "− 3,445 already declared at T")
    box(ax, 0.495, "scoring population", "91,437 live, undeclared crates")
    arrow(ax, 0.495, 0.425, "observed in (T, snapshot]")

    labels = [("archived", "1,950"), ("advisory", "100"),
              ("alive", "73"), ("unlabeled", "89,314")]
    w, gap, x = 0.215, 0.022, 0.04
    for i, (name, n) in enumerate(labels):
        xi = x + i * (w + gap)
        ax.add_patch(plt.Rectangle((xi, 0.315), w, 0.11, facecolor="#EDF2F7",
                                   edgecolor=BLUE, linewidth=1.0))
        ax.text(xi + w / 2, 0.385, n, ha="center", fontsize=8, fontweight="bold")
        ax.text(xi + w / 2, 0.345, name, ha="center", fontsize=6.6, color=MUTED)

    ax.text(0.04, 0.22, "Repository tier: 147,117 repos mined at the snapshot;"
                        "\n52,867 with as-of-T history; 6,058 archival dates.",
            fontsize=7.2, color=INK, va="top")
    ax.text(0.04, 0.10, "Gold set: 300 crates sampled in 3 reverse-dependency"
                        "\nstrata; 268 decided (human/LLM, κ = 0.438).",
            fontsize=7.2, color=INK, va="top")

    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"dataset_funnel.{ext}", dpi=300)
    plt.close(fig)


def dev_fold() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gold dev fold: calibration index, predicted probability, label, weight."""
    cal = json.load(open(DERIVED / f"calibration_{T}.json"))
    split = pd.read_parquet(DERIVED / f"goldset_split_{T}.parquet")
    dev = split[split["fold"] == "dev"].copy()
    scores = pd.read_parquet(
        DERIVED / f"sweep/scores_{T}_adv_d2_lr0.03_n200_m20_final.parquet")
    feats = pd.read_parquet(DERIVED / f"features_at_{T}.parquet")[
        ["name", "reverse_deps", "crate_age_days"]]
    dev = dev.merge(scores, on="name").merge(feats, on="name")

    idx = (cal["coef_score"] * dev["risk_score"]
           + cal["coef_age_years"] * dev["crate_age_days"] / 365.25
           + cal["intercept"])
    curve = pd.DataFrame(cal["iso_index_curve"])
    p = np.interp(idx, curve["index"], curve["p"])
    y = (dev["gold_label"] == "abandoned").to_numpy().astype(float)

    pop = feats.merge(scores[["name"]], on="name")
    bins, labs = [-1, 0, 9, np.inf], ["revdeps_0", "revdeps_1_9", "revdeps_10plus"]
    pop_share = pd.cut(pop["reverse_deps"], bins, labels=labs).value_counts(normalize=True)
    dev_share = dev["stratum"].value_counts(normalize=True)
    w = dev["stratum"].map(pop_share / dev_share).astype(float).to_numpy()
    return idx.to_numpy(), np.asarray(p), y, w


def calibration_figure() -> None:
    # quintiles of the continuous calibration index: isotonic ties in p would
    # otherwise collapse bins into unequal (and tiny) groups
    idx, p, y, w = dev_fold()
    qs = np.quantile(idx, np.linspace(0, 1, 6))
    qs[0] -= 1e-9
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=1)
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (idx > lo) & (idx <= hi)
        if not m.any():
            continue
        pred = np.average(p[m], weights=w[m])
        obs = np.average(y[m], weights=w[m])
        ax.scatter(pred, obs, s=36, color=BLUE, zorder=3)
    ax.set_xlabel("predicted P(abandoned within ~2 years)")
    ax.set_ylabel("observed abandoned fraction")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, lw=0.3, color="#DDDDDD", zorder=0)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"calibration.{ext}", dpi=300)
    plt.close(fig)


funnel_figure()
calibration_figure()
print(f"figures written to {FIGS}")
