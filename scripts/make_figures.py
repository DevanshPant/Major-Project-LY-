"""Generate publication figures from the measured result files.

Outputs vector PDF (preferred by IEEE) plus 600 dpi PNG into paper/figs/.
Everything is drawn from data/eval_results.json and
data/adversarial_results.json, so no number is hand-entered.

Run:
    python -m scripts.make_figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# IEEE single column is 3.5 in wide. Keep text >= 8 pt at final size.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
})

# Grayscale-safe: distinct lightness, plus hatching for print.
GRAY = ["#d9d9d9", "#a6a6a6", "#6b6b6b", "#1f1f1f"]
HATCH = ["///", "...", "xxx", ""]
LABELS = {"tfidf": "TF-IDF", "bi": "Bi-encoder",
          "full_pre": "+ CE (pretrained)", "full_ft": "+ CE (fine-tuned)"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig_retrieval():
    p = ROOT / "data" / "eval_results.json"
    if not p.exists():
        print("  skip retrieval figure (no eval_results.json)")
        return
    df = pd.read_json(p)
    order = ["tfidf", "bi", "full_pre", "full_ft"]
    metrics = [("p_at_k", "P@10"), ("r_at_k", "R@10"),
               ("ndcg", "NDCG@10"), ("mrr", "MRR")]
    means = df.groupby("system").mean(numeric_only=True).reindex(order)
    sems = df.groupby("system").sem(numeric_only=True).reindex(order)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(metrics))
    w = 0.20
    for i, s in enumerate(order):
        vals = [means.loc[s, m] for m, _ in metrics]
        errs = [sems.loc[s, m] for m, _ in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=2,
               color=GRAY[i], hatch=HATCH[i], edgecolor="black",
               linewidth=0.5, label=LABELS[s])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.30), handlelength=1.4)
    save(fig, "fig_retrieval")


def fig_perquery():
    p = ROOT / "data" / "eval_results.json"
    if not p.exists():
        return
    df = pd.read_json(p)
    piv = df.pivot(index="query", columns="system", values="p_at_k")
    piv = piv.assign(delta=piv["full_ft"] - piv["full_pre"]).sort_values("delta")

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    y = np.arange(len(piv))
    colors = ["#252525" if d > 0 else "#bdbdbd" for d in piv["delta"]]
    ax.barh(y, piv["delta"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([q.replace("-", " ").title()[:22] for q in piv.index], fontsize=6.5)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("P@10 change from domain adaptation")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_perquery")


def fig_adversarial():
    p = ROOT / "data" / "adversarial_results.json"
    if not p.exists():
        print("  skip adversarial figure (no adversarial_results.json)")
        return
    d = json.load(open(p))
    rec = pd.DataFrame(d["ranking"])
    det = pd.DataFrame(d["detection"])

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.3))
    fig.subplots_adjust(hspace=0.62)

    # (a) attacker share of top-5, defence off vs on
    ax = axes[0]
    vals, errs = [], []
    for defence in (False, True):
        sub = rec[rec["defence"] == defence]
        per_role = (sub[sub["rank"] <= 5]
                    .groupby("role")["label"]
                    .apply(lambda s: s.str.startswith("attack").mean()))
        vals.append(per_role.mean())
        errs.append(per_role.sem() if len(per_role) > 1 else 0.0)
    ax.bar(["Defence off", "Defence on"], vals, 0.5, yerr=errs, capsize=3,
           color=[GRAY[0], GRAY[2]], hatch=["///", ""],
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Attacker share of top-5")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.04, f"{v*100:.0f}%", ha="center", fontsize=8)
    ax.set_title("(a) Attack success", fontsize=9)

    # (b) detection rates by file class
    ax = axes[1]
    classes = [("attack_white", "White text"), ("attack_tiny", "Micro-font"),
               ("clean_relevant", "Clean rel."), ("clean_offtopic", "Clean off.")]
    rates, ns = [], []
    for key, _ in classes:
        sub = det[det["label"] == key]
        rates.append(sub["flagged"].mean() if len(sub) else 0.0)
        ns.append(len(sub))
    cols = [GRAY[2], GRAY[2], GRAY[0], GRAY[0]]
    hats = ["", "", "...", "..."]
    ax.bar(range(len(classes)), rates, 0.6, color=cols, hatch=hats,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([lab for _, lab in classes], fontsize=7.5)
    ax.set_ylabel("Flagged as suspicious")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    for i, (v, n) in enumerate(zip(rates, ns)):
        ax.text(i, v + 0.03, f"{v*100:.0f}%\nn={n}", ha="center", fontsize=6.5)
    ax.set_title("(b) Detection", fontsize=9)

    save(fig, "fig_adversarial")


def fig_rankshift():
    p = ROOT / "data" / "adversarial_results.json"
    if not p.exists():
        return
    rec = pd.DataFrame(json.load(open(p))["ranking"])
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    groups = [("attack_white", "White-text attack"),
              ("attack_tiny", "Micro-font attack"),
              ("clean_relevant", "Genuine relevant")]
    width = 0.35
    x = np.arange(len(groups))
    for i, defence in enumerate((False, True)):
        sub = rec[rec["defence"] == defence]
        means = [sub[sub["label"] == k]["rank"].mean() for k, _ in groups]
        errs = [sub[sub["label"] == k]["rank"].sem() for k, _ in groups]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=2,
               color=GRAY[0] if not defence else GRAY[2],
               hatch="///" if not defence else "",
               edgecolor="black", linewidth=0.5,
               label="Defence off" if not defence else "Defence on")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in groups], fontsize=7.5)
    ax.set_ylabel("Mean rank (lower is better)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "fig_rankshift")


def fig_corpus():
    p = ROOT / "data" / "corpus_stats.json"
    if not p.exists():
        print("  skip corpus figure (no corpus_stats.json)")
        return
    st = json.load(open(p))
    deg = st["degree"]
    order = ["phd", "masters", "bachelors", "diploma", "none"]
    nice = {"phd": "PhD", "masters": "Master", "bachelors": "Bachelor",
            "diploma": "Diploma", "none": "Not stated"}
    vals = [deg.get(k, 0) for k in order]
    total = sum(vals)

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.4))
    fig.subplots_adjust(hspace=0.60)
    ax = axes[0]
    ramp = ["#252525", "#525252", "#737373", "#a8a8a8", "#d9d9d9"]
    wedges, _ = ax.pie(vals, colors=ramp, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1))
    ax.set_title("(a) Highest degree detected", fontsize=9)
    ax.legend(wedges, [f"{nice[k]} ({v/total*100:.1f}%)" for k, v in zip(order, vals)],
              frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.92, 0.5))

    ax = axes[1]
    ex = st["experience"]
    keys = ["0", "1-2", "3-5", "6-10", "11+"]
    v2 = [ex.get(k, 0) for k in keys]
    ax.bar(range(len(keys)), v2, 0.6, color="#737373",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_xlabel("Years of experience extracted")
    ax.set_ylabel("Resumes")
    ax.set_title("(b) Experience distribution", fontsize=9)
    save(fig, "fig_corpus")


def fig_latency():
    p = ROOT / "data" / "latency.json"
    if not p.exists():
        print("  skip latency figure (no latency.json)")
        return
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, (label, series) in enumerate(d.items()):
        n = [pt["n"] for pt in series]
        t = [pt["sec"] for pt in series]
        ax.plot(n, t, marker="os^"[i % 3], markersize=4, linewidth=1.4,
                color=GRAY[i % 3], label=label, markeredgecolor="black",
                markeredgewidth=0.4)
    ax.set_xlabel("Resumes in batch")
    ax.set_ylabel("End-to-end latency (s)")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "fig_latency")


if __name__ == "__main__":
    print("generating figures ->", OUT)
    fig_retrieval()
    fig_perquery()
    fig_adversarial()
    fig_rankshift()
    fig_corpus()
    fig_latency()
    print("done")
