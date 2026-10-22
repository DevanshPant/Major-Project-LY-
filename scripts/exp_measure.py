"""Corpus statistics and latency measurement for the paper.

Writes:
    data/corpus_stats.json   degree / experience / category / format counts
    data/latency.json        end-to-end latency vs batch size, per configuration

Run:
    python -m scripts.exp_measure --what all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.extractor import extract
from app.core.pipeline import Candidate, run_match, run_baseline


def corpus_stats(out: Path):
    df = pd.read_parquet(ROOT / "data" / "corpus.parquet")
    deg, exp_b = {}, {"0": 0, "1-2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    years_all = []
    for t in df["text"]:
        e = extract(str(t))
        deg[e.degree] = deg.get(e.degree, 0) + 1
        y = e.years
        years_all.append(y)
        if y == 0:
            exp_b["0"] += 1
        elif y <= 2:
            exp_b["1-2"] += 1
        elif y <= 5:
            exp_b["3-5"] += 1
        elif y <= 10:
            exp_b["6-10"] += 1
        else:
            exp_b["11+"] += 1

    st = {
        "n_resumes": int(len(df)),
        "n_categories": int(df["category"].nunique()),
        "format": {k: int(v) for k, v in df["format"].value_counts().items()},
        "degree": {k: int(v) for k, v in deg.items()},
        "experience": exp_b,
        "mean_chars": float(df["text"].str.len().mean()),
        "median_chars": float(df["text"].str.len().median()),
        "category_counts": {k: int(v) for k, v in df["category"].value_counts().items()},
        "years_mean_nonzero": float(
            pd.Series([y for y in years_all if y > 0]).mean()),
    }
    json.dump(st, open(out, "w"), indent=2)
    print(f"corpus_stats -> {out}")
    print(f"  n={st['n_resumes']} cats={st['n_categories']} "
          f"mean_chars={st['mean_chars']:.0f}")
    print(f"  degree={st['degree']}")


def latency(out: Path, sizes=(10, 25, 50, 100, 200), reps=3):
    df = pd.read_parquet(ROOT / "data" / "corpus.parquet")
    jd = ("Senior software engineer with 5 years experience. Python, Java, SQL, "
          "cloud, REST APIs. Bachelor degree required.")
    pool = df["text"].astype(str).tolist()

    configs = {
        "TF-IDF": lambda c: run_baseline(jd, c, top_n=10),
        "Bi-encoder": lambda c: run_match(jd, c, top_n=10, use_reranker=False,
                                          integrity_check=False),
        "Proposed": lambda c: run_match(jd, c, top_n=10, use_reranker=True,
                                        integrity_check=False),
    }

    # warm the models so the first timed run is not a model load
    run_match(jd, [Candidate(file="w", text=pool[0])], top_n=1,
              use_reranker=True, integrity_check=False)

    res = {name: [] for name in configs}
    for n in sizes:
        texts = (pool * ((n // len(pool)) + 1))[:n]
        for name, fn in configs.items():
            best = None
            for _ in range(reps):
                cands = [Candidate(file=f"r{i}", text=t) for i, t in enumerate(texts)]
                t0 = time.perf_counter()
                fn(cands)
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            res[name].append({"n": n, "sec": round(best, 3)})
            print(f"  n={n:4d} {name:12s} {best:7.3f}s")
    json.dump(res, open(out, "w"), indent=2)
    print(f"latency -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["all", "corpus", "latency"])
    a = ap.parse_args()
    d = ROOT / "data"
    d.mkdir(exist_ok=True)
    if a.what in ("all", "corpus"):
        corpus_stats(d / "corpus_stats.json")
    if a.what in ("all", "latency"):
        latency(d / "latency.json")
