"""Retrieval evaluation: Precision@k, Recall@k, NDCG@k, MRR.

Ground truth: a resume is relevant to a JD if its dataset category
matches the JD's target category. This is a coarse proxy label, but it
is identical for every system under test, so the comparison between
systems is fair even though absolute values are approximate.

Three systems are compared on the same pools:
  tfidf  - TF-IDF + cosine (the keyword baseline)
  bi     - MiniLM bi-encoder retrieval only
  full   - MiniLM retrieval + cross-encoder rerank + weighted scoring

Run:
    python -m scripts.evaluate --k 10 --pool 100 --pos 20 --seed 1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DATASET_CSV
from app.core.pipeline import Candidate, run_match, run_baseline
from scripts.train_reranker import CATEGORY_JD


def precision_at_k(hits, k):
    return sum(hits[:k]) / k if k else 0.0


def recall_at_k(hits, k, n_relevant):
    return (sum(hits[:k]) / n_relevant) if n_relevant else 0.0


def ndcg_at_k(hits, k, n_relevant):
    """Correct NDCG: IDCG uses the number of relevant items AVAILABLE,
    capped at k, not the number actually retrieved."""
    dcg = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hits[:k]) if h)
    ideal = min(n_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(hits):
    for i, h in enumerate(hits):
        if h:
            return 1.0 / (i + 1)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--pos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/eval_results.json")
    args = ap.parse_args()

    df = pd.read_csv(DATASET_CSV)
    text_col = "Resume_str" if "Resume_str" in df.columns else df.columns[0]
    df = df.rename(columns={text_col: "text"})
    if "Category" in df.columns:
        df = df.rename(columns={"Category": "category"})
    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 150].reset_index(drop=True)

    cats = sorted(df["category"].unique().tolist())
    K = args.k
    rows = []

    print(f"{len(cats)} queries | pool={args.pool} | positives={args.pos} | k={K}")
    print()

    for qi, cat in enumerate(cats, 1):
        jd = CATEGORY_JD.get(cat) or f"Professional in {cat.lower().replace('-', ' ')}."
        avail = int((df["category"] == cat).sum())
        n_pos = min(args.pos, avail)
        pos_pool = df[df["category"] == cat].sample(n_pos, random_state=args.seed)
        neg_pool = df[df["category"] != cat].sample(
            min(args.pool - n_pos, int((df["category"] != cat).sum())),
            random_state=args.seed)
        pool = pd.concat([pos_pool, neg_pool]).sample(
            frac=1, random_state=args.seed).reset_index(drop=True)

        cands_src = [(f"r{i}", t) for i, t in enumerate(pool["text"])]
        gold = set(pool.index[pool["category"] == cat])
        n_rel = len(gold)

        def mk():
            return [Candidate(file=n, text=t) for n, t in cands_src]

        t0 = time.perf_counter()
        m_full = run_match(jd, mk(), top_n=K, use_reranker=True, integrity_check=False)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        m_bi = run_match(jd, mk(), top_n=K, use_reranker=False, integrity_check=False)
        t_bi = time.perf_counter() - t0

        t0 = time.perf_counter()
        m_tf = run_baseline(jd, mk(), top_n=K)
        t_tf = time.perf_counter() - t0

        for name, ms, el in (("full", m_full, t_full), ("bi", m_bi, t_bi), ("tfidf", m_tf, t_tf)):
            hits = [int(m.candidate.file[1:]) in gold for m in ms]
            rows.append({
                "query": cat, "system": name,
                "p_at_k": precision_at_k(hits, K),
                "r_at_k": recall_at_k(hits, K, n_rel),
                "ndcg": ndcg_at_k(hits, K, n_rel),
                "mrr": mrr(hits),
                "latency_s": el,
            })
        pf = [r for r in rows if r["query"] == cat and r["system"] == "full"][0]
        pt = [r for r in rows if r["query"] == cat and r["system"] == "tfidf"][0]
        print(f"  [{qi:2d}/{len(cats)}] {cat:26s} P@{K} full={pf['p_at_k']:.2f} "
              f"tfidf={pt['p_at_k']:.2f}  NDCG full={pf['ndcg']:.2f} tfidf={pt['ndcg']:.2f}")

    res = pd.DataFrame(rows)
    print()
    print("=" * 72)
    summary = res.groupby("system")[["p_at_k", "r_at_k", "ndcg", "mrr", "latency_s"]].mean()
    order = ["tfidf", "bi", "full"]
    summary = summary.reindex(order)
    print(f"MEAN OVER {len(cats)} QUERIES (k={K})")
    print(f"{'system':8s} {'P@k':>7s} {'R@k':>7s} {'NDCG@k':>8s} {'MRR':>7s} {'sec/query':>10s}")
    for s in order:
        r = summary.loc[s]
        print(f"{s:8s} {r['p_at_k']:7.3f} {r['r_at_k']:7.3f} {r['ndcg']:8.3f} "
              f"{r['mrr']:7.3f} {r['latency_s']:10.2f}")
    print("=" * 72)

    # paired improvement of full over tfidf
    piv = res.pivot(index="query", columns="system", values="p_at_k")
    d = piv["full"] - piv["tfidf"]
    print(f"P@{K} improvement (full - tfidf): mean={d.mean():+.3f}  "
          f"wins={int((d>0).sum())}  ties={int((d==0).sum())}  losses={int((d<0).sum())}")
    try:
        from scipy import stats
        t, p = stats.ttest_rel(piv["full"], piv["tfidf"])
        print(f"paired t-test: t={t:.3f}  p={p:.5f}")
    except Exception:
        print("(scipy not installed - skipping t-test)")

    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    res.to_json(outp, orient="records", indent=2)
    print(f"\nper-query results -> {outp}")


if __name__ == "__main__":
    main()
