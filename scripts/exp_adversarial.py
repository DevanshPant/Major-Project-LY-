"""Adversarial keyword-injection experiment.

Measures whether rendering-layer integrity verification defeats the
invisible-text attack, and at what false-positive cost.

Design
------
For a target role R we build a pool of:
  - N genuine resumes from category R        (honest, relevant)
  - N genuine resumes from other categories  (honest, irrelevant)
  - N attacked copies of the off-category resumes, injected with
    role-R keywords rendered invisibly (white-on-white)
  - N attacked copies using 1pt micro-font instead

The attack goal is to rank an irrelevant resume above the honest
relevant ones. With the defence OFF and ON we report:
  - attacker capture of rank-1 and share of the top-5
  - detection TPR on attacked files, FPR on clean files
  - mean rank of attacked vs relevant files

Run:
    python -m scripts.exp_adversarial --per_role 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DATASET_CSV
from app.core.integrity import analyse
from app.core.parser import parse_bytes
from app.core.pipeline import Candidate, run_match

ROLE_JD = {
    "INFORMATION-TECHNOLOGY":
        "Senior software engineer with 5 years experience. Python, Java, SQL, "
        "cloud, REST APIs, Linux. Bachelor degree required.",
    "ACCOUNTANT":
        "Senior accountant with 5 years experience. Bookkeeping, audit, tax "
        "preparation, GAAP, reconciliation. Bachelor degree required.",
    "HEALTHCARE":
        "Healthcare professional with 5 years experience. Patient care, "
        "clinical practice, medical records, treatment planning. Degree required.",
}

STUFF = {
    "INFORMATION-TECHNOLOGY":
        "senior software engineer python java sql aws cloud rest api linux docker "
        "kubernetes microservices git agile 10 years experience bachelor computer science",
    "ACCOUNTANT":
        "senior accountant cpa gaap ifrs audit tax preparation bookkeeping "
        "reconciliation financial reporting 10 years experience bachelor accounting",
    "HEALTHCARE":
        "healthcare clinical patient care registered nurse medical records treatment "
        "planning hospital ward 10 years experience bachelor nursing",
}


def _body(pdf, text):
    for line in text.split("\n")[:70]:
        chunk = line.encode("latin-1", "ignore").decode("latin-1")[:95]
        pdf.cell(0, 5, chunk, new_x="LMARGIN", new_y="NEXT")


def _pdf_clean(text: str) -> bytes:
    from fpdf import FPDF
    p = FPDF(); p.add_page(); p.set_font("Helvetica", size=10)
    p.set_text_color(0, 0, 0)
    _body(p, text)
    return bytes(p.output())


def _pdf_white_attack(text: str, payload: str) -> bytes:
    from fpdf import FPDF
    p = FPDF(); p.add_page(); p.set_font("Helvetica", size=10)
    p.set_text_color(0, 0, 0)
    _body(p, text)
    p.set_text_color(255, 255, 255)      # invisible on white paper
    for i in range(0, len(payload), 90):
        p.cell(0, 5, payload[i:i + 90], new_x="LMARGIN", new_y="NEXT")
    return bytes(p.output())


def _pdf_tiny_attack(text: str, payload: str) -> bytes:
    from fpdf import FPDF
    p = FPDF(); p.add_page(); p.set_font("Helvetica", size=10)
    p.set_text_color(0, 0, 0)
    _body(p, text)
    p.set_font("Helvetica", size=1)      # invisible to a human reader
    big = (payload + " ") * 3
    for i in range(0, len(big), 300):
        p.cell(0, 1, big[i:i + 300], new_x="LMARGIN", new_y="NEXT")
    return bytes(p.output())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_role", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/adversarial_results.json")
    args = ap.parse_args()

    df = pd.read_csv(DATASET_CSV)
    tc = "Resume_str" if "Resume_str" in df.columns else df.columns[0]
    df = df.rename(columns={tc: "text"})
    if "Category" in df.columns:
        df = df.rename(columns={"Category": "category"})
    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 400].reset_index(drop=True)

    records = []
    detect_rows = []

    for role, jd in ROLE_JD.items():
        if int((df["category"] == role).sum()) < args.per_role:
            print(f"  skip {role}: not enough samples")
            continue
        payload = STUFF[role]
        rel = df[df["category"] == role].sample(args.per_role, random_state=args.seed)
        off = df[df["category"] != role].sample(args.per_role, random_state=args.seed)

        pool = []
        for i, t in enumerate(rel["text"]):
            pool.append((f"clean_rel_{i}.pdf", _pdf_clean(t), "clean_relevant"))
        for i, t in enumerate(off["text"]):
            pool.append((f"clean_off_{i}.pdf", _pdf_clean(t), "clean_offtopic"))
        for i, t in enumerate(off["text"]):
            pool.append((f"atk_white_{i}.pdf", _pdf_white_attack(t, payload), "attack_white"))
        for i, t in enumerate(off["text"]):
            pool.append((f"atk_tiny_{i}.pdf", _pdf_tiny_attack(t, payload), "attack_tiny"))

        texts, reports = {}, {}
        for name, b, label in pool:
            txt = parse_bytes(name, b)
            texts[name] = txt
            rep = analyse(name, b, txt)
            reports[name] = rep
            detect_rows.append({
                "role": role, "file": name, "label": label,
                "flagged": bool(rep.suspicious),
                "penalty": float(rep.score_penalty),
                "reasons": "; ".join(rep.reasons),
            })

        for defence in (False, True):
            cands = []
            for name, b, label in pool:
                c = Candidate(file=name, text=texts[name])
                if defence:
                    c.integrity = reports[name]
                cands.append(c)
            ms = run_match(jd, cands, top_n=len(cands),
                           use_reranker=True, integrity_check=defence)
            labmap = {n: l for n, _b, l in pool}
            for rank, m in enumerate(ms, 1):
                records.append({
                    "role": role, "defence": defence, "file": m.candidate.file,
                    "label": labmap[m.candidate.file], "rank": rank,
                    "score": float(m.score.final),
                })
        print(f"  {role}: {len(pool)} files scored under both conditions")

    rec = pd.DataFrame(records)
    det = pd.DataFrame(detect_rows)

    print()
    print("=" * 74)
    print("DETECTION (rendering-layer integrity verification)")
    for lab in ["clean_relevant", "clean_offtopic", "attack_white", "attack_tiny"]:
        sub = det[det["label"] == lab]
        if len(sub) == 0:
            continue
        kind = "TPR" if lab.startswith("attack") else "FPR"
        print(f"  {lab:16s} n={len(sub):3d}  flagged={sub['flagged'].mean()*100:5.1f}%  ({kind})")

    print()
    print("ATTACKER SUCCESS (lower is better for the defence)")
    for defence in (False, True):
        sub = rec[rec["defence"] == defence]
        tag = "ON " if defence else "OFF"
        r1 = sub[sub["rank"] == 1]["label"].str.startswith("attack").mean()
        t5 = sub[sub["rank"] <= 5]["label"].str.startswith("attack").mean()
        mr_a = sub[sub["label"].str.startswith("attack")]["rank"].mean()
        mr_r = sub[sub["label"] == "clean_relevant"]["rank"].mean()
        print(f"  defence {tag}: rank-1 held by attacker in {r1*100:5.1f}% of roles | "
              f"attacker share of top-5 = {t5*100:5.1f}% | "
              f"mean rank attack={mr_a:5.1f} relevant={mr_r:5.1f}")
    print("=" * 74)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ranking": records, "detection": detect_rows}, open(out, "w"), indent=2)
    print(f"\nraw results -> {out}")


if __name__ == "__main__":
    main()
