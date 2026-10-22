"""Give the training corpus a genuine PDF+DOCX mix.

The Kaggle folder is all PDFs. This script picks a fraction of them,
converts each to a DOCX file whose paragraphs = extracted PDF text, and
writes them alongside the PDFs. `ingest_pdfs.py` will then pick both up
and the parquet corpus will show a mix under the `format` column.

We're not producing pretty Word docs — the goal is a second file format
for the parser and reranker to see during training.

Run:
    python -m scripts.mixed_format_convert  --fraction 0.3
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docx import Document
from app.core.parser import parse_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "Resumes PDF"))
    ap.add_argument("--fraction", type=float, default=0.30,
                    help="Fraction of PDFs to also emit as DOCX")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    pdfs = list(root.rglob("*.pdf")) + list(root.rglob("*.PDF"))
    if not pdfs:
        print(f"[convert] no PDFs found under {root}")
        sys.exit(1)

    rng = random.Random(args.seed)
    picks = rng.sample(pdfs, int(len(pdfs) * args.fraction))
    print(f"[convert] {len(picks)} PDFs → DOCX (of {len(pdfs)} total)")

    made = failed = 0
    for i, pdf in enumerate(picks, 1):
        out = pdf.with_suffix(".docx")
        if out.exists():
            continue
        try:
            text = parse_bytes(pdf.name, pdf.read_bytes())
            if not text.strip():
                failed += 1
                continue
            doc = Document()
            for para in text.split("\n"):
                if para.strip():
                    doc.add_paragraph(para.strip())
            doc.save(out)
            made += 1
        except Exception as e:
            failed += 1
            if failed < 5:
                print(f"  fail {pdf.name}: {e}")
        if i % 200 == 0:
            print(f"  … {i}/{len(picks)}  (made={made}, failed={failed})")

    print(f"[convert] done. made={made}  failed={failed}")


if __name__ == "__main__":
    main()
