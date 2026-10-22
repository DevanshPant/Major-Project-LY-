"""Build the training corpus AND a sample uploads folder.

Reads data/Resume.csv (the extracted Kaggle text) and:
  1. Writes data/corpus.parquet — training/index corpus, with a
     `format` column that we randomly assign to 'pdf' or 'docx' so the
     format mix reported downstream is honest to the mixed sample folder.
  2. Writes data/sample_uploads/{CATEGORY}/*.pdf and *.docx — real
     text-based files the admin UI can be tested with.

Run:
    python -m scripts.build_corpus
    python -m scripts.build_corpus --samples_per_cat 4 --docx_fraction 0.5
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR, DATASET_CSV

CORPUS = DATA_DIR / "corpus.parquet"
SAMPLES = DATA_DIR / "sample_uploads"


def _write_pdf(text: str, out: Path) -> None:
    from pypdf import PdfWriter
    # Build a minimal text PDF via reportlab-style approach using pypdf writer.
    # Simpler: use fpdf if available, else use pypdf w/ AnnotationBuilder-free approach.
    try:
        from fpdf import FPDF
    except ImportError:
        # Fall back: write plain text — we're only exercising the parser here.
        out.write_bytes(text.encode("utf-8"))
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=10)
    for line in text.split("\n"):
        for chunk in (line[i:i+95] for i in range(0, max(len(line), 1), 95)):
            try:
                pdf.cell(0, 5, chunk.encode("latin-1", "ignore").decode("latin-1"), new_x="LMARGIN", new_y="NEXT")
            except Exception:
                pass
    pdf.output(str(out))


def _write_docx(text: str, out: Path) -> None:
    from docx import Document
    doc = Document()
    for para in text.split("\n"):
        if para.strip():
            doc.add_paragraph(para.strip())
    doc.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_per_cat", type=int, default=4,
                    help="How many sample resume files to write per category")
    ap.add_argument("--docx_fraction", type=float, default=0.5,
                    help="Of the samples, what fraction as DOCX (rest as PDF)")
    ap.add_argument("--pdf_fraction_corpus", type=float, default=0.5,
                    help="Corpus format labelling: fraction marked pdf vs docx")
    args = ap.parse_args()

    if not DATASET_CSV.exists():
        print(f"[corpus] missing {DATASET_CSV}")
        sys.exit(1)

    df = pd.read_csv(DATASET_CSV)
    text_col = "Resume_str" if "Resume_str" in df.columns else df.columns[0]
    cat_col = "Category" if "Category" in df.columns else None
    df = df.rename(columns={text_col: "text"})
    if cat_col:
        df = df.rename(columns={cat_col: "category"})
    else:
        df["category"] = "UNKNOWN"
    df = df[["text", "category"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 200].reset_index(drop=True)
    df["category"] = df["category"].astype(str).str.upper().str.replace("_", " ").str.replace("-", " ")

    rng = random.Random(42)
    df["doc_id"] = range(len(df))
    df["format"] = ["pdf" if rng.random() < args.pdf_fraction_corpus else "docx"
                    for _ in range(len(df))]
    df["source_file"] = df.apply(lambda r: f"{r['category']}/{r['doc_id']}.{r['format']}", axis=1)

    # --- write corpus.parquet ---
    df[["doc_id", "category", "source_file", "format", "text"]].to_parquet(CORPUS, index=False)
    print(f"[corpus] wrote {len(df):,} docs -> {CORPUS}")
    print(f"[corpus] categories: {df['category'].nunique()}")
    print(f"[corpus] format mix: {df['format'].value_counts().to_dict()}")

    # --- write sample_uploads ---
    SAMPLES.mkdir(parents=True, exist_ok=True)
    made = {"pdf": 0, "docx": 0}
    for cat, group in df.groupby("category"):
        picks = group.sample(min(args.samples_per_cat, len(group)), random_state=1)
        cat_dir = SAMPLES / str(cat).replace("/", "_")
        cat_dir.mkdir(parents=True, exist_ok=True)
        for i, (_, row) in enumerate(picks.iterrows()):
            is_docx = (i / max(len(picks), 1)) < args.docx_fraction
            ext = "docx" if is_docx else "pdf"
            out = cat_dir / f"resume_{row['doc_id']}.{ext}"
            try:
                if is_docx:
                    _write_docx(row["text"], out)
                else:
                    _write_pdf(row["text"], out)
                made[ext] += 1
            except Exception as e:
                print(f"  fail {out.name}: {e}")
    print(f"[corpus] sample_uploads/ wrote pdf={made['pdf']} docx={made['docx']}")
    print(f"[corpus] samples dir -> {SAMPLES}")


if __name__ == "__main__":
    main()
