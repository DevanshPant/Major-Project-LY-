"""Ingest every PDF under `Resumes PDF/` into a single Parquet corpus.

- Walks each category folder
- Extracts text via pypdf
- Normalises category names (e.g. "Accountant resumes" and
  "Accountant" collapse into "ACCOUNTANT")
- Skips empty extractions but logs them
- Writes  data/corpus.parquet   with columns:
      doc_id, category, source_file, format, text

Run:
    python -m scripts.ingest_pdfs
    python -m scripts.ingest_pdfs --root "D:/major porject/resume_ai/Resumes PDF"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR
from app.core.parser import parse_bytes

DEFAULT_ROOT = ROOT / "Resumes PDF"
OUT_PATH = DATA_DIR / "corpus.parquet"


# Collapse minor category variants (Accountant / Accountant resumes / accountants)
def normalise_category(name: str) -> str:
    n = name.strip()
    n = re.sub(r"\s*resumes?\s*$", "", n, flags=re.IGNORECASE)  # trailing "resumes"
    n = re.sub(r"[_\-\s]+", " ", n)                               # collapse separators
    n = re.sub(r"\s+", " ", n).strip().upper()
    aliases = {
        "AGRICULTURAL": "AGRICULTURE",
        "ARCHITECTS": "ARCHITECT",
        "AUTOMOBILES": "AUTOMOBILE",
        "CONSULT": "CONSULTANT",
        "DATASCIENCE": "DATA SCIENCE",
        "CIVILENGINEER": "CIVIL ENGINEER",
        "BUSINESSANALYST": "BUSINESS ANALYST",
        "PYTHONDEVELOPER": "PYTHON DEVELOPER",
        "SAPDEVELOPER": "SAP DEVELOPER",
        "WEBDESIGNING": "WEB DESIGN",
        "BUILDING CONSTRUCTION": "CONSTRUCTION",
        "MECHANICALENGINEER": "MECHANICAL ENGINEER",
        "ELECTRICALENGINEERING": "ELECTRICAL ENGINEER",
        "NETWORKSECURITYENGINEER": "NETWORK SECURITY ENGINEER",
        "OPERATIONSMANAGER": "OPERATIONS MANAGER",
        "PMO": "PROJECT MANAGEMENT",
        "PROJECTMANAGEMENT": "PROJECT MANAGEMENT",
        "HADOOP": "BIG DATA",
        "ETL DEVELOPER": "DATA ENGINEER",
    }
    return aliases.get(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="Folder containing category subfolders of PDFs")
    ap.add_argument("--min_chars", type=int, default=200,
                    help="Skip extractions shorter than this")
    ap.add_argument("--limit_per_cat", type=int, default=0,
                    help="Cap resumes per category (0 = no cap)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ingest] root not found: {root}")
        sys.exit(1)

    rows: list[dict] = []
    skipped_empty = skipped_short = failed = 0
    per_cat: dict[str, int] = {}

    subfolders = [d for d in root.iterdir() if d.is_dir()]
    print(f"[ingest] scanning {len(subfolders)} folders in {root}")

    t0 = time.time()
    for folder in sorted(subfolders):
        cat = normalise_category(folder.name)
        pdfs = list(folder.rglob("*.pdf")) + list(folder.rglob("*.PDF"))
        pdfs += list(folder.rglob("*.docx")) + list(folder.rglob("*.DOCX"))
        if not pdfs:
            continue

        for i, pdf in enumerate(pdfs):
            if args.limit_per_cat and per_cat.get(cat, 0) >= args.limit_per_cat:
                break
            try:
                data = pdf.read_bytes()
                text = parse_bytes(pdf.name, data)
            except Exception:
                failed += 1
                continue
            if not text:
                skipped_empty += 1
                continue
            if len(text) < args.min_chars:
                skipped_short += 1
                continue
            rows.append({
                "category":     cat,
                "source_file":  str(pdf.relative_to(root)),
                "format":       pdf.suffix.lower().lstrip("."),
                "text":         text,
            })
            per_cat[cat] = per_cat.get(cat, 0) + 1
        print(f"  {folder.name:40s}  -> {cat:28s}  ({per_cat.get(cat,0)} kept)")

    if not rows:
        print("[ingest] nothing to write")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.insert(0, "doc_id", range(len(df)))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    dur = time.time() - t0
    print()
    print(f"[ingest] wrote {len(df):,} docs → {OUT_PATH}")
    print(f"[ingest] {df['category'].nunique()} unique categories")
    print(f"[ingest] format mix: {df['format'].value_counts().to_dict()}")
    print(f"[ingest] skipped: empty={skipped_empty}  short={skipped_short}  failed={failed}")
    print(f"[ingest] took {dur:.1f}s")


if __name__ == "__main__":
    main()
