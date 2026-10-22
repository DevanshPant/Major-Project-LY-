"""Copy every already-OCR'd resume into data/demo_ready/.

Anything in that folder is guaranteed to be a cache hit, so it loads in
milliseconds. Use it for a live demo where you cannot risk hitting an
un-cached file and waiting on OCR.

Run:
    python -m scripts.make_demo_folder --dir "Resume_Testing/resumes"
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import ocr_cache
from app.core.parser import _read_pdf_pypdf, _read_pdf_pymupdf

OUT = ROOT / "data" / "demo_ready"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.dir)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        print(f"[demo] not found: {src}")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        if old.is_file():
            old.unlink()

    seen: set[str] = set()
    pdfs: list[Path] = []
    for pat in ("*.pdf", "*.PDF"):
        for p in src.rglob(pat):
            k = str(p.resolve()).lower()
            if k not in seen:
                seen.add(k)
                pdfs.append(p)

    copied = 0
    for p in sorted(pdfs):
        if args.limit and copied >= args.limit:
            break
        try:
            data = p.read_bytes()
        except Exception:
            continue
        # Fast if: already OCR-cached, OR readable without OCR at all.
        fast = bool(ocr_cache.get(data)) \
            or len(_read_pdf_pypdf(data)) >= 40 \
            or len(_read_pdf_pymupdf(data)) >= 40
        if fast:
            shutil.copy2(p, OUT / p.name)
            copied += 1

    print(f"[demo] copied {copied} instant-load resumes -> {OUT}")
    print("[demo] every file here loads in milliseconds. Safe for a live demo.")


if __name__ == "__main__":
    main()
