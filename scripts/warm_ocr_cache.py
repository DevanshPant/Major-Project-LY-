"""One-time OCR warm-up pass.

OCRs every scanned PDF in a folder and stores the text in
models/ocr_cache.db. After this runs, the app resolves those same
files instantly (cache hit by SHA-256 of the file bytes), so a live
demo over a scanned-PDF corpus is fast.

Uses multiprocessing — EasyOCR is CPU-bound and holds the GIL, so
threads do not help but separate processes do.

Run:
    python -m scripts.warm_ocr_cache --dir "Resume_Testing/resumes"
    python -m scripts.warm_ocr_cache --dir "Test files/resumes" --workers 6
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Keep each worker single-threaded so N processes don't oversubscribe the CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _ocr_one(path_str: str) -> tuple[str, int]:
    """Runs inside a worker process. Returns (path, chars_extracted)."""
    import torch
    torch.set_num_threads(1)

    from app.core import ocr, ocr_cache
    from app.core.parser import _read_pdf_pypdf, _read_pdf_pymupdf

    p = Path(path_str)
    try:
        data = p.read_bytes()
    except Exception:
        return path_str, -1

    # Skip if already cached.
    if ocr_cache.get(data):
        return path_str, 0

    # Skip if a plain extractor can already read it.
    if len(_read_pdf_pypdf(data)) >= 40 or len(_read_pdf_pymupdf(data)) >= 40:
        return path_str, 0

    try:
        text = ocr.ocr_pdf(data, max_pages=1, max_width=1000)
    except Exception:
        return path_str, -1
    if text:
        ocr_cache.put(data, text)
        return path_str, len(text)
    return path_str, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of PDFs to warm")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0, help="cap files (0 = all)")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_absolute():
        root = ROOT / root
    if not root.exists():
        print(f"[warm] folder not found: {root}")
        sys.exit(1)

    # Case-insensitive filesystems (Windows) match *.pdf and *.PDF to the
    # same files, so de-duplicate by resolved path.
    seen: set[str] = set()
    pdfs: list[str] = []
    for pat in ("*.pdf", "*.PDF"):
        for p in root.rglob(pat):
            rp = str(p.resolve()).lower()
            if rp not in seen:
                seen.add(rp)
                pdfs.append(str(p))
    pdfs.sort()
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"[warm] no PDFs under {root}")
        sys.exit(1)

    print(f"[warm] {len(pdfs)} PDFs | {args.workers} worker processes")
    print(f"[warm] cache -> {ROOT / 'models' / 'ocr_cache.db'}")

    t0 = time.time()
    done = ok = skipped = failed = 0
    with Pool(processes=args.workers) as pool:
        for path, chars in pool.imap_unordered(_ocr_one, pdfs, chunksize=1):
            done += 1
            if chars > 0:
                ok += 1
            elif chars == 0:
                skipped += 1
            else:
                failed += 1
            if done % 10 == 0 or done == len(pdfs):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(pdfs) - done) / rate if rate else 0
                print(f"  {done}/{len(pdfs)}  ok={ok} skip={skipped} fail={failed}  "
                      f"{rate:.2f} files/s  ETA {eta/60:.1f} min", flush=True)

    from app.core import ocr_cache
    s = ocr_cache.stats()
    print()
    print(f"[warm] finished in {(time.time()-t0)/60:.1f} min")
    print(f"[warm] cache now holds {s['entries']} entries ({s['db_mb']} MB)")


if __name__ == "__main__":
    main()
