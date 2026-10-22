"""Optional OCR fallback for scanned/image-only PDFs.

Uses EasyOCR — pure Python, downloads a ~64 MB detection + recognition
model on first use. Skipped entirely if EasyOCR is not installed.

Usage:
    from .ocr import ocr_pdf_if_scanned
    text = ocr_pdf_if_scanned(pdf_bytes)   # "" if not scanned or OCR unavailable
"""
from __future__ import annotations

import io
import logging
from functools import lru_cache


log = logging.getLogger(__name__)


def _has_easyocr() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def looks_scanned(pdf_bytes: bytes) -> bool:
    """A cheap heuristic — if pypdf's text extraction returned <20 chars
    for a non-trivial PDF, it's almost certainly scanned."""
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(pdf_bytes))
        chars = 0
        for p in r.pages[:3]:
            chars += len((p.extract_text() or "").strip())
        return chars < 20
    except Exception:
        return False


def ocr_pdf(pdf_bytes: bytes, max_pages: int = 1, max_width: int = 1000) -> str:
    """OCR up to `max_pages` from the PDF (default 1 — resume page 1 has
    enough context to rank). Images are downscaled to at most
    `max_width` px on the long side, which cuts EasyOCR runtime ~2×
    without losing text recognition."""
    if not _has_easyocr():
        return ""
    # Prefer PyMuPDF for rasterisation (fast, no poppler); fall back to
    # pulling embedded images out of pypdf.
    imgs = _rasterise_pymupdf(pdf_bytes, max_pages) or \
           _images_from_pypdf(pdf_bytes, max_pages)
    if not imgs:
        return ""

    reader = _reader()
    out: list[str] = []
    for img in imgs:
        try:
            img = _downscale(img, max_width)
            res = reader.readtext(_to_np(img), detail=0, paragraph=True)
            out.extend(res)
        except Exception:
            continue
    return "\n".join(out).strip()


def _rasterise_pymupdf(pdf_bytes: bytes, max_pages: int):
    """Turn each page into a raster image. Much faster than image-mining."""
    try:
        import pymupdf
        from PIL import Image
    except Exception:
        return []
    out = []
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc[:max_pages]:
                # 150 DPI is fine for OCR of clean scans (~1275 px wide letter).
                pix = page.get_pixmap(dpi=150)
                mode = "RGB" if pix.n < 5 else "RGBA"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                out.append(img)
    except Exception:
        return []
    return out


def _images_from_pypdf(pdf_bytes: bytes, max_pages: int):
    try:
        from pypdf import PdfReader
        from PIL import Image
    except Exception:
        return []
    out = []
    try:
        r = PdfReader(io.BytesIO(pdf_bytes))
        for page in r.pages[:max_pages]:
            for image_file_object in page.images:
                try:
                    out.append(Image.open(io.BytesIO(image_file_object.data)))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def _downscale(img, max_width: int):
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    return img.resize((max_width, int(img.height * ratio)))


def _to_np(img):
    import numpy as np
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return np.array(img)


def ocr_pdf_if_scanned(pdf_bytes: bytes) -> str:
    if not looks_scanned(pdf_bytes):
        return ""
    if not _has_easyocr():
        return ""
    return ocr_pdf(pdf_bytes)


def is_available() -> bool:
    return _has_easyocr()
