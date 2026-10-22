"""Parse resumes from bytes/files.

Supports PDF, DOCX, TXT, and ZIP (recursive). Returns list of
(filename, plain_text). Empty text -> the file could not be read, but
its filename is still returned so the caller sees what was skipped.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from docx import Document

TEXT_EXTS = {".txt", ".md"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}


# Set via API to enable OCR fallback for scanned PDFs (slow: ~20s/file).
# Off by default so a normal short PDF doesn't accidentally trigger OCR.
OCR_ENABLED = False


def _read_pdf(b: bytes) -> str:
    # Extractor cascade, cheapest first:
    #   1. pypdf          — fast, handles most text PDFs
    #   2. pymupdf        — catches text layers pypdf misses
    #   3. OCR cache      — instant if this file was OCR'd before
    #   4. live OCR       — only when explicitly enabled (slow)
    text = _read_pdf_pypdf(b)
    if len(text) >= 40:
        return text
    text2 = _read_pdf_pymupdf(b)
    if len(text2) >= 40:
        return text2
    text = text2 or text

    # Cache hit — this is what makes scanned-PDF batches instant after
    # a one-time warm-up pass.
    try:
        from . import ocr_cache
        cached = ocr_cache.get(b)
        if cached:
            return cached
    except Exception:
        pass

    if OCR_ENABLED:
        try:
            from . import ocr
            ocr_text = ocr.ocr_pdf_if_scanned(b)
            if ocr_text:
                try:
                    from . import ocr_cache as _oc
                    _oc.put(b, ocr_text)      # remember for next time
                except Exception:
                    pass
                return ocr_text
        except Exception:
            pass
    return text


def _read_pdf_pypdf(b: bytes) -> str:
    try:
        r = PdfReader(io.BytesIO(b))
        return "\n".join((p.extract_text() or "") for p in r.pages).strip()
    except Exception:
        return ""


def _read_pdf_pymupdf(b: bytes) -> str:
    try:
        import pymupdf  # noqa
    except ImportError:
        return ""
    try:
        with pymupdf.open(stream=b, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc).strip()
    except Exception:
        return ""


def _read_docx(b: bytes) -> str:
    try:
        d = Document(io.BytesIO(b))
        return "\n".join(p.text for p in d.paragraphs).strip()
    except Exception:
        return ""


def _read_text(b: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return b.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return ""


def parse_bytes(name: str, data: bytes) -> str:
    ext = Path(name).suffix.lower()
    if ext in PDF_EXTS:
        return _read_pdf(data)
    if ext in DOCX_EXTS:
        return _read_docx(data)
    if ext in TEXT_EXTS:
        return _read_text(data)
    # try text as a fallback
    return _read_text(data)


def parse_upload_with_bytes(name: str, data: bytes) -> list[tuple[str, str, bytes]]:
    """Like parse_upload, but also returns the original bytes of each
    inner file (so an integrity check can inspect PDF colour/font data).

    Returns [(filename, text, raw_bytes), ...].
    """
    ext = Path(name).suffix.lower()
    if ext == ".zip":
        out: list[tuple[str, str, bytes]] = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner_ext = Path(info.filename).suffix.lower()
                    if inner_ext not in (PDF_EXTS | DOCX_EXTS | TEXT_EXTS):
                        continue
                    with zf.open(info) as fh:
                        inner = fh.read()
                        out.append((Path(info.filename).name,
                                    parse_bytes(info.filename, inner),
                                    inner))
        except zipfile.BadZipFile:
            return [(name, "", data)]
        return out
    return [(name, parse_bytes(name, data), data)]


def parse_upload(name: str, data: bytes) -> list[tuple[str, str]]:
    return [(n, t) for n, t, _ in parse_upload_with_bytes(name, data)]


def parse_uploads(files: Iterable[tuple[str, bytes]]) -> list[tuple[str, str]]:
    return [(n, t) for n, t, _ in parse_uploads_with_bytes(files)]


def parse_uploads_with_bytes(files: Iterable[tuple[str, bytes]]
                             ) -> list[tuple[str, str, bytes]]:
    """Parse a batch in parallel — PDF text extraction and DOCX unzipping
    are IO- and CPU-light-ish; a threadpool gives a real speedup for
    large batches without the multiprocessing overhead."""
    from concurrent.futures import ThreadPoolExecutor
    file_list = list(files)
    if len(file_list) <= 4:
        out: list[tuple[str, str, bytes]] = []
        for name, data in file_list:
            out.extend(parse_upload_with_bytes(name, data))
        return out
    with ThreadPoolExecutor(max_workers=min(16, len(file_list))) as ex:
        results = list(ex.map(lambda nd: parse_upload_with_bytes(*nd), file_list))
    return [item for sub in results for item in sub]
