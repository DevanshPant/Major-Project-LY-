"""HTTP endpoints. Kept small — the pipeline module does the real work."""
from __future__ import annotations

import csv
import io
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..config import MAX_UPLOAD_MB, MAX_RESUMES_PER_REQUEST, DEFAULT_TOP_N
from ..core import parser, pipeline, reranker
from ..core.integrity import analyse as integrity_analyse
from ..core.scorer import quality_label
from ..auth import require_admin

router = APIRouter()


def admin_dep(request: Request):
    require_admin(request)


def _estimate_file_count(raw: list[tuple[str, bytes]]) -> int:
    """Peek inside ZIPs so the OCR decision reflects total files, not upload count."""
    import zipfile, io
    total = 0
    for name, data in raw:
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    total += sum(1 for i in zf.infolist() if not i.is_dir())
            except Exception:
                total += 1
        else:
            total += 1
    return total


def _count_scanned_pdfs(unread_files) -> int:
    """Best-effort classification of empty files as scanned PDFs."""
    hits = 0
    for n, b in unread_files:
        if n.lower().endswith(".pdf"):
            try:
                from ..core.ocr import looks_scanned
                if looks_scanned(b):
                    hits += 1
            except Exception:
                pass
    return hits


def _resolve_jd(jd_text: str, jd_file: UploadFile | None) -> str:
    """A JD may arrive as pasted text OR as an uploaded PDF/DOCX/TXT.
    If a file is present, it wins. Empty result => 400."""
    text = (jd_text or "").strip()
    if jd_file and jd_file.filename:
        data = jd_file.file.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "JD file exceeds 20 MB")
        from ..core.parser import parse_bytes
        extracted = parse_bytes(jd_file.filename, data).strip()
        if extracted:
            return extracted
        # File uploaded but nothing extractable — fall back to pasted text.
        if not text:
            raise HTTPException(422,
                f"Could not read text from JD file '{jd_file.filename}'. "
                "Try a text-based PDF or paste the JD instead.")
    if not text:
        raise HTTPException(400, "Provide a job description (paste text or upload a JD file)")
    return text


def _read_uploads(files: List[UploadFile]) -> list[tuple[str, bytes]]:
    total = 0
    cap = MAX_UPLOAD_MB * 1024 * 1024
    out: list[tuple[str, bytes]] = []
    for f in files:
        data = f.file.read()
        total += len(data)
        if total > cap:
            raise HTTPException(
                413,
                f"Batch exceeds {MAX_UPLOAD_MB} MB total "
                f"({total / (1024*1024):.0f} MB uploaded). "
                "Split into two batches or contact the admin to raise the limit.",
            )
        out.append((f.filename or "unnamed", data))
    return out


def _matches_to_payload(matches) -> list[dict]:
    return [m.to_row() for m in matches]


@router.post("/api/match", dependencies=[Depends(admin_dep)])
async def match(
    jd: str = Form(""),
    top_n: int = Form(DEFAULT_TOP_N),
    use_reranker: bool = Form(True),
    bias_check: bool = Form(False),
    files: List[UploadFile] = File(...),
    jd_file: UploadFile | None = File(None),
):
    """Robust match endpoint.

    OCR and integrity checks are handled automatically — no user toggles:
      - OCR auto-enables when the batch is small (<=20 files) so scanned
        PDFs still work but a 2,000-file batch doesn't take hours.
      - Integrity check always runs on the top-N shortlist (cheap).
      - Files that can't be parsed are counted and reported, never crash
        the request. Only fails hard if every single file is empty.
    """
    jd = _resolve_jd(jd, jd_file)
    if not files:
        raise HTTPException(400, "Upload at least one resume file or a ZIP")
    top_n = max(1, min(int(top_n), 1500))

    raw = _read_uploads(files)

    # Auto-OCR only for very small batches (<=10 files). With the new
    # cascade (pypdf -> pymupdf -> OCR of page 1 only, downscaled)
    # scanned files now take ~5-8 s each instead of ~20 s, so small
    # mixed batches stay responsive.
    n_uploads_hint = _estimate_file_count(raw)
    parser.OCR_ENABLED = n_uploads_hint <= 10

    parsed_all = parser.parse_uploads_with_bytes(raw)   # [(name, text, raw_bytes)]
    unread_files = [(n, b) for n, t, b in parsed_all if not t.strip()]
    unread = len(unread_files)
    parsed = [(n, t, b) for n, t, b in parsed_all if t.strip()]

    ocr_partial_note = ""
    scanned_count = _count_scanned_pdfs(unread_files[:40])
    scanned_ratio_note = ""
    if unread and unread > 0:
        # Extrapolate the sample to give the user a realistic hint.
        est_scanned = int(round(unread * scanned_count / max(len(unread_files[:40]), 1)))
        if est_scanned:
            scanned_ratio_note = (
                f"{est_scanned} of the {unread} unreadable file(s) appear to "
                "be scanned/image PDFs — save them as DOCX from the source "
                "tool, or split them into a batch of 20 or fewer for OCR."
            )

    if not parsed:
        # Nothing parsed on the fast path — try OCR on a small sample of the
        # scanned PDFs as a partial-success fallback, so a big scanned-only
        # batch still returns something useful instead of a dead-end error.
        scanned_hits = scanned_count
        try:
            from ..core import ocr as _ocr
            ocr_available = _ocr.is_available()
        except Exception:
            ocr_available = False

        if scanned_hits > 0 and ocr_available:
            # HARD time budget. OCR is ~15 s/file on CPU, so we stop
            # after OCR_SECONDS regardless of how many are left — a web
            # request must never hang for minutes.
            OCR_SECONDS = 45
            OCR_MAX_FILES = 12
            import time as _time
            deadline = _time.monotonic() + OCR_SECONDS
            queue = [(n, b) for n, b in unread_files
                     if n.lower().endswith(".pdf")][:OCR_MAX_FILES]

            from concurrent.futures import ThreadPoolExecutor, as_completed
            def _one(nb):
                n, b = nb
                try:
                    return n, _ocr.ocr_pdf(b).strip(), b
                except Exception:
                    return n, "", b

            done_results = []
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = {ex.submit(_one, nb): nb for nb in queue}
                for fut in as_completed(futures, timeout=None):
                    try:
                        done_results.append(fut.result(timeout=1))
                    except Exception:
                        pass
                    if _time.monotonic() > deadline:
                        for f in futures:
                            f.cancel()
                        break

            parsed = [t for t in done_results if t[1]]
            if parsed:
                # Cache what we just paid for so the next run is instant.
                for n, txt, b in parsed:
                    try:
                        from ..core import ocr_cache as _oc
                        _oc.put(b, txt)
                    except Exception:
                        pass
                ocr_partial_note = (
                    f"{len(parsed)} of {unread} scanned PDF(s) were read via OCR "
                    f"within the {OCR_SECONDS}s budget. Run "
                    "`python -m scripts.warm_ocr_cache` to pre-process the rest "
                    "so they load instantly."
                )
                unread = unread - len(parsed)

        if not parsed:
            if scanned_hits > 0 and not ocr_available:
                raise HTTPException(
                    422,
                    f"All {unread} files appear to be scanned/image PDFs and "
                    "no OCR engine is installed. Install EasyOCR "
                    "(`pip install easyocr`) or upload text-based files."
                )
            if scanned_hits > 0:
                raise HTTPException(
                    422,
                    f"All {unread} PDFs are scanned images and even OCR "
                    "couldn't recover text (some scanners produce vector "
                    "output that isn't rasterised). Try DOCX exports."
                )
            raise HTTPException(
                422,
                f"None of the {unread} uploaded file(s) contained readable text. "
                "Supported formats: PDF (text-based), DOCX, TXT, or a ZIP of these."
            )
    if len(parsed) > MAX_RESUMES_PER_REQUEST:
        raise HTTPException(
            413,
            f"{len(parsed)} resumes in batch — the per-request cap is "
            f"{MAX_RESUMES_PER_REQUEST}. Split into smaller batches.",
        )

    # Candidates first — integrity check happens later, only on the
    # shortlist that survives semantic ranking (huge speedup for big
    # batches).  The raw bytes stay in `raw_by_name` so we can inspect
    # any flagged file.
    cands: list[pipeline.Candidate] = []
    raw_by_name: dict[str, bytes] = {}
    for n, t, data in parsed:
        cands.append(pipeline.Candidate(file=n, text=t))
        raw_by_name[n] = data
    suspicious_count = 0

    try:
        # First-pass ranking over a wide shortlist so a stuffed resume
        # can't hide outside the integrity check that follows.
        shortlist_n = max(top_n, min(200, len(cands)))
        matches = pipeline.run_match(
            jd_text=jd, candidates=cands, top_n=shortlist_n,
            use_reranker=use_reranker, bias_check=bias_check,
            integrity_check=False,
        )
    except ModuleNotFoundError as e:
        raise HTTPException(
            503,
            f"ML dependency missing: {e.name}. Run: pip install -r requirements.txt",
        )

    # Silent integrity check on the shortlist — no user toggle. Runs in
    # parallel and only scans the top-200 files, so it's cheap even on
    # huge batches.
    if matches:
        from concurrent.futures import ThreadPoolExecutor
        def _check(m):
            data = raw_by_name.get(m.candidate.file, b"")
            return m, integrity_analyse(m.candidate.file, data, m.candidate.text)
        with ThreadPoolExecutor(max_workers=min(16, len(matches))) as ex:
            for m, report in ex.map(_check, matches):
                m.candidate.integrity = report
                if report.suspicious:
                    suspicious_count += 1
                    if report.score_penalty > 0:
                        m.score.final = max(0.0, m.score.final - report.score_penalty * 100)
                        m.quality = quality_label(m.score.final)
        matches.sort(key=lambda m: m.score.final, reverse=True)
        for r, m in enumerate(matches, start=1):
            m.rank = r

    matches = matches[:top_n]
    return {
        "count_uploaded": len(parsed),
        "count_unreadable": unread,
        "count_suspicious": suspicious_count,
        "ocr_note": ocr_partial_note or scanned_ratio_note,
        "top_n": top_n,
        "reranker": "finetuned" if reranker.is_finetuned() else "pretrained",
        "results": _matches_to_payload(matches),
    }


@router.post("/api/compare", dependencies=[Depends(admin_dep)])
async def compare(
    jd: str = Form(""),
    top_n: int = Form(DEFAULT_TOP_N),
    files: List[UploadFile] = File(...),
    jd_file: UploadFile | None = File(None),
):
    """Return TF-IDF baseline vs MiniLM+CE rankings side by side."""
    jd = _resolve_jd(jd, jd_file)
    raw = _read_uploads(files)
    parsed = [(n, t) for n, t in parser.parse_uploads(raw) if t.strip()]
    if not parsed:
        raise HTTPException(422, "No readable resumes found")
    cands = [pipeline.Candidate(file=n, text=t) for n, t in parsed]
    new_matches = pipeline.run_match(jd, cands, top_n=top_n, use_reranker=True)
    old_matches = pipeline.run_baseline(jd, cands, top_n=top_n)
    return {
        "new_model": _matches_to_payload(new_matches),
        "baseline":  _matches_to_payload(old_matches),
    }


@router.post("/api/export", dependencies=[Depends(admin_dep)])
async def export_csv(payload: str = Form(...)):
    """`payload` is a JSON string of the results array from /api/match."""
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid payload")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(400, "Empty payload")

    # Flatten the explanation object for CSV consumers.
    flat_rows = []
    for r in rows:
        ex = r.get("explanation") or {}
        flat_rows.append({
            **r,
            "why_chosen": ex.get("headline", ""),
            "strengths":  " | ".join(ex.get("strengths", [])),
            "concerns":   " | ".join(ex.get("concerns", [])),
            "integrity_flags": " | ".join(r.get("flags") or []),
        })
    buf = io.StringIO()
    cols = ["rank", "file", "final_score", "quality", "why_chosen",
            "strengths", "concerns",
            "semantic", "skills", "experience", "education", "years", "degree",
            "matched_skills", "missing_skills", "integrity_flags"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in flat_rows:
        w.writerow(r)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="shortlist.csv"'},
    )


@router.get("/api/health")
async def health():
    try:
        from ..core import ocr as ocr_mod
        ocr_enabled = ocr_mod.is_available()
    except Exception:
        ocr_enabled = False
    try:
        from ..core import ocr_cache
        cache = ocr_cache.stats()
    except Exception:
        cache = {"entries": 0}
    return {
        "ok": True,
        "reranker": "finetuned" if reranker.is_finetuned() else "pretrained",
        "ocr_available": ocr_enabled,
        "ocr_cache": cache,
    }
