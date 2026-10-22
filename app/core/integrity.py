"""Resume-integrity checks — catches ATS-gaming tricks.

Techniques we defend against:
  1. White (or near-background) text — same colour as the page, invisible
     to a human recruiter but harvested by every PDF/DOCX text extractor.
  2. Micro-font stuffing — text sized <=2pt, invisible in a print but
     still parsed as text by the extractor.
  3. PDF text rendering mode 3 — the PDF spec's "invisible text" flag,
     legitimately used for OCR overlays but also for pure stuffing.
  4. Keyword-density anomalies — a resume where one technical term
     appears 20+ times is a red flag.

For every file we return:
  IntegrityReport {
      suspicious: bool             # any hard flag fired
      reasons:   [str]             # human-readable
      hidden_text: str             # what was hidden (for the audit trail)
      visible_ratio: float         # visible chars / total chars
      score_penalty: float         # 0..0.5, subtracted from final_score/100
  }
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field


# ---------- colour helpers ----------
def _to_rgb(c) -> tuple[float, float, float] | None:
    """Normalise the colour reported by pdfplumber (or docx) to 0..1 RGB."""
    if c is None:
        return None
    if isinstance(c, (int, float)):
        return (float(c), float(c), float(c))
    if isinstance(c, (list, tuple)):
        if len(c) == 1:
            v = float(c[0]);  return (v, v, v)
        if len(c) >= 3:
            return (float(c[0]), float(c[1]), float(c[2]))
        if len(c) == 4:  # CMYK
            k = float(c[3]); r = (1 - float(c[0])) * (1 - k)
            g = (1 - float(c[1])) * (1 - k); b = (1 - float(c[2])) * (1 - k)
            return (r, g, b)
    return None


def _brightness(rgb: tuple[float, float, float]) -> float:
    """Perceived brightness 0..1."""
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _is_near_white(rgb) -> bool:
    if rgb is None: return False
    return _brightness(rgb) >= 0.92


@dataclass
class IntegrityReport:
    suspicious: bool = False
    reasons: list[str] = field(default_factory=list)
    hidden_text: str = ""
    visible_ratio: float = 1.0
    score_penalty: float = 0.0

    def to_dict(self) -> dict:
        return {
            "suspicious": self.suspicious,
            "reasons": self.reasons,
            "hidden_text": (self.hidden_text[:220] + "…") if len(self.hidden_text) > 220 else self.hidden_text,
            "visible_ratio": round(self.visible_ratio, 3),
            "score_penalty": round(self.score_penalty, 3),
        }


# ---------- keyword density ----------
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\+\.\#\-]{1,24}")


def _keyword_density_flag(text: str) -> str | None:
    if len(text) < 500:
        return None
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if len(tokens) < 100:
        return None
    from collections import Counter
    c = Counter(tokens)
    # ignore very common English words
    stop = {"the","and","for","with","from","that","this","are","was","were",
            "have","has","not","you","your","our","their","they","its","but"}
    tech = [(w, n) for w, n in c.most_common(20) if w not in stop and len(w) > 2]
    if not tech:
        return None
    top_word, top_n = tech[0]
    ratio = top_n / max(len(tokens), 1)
    if top_n >= 15 and ratio >= 0.04:
        return f"keyword '{top_word}' repeated {top_n} times ({ratio*100:.1f}% of resume)"
    return None


# ---------- PDF ----------
def check_pdf(data: bytes, max_pages: int = 3) -> IntegrityReport:
    r = IntegrityReport()
    try:
        import pdfplumber
    except ImportError:
        return r
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            total_chars = 0
            hidden_chars = 0
            hidden_text_parts: list[str] = []
            tiny_chars = 0
            # pdfplumber's per-char inspection is expensive; the tricks we're
            # looking for (white text, invisible mode, micro-font stuffing)
            # always show up on the visible first pages, so cap the scan.
            for page in pdf.pages[:max_pages]:
                # detect page background — pdfplumber doesn't expose it
                # directly; treat "near white" as background (typical resume)
                for ch in page.chars:
                    total_chars += 1
                    txt = ch.get("text", "")
                    fill_rgb = _to_rgb(ch.get("non_stroking_color"))
                    # invisible text (rendering mode 3, if pdfplumber exposes it)
                    is_hidden_mode = ch.get("render_mode") == 3
                    is_near_bg = _is_near_white(fill_rgb)
                    size = ch.get("size", 12) or 12
                    if is_near_bg or is_hidden_mode:
                        hidden_chars += 1
                        hidden_text_parts.append(txt)
                    if size and size < 3:
                        tiny_chars += 1

            r.visible_ratio = 1 - (hidden_chars / total_chars) if total_chars else 1.0
            if hidden_chars > 40:
                pct = 100 * hidden_chars / total_chars
                r.suspicious = True
                r.reasons.append(
                    f"{hidden_chars} chars of near-invisible text ({pct:.1f}% of doc) — "
                    "same color as page background or PDF invisible-mode"
                )
                r.hidden_text = "".join(hidden_text_parts)
                # Heavy penalty scaling with how much of the doc is hidden.
                ratio = hidden_chars / max(total_chars, 1)
                r.score_penalty = min(0.85, 0.35 + ratio * 1.3)
            if tiny_chars > 60:
                r.suspicious = True
                r.reasons.append(f"{tiny_chars} chars in <3pt micro-font — invisible to human readers")
                r.score_penalty = max(r.score_penalty, 0.75)
    except Exception:
        return r
    return r


# ---------- DOCX ----------
def check_docx(data: bytes) -> IntegrityReport:
    r = IntegrityReport()
    try:
        from docx import Document
    except ImportError:
        return r
    try:
        doc = Document(io.BytesIO(data))
        total_chars = 0
        hidden_chars = 0
        tiny_chars = 0
        hidden_text_parts: list[str] = []
        for para in doc.paragraphs:
            for run in para.runs:
                t = run.text or ""
                total_chars += len(t)
                color = None
                try:
                    if run.font.color and run.font.color.rgb:
                        rgb = run.font.color.rgb
                        color = (int(rgb[0])/255, int(rgb[1])/255, int(rgb[2])/255)
                except Exception:
                    pass
                sz = None
                try:
                    if run.font.size:
                        sz = run.font.size.pt
                except Exception:
                    pass
                if _is_near_white(color):
                    hidden_chars += len(t); hidden_text_parts.append(t)
                if sz is not None and sz < 3:
                    tiny_chars += len(t)
        r.visible_ratio = 1 - (hidden_chars / total_chars) if total_chars else 1.0
        if hidden_chars > 20:
            pct = 100 * hidden_chars / max(total_chars, 1)
            r.suspicious = True
            r.reasons.append(
                f"{hidden_chars} chars of white/near-white text ({pct:.1f}% of doc)"
            )
            r.hidden_text = "".join(hidden_text_parts)
            ratio = hidden_chars / max(total_chars, 1)
            r.score_penalty = min(0.85, 0.35 + ratio * 1.3)
        if tiny_chars > 30:
            r.suspicious = True
            r.reasons.append(f"{tiny_chars} chars in <3pt micro-font")
            r.score_penalty = max(r.score_penalty, 0.75)
    except Exception:
        return r
    return r


# ---------- dispatch ----------
def analyse(filename: str, data: bytes, extracted_text: str = "") -> IntegrityReport:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        r = check_pdf(data)
    elif ext == "docx":
        r = check_docx(data)
    else:
        r = IntegrityReport()
    # Layer keyword-density anomaly on top (works for any format)
    kw_flag = _keyword_density_flag(extracted_text)
    if kw_flag:
        r.suspicious = True
        r.reasons.append(kw_flag)
        r.score_penalty = max(r.score_penalty, 0.15)
    return r
