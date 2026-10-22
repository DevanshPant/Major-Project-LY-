"""Demo generator for the integrity check.

Creates three variants of the SAME resume so you can show detection
working live in your viva:

    data/stuffed_demo/
        legit.pdf         # a normal resume
        legit.docx
        stuffed_white.pdf # legit content + hidden white-on-white keywords
        stuffed_white.docx
        stuffed_tiny.pdf  # legit content + micro-font keyword stuffing

Then upload the folder to /admin and turn Integrity check ON — the
stuffed versions will be flagged with the specific reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from docx import Document
from docx.shared import RGBColor, Pt

OUT = ROOT / "data" / "stuffed_demo"
OUT.mkdir(parents=True, exist_ok=True)

REAL_RESUME = """JANE DOE - Junior Accountant

Summary
Accounts trainee with 1 year of bookkeeping and month-end close experience
at a small firm. Comfortable with Excel, QuickBooks and basic tax filing.

Experience
2024 - Present   Assistant Accountant, Fair & Co
                 Reconciled ledgers, prepared invoices, supported audit prep.

Education
2020-2023        B.Com, Accounting

Skills
Excel | QuickBooks | basic tax | bookkeeping
"""

# Keywords a resume-stuffer would sprinkle in to game an ATS
STUFF = (
    "senior accountant CPA CA MBA GAAP IFRS SOX audit tax preparation "
    "financial reporting management accounting cost accounting variance analysis "
    "budget forecasting Big 4 Deloitte PwC EY KPMG SAP Oracle Hyperion "
    "10+ years experience"
)


# ---------- PDF variants ----------
def _pdf_normal(path: Path):
    p = FPDF(); p.add_page(); p.set_font("Helvetica", size=11)
    for line in REAL_RESUME.split("\n"):
        p.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    p.output(str(path))


def _pdf_white_stuffed(path: Path):
    p = FPDF(); p.add_page()
    p.set_font("Helvetica", size=11)
    p.set_text_color(0, 0, 0)
    for line in REAL_RESUME.split("\n"):
        p.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    # append a wall of white-on-white keywords
    p.ln(4)
    p.set_text_color(255, 255, 255)      # invisible on white paper
    for chunk in (STUFF[i:i+95] for i in range(0, len(STUFF), 95)):
        p.cell(0, 5, chunk, new_x="LMARGIN", new_y="NEXT")
    p.output(str(path))


def _pdf_tiny_stuffed(path: Path):
    p = FPDF(); p.add_page()
    p.set_font("Helvetica", size=11); p.set_text_color(0, 0, 0)
    for line in REAL_RESUME.split("\n"):
        p.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    p.ln(4)
    # micro-font stuffing — visible-black but 2pt high
    p.set_font("Helvetica", size=2)
    for chunk in (STUFF[i:i+300] for i in range(0, len(STUFF)*3, 300)):
        p.cell(0, 1, chunk, new_x="LMARGIN", new_y="NEXT")
    p.output(str(path))


# ---------- DOCX variants ----------
def _docx_normal(path: Path):
    d = Document()
    for line in REAL_RESUME.split("\n"):
        d.add_paragraph(line)
    d.save(str(path))


def _docx_white_stuffed(path: Path):
    d = Document()
    for line in REAL_RESUME.split("\n"):
        d.add_paragraph(line)
    p = d.add_paragraph()
    r = p.add_run(STUFF)
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)     # white text
    d.save(str(path))


def _docx_tiny_stuffed(path: Path):
    d = Document()
    for line in REAL_RESUME.split("\n"):
        d.add_paragraph(line)
    p = d.add_paragraph()
    r = p.add_run(STUFF * 3)
    r.font.size = Pt(1)
    d.save(str(path))


def main():
    _pdf_normal(OUT / "legit.pdf")
    _pdf_white_stuffed(OUT / "stuffed_white.pdf")
    _pdf_tiny_stuffed(OUT / "stuffed_tiny.pdf")
    _docx_normal(OUT / "legit.docx")
    _docx_white_stuffed(OUT / "stuffed_white.docx")
    _docx_tiny_stuffed(OUT / "stuffed_tiny.docx")
    print(f"[demo] wrote 6 files to {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
