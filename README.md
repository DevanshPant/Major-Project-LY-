# ResumeAI — Semantic Resume Screening & Shortlisting

Group 29 · K.J. Somaiya · Final Year Project

Editorial-brutalist UI (Prmpt-inspired) on top of a **MiniLM-L6-v2 + FAISS + ms-marco cross-encoder** pipeline. Mixed-format resume support (PDF + DOCX). Fine-tuneable reranker. Live TF-IDF baseline for viva comparisons.

---

## USPs (defensible in viva)

1. **Semantic understanding** — sentence-BERT embeddings, not keyword bingo.
2. **Two-stage retrieval** — FAISS shortlists 40 → cross-encoder picks top-N.
3. **Fixes the 80% experience-extraction gap** shown in your slides.
4. **Explainable scores** — semantic / skills / experience / education bars.
5. **Bias-check toggle** — strips name / email / phone before scoring.
6. **Live model comparison** — TF-IDF vs MiniLM+CE on the same JD.
7. **Fine-tuned reranker** — weak-supervision training script included.
8. **Mixed-format support** — PDF, DOCX, TXT, or a ZIP folder.
9. **CSV export** — one click, ready for HR handoff.

---

## Setup (Windows)

```powershell
cd "D:\major porject\resume_ai"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare the data

Your extracted CSV is already in `D:\major porject\Resume.csv`. This one command copies it into place, builds the corpus, and writes **48 real PDF + 48 real DOCX** sample files for admin-flow testing:

```powershell
copy "D:\major porject\Resume.csv" "data\Resume.csv"
python -m scripts.build_corpus --samples_per_cat 4
```

Output:

```
data/Resume.csv                      # original Kaggle CSV
data/corpus.parquet                  # 2,483 docs, 24 categories, 51% docx / 49% pdf
data/sample_uploads/{CATEGORY}/*.pdf,*.docx    # 96 real files for demo uploads
```

## Build the FAISS index

```powershell
python -m scripts.build_index
```

Downloads MiniLM (~90 MB) on first run. Writes:

```
models/resumes.faiss           # 2483 × 384 vectors
models/resumes.meta.parquet
```

## Fine-tune the cross-encoder (your "own model" piece)

```powershell
python -m scripts.train_reranker --epochs 1 --limit 1500 --batch_size 8
```

Weak supervision: synthetic positive pairs (JD template + resume from that
category) vs negatives (JD template + resume from a different category).
Result lives at `models/cross_encoder_finetuned/` and is loaded
automatically by the app on next request.

## Evaluate

```powershell
python -m scripts.evaluate --k 10 --queries 24
```

Prints Precision@k and NDCG@k for **new model vs TF-IDF baseline** — copy these numbers straight into your report.

## Run the app

```powershell
python run.py
```

Open <http://127.0.0.1:8000>.

---

## Admin flow

1. `/` → click **Screen resumes**.
2. Paste JD.
3. Drop resumes — PDF, DOCX, TXT, or a ZIP. (Try dragging in a category folder from `data/sample_uploads/` — mix of PDF and DOCX.)
4. Pick top-N (1–200).
5. Hit **Run matching**.
6. **Download CSV**.

## Project layout

```
resume_ai/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py            # tunables
│   ├── api/routes.py        # /api/match /api/compare /api/export /api/health
│   ├── core/
│   │   ├── parser.py        # PDF / DOCX / TXT / ZIP
│   │   ├── extractor.py     # skills + YOE + degree
│   │   ├── embedder.py      # MiniLM bi-encoder
│   │   ├── vector_store.py  # FAISS
│   │   ├── reranker.py      # ms-marco cross-encoder (+ fine-tuned)
│   │   ├── baseline.py      # TF-IDF (Model 1)
│   │   ├── scorer.py        # weighted final score
│   │   └── pipeline.py      # orchestrator
│   ├── templates/           # landing / admin / compare
│   └── static/              # style.css + app.js
├── scripts/
│   ├── build_corpus.py      # build corpus.parquet + real PDF/DOCX samples
│   ├── build_index.py       # embed → FAISS
│   ├── train_reranker.py    # fine-tune cross-encoder
│   ├── ingest_pdfs.py       # (for text-extractable PDFs, not scanned ones)
│   └── evaluate.py          # P@k, NDCG@k
├── data/
│   ├── Resume.csv           # source text
│   ├── corpus.parquet       # training corpus
│   └── sample_uploads/      # real PDF+DOCX files for demo
├── models/                  # FAISS index + fine-tuned weights
├── Resumes PDF/             # 8,905 scanned PDFs (Kaggle) — not text-extractable
└── tests/
```

## Notes

- The `Resumes PDF/` folder from Kaggle is **scanned images**, not text-based. `pypdf` returns empty for them — matching your PPT's dataset description ("Scanned Images converted to PDFs"). The extracted text lives in `Resume.csv`, which is what we train on.
- To turn a scanned PDF folder into text you would need OCR (EasyOCR / Tesseract). It's not required for training since we already have the text.
- All body text ≥ 4.5:1 contrast on the background. No text is ever the background color.
