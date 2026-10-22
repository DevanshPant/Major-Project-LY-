"""Build the FAISS index from the ingested corpus.

Prefers  data/corpus.parquet  (produced by ingest_pdfs.py).
Falls back to data/Resume.csv (Kaggle) if the parquet is missing.

Run:
    python -m scripts.build_index
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR, DATASET_CSV, FAISS_INDEX, FAISS_META
from app.core.embedder import encode
from app.core.vector_store import VectorStore

CORPUS_PARQUET = DATA_DIR / "corpus.parquet"


def load_corpus() -> pd.DataFrame:
    if CORPUS_PARQUET.exists():
        df = pd.read_parquet(CORPUS_PARQUET)
        print(f"[build_index] Using corpus.parquet ({len(df):,} docs)")
        return df

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            "No corpus found. Run one of:\n"
            "  python -m scripts.ingest_pdfs      # for the PDF folder tree\n"
            "  python -m scripts.download_data    # for the Kaggle CSV"
        )

    df = pd.read_csv(DATASET_CSV)
    text_col = "Resume_str" if "Resume_str" in df.columns else df.columns[0]
    cat_col = "Category" if "Category" in df.columns else None
    df = df.rename(columns={text_col: "text"})
    if cat_col:
        df = df.rename(columns={cat_col: "category"})
    else:
        df["category"] = "unknown"
    df = df[["text", "category"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 100].reset_index(drop=True)
    df["doc_id"] = df.index
    df["source_file"] = df["doc_id"].astype(str) + ".csv"
    df["format"] = "csv"
    print(f"[build_index] Using {DATASET_CSV.name} ({len(df):,} docs)")
    return df


def main():
    df = load_corpus()
    print(f"[build_index] {df['category'].nunique()} categories | "
          f"format mix: {df['format'].value_counts().to_dict()}")
    print("[build_index] encoding — first run downloads ~90 MB for MiniLM…")
    vecs = encode(df["text"].tolist(), batch_size=32)
    print(f"[build_index] embeddings: {vecs.shape}")
    store = VectorStore()
    store.build(vecs, df[["doc_id", "category", "source_file", "format"]])
    store.save(FAISS_INDEX, FAISS_META)
    print(f"[build_index] wrote → {FAISS_INDEX}")
    print(f"[build_index] wrote → {FAISS_META}")


if __name__ == "__main__":
    main()
