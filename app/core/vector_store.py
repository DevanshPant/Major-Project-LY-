"""FAISS wrapper. Uses IndexFlatIP over L2-normalised vectors, so the
inner product IS the cosine similarity."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import FAISS_INDEX, FAISS_META, EMBED_DIM


class VectorStore:
    def __init__(self):
        import faiss
        self._faiss = faiss
        self.index: Optional["faiss.Index"] = None
        self.meta: Optional[pd.DataFrame] = None

    # ---------- build ----------
    def build(self, embeddings: np.ndarray, meta: pd.DataFrame) -> None:
        assert embeddings.shape[1] == EMBED_DIM, (
            f"expected {EMBED_DIM}-dim vectors, got {embeddings.shape[1]}"
        )
        assert len(meta) == embeddings.shape[0], "meta / embeddings length mismatch"
        idx = self._faiss.IndexFlatIP(EMBED_DIM)
        idx.add(embeddings.astype("float32"))
        self.index = idx
        self.meta = meta.reset_index(drop=True)

    def save(self, index_path: Path = FAISS_INDEX, meta_path: Path = FAISS_META) -> None:
        assert self.index is not None and self.meta is not None
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(index_path))
        self.meta.to_parquet(meta_path, index=False)

    def load(self, index_path: Path = FAISS_INDEX, meta_path: Path = FAISS_META) -> bool:
        if not index_path.exists() or not meta_path.exists():
            return False
        self.index = self._faiss.read_index(str(index_path))
        self.meta = pd.read_parquet(meta_path)
        return True

    # ---------- query ----------
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(row_id, similarity)] sorted by similarity desc."""
        assert self.index is not None, "index not loaded"
        q = query_vec.reshape(1, -1).astype("float32")
        k = min(k, self.index.ntotal)
        sims, ids = self.index.search(q, k)
        return list(zip(ids[0].tolist(), sims[0].tolist()))
