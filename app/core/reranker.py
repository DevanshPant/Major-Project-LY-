"""Cross-encoder reranker. Prefers a fine-tuned checkpoint under
models/cross_encoder_finetuned/ if present, else falls back to the
pretrained ms-marco checkpoint."""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..config import CROSS_ENCODER, FINETUNED_CE_DIR


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    path = str(FINETUNED_CE_DIR) if FINETUNED_CE_DIR.exists() else CROSS_ENCODER
    return CrossEncoder(path)


def is_finetuned() -> bool:
    return FINETUNED_CE_DIR.exists()


def score(jd: str, resumes: Sequence[str]) -> np.ndarray:
    """Return relevance scores; higher = better. Sigmoid-normalised to (0, 1)."""
    if not resumes:
        return np.zeros(0, dtype="float32")
    with _lock:
        m = _model()
        pairs = [[jd, r] for r in resumes]
        raw = np.asarray(m.predict(pairs, show_progress_bar=False), dtype="float32")
    # ms-marco cross-encoders emit logits; sigmoid gives a clean 0-1 relevance.
    return 1.0 / (1.0 + np.exp(-raw))
