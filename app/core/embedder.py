"""Bi-encoder wrapper. Lazy-loaded singleton so the model downloads
once and is reused across requests."""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from ..config import BI_ENCODER


_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model():
    # Import inside the function so `import app` doesn't force torch on
    # machines that only want to inspect the code.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BI_ENCODER)


# MiniLM-L6-v2 has a 256-token context window; anything past ~2,500
# characters is silently dropped inside encode(). Truncating up-front
# saves tokenisation time on long resumes.
_MAX_ENCODE_CHARS = 2500


def encode(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return L2-normalised embeddings (float32, shape [N, 384])."""
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    trimmed = [t[:_MAX_ENCODE_CHARS] if len(t) > _MAX_ENCODE_CHARS else t for t in texts]
    with _lock:
        m = _model()
        vecs = m.encode(
            trimmed,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    return vecs.astype("float32")


def encode_one(text: str) -> np.ndarray:
    return encode([text])[0]
