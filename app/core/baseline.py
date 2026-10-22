"""TF-IDF + cosine baseline. Kept for the live model-comparison view —
this is exactly the 'Model 1' from the PPT."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfBaseline:
    def __init__(self, max_features: int = 20000):
        self.vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self._corpus_matrix = None

    def fit(self, corpus: list[str]) -> None:
        self._corpus_matrix = self.vec.fit_transform(corpus)

    def score(self, jd: str, resumes: list[str]) -> np.ndarray:
        """Score `resumes` against `jd`. Independent of any prior fit."""
        if not resumes:
            return np.zeros(0, dtype="float32")
        m = self.vec.fit_transform([jd] + resumes)
        sims = cosine_similarity(m[0:1], m[1:]).ravel()
        return sims.astype("float32")
