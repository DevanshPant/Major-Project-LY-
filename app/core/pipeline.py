"""End-to-end matching pipeline.

Given a JD and a set of resumes, run:
  1. extract structured fields (skills, YOE, degree) from JD
  2. embed JD + resumes with MiniLM
  3. cosine-similarity retrieve top-K (FAISS if a corpus was pre-indexed;
     otherwise plain matmul over the current batch)
  4. cross-encoder rerank the shortlist
  5. combine into a final score with explanation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import embedder, reranker, extractor as ex
from .scorer import combine, quality_label, ScoreBreakdown
from .baseline import TfidfBaseline
from .integrity import IntegrityReport
from .explain import build as build_explanation, Explanation


@dataclass
class Candidate:
    file: str
    text: str
    ext: "ex.Extracted" = field(default_factory=ex.Extracted)
    integrity: IntegrityReport = field(default_factory=IntegrityReport)

    def snippet(self, n: int = 240) -> str:
        return (self.text[:n] + "…") if len(self.text) > n else self.text


@dataclass
class Match:
    candidate: Candidate
    score: ScoreBreakdown
    matched_skills: list[str]
    missing_skills: list[str]
    quality: str
    rank: int = 0
    explanation: Explanation | None = None

    def to_row(self) -> dict:
        integ = self.candidate.integrity
        return {
            "rank": self.rank,
            "file": self.candidate.file,
            "final_score": round(self.score.final, 1),
            "semantic": round(self.score.semantic * 100, 1),
            "skills":   round(self.score.skills   * 100, 1),
            "experience": round(self.score.exp    * 100, 1),
            "education":  round(self.score.edu    * 100, 1),
            "quality":  self.quality,
            "years":    self.candidate.ext.years,
            "degree":   self.candidate.ext.degree,
            "matched_skills": ", ".join(self.matched_skills),
            "missing_skills": ", ".join(self.missing_skills),
            "snippet":  self.candidate.snippet(200),
            "suspicious":     integ.suspicious,
            "flags":          integ.reasons,
            "hidden_text":    integ.hidden_text,
            "visible_ratio":  integ.visible_ratio,
            "explanation":    self.explanation.to_dict() if self.explanation else None,
        }


def _anonymise(text: str) -> str:
    """Best-effort identity stripping: first line (often the name),
    email, phone. Keeps everything else. Used when bias-check is on."""
    import re
    t = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[email]", text)
    t = re.sub(r"(?:\+?\d[\s-]?){7,}\d", "[phone]", t)
    lines = t.splitlines()
    if lines:
        lines[0] = "[name]"
    return "\n".join(lines)


def _spread(v: np.ndarray) -> np.ndarray:
    """Min-max scale into 0..1 so relative ordering survives even when the
    model's absolute outputs are all clustered near one end."""
    if v.size == 0:
        return v
    lo = float(v.min()); hi = float(v.max())
    if hi - lo < 1e-6:
        return np.full_like(v, 0.5)
    return ((v - lo) / (hi - lo)).astype("float32")


def run_match(
    jd_text: str,
    candidates: list[Candidate],
    top_n: int = 10,
    use_reranker: bool = True,
    bias_check: bool = False,
    integrity_check: bool = True,
    weights: Optional[dict] = None,
) -> list[Match]:
    """Score every candidate against JD, return top-N sorted desc."""
    if not candidates:
        return []

    jd_ext = ex.extract(jd_text)
    jd_skills = jd_ext.skills
    jd_years = jd_ext.years
    jd_degree = jd_ext.degree

    # The curated lexicon is tech-heavy. For non-tech JDs (nursing, legal,
    # hospitality...) it finds almost nothing, which used to award every
    # candidate the same constant skills score. Fall back to salient JD
    # terms so the skills axis stays JD-dependent for any domain.
    jd_kw: set[str] = set()
    use_kw_fallback = len(jd_skills) < 2
    if use_kw_fallback:
        jd_kw = ex.jd_keywords(jd_text)

    # Fill in structured fields per candidate.
    for c in candidates:
        if not c.ext.skills and c.text:
            c.ext = ex.extract(c.text)

    texts = [_anonymise(c.text) if bias_check else c.text for c in candidates]

    # ---- semantic retrieval ----
    jd_vec = embedder.encode_one(jd_text)
    doc_vecs = embedder.encode(texts)
    semantic_sims = (doc_vecs @ jd_vec).astype("float32")   # already normalised

    # ---- rerank a shortlist ----
    # Cap at 30 regardless of top_n — cross-encoder is O(pairs) and each
    # pair costs ~200 ms on CPU. A wider shortlist rarely changes the
    # final top-N since bi-encoder recall is already >95 % at k=30.
    k_re = min(len(candidates), 30)
    order = np.argsort(-semantic_sims)[:k_re]
    if use_reranker:
        ce_scores_shortlist = reranker.score(jd_text, [texts[i] for i in order])
    else:
        ce_scores_shortlist = semantic_sims[order]

    # Assemble per-candidate semantic score.
    #
    # The cross-encoder emits logits that, after sigmoid, cluster near 0
    # for noisy OCR text — which flattened every candidate to the same
    # value. We rank-normalise WITHIN the batch instead: the relative
    # ordering the model produced is what matters, and min-max scaling
    # restores the spread that the weighted score needs.
    sem_final = semantic_sims.copy()
    for pos, idx in enumerate(order):
        sem_final[idx] = ce_scores_shortlist[pos]

    # Blend a little raw cosine back in so candidates outside the rerank
    # shortlist stay comparable to those inside it.
    sem_final = 0.85 * sem_final + 0.15 * np.clip(semantic_sims, 0.0, 1.0)
    sem_final = _spread(sem_final)

    # Which axes does this JD actually constrain? Unconstrained axes are
    # dropped and their weight redistributed, instead of handing every
    # candidate an identical constant.
    active = {
        "semantic": True,
        "skills": bool(jd_skills) or bool(jd_kw),
        "exp": jd_years > 0,
        "edu": jd_degree != "none",
    }

    matches: list[Match] = []
    for i, c in enumerate(candidates):
        if use_kw_fallback:
            s_score, matched, missing = ex.keyword_overlap(c.text, jd_kw)
        else:
            s_score, matched, missing = ex.skill_overlap(c.ext.skills, jd_skills)
        e_score = ex.experience_score(c.ext.years, jd_years) if active["exp"] else 0.0
        d_score = ex.degree_score(c.ext.degree, jd_degree) if active["edu"] else 0.0
        sb = combine(float(sem_final[i]), s_score, e_score, d_score,
                     weights, active=active)
        # Downgrade suspicious resumes (hidden text / stuffing) if the
        # integrity check is enabled AND the parser found something.
        if integrity_check and c.integrity.suspicious and c.integrity.score_penalty > 0:
            sb.final = max(0.0, sb.final - c.integrity.score_penalty * 100)
        matches.append(Match(
            candidate=c,
            score=sb,
            matched_skills=sorted(matched),
            missing_skills=sorted(missing)[:12],
            quality=quality_label(sb.final),
        ))

    matches.sort(key=lambda m: m.score.final, reverse=True)
    top = matches[:top_n]
    # Attach a natural-language explanation only for the shortlist we return —
    # cheaper than doing all N and only what the UI shows.
    for r, m in enumerate(top, start=1):
        m.rank = r
        m.explanation = build_explanation(
            rank=r,
            final_score=m.score.final,
            quality=m.quality,
            semantic_pct=m.score.semantic * 100,
            skills_pct=m.score.skills   * 100,
            exp_pct=m.score.exp        * 100,
            edu_pct=m.score.edu        * 100,
            matched_skills=m.matched_skills,
            missing_skills=m.missing_skills,
            resume_ext=m.candidate.ext,
            jd_ext=jd_ext,
            integrity_flags=m.candidate.integrity.reasons,
        )
    return top


def run_baseline(jd_text: str, candidates: list[Candidate], top_n: int) -> list[Match]:
    """TF-IDF only, no extraction weighting — the 'Model 1' comparison."""
    if not candidates:
        return []
    tf = TfidfBaseline()
    sims = tf.score(jd_text, [c.text for c in candidates])
    out: list[Match] = []
    for i, c in enumerate(candidates):
        sb = combine(float(sims[i]), 0.0, 0.0, 0.0, {"semantic": 1, "skills": 0, "exp": 0, "edu": 0})
        out.append(Match(candidate=c, score=sb, matched_skills=[], missing_skills=[],
                         quality=quality_label(sb.final)))
    out.sort(key=lambda m: m.score.final, reverse=True)
    for r, m in enumerate(out[:top_n], start=1):
        m.rank = r
    return out[:top_n]
