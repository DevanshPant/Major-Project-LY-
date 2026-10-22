"""Final weighted score. If a trained weights file exists it is used;
otherwise the sensible defaults from config.DEFAULT_WEIGHTS apply."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config import DEFAULT_WEIGHTS, SCORER_WEIGHTS


def _load_weights() -> dict:
    if SCORER_WEIGHTS.exists():
        try:
            w = json.loads(SCORER_WEIGHTS.read_text())
            # sanity: keys and non-negative
            keys = {"semantic", "skills", "exp", "edu"}
            if set(w) == keys and all(v >= 0 for v in w.values()):
                total = sum(w.values()) or 1.0
                return {k: v / total for k, v in w.items()}
        except (json.JSONDecodeError, ValueError):
            pass
    return dict(DEFAULT_WEIGHTS)


@dataclass
class ScoreBreakdown:
    semantic: float = 0.0   # 0..1
    skills: float = 0.0     # 0..1
    exp: float = 0.0        # 0..1
    edu: float = 0.0        # 0..1
    final: float = 0.0      # 0..100
    weights: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "semantic": round(self.semantic * 100, 1),
            "skills":   round(self.skills * 100, 1),
            "exp":      round(self.exp * 100, 1),
            "edu":      round(self.edu * 100, 1),
            "final":    round(self.final, 1),
            "weights":  self.weights,
        }


def combine(semantic: float, skills: float, exp: float, edu: float,
            weights: dict | None = None,
            active: dict | None = None) -> ScoreBreakdown:
    """Combine sub-scores into a final 0-100 score.

    `active` marks which components the JD actually constrains, e.g.
    {"semantic": True, "skills": True, "exp": False, "edu": False} when
    the JD states no year or degree requirement. Inactive components are
    dropped and their weight is redistributed over the active ones.

    Without this, an unstated requirement awards every candidate the same
    constant, which flattens the ranking and makes different JDs return
    identical shortlists.
    """
    w = dict(weights or _load_weights())
    if active:
        live = {k: v for k, v in w.items() if active.get(k, True)}
        total = sum(live.values())
        if total > 0:
            w = {k: (live.get(k, 0.0) / total) for k in w}
        # if nothing is active, fall back to semantic only
        if total <= 0:
            w = {"semantic": 1.0, "skills": 0.0, "exp": 0.0, "edu": 0.0}
    final = (
        w["semantic"] * semantic +
        w["skills"]   * skills +
        w["exp"]      * exp +
        w["edu"]      * edu
    ) * 100.0
    return ScoreBreakdown(
        semantic=semantic, skills=skills, exp=exp, edu=edu,
        final=final, weights=w,
    )


def quality_label(final_score: float) -> str:
    if final_score >= 80: return "Excellent"
    if final_score >= 65: return "Strong"
    if final_score >= 50: return "Good"
    if final_score >= 35: return "Moderate"
    return "Weak"
