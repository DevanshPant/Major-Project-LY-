"""Turn each ranking into a plain-English "why chosen" explanation.

Nothing is fabricated — every sentence maps to a real number produced
by the pipeline (sub-scores, matched skills, YOE, degree). If the
integrity check fired, that becomes a caveat, not a hidden factor.
"""
from __future__ import annotations

from dataclasses import dataclass

from .extractor import Extracted


@dataclass
class Explanation:
    headline: str            # one-line summary shown at the top
    strengths: list[str]     # bullets
    concerns:  list[str]     # bullets

    def to_dict(self) -> dict:
        return {
            "headline":  self.headline,
            "strengths": self.strengths,
            "concerns":  self.concerns,
        }


def _label(pct: float) -> str:
    if pct >= 80: return "strong"
    if pct >= 60: return "solid"
    if pct >= 40: return "moderate"
    if pct >= 20: return "weak"
    return "very weak"


def build(
    *,
    rank: int,
    final_score: float,
    quality: str,
    semantic_pct: float,     # 0-100
    skills_pct: float,       # 0-100
    exp_pct: float,          # 0-100
    edu_pct: float,          # 0-100
    matched_skills: list[str],
    missing_skills: list[str],
    resume_ext: Extracted,
    jd_ext: Extracted,
    integrity_flags: list[str],
) -> Explanation:
    """Assemble a per-candidate explanation from real numbers."""
    strengths: list[str] = []
    concerns:  list[str] = []

    # ---- semantic ----
    strengths.append(
        f"{_label(semantic_pct).capitalize()} semantic fit with the JD "
        f"({semantic_pct:.0f}/100) — the cross-encoder read this resume "
        "and the JD together and rated their overall relevance."
    ) if semantic_pct >= 60 else concerns.append(
        f"Only {_label(semantic_pct)} semantic fit ({semantic_pct:.0f}/100) "
        "— the resume's overall narrative isn't a close topical match."
    )

    # ---- skills ----
    if matched_skills:
        top = matched_skills[:6]
        strengths.append(
            f"Matches {len(matched_skills)} of the {len(matched_skills)+len(missing_skills)} "
            f"skills in the JD, including {', '.join(top)}"
            + (f" and {len(matched_skills)-len(top)} more." if len(matched_skills) > len(top) else ".")
        )
    if missing_skills:
        head = missing_skills[:6]
        concerns.append(
            f"Missing {len(missing_skills)} JD skills"
            + (": " + ", ".join(head) + ("…" if len(missing_skills) > len(head) else "."))
        )

    # ---- experience ----
    if jd_ext.years > 0:
        if resume_ext.years >= jd_ext.years:
            strengths.append(
                f"{resume_ext.years} years of experience — meets the "
                f"{jd_ext.years}+ year JD requirement."
            )
        else:
            gap = jd_ext.years - resume_ext.years
            concerns.append(
                f"{resume_ext.years} years of experience — {gap} year"
                f"{'s' if gap != 1 else ''} short of the {jd_ext.years}+ year requirement."
            )
    else:
        if resume_ext.years >= 3:
            strengths.append(f"{resume_ext.years} years of experience.")
        elif resume_ext.years > 0:
            strengths.append(f"{resume_ext.years} year(s) of experience.")

    # ---- education ----
    if jd_ext.degree != "none":
        if edu_pct >= 100:
            strengths.append(
                f"Holds a {resume_ext.degree} — meets or exceeds the "
                f"JD's {jd_ext.degree} requirement."
            )
        else:
            concerns.append(
                f"Highest degree found is {resume_ext.degree}, JD asks for {jd_ext.degree}."
            )

    # ---- integrity ----
    if integrity_flags:
        concerns.append(
            "Resume-integrity check flagged this candidate — "
            + integrity_flags[0]
            + ("; the score has been penalised accordingly." if len(integrity_flags) == 1
               else f"; +{len(integrity_flags)-1} more issue(s).")
        )

    # ---- headline ----
    if rank == 1:
        headline = (
            f"Top pick ({final_score:.0f}/100, {quality}). "
            + _headline_reason(semantic_pct, skills_pct, exp_pct, edu_pct)
        )
    else:
        headline = (
            f"Ranked #{rank} ({final_score:.0f}/100, {quality}). "
            + _headline_reason(semantic_pct, skills_pct, exp_pct, edu_pct)
        )

    return Explanation(headline=headline, strengths=strengths, concerns=concerns)


def _headline_reason(sem: float, sk: float, ex: float, ed: float) -> str:
    """Pick the single biggest driver of the score."""
    parts = [
        ("semantic match", sem),
        ("skills coverage", sk),
        ("experience fit", ex),
        ("education fit", ed),
    ]
    parts.sort(key=lambda p: -p[1])
    top_name, top_val = parts[0]
    if top_val < 30:
        return "No sub-score is strong — surfaced because everything else was weaker."
    return f"Driven mainly by {top_name} ({top_val:.0f}/100)."
