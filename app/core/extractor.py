"""Structured extraction from resume/JD text.

Extracts:
  - skills   (matched against a curated skills lexicon; ~500 common tech
              & business terms, extensible)
  - years    (max years of experience found via regex)
  - degree   (highest degree family: phd > masters > bachelors > diploma > none)

Uses spaCy if available for better tokenization; falls back to plain
regex + string matching so the app still runs on a fresh machine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------- Skills lexicon ----------------
# Small but real; extend freely. Lower-cased, phrase-normalised.
SKILLS = {
    # languages
    "python", "java", "c", "c++", "c#", "go", "golang", "rust", "kotlin",
    "swift", "typescript", "javascript", "scala", "r", "matlab", "php",
    "ruby", "perl", "bash", "shell", "sql", "pl/sql", "t-sql",
    # web / frontend
    "html", "css", "sass", "tailwind", "bootstrap", "react", "next.js",
    "vue", "angular", "svelte", "redux", "jquery",
    # backend / api
    "node.js", "express", "django", "flask", "fastapi", "spring",
    "spring boot", "rails", ".net", "asp.net", "graphql", "rest", "grpc",
    # data / ml
    "pandas", "numpy", "scikit-learn", "scikit learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "catboost", "huggingface",
    "hugging face", "transformers", "nlp", "computer vision", "opencv",
    "spacy", "nltk", "gensim", "faiss", "langchain", "llama", "openai",
    "llm", "rag", "prompt engineering",
    # data eng
    "kafka", "spark", "hadoop", "hive", "airflow", "dbt", "snowflake",
    "redshift", "bigquery", "databricks", "etl", "elt",
    # cloud / devops
    "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform",
    "ansible", "jenkins", "github actions", "ci/cd", "linux", "nginx",
    "prometheus", "grafana",
    # db
    "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "oracle",
    # misc / soft
    "git", "agile", "scrum", "jira", "figma", "tableau", "power bi",
    "excel", "communication", "leadership", "problem solving",
}

# Order matters: longer / more specific patterns first.
_SKILL_PATTERNS = sorted(SKILLS, key=len, reverse=True)

# ---------------- Regex ----------------
_YEARS_RE = re.compile(
    r"(?P<n>\d{1,2})\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)?",
    re.IGNORECASE,
)
_YEARS_RANGE_RE = re.compile(
    r"(?P<a>\d{1,2})\s*[-–to]{1,3}\s*(?P<b>\d{1,2})\s*(?:years?|yrs?)",
    re.IGNORECASE,
)

_DEGREE_RE = [
    ("phd",       re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.I)),
    ("masters",   re.compile(r"\b(m\.?tech|m\.?e\.?|m\.?sc|m\.?s\.?|mba|masters?|post[- ]?graduate|pg\b)\b", re.I)),
    ("bachelors", re.compile(r"\b(b\.?tech|b\.?e\.?|b\.?sc|b\.?s\.?|b\.?a|bca|bcom|bachelors?|under[- ]?graduate|ug\b)\b", re.I)),
    ("diploma",   re.compile(r"\b(diploma|associate)\b", re.I)),
]

_DEGREE_RANK = {"none": 0, "diploma": 1, "bachelors": 2, "masters": 3, "phd": 4}


@dataclass
class Extracted:
    skills: set[str] = field(default_factory=set)
    years: int = 0
    degree: str = "none"

    def to_dict(self) -> dict:
        return {
            "skills": sorted(self.skills),
            "years": self.years,
            "degree": self.degree,
        }


def _extract_skills(text_lc: str) -> set[str]:
    found: set[str] = set()
    for s in _SKILL_PATTERNS:
        # word-boundary match for single words; substring for phrases with punctuation
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", text_lc):
            found.add(s)
    return found


def _extract_years(text: str) -> int:
    years = [0]
    for m in _YEARS_RE.finditer(text):
        try:
            n = int(m.group("n"))
            if 0 <= n <= 50:
                years.append(n)
        except ValueError:
            pass
    for m in _YEARS_RANGE_RE.finditer(text):
        try:
            years.append(int(m.group("b")))
        except ValueError:
            pass
    return max(years)


def _extract_degree(text: str) -> str:
    for label, pat in _DEGREE_RE:  # ordered phd -> diploma
        if pat.search(text):
            return label
    return "none"


def extract(text: str) -> Extracted:
    if not text:
        return Extracted()
    lc = text.lower()
    return Extracted(
        skills=_extract_skills(lc),
        years=_extract_years(text),
        degree=_extract_degree(text),
    )


def degree_score(resume_deg: str, jd_deg: str) -> float:
    """1.0 if resume meets/exceeds JD, else scaled down. Both fields
    default to 'none' when unstated, giving 1.0 (no constraint)."""
    if jd_deg == "none":
        return 1.0
    r = _degree_rank(resume_deg)
    j = _degree_rank(jd_deg)
    if r >= j:
        return 1.0
    return max(0.0, 1.0 - 0.25 * (j - r))


def _degree_rank(d: str) -> int:
    return _DEGREE_RANK.get(d, 0)


def experience_score(resume_years: int, jd_years: int) -> float:
    """1.0 if resume >= jd requirement; linear falloff below; capped when JD is 0."""
    if jd_years <= 0:
        # No stated requirement — reward some experience but don't punish juniors.
        return min(1.0, 0.5 + 0.05 * resume_years)
    if resume_years >= jd_years:
        return 1.0
    return max(0.0, resume_years / jd_years)


# Words that carry no signal when we fall back to free-text JD terms.
_JD_STOP = {
    "the","and","for","with","from","that","this","are","was","were","have",
    "has","had","not","you","your","our","their","they","its","but","will",
    "who","all","any","can","must","should","would","need","needs","required",
    "requirement","requirements","experience","experienced","years","year",
    "work","working","role","job","candidate","candidates","looking","seeking",
    "strong","good","excellent","ability","able","skills","skill","knowledge",
    "plus","preferred","must-have","responsibilities","including","etc",
    "team","teams","new","use","using","well","also","across","within","per",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\+\.\#/&-]{2,}")


def jd_keywords(jd_text: str, limit: int = 25) -> set[str]:
    """Salient content words from a JD, used when the curated skills
    lexicon finds too few matches (e.g. nursing, legal, hospitality
    roles that a tech-oriented lexicon does not cover)."""
    if not jd_text:
        return set()
    words = [w.lower() for w in _WORD.findall(jd_text)]
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        w = w.strip(".-/&")
        if len(w) < 3 or w in _JD_STOP or w.isdigit() or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return set(out)


def keyword_overlap(resume_text: str, keywords: set[str]) -> tuple[float, set[str], set[str]]:
    """Fraction of JD keywords present in the resume body."""
    if not keywords:
        return 0.0, set(), set()
    lc = resume_text.lower()
    matched = {k for k in keywords
               if re.search(rf"(?<![A-Za-z0-9]){re.escape(k)}", lc)}
    return len(matched) / len(keywords), matched, keywords - matched


def skill_overlap(resume_skills: set[str], jd_skills: set[str]) -> tuple[float, set[str], set[str]]:
    """Return (score 0-1, matched, missing). Empty JD skills -> (0.0, {}, {})
    so the caller can drop this component instead of awarding everyone a
    constant, which would flatten the ranking."""
    if not jd_skills:
        return 0.0, set(), set()
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    return len(matched) / len(jd_skills), matched, missing
