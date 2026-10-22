"""Smoke tests — no model downloads required.

The heavy tests (embedder/reranker) are skipped unless RUN_HEAVY=1
is set, so `pytest` stays fast on CI.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.core import parser, extractor, scorer
from app.core.baseline import TfidfBaseline


def test_parser_txt():
    out = parser.parse_bytes("r.txt", b"Hello, this is my resume with Python and Java skills.")
    assert "Python" in out


def test_extractor_skills_years_degree():
    text = "Software engineer with 5 years of experience. Skills: Python, Docker, AWS. B.Tech in CSE."
    ex = extractor.extract(text)
    assert "python" in ex.skills
    assert "docker" in ex.skills
    assert "aws" in ex.skills
    assert ex.years == 5
    assert ex.degree == "bachelors"


def test_degree_score_hierarchy():
    assert extractor.degree_score("masters", "bachelors") == 1.0
    assert extractor.degree_score("bachelors", "masters") < 1.0
    assert extractor.degree_score("bachelors", "none") == 1.0


def test_experience_score():
    assert extractor.experience_score(5, 3) == 1.0
    assert extractor.experience_score(1, 4) == pytest.approx(0.25)
    assert 0.5 <= extractor.experience_score(2, 0) <= 1.0


def test_scorer_combine():
    sb = scorer.combine(0.8, 0.6, 1.0, 1.0)
    assert 0 <= sb.final <= 100
    assert sb.final > 60


def test_baseline_ranks_matching_resume_higher():
    tf = TfidfBaseline()
    jd = "Looking for a Python and Django backend engineer"
    resumes = [
        "I love Django and Python and Flask, built REST APIs",
        "Frontend React developer, JavaScript, HTML/CSS",
    ]
    sims = tf.score(jd, resumes)
    assert sims[0] > sims[1]


@pytest.mark.skipif(os.getenv("RUN_HEAVY") != "1", reason="downloads MiniLM")
def test_pipeline_end_to_end():
    from app.core.pipeline import Candidate, run_match
    jd = "Senior ML engineer with Python and PyTorch, 3+ years exp, Master's"
    cands = [
        Candidate(file="a.txt", text="ML engineer with 5 years exp. Python, PyTorch, NLP. M.Sc CSE."),
        Candidate(file="b.txt", text="Frontend developer, React, Redux, JavaScript, 2 years."),
    ]
    matches = run_match(jd, cands, top_n=2)
    assert matches[0].candidate.file == "a.txt"
    assert matches[0].score.final > matches[1].score.final
