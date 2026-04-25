"""Tests for scoring/scoring_v1.py (v1.1)

Covers:
  - Perfect overlap (score = 100)
  - Zero overlap (score = 0)
  - Importance weighting (must-have vs nice-to-have)
  - Case insensitivity
  - Empty fields
  - Score stays in 0-100 bounds
  - output schema formatting
"""

from __future__ import annotations

import pytest

from scoring.scoring_v1 import score_match, build_job_profile, _normalise_list


# ---------------------------------------------------------------------------
# Basic score_match tests
# ---------------------------------------------------------------------------

class TestScoreMatch:
    def _resume(self, skills=None, tools_tech=None, keywords=None):
        return {
            "skills": skills or [],
            "tools_tech": tools_tech or [],
            "keywords": keywords or [],
        }

    def _jd(self, skills=None, tools_tech=None, keywords=None):
        return {
            "skills": skills or [],
            "tools_tech": tools_tech or [],
            "keywords": keywords or [],
        }

    # --- Perfect overlap ---

    def test_perfect_overlap_score_is_100(self):
        r = self._resume(["Python"], ["Docker"], ["backend"])
        # JD with flat strings -> treated as must-have
        j = self._jd(["Python"], ["Docker"], ["backend"])
        result = score_match(r, j)
        assert result["score"] == 100

    def test_perfect_overlap_matching_is_all_terms(self):
        r = self._resume(["python", "sql"], ["docker"], ["backend"])
        j = self._jd(["python", "sql"], ["docker"], ["backend"])
        result = score_match(r, j)
        assert "python" in result["matching"]
        assert "sql" in result["matching"]
        assert result["missing"] == []

    # --- Zero overlap ---

    def test_zero_overlap_score_is_0(self):
        a = self._resume(["Python"], ["Docker"], ["backend"])
        b = self._jd(["Java"], ["Kubernetes"], ["frontend"])
        result = score_match(a, b)
        assert result["score"] == 0

    def test_zero_overlap_missing_contains_all_b_terms_with_importance(self):
        a = self._resume(["Python"])
        # Provide tagged dicts for JD
        b = self._jd(skills=[
            {"name": "Java", "importance": "must-have"},
            {"name": "Scala", "importance": "nice-to-have"}
        ])
        result = score_match(a, b)
        assert {"name": "java", "importance": "must-have"} in result["missing"]
        assert {"name": "scala", "importance": "nice-to-have"} in result["missing"]

    # --- Importance weighting ---

    def test_missing_must_have_penalises_heavily(self):
        # 1 nice-to-have matched, 1 must-have missing
        r = self._resume(skills=["Python"])
        j = self._jd(skills=[
            {"name": "Java", "importance": "must-have"},
            {"name": "Python", "importance": "nice-to-have"}
        ])
        result = score_match(r, j)
        # must_ratio = 0/1, nice_ratio = 1/1
        # score = 70*0 + 30*1 = 30
        assert result["score"] == 30

    def test_missing_nice_to_have_penalises_lightly(self):
        # 1 must-have matched, 1 nice-to-have missing
        r = self._resume(skills=["Java"])
        j = self._jd(skills=[
            {"name": "Java", "importance": "must-have"},
            {"name": "Python", "importance": "nice-to-have"}
        ])
        result = score_match(r, j)
        # must_ratio = 1/1, nice_ratio = 0/1
        # score = 70*1 + 30*0 = 70
        assert result["score"] == 70
        
    def test_flat_strings_treated_as_must_have(self):
        r = self._resume(skills=["Python"])
        j = self._jd(skills=["Java", "Python"]) # flat strings
        result = score_match(r, j)
        # must_ratio = 1/2, nice_ratio = 1.0 (empty)
        # score = 70*0.5 + 30*1.0 = 35 + 30 = 65
        assert result["score"] == 65
        assert {"name": "java", "importance": "must-have"} in result["missing"]

    # --- Case insensitivity ---

    def test_matching_is_case_insensitive(self):
        a = self._resume(["PYTHON"])
        b = self._jd(["python"])
        result = score_match(a, b)
        assert result["score"] == 100

    # --- Empty fields ---

    def test_empty_resume_gives_zero_score_if_jd_has_both_types(self):
        a = self._resume()
        b = self._jd(
            skills=[{"name": "Java", "importance": "must-have"}],
            keywords=["backend"] # keywords are nice-to-have
        )
        result = score_match(a, b)
        assert result["score"] == 0

    def test_empty_jd_gives_full_score(self):
        """If JD requires nothing, resume matches everything."""
        a = self._resume(["Python"])
        b = self._jd()
        result = score_match(a, b)
        assert result["score"] == 100
        assert "python" in result["extra"]

    def test_both_empty_gives_full_score(self):
        result = score_match(self._resume(), self._jd())
        assert result["score"] == 100

    # --- Output structure ---

    def test_result_has_required_keys(self):
        result = score_match(self._resume(), self._jd())
        assert "score" in result
        assert "matching" in result
        assert "missing" in result
        assert "extra" in result
        assert "breakdown" in result

    def test_breakdown_has_all_keys(self):
        result = score_match(self._resume(), self._jd())
        for k in ["must_have_score", "nice_to_have_score", "must_have_matched", 
                  "must_have_total", "nice_to_have_matched", "nice_to_have_total"]:
            assert k in result["breakdown"]


# ---------------------------------------------------------------------------
# _normalise_list helper tests
# ---------------------------------------------------------------------------

class TestNormaliseList:
    def test_lowercases_all_items(self):
        result = _normalise_list(["Python", "SQL", "DOCKER"])
        assert result == {"python", "sql", "docker"}

    def test_strips_whitespace(self):
        result = _normalise_list(["  python  ", "sql"])
        assert "python" in result

    def test_deduplicates(self):
        result = _normalise_list(["python", "Python", "PYTHON"])
        assert result == {"python"}


# ---------------------------------------------------------------------------
# build_job_profile tests (Agent 3 compat)
# ---------------------------------------------------------------------------

class TestBuildJobProfile:
    def _job(self, title="Engineer", company="Acme", tags=None, description=""):
        return {
            "title": title,
            "company": company,
            "location": "Remote",
            "tags": tags or [],
            "description": description,
            "url": "https://example.com",
            "remote": True,
            "source": "Test",
        }

    def test_tags_become_skills_and_keywords(self):
        job = self._job(tags=["python", "docker", "backend"])
        profile = build_job_profile(job)
        assert "python" in profile["skills"]
        assert "docker" in profile["skills"]
        assert "python" in profile["keywords"]

    def test_profile_is_scoreable_with_score_match(self):
        job = self._job(tags=["python", "machine learning"])
        resume = {
            "skills": ["Python", "Machine Learning"],
            "tools_tech": [],
            "keywords": ["python"],
        }
        profile = build_job_profile(job)
        result = score_match(resume, profile)
        assert result["score"] == 100
        assert "python" in result["matching"]
