"""Tests for agents/matcher.py

Covers:
- Synonym reconciliation
- Experience assessment parsing
"""

import json
import pytest
from unittest.mock import MagicMock

from agents.matcher import run
from llm.base import LLMProvider

class DummyLLM(LLMProvider):
    def __init__(self, response_json: dict):
        self.response_json = response_json

    @property
    def provider_name(self) -> str:
        return "Dummy"

    def generate(self, prompt: str, system: str = None) -> str:
        return json.dumps(self.response_json)

    def test_connection(self):
        return True, "OK"

def test_matcher_synonym_reconciliation():
    resume_profile = {"skills": ["container orchestration"]}
    jd_profile = {"skills": [{"name": "kubernetes", "importance": "must-have"}]}

    # Mock the LLM to move kubernetes from missing to matching
    llm = DummyLLM({
        "matching_skills": ["container orchestration", "kubernetes"],
        "missing_skills": [],
        "experience_verdict": "Meets",
        "experience_justification": "Has relevant experience",
        "suggestions": ["Good"],
        "rewritten_bullets": ["Bullet"]
    })

    result = run(llm, resume_profile, jd_profile)
    
    assert "kubernetes" in result["matching_skills"]
    assert len(result["missing_skills"]) == 0
    assert result["experience_verdict"] == "Meets"

def test_matcher_preserves_missing_importance_tags():
    resume_profile = {"skills": []}
    jd_profile = {"skills": [{"name": "kubernetes", "importance": "must-have"}]}

    llm = DummyLLM({
        "matching_skills": [],
        "missing_skills": [{"name": "kubernetes", "importance": "must-have"}],
        "experience_verdict": "Below",
        "experience_justification": "No exp",
        "suggestions": [],
        "rewritten_bullets": []
    })

    result = run(llm, resume_profile, jd_profile)
    
    assert len(result["missing_skills"]) == 1
    assert result["missing_skills"][0]["name"] == "kubernetes"
    assert result["missing_skills"][0]["importance"] == "must-have"

def test_matcher_handles_malformed_json_gracefully():
    class BadLLM(LLMProvider):
        @property
        def provider_name(self) -> str:
            return "Bad"
            
        def generate(self, prompt: str, system: str = None) -> str:
            return "This is not JSON at all."
            
        def test_connection(self):
            return True, "OK"

    result = run(BadLLM(), {}, {})
    
    assert result["experience_verdict"] == "Unclear from JD"
    assert "could not parse" in result["suggestions"][0]
