"""Tests for tools/mcp_job_search.py

Covers:
  - Normalisation of Arbeitnow raw records to common schema
  - Normalisation of RemoteOK raw records to common schema
  - Deduplication by title+company fingerprint (case-insensitive)
  - HTML stripping from descriptions
  - Graceful handling of missing or None fields
  - RemoteOK metadata-object (first element) is skipped
"""

from __future__ import annotations

import pytest

# Import the internal helpers directly for unit testing
from tools.mcp_job_search import (
    _normalise_arbeitnow,
    _normalise_remoteok,
    _deduplicate,
    _strip_html,
    _fingerprint,
)


# ---------------------------------------------------------------------------
# Arbeitnow normalisation
# ---------------------------------------------------------------------------

class TestNormaliseArbeitnow:
    def _raw(self, **kwargs):
        base = {
            "title": "Senior Python Engineer",
            "company_name": "Acme Corp",
            "location": "Berlin, Germany",
            "description": "<p>We need Python skills.</p>",
            "url": "https://arbeitnow.com/jobs/acme/123",
            "remote": True,
            "tags": ["Python", "Django"],
            "job_types": ["full-time"],
        }
        base.update(kwargs)
        return base

    def test_maps_title(self):
        result = _normalise_arbeitnow(self._raw())
        assert result["title"] == "Senior Python Engineer"

    def test_maps_company_name(self):
        result = _normalise_arbeitnow(self._raw())
        assert result["company"] == "Acme Corp"

    def test_maps_location(self):
        result = _normalise_arbeitnow(self._raw())
        assert result["location"] == "Berlin, Germany"

    def test_strips_html_from_description(self):
        result = _normalise_arbeitnow(self._raw(description="<p>We need <b>Python</b> skills.</p>"))
        assert "<p>" not in result["description"]
        assert "Python" in result["description"]

    def test_maps_url(self):
        result = _normalise_arbeitnow(self._raw())
        assert result["url"] == "https://arbeitnow.com/jobs/acme/123"

    def test_maps_remote_bool_true(self):
        result = _normalise_arbeitnow(self._raw(remote=True))
        assert result["remote"] is True

    def test_maps_remote_bool_false(self):
        result = _normalise_arbeitnow(self._raw(remote=False))
        assert result["remote"] is False

    def test_tags_lowercased(self):
        result = _normalise_arbeitnow(self._raw(tags=["Python", "DJANGO", "REST"]))
        assert "python" in result["tags"]
        assert "django" in result["tags"]

    def test_source_is_arbeitnow(self):
        result = _normalise_arbeitnow(self._raw())
        assert result["source"] == "Arbeitnow"

    def test_missing_title_returns_none(self):
        result = _normalise_arbeitnow(self._raw(title=""))
        assert result is None

    def test_missing_company_returns_none(self):
        result = _normalise_arbeitnow(self._raw(company_name=""))
        assert result is None

    def test_none_company_name_returns_none(self):
        raw = self._raw()
        raw["company_name"] = None
        result = _normalise_arbeitnow(raw)
        assert result is None

    def test_empty_tags_yields_empty_list(self):
        result = _normalise_arbeitnow(self._raw(tags=[]))
        assert result["tags"] == []

    def test_missing_location_defaults_to_remote(self):
        raw = self._raw()
        raw["location"] = None
        result = _normalise_arbeitnow(raw)
        assert result["location"] == "Remote"

    def test_result_has_all_schema_keys(self):
        result = _normalise_arbeitnow(self._raw())
        for key in ["title", "company", "location", "description", "url", "remote", "tags", "source"]:
            assert key in result


# ---------------------------------------------------------------------------
# RemoteOK normalisation
# ---------------------------------------------------------------------------

class TestNormaliseRemoteOK:
    def _raw(self, **kwargs):
        base = {
            "id": "123456",
            "position": "Full-Stack Developer",
            "company": "Remote Co",
            "location": "Worldwide",
            "description": "<p>Build <b>React</b> apps.</p>",
            "url": "https://remoteok.com/remote-jobs/123456",
            "apply_url": "https://remoteok.com/remote-jobs/123456",
            "tags": ["react", "node", "javascript"],
            "salary_min": 0,
            "salary_max": 0,
        }
        base.update(kwargs)
        return base

    def test_maps_position_to_title(self):
        result = _normalise_remoteok(self._raw())
        assert result["title"] == "Full-Stack Developer"

    def test_maps_company(self):
        result = _normalise_remoteok(self._raw())
        assert result["company"] == "Remote Co"

    def test_maps_location(self):
        result = _normalise_remoteok(self._raw())
        assert result["location"] == "Worldwide"

    def test_strips_html_from_description(self):
        result = _normalise_remoteok(self._raw())
        assert "<p>" not in result["description"]
        assert "React" in result["description"]

    def test_remote_is_always_true(self):
        result = _normalise_remoteok(self._raw())
        assert result["remote"] is True

    def test_source_is_remoteok(self):
        result = _normalise_remoteok(self._raw())
        assert result["source"] == "RemoteOK"

    def test_missing_position_returns_none(self):
        result = _normalise_remoteok(self._raw(position=""))
        assert result is None

    def test_none_position_returns_none(self):
        raw = self._raw()
        raw["position"] = None
        result = _normalise_remoteok(raw)
        assert result is None

    def test_missing_company_returns_none(self):
        result = _normalise_remoteok(self._raw(company=""))
        assert result is None

    def test_empty_location_defaults_to_remote(self):
        result = _normalise_remoteok(self._raw(location=""))
        assert result["location"] == "Remote"

    def test_none_location_defaults_to_remote(self):
        raw = self._raw()
        raw["location"] = None
        result = _normalise_remoteok(raw)
        assert result["location"] == "Remote"

    def test_tags_preserved(self):
        result = _normalise_remoteok(self._raw())
        assert "react" in result["tags"]
        assert "node" in result["tags"]

    def test_result_has_all_schema_keys(self):
        result = _normalise_remoteok(self._raw())
        for key in ["title", "company", "location", "description", "url", "remote", "tags", "source"]:
            assert key in result


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def _job(self, title, company, source="Test"):
        return {
            "title": title,
            "company": company,
            "location": "Remote",
            "description": "",
            "url": "https://example.com",
            "remote": True,
            "tags": [],
            "source": source,
        }

    def test_identical_jobs_deduplicated(self):
        jobs = [
            self._job("Python Engineer", "Acme", "Arbeitnow"),
            self._job("Python Engineer", "Acme", "RemoteOK"),
        ]
        result = _deduplicate(jobs)
        assert len(result) == 1

    def test_case_insensitive_dedup(self):
        jobs = [
            self._job("python engineer", "acme"),
            self._job("Python Engineer", "Acme"),
        ]
        result = _deduplicate(jobs)
        assert len(result) == 1

    def test_different_companies_not_deduped(self):
        jobs = [
            self._job("Python Engineer", "Acme"),
            self._job("Python Engineer", "BetaCorp"),
        ]
        result = _deduplicate(jobs)
        assert len(result) == 2

    def test_different_titles_not_deduped(self):
        jobs = [
            self._job("Python Engineer", "Acme"),
            self._job("Senior Python Engineer", "Acme"),
        ]
        result = _deduplicate(jobs)
        assert len(result) == 2

    def test_first_occurrence_preserved(self):
        j1 = self._job("Engineer", "Acme", "Arbeitnow")
        j2 = self._job("Engineer", "Acme", "RemoteOK")
        result = _deduplicate([j1, j2])
        assert result[0]["source"] == "Arbeitnow"

    def test_empty_list_returns_empty(self):
        assert _deduplicate([]) == []

    def test_single_item_list_unchanged(self):
        jobs = [self._job("Engineer", "Acme")]
        result = _deduplicate(jobs)
        assert len(result) == 1

    def test_all_unique_preserved(self):
        jobs = [
            self._job("Engineer", "Acme"),
            self._job("Designer", "BetaCorp"),
            self._job("Manager", "GammaCo"),
        ]
        result = _deduplicate(jobs)
        assert len(result) == 3

    def test_order_preserved(self):
        jobs = [
            self._job("Alpha", "Z Corp"),
            self._job("Beta", "A Corp"),
            self._job("Gamma", "M Corp"),
        ]
        result = _deduplicate(jobs)
        assert [r["title"] for r in result] == ["Alpha", "Beta", "Gamma"]


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_removes_basic_tags(self):
        result = _strip_html("<p>Hello world</p>")
        assert result == "Hello world"

    def test_removes_nested_tags(self):
        result = _strip_html("<div><p>Hello <b>world</b></p></div>")
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_decodes_html_entities(self):
        result = _strip_html("&lt;Python&gt; &amp; SQL")
        assert "<Python>" in result
        assert "&" in result

    def test_collapses_whitespace(self):
        result = _strip_html("<p>  Hello   world  </p>")
        assert "  " not in result

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_plain_text_unchanged(self):
        result = _strip_html("No HTML here")
        assert result == "No HTML here"

    def test_list_items_readable(self):
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = _strip_html(html)
        assert "Item 1" in result
        assert "Item 2" in result


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_same_inputs_same_hash(self):
        assert _fingerprint("Engineer", "Acme") == _fingerprint("Engineer", "Acme")

    def test_case_insensitive(self):
        assert _fingerprint("engineer", "acme") == _fingerprint("ENGINEER", "ACME")

    def test_different_title_different_hash(self):
        assert _fingerprint("Engineer", "Acme") != _fingerprint("Manager", "Acme")

    def test_different_company_different_hash(self):
        assert _fingerprint("Engineer", "Acme") != _fingerprint("Engineer", "BetaCorp")

    def test_returns_string(self):
        assert isinstance(_fingerprint("a", "b"), str)
