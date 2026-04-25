"""MCP Tool: search_jobs

Queries Arbeitnow and RemoteOK APIs, merges results, deduplicates,
and normalises them into a common schema.

Confirmed field names from live API probes (2026-08-22):

  Arbeitnow  (https://arbeitnow.com/api/job-board-api):
    title, company_name, location, description, url, remote, tags

  RemoteOK   (https://remoteok.com/api):
    position, company, location, description (HTML), url, tags
    First element of the array is a metadata object (not a job) — skip it.

Common schema output:
  {title, company, location, description, url, remote, tags, source}
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARBEITNOW_URL = "https://arbeitnow.com/api/job-board-api"
_REMOTEOK_URL = "https://remoteok.com/api"
_USER_AGENT = "ResumeMatchApp/1.0 (job-search-tool; github.com/resume-jd-matcher)"
_TIMEOUT = 10  # seconds per API call
_MAX_PAGES_ARBEITNOW = 3  # ≤75 jobs per run (25/page)


# ---------------------------------------------------------------------------
# Public MCP tool entrypoint
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)  # cache for 5 minutes to protect free APIs
def search_jobs(
    query: str = "",
    remote_only: bool = False,
    location: str = "",
    max_results: int = 30,
) -> list[dict]:
    """Fetch, merge, deduplicate and normalise jobs from Arbeitnow + RemoteOK.

    Parameters
    ----------
    query:
        Keyword / role to search for (passed as ``q`` to Arbeitnow).
    remote_only:
        If True, filter to remote jobs only.
    location:
        Free-text location filter (Arbeitnow ``location`` param).
    max_results:
        Cap total results before scoring (default 30 for LLM cost control).

    Returns
    -------
    List of normalised job dicts with keys:
        title, company, location, description, url, remote, tags, source
    """
    jobs: list[dict] = []

    arbeitnow_jobs = _fetch_arbeitnow(query=query, remote_only=remote_only, location=location)
    jobs.extend(arbeitnow_jobs)

    remoteok_jobs = _fetch_remoteok(query=query)
    jobs.extend(remoteok_jobs)

    # Deduplicate by title+company fingerprint
    jobs = _deduplicate(jobs)

    # Apply remote filter post-fetch (RemoteOK is always remote)
    if remote_only:
        jobs = [j for j in jobs if j.get("remote", False)]

    # Apply location filter (case-insensitive substring)
    if location.strip():
        loc_lower = location.strip().lower()
        jobs = [
            j for j in jobs
            if loc_lower in (j.get("location") or "").lower()
            or j.get("remote", False)  # remote jobs match any location
        ]

    return jobs[:max_results]


# ---------------------------------------------------------------------------
# Arbeitnow fetcher
# ---------------------------------------------------------------------------

def _fetch_arbeitnow(
    query: str = "",
    remote_only: bool = False,
    location: str = "",
) -> list[dict]:
    """Fetch up to _MAX_PAGES_ARBEITNOW pages from Arbeitnow."""
    all_jobs: list[dict] = []
    headers = {"User-Agent": _USER_AGENT}

    for page in range(1, _MAX_PAGES_ARBEITNOW + 1):
        params: dict = {"page": page}
        if query.strip():
            params["q"] = query.strip()
        if remote_only:
            params["remote"] = "true"
        if location.strip():
            params["location"] = location.strip()

        try:
            resp = requests.get(
                _ARBEITNOW_URL, params=params, headers=headers, timeout=_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            raise RuntimeError("Arbeitnow API timed out — the service may be slow. Try again shortly.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Could not connect to Arbeitnow API. Check your internet connection.")
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Arbeitnow API returned an error: {exc.response.status_code}") from exc
        except Exception as exc:
            raise RuntimeError(f"Arbeitnow API unexpected error: {exc}") from exc

        jobs_on_page = data.get("data", [])
        if not jobs_on_page:
            break  # No more results

        for raw in jobs_on_page:
            normalised = _normalise_arbeitnow(raw)
            if normalised:
                all_jobs.append(normalised)

        # Respect pagination metadata
        links = data.get("links", {})
        if not links.get("next"):
            break  # No next page

    return all_jobs


def _normalise_arbeitnow(raw: dict) -> dict | None:
    """Map an Arbeitnow job record to the common schema."""
    title = (raw.get("title") or "").strip()
    company = (raw.get("company_name") or "").strip()
    if not title or not company:
        return None

    return {
        "title": title,
        "company": company,
        "location": (raw.get("location") or "Remote").strip(),
        "description": _strip_html(raw.get("description") or ""),
        "url": (raw.get("url") or "").strip(),
        "remote": bool(raw.get("remote", False)),
        "tags": [t.lower() for t in (raw.get("tags") or [])],
        "source": "Arbeitnow",
    }


# ---------------------------------------------------------------------------
# RemoteOK fetcher
# ---------------------------------------------------------------------------

def _fetch_remoteok(query: str = "") -> list[dict]:
    """Fetch jobs from RemoteOK API.

    The API returns an array where the first element is a metadata/legal
    object (not a job) — we skip it. A proper User-Agent is required.
    """
    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.get(_REMOTEOK_URL, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        raw_list = resp.json()
    except requests.exceptions.Timeout:
        raise RuntimeError("RemoteOK API timed out — the service may be slow. Try again shortly.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to RemoteOK API. Check your internet connection.")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"RemoteOK API returned an error: {exc.response.status_code}") from exc
    except Exception as exc:
        raise RuntimeError(f"RemoteOK API unexpected error: {exc}") from exc

    # First element is a metadata object — skip items without 'id'
    jobs: list[dict] = []
    query_lower = query.lower().strip()

    for raw in raw_list:
        if not isinstance(raw, dict) or "id" not in raw:
            continue
        normalised = _normalise_remoteok(raw)
        if not normalised:
            continue
        # Client-side keyword filter (RemoteOK has no server-side search)
        if query_lower:
            searchable = (
                f"{normalised['title']} {normalised['company']} "
                f"{' '.join(normalised['tags'])} {normalised['description'][:500]}"
            ).lower()
            if query_lower not in searchable:
                continue
        jobs.append(normalised)

    return jobs


def _normalise_remoteok(raw: dict) -> dict | None:
    """Map a RemoteOK job record to the common schema."""
    title = (raw.get("position") or "").strip()
    company = (raw.get("company") or "").strip()
    if not title or not company:
        return None

    return {
        "title": title,
        "company": company,
        "location": (raw.get("location") or "Remote").strip() or "Remote",
        "description": _strip_html(raw.get("description") or ""),
        "url": (raw.get("url") or raw.get("apply_url") or "").strip(),
        "remote": True,  # RemoteOK is always remote-first
        "tags": [t.lower() for t in (raw.get("tags") or [])],
        "source": "RemoteOK",
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(jobs: list[dict]) -> list[dict]:
    """Remove duplicate jobs by title+company fingerprint (case-insensitive)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for job in jobs:
        key = _fingerprint(job.get("title", ""), job.get("company", ""))
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def _fingerprint(title: str, company: str) -> str:
    """Create a stable dedup key from title + company."""
    normalised = f"{title.lower().strip()}|{company.lower().strip()}"
    # Remove common suffixes that don't differentiate roles
    normalised = re.sub(r"\s*(–|—|-)\s*.*$", "", normalised)
    return hashlib.md5(normalised.encode()).hexdigest()


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(raw: str) -> str:
    """Remove HTML tags and decode entities; collapse whitespace."""
    text = _HTML_TAG_RE.sub(" ", raw)
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
