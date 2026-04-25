"""Agent 3 — Job Finder

Searches live job APIs, scores results against the resume profile, and
generates a short match-reason for each top job.

Pipeline:
  1. Call mcp_job_search.search_jobs() — Arbeitnow + RemoteOK, cached 5 min
  2. For each returned job (up to 30), build a job_profile and score it
     against resume_profile using scoring_v1.score_match() (same module as Agent 2)
  3. Sort by score, take the top 15
  4. Make ONE batched LLM call to generate a 2-line "why this matches" reason
     for all top 15 jobs simultaneously
  5. Return ranked list of {title, company, location, url, fit_score, reason}

Agent design rule: this file imports ONLY from llm.base, never from a
specific LLM SDK. The llm parameter is injected by the caller (app.py).
"""

from __future__ import annotations

import json
import re

from llm.base import LLMProvider
from scoring.scoring_v1 import score_match, build_job_profile
from tools.mcp_job_search import search_jobs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_FETCH = 30   # jobs fetched for scoring
_TOP_N = 15       # top jobs sent to LLM for reason generation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    llm: LLMProvider,
    resume_profile: dict,
    keyword: str = "",
    remote_only: bool = False,
    location: str = "",
) -> list[dict]:
    """Search, score and rank jobs against the resume profile.

    Parameters
    ----------
    llm:
        A concrete LLMProvider (injected by app.py).
    resume_profile:
        Structured resume profile from Agent 1.
    keyword:
        Search keyword / role title for the job APIs.
    remote_only:
        Filter to remote jobs only.
    location:
        Location filter (passed to Arbeitnow; RemoteOK is always remote).

    Returns
    -------
    List of up to 15 job dicts, sorted by fit_score descending:
        {title, company, location, url, fit_score, reason, source}
    """
    # Step 1: Fetch jobs (cached 5 min by mcp_job_search)
    jobs = search_jobs(
        query=keyword,
        remote_only=remote_only,
        location=location,
        max_results=_MAX_FETCH,
    )

    if not jobs:
        return []

    # Step 2: Score each job against the resume profile
    scored: list[dict] = []
    for job in jobs:
        job_profile = build_job_profile(job)
        score_result = score_match(resume_profile, job_profile)
        scored.append({
            **job,
            "fit_score": score_result["score"],
            "_matching": score_result["matching"],
        })

    # Step 3: Sort and take top N
    scored.sort(key=lambda j: j["fit_score"], reverse=True)
    top_jobs = scored[:_TOP_N]

    if not top_jobs:
        return []

    # Step 4: Single batched LLM call for reasons
    reasons = _generate_reasons(llm, resume_profile, top_jobs)

    # Step 5: Build output list
    results: list[dict] = []
    for i, job in enumerate(top_jobs):
        results.append({
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "url": job["url"],
            "fit_score": job["fit_score"],
            "reason": reasons.get(i, "Strong profile alignment with job requirements."),
            "source": job.get("source", ""),
            "remote": job.get("remote", False),
        })

    return results


# ---------------------------------------------------------------------------
# Batched reason generation (single LLM call for all top-N jobs)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a concise career advisor helping a job seeker understand "
    "why specific job postings match their background. "
    "Be specific — reference the candidate's actual skills and the job's requirements. "
    "Always respond with valid JSON only — no explanation, no markdown fences."
)

_REASONS_PROMPT_TEMPLATE = """Given this candidate's profile and the following job listings, write a 2-sentence "why this matches" reason for each job.

CANDIDATE PROFILE:
- Skills: {skills}
- Tools & Tech: {tools}
- Experience: {exp_years} years, {seniority} level
- Target Roles: {titles}

JOBS (indexed 0 to {last_idx}):
{jobs_text}

Return a single JSON object mapping each job index (as a string) to a 2-sentence reason string.
Each reason must:
1. Reference at least one specific skill or tool from the candidate's profile.
2. Explain what specifically about the job aligns with their background.
Keep each reason to 2 sentences maximum.

Example format:
{{
  "0": "Your Python and Django expertise directly matches this role's core stack. The senior-level position aligns with your 7 years of backend engineering experience.",
  "1": "...",
  ...
}}

Return ONLY the JSON object."""


def _generate_reasons(
    llm: LLMProvider,
    resume_profile: dict,
    top_jobs: list[dict],
) -> dict[int, str]:
    """Generate match reasons for all top jobs in a single LLM call."""
    # Build jobs text block
    jobs_lines: list[str] = []
    for i, job in enumerate(top_jobs):
        matching = ", ".join(job.get("_matching", [])[:5]) or "general overlap"
        jobs_lines.append(
            f"[{i}] {job['title']} at {job['company']} ({job['location']}) — "
            f"fit score: {round(job['fit_score'] * 100)}% — "
            f"matching terms: {matching}\n"
            f"    Description excerpt: {job.get('description', '')[:200]}"
        )

    prompt = _REASONS_PROMPT_TEMPLATE.format(
        skills=", ".join(resume_profile.get("skills", [])[:10]) or "general",
        tools=", ".join(resume_profile.get("tools_tech", [])[:8]) or "various",
        exp_years=resume_profile.get("experience_years", 0),
        seniority=resume_profile.get("seniority_level", "mid") or "mid",
        titles=", ".join(resume_profile.get("target_titles", [])[:3]) or "software professional",
        jobs_text="\n\n".join(jobs_lines),
        last_idx=len(top_jobs) - 1,
    )

    try:
        raw_response = llm.generate(prompt, system=_SYSTEM_PROMPT)
    except RuntimeError:
        raise

    return _parse_reasons_json(raw_response, len(top_jobs))


def _parse_reasons_json(raw: str, n_jobs: int) -> dict[int, str]:
    """Parse the batched LLM reasons response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: return generic reasons for all jobs
        return {i: "Strong alignment with your profile and target role." for i in range(n_jobs)}

    # Map string keys → int keys, with fallback
    reasons: dict[int, str] = {}
    for i in range(n_jobs):
        reason = data.get(str(i)) or data.get(i) or "Strong alignment with your profile."
        reasons[i] = str(reason).strip()

    return reasons
