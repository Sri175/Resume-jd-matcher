"""Agent 2 — Matcher / Advisor

Compares a resume profile against a JD profile and produces:
  1. A numeric match score out of 100 (from scoring_v1.py — no LLM call)
  2. Matching skills — in both resume and JD (post-synonym reconciliation)
  3. Missing skills — in JD but absent from resume (post-synonym reconciliation)
  4. Extra skills — in resume but NOT required by JD
  5. Experience match — LLM reasoned verdict + justification
  6. Actionable suggestions (LLM call)
  7. 2-3 rewritten resume bullet points (LLM call)

Agent design rule: this file imports ONLY from llm.base, never from a
specific LLM SDK. The llm parameter is injected by the caller (app.py).
"""

from __future__ import annotations

import json
import re

from llm.base import LLMProvider
from scoring.scoring_v1 import score_match


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    llm: LLMProvider,
    resume_profile: dict,
    jd_profile: dict,
) -> dict:
    """Compare resume_profile against jd_profile and produce match advice.

    Parameters
    ----------
    llm:
        A concrete LLMProvider (injected by app.py).
    resume_profile:
        Structured profile extracted by Agent 1.
    jd_profile:
        Structured JD profile extracted by Agent 1.

    Returns
    -------
    dict with keys:
        ``score``                    – int 0–100
        ``matching_skills``          – list[str]
        ``missing_skills``           – list[dict] {"name": str, "importance": str}
        ``extra_skills``             – list[str]
        ``experience_verdict``       – str
        ``experience_justification`` – str
        ``suggestions``              – list[str]
        ``rewritten_bullets``        – list[str]
    """
    # Step 1: Local deterministic scoring
    score_result = score_match(resume_profile, jd_profile)

    # Step 2: LLM reasoning (synonyms, experience, advice)
    advice = _generate_advice(llm, resume_profile, jd_profile, score_result)

    return {
        "score": score_result["score"],
        "matching_skills": advice.get("matching_skills", score_result["matching"]),
        "missing_skills": advice.get("missing_skills", score_result["missing"]),
        "extra_skills": score_result["extra"],
        "experience_verdict": advice.get("experience_verdict", "Unclear from JD"),
        "experience_justification": advice.get("experience_justification", ""),
        "suggestions": advice.get("suggestions", []),
        "rewritten_bullets": advice.get("rewritten_bullets", []),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert technical recruiter and resume coach. "
    "You provide precise skill reconciliation and actionable advice. "
    "Always respond with valid JSON only — no explanation, no markdown fences."
)

_ADVICE_PROMPT_TEMPLATE = """\
You are reviewing a candidate's resume against a job description.

RESUME PROFILE:
{resume_profile}

JOB DESCRIPTION PROFILE:
{jd_profile}

PRELIMINARY KEYWORD MATCH (Automated):
- Exact Matching: {matching}
- Missing: {missing}

Your tasks:
1. SYNONYM RECONCILIATION:
   Review the "Missing" skills. Are any of them actually present in the resume
   under a different name or closely related technology (e.g. JD requires
   "Kubernetes", resume has "container orchestration" or "EKS")?
   If yes, MOVE them from missing_skills to matching_skills. Ensure you keep the
   importance tags on missing_skills.

2. EXPERIENCE ASSESSMENT:
   Compare the candidate's experience/seniority against the JD requirements.
   Even if the JD uses words like "senior" instead of numbers, reason contextually.
   Verdict MUST be exactly one of: "Meets", "Below", "Exceeds", or "Unclear from JD".

3. ADVICE & BULLETS:
   Provide specific, actionable suggestions referencing actual skills/gaps.
   Rewrite 2-3 resume bullet points using the STAR method to better highlight
   relevant experience and incorporate missing keywords naturally (do not fabricate).

Return a single JSON object EXACTLY matching this schema:
{{
  "matching_skills": ["list of strings (include automated matching + reconciled synonyms)"],
  "missing_skills": [
    {{"name": "skill name", "importance": "must-have"}},
    {{"name": "skill name", "importance": "nice-to-have"}}
  ],
  "experience_verdict": "Meets | Below | Exceeds | Unclear from JD",
  "experience_justification": "One brief sentence explaining the verdict.",
  "suggestions": [
    "Specific, actionable suggestion 1",
    "Specific, actionable suggestion 2"
  ],
  "rewritten_bullets": [
    "• Rewritten bullet point 1",
    "• Rewritten bullet point 2"
  ]
}}

Rules:
- missing_skills MUST be a list of objects with "name" and "importance".
- Return ONLY the JSON object.
"""


def _generate_advice(
    llm: LLMProvider,
    resume_profile: dict,
    jd_profile: dict,
    score_result: dict,
) -> dict:
    """Call the LLM to generate suggestions, reconcile synonyms, and assess experience."""
    prompt = _ADVICE_PROMPT_TEMPLATE.format(
        resume_profile=json.dumps(resume_profile, indent=2),
        jd_profile=json.dumps(jd_profile, indent=2),
        matching=json.dumps(score_result.get("matching", [])),
        missing=json.dumps(score_result.get("missing", [])),
    )

    try:
        raw_response = llm.generate(prompt, system=_SYSTEM_PROMPT)
    except RuntimeError:
        raise

    return _parse_advice_json(raw_response)


def _parse_advice_json(raw: str) -> dict:
    """Parse LLM JSON advice response with error handling."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: return a structured error rather than crashing
        return {
            "matching_skills": [],
            "missing_skills": [],
            "experience_verdict": "Unclear from JD",
            "experience_justification": "Failed to parse AI response.",
            "suggestions": [
                "The AI advisor could not parse its response. "
                "Please try again — this sometimes happens with complex profiles."
            ],
            "rewritten_bullets": [],
        }

    # Clean up and validate missing_skills (list of dicts)
    missing = []
    for item in data.get("missing_skills", []):
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            imp = str(item.get("importance", "must-have")).strip().lower()
            if imp not in ("must-have", "nice-to-have"):
                imp = "must-have"
            if name:
                missing.append({"name": name, "importance": imp})

    # Validate experience verdict
    verdict = str(data.get("experience_verdict", "")).strip()
    if verdict not in ("Meets", "Below", "Exceeds", "Unclear from JD"):
        verdict = "Unclear from JD"

    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)]

    bullets = data.get("rewritten_bullets", [])
    if not isinstance(bullets, list):
        bullets = [str(bullets)]

    matching = data.get("matching_skills", [])
    if not isinstance(matching, list):
        matching = [str(matching)]

    return {
        "matching_skills": [str(s).strip() for s in matching if str(s).strip()],
        "missing_skills": missing,
        "experience_verdict": verdict,
        "experience_justification": str(data.get("experience_justification", "")).strip(),
        "suggestions": [str(s).strip() for s in suggestions if str(s).strip()],
        "rewritten_bullets": [str(b).strip() for b in bullets if str(b).strip()],
    }
