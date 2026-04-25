"""Agent 1 — Extractor

Extracts structured profiles from resume text and JD text using LLM reasoning.
The LLM is the ONLY mechanism for identifying skills — no hardcoded skill
dictionaries, fixed keyword lists, or regex-based skill matching.

The LLM reads whatever is actually written and identifies skills, tools, and
keywords contextually — including niche or uncommon technologies.

Two separate prompts are used:

  RESUME prompt  -> flat string lists (simple schema)
  JD prompt      -> importance-tagged schema (must-have / nice-to-have)

Pipeline:
  1. mcp_extract.extract_structured_info() — clean + section-split text
  2. Call LLM with appropriate prompt (resume or JD)
  3. Parse + validate the JSON response
  4. Return resume_profile and jd_profile

Profile schemas:

  resume_profile = {
    "skills":           [str, ...],   # LLM-identified competencies
    "experience_years": int,          # total years of experience
    "seniority_level":  str,          # "junior"|"mid"|"senior"|"lead"|""
    "target_titles":    [str, ...],
    "tools_tech":       [str, ...],
    "keywords":         [str, ...],
  }

  jd_profile = {
    "skills":                    [{"name": str, "importance": "must-have"|"nice-to-have"}, ...],
    "tools_tech":                [{"name": str, "importance": "must-have"|"nice-to-have"}, ...],
    "keywords":                  [str, ...],
    "experience_years_required": str,   # free text: "5+ years", "not specified"
    "seniority_level_required":  str,   # free text: "senior", "not specified"
    "target_titles":             [str, ...],
  }

Agent design rule: imports ONLY from llm.base — never from a specific SDK.
The llm parameter is injected by the caller (app.py).
"""

from __future__ import annotations

import json
import re

from llm.base import LLMProvider
from tools.mcp_extract import extract_structured_info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    llm: LLMProvider,
    resume_text: str,
    jd_text: str = "",
) -> dict:
    """Extract structured profiles from resume and JD text.

    Parameters
    ----------
    llm:
        A concrete LLMProvider injected by app.py.
    resume_text:
        Raw resume text (plain text, already extracted from PDF if needed).
    jd_text:
        Job description text. If empty, jd_profile is not returned.

    Returns
    -------
    dict:
        ``resume_profile`` -- structured resume profile
        ``jd_profile``     -- structured JD profile (only if jd_text != "")
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty — cannot extract profile.")

    # Step 1: MCP preprocessing
    resume_doc = extract_structured_info(resume_text.strip(), doc_type="resume")
    resume_profile = _extract_resume_profile(llm, resume_doc)

    result: dict = {"resume_profile": resume_profile}

    if jd_text and jd_text.strip():
        jd_doc = extract_structured_info(jd_text.strip(), doc_type="jd")
        jd_profile = _extract_jd_profile(llm, jd_doc)
        result["jd_profile"] = jd_profile

    return result


# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise document analysis assistant. "
    "You extract structured information from resumes and job descriptions. "
    "Identify skills, tools, and keywords contextually from what is written — "
    "including niche, uncommon, or domain-specific technologies not on any standard list. "
    "Always respond with valid JSON only — no explanation, no markdown fences."
)


# ---------------------------------------------------------------------------
# Resume extraction (flat schema)
# ---------------------------------------------------------------------------

_RESUME_PROMPT = """\
Extract structured information from this RESUME.

Identify all skills, tools, and keywords that are ACTUALLY MENTIONED in the text.
Do not add skills that are not present. Do not use a fixed skill list —
extract whatever the candidate has actually written, including niche technologies.

Return a single JSON object with EXACTLY these keys:
{{
  "skills": ["list of skill names: programming languages, frameworks, methodologies, soft skills"],
  "experience_years": <integer: candidate's total years of work experience. 0 if unclear>,
  "seniority_level": "<one of: junior, mid, senior, lead, executive, or empty string if unclear>",
  "target_titles": ["list of job titles the candidate has held or is targeting"],
  "tools_tech": ["list of specific tools, technologies, platforms, cloud services"],
  "keywords": ["list of domain/industry keywords and methodologies not already in skills"]
}}

Rules:
- Extract ONLY what is explicitly stated or clearly implied in the text.
- Do NOT invent skills or infer from general job titles.
- All lists must contain only strings. Deduplicate. Keep them lowercase except proper names.
- Return ONLY the JSON object, nothing else.

RESUME TEXT:
---
{text}
---"""


def _extract_resume_profile(llm: LLMProvider, doc: dict) -> dict:
    """Call the LLM to extract a flat resume profile."""
    text = _build_section_text(doc)
    prompt = _RESUME_PROMPT.format(text=text[:8000])
    raw = llm.generate(prompt, system=_SYSTEM_PROMPT)
    return _parse_resume_json(raw)


def _parse_resume_json(raw: str) -> dict:
    """Parse and validate a resume profile JSON response."""
    data = _parse_json_response(raw)
    return {
        "skills": _ensure_str_list(data.get("skills", [])),
        "experience_years": _ensure_int(data.get("experience_years", 0)),
        "seniority_level": str(data.get("seniority_level", "")).lower().strip(),
        "target_titles": _ensure_str_list(data.get("target_titles", [])),
        "tools_tech": _ensure_str_list(data.get("tools_tech", [])),
        "keywords": _ensure_str_list(data.get("keywords", [])),
    }


# ---------------------------------------------------------------------------
# JD extraction (importance-tagged schema)
# ---------------------------------------------------------------------------

_JD_PROMPT = """\
Extract structured information from this JOB DESCRIPTION.

Identify ALL skills and tools mentioned. For EACH skill/tool, classify its importance:
- "must-have": the JD uses language like "required", "must have", "essential",
  "you will need", "we require", or states it as a core responsibility.
- "nice-to-have": the JD uses language like "bonus", "plus", "preferred",
  "nice to have", "ideally", "advantageous", or lists it as optional.
- If the JD does not make it clear: default to "must-have".

Return a single JSON object with EXACTLY these keys:
{{
  "skills": [
    {{"name": "skill name", "importance": "must-have"}},
    {{"name": "another skill", "importance": "nice-to-have"}}
  ],
  "tools_tech": [
    {{"name": "tool or tech name", "importance": "must-have"}},
    {{"name": "optional tool", "importance": "nice-to-have"}}
  ],
  "keywords": ["list of flat domain/industry keywords: role level, methodologies, domain terms"],
  "experience_years_required": "<free text: e.g. '5+ years', '3-5 years', 'not specified'>",
  "seniority_level_required": "<free text: e.g. 'senior', 'mid-level', 'not specified'>",
  "target_titles": ["list of job titles this role is advertising or similar to"]
}}

Rules:
- Extract ONLY what is in the text. Do NOT fabricate requirements.
- Skills and tools_tech must be lists of objects with "name" and "importance" keys.
- Keywords must be a flat list of strings (no importance tag needed).
- For experience_years_required and seniority_level_required: if not mentioned, use "not specified".
- Return ONLY the JSON object, nothing else.

JOB DESCRIPTION TEXT:
---
{text}
---"""


def _extract_jd_profile(llm: LLMProvider, doc: dict) -> dict:
    """Call the LLM to extract an importance-tagged JD profile."""
    text = _build_section_text(doc)
    prompt = _JD_PROMPT.format(text=text[:8000])
    raw = llm.generate(prompt, system=_SYSTEM_PROMPT)
    return _parse_jd_json(raw)


def _parse_jd_json(raw: str) -> dict:
    """Parse and validate a JD profile JSON response."""
    data = _parse_json_response(raw)

    def _parse_tagged_list(items: list) -> list[dict]:
        """Coerce a list of items into [{name, importance}] format."""
        result: list[dict] = []
        for item in items:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                imp = str(item.get("importance", "must-have")).strip().lower()
                if name:
                    if imp not in ("must-have", "nice-to-have"):
                        imp = "must-have"
                    result.append({"name": name, "importance": imp})
            elif isinstance(item, str) and item.strip():
                # Flat string fallback — treat as must-have
                result.append({"name": item.strip(), "importance": "must-have"})
        return result

    return {
        "skills": _parse_tagged_list(data.get("skills", [])),
        "tools_tech": _parse_tagged_list(data.get("tools_tech", [])),
        "keywords": _ensure_str_list(data.get("keywords", [])),
        "experience_years_required": str(
            data.get("experience_years_required", "not specified") or "not specified"
        ).strip(),
        "seniority_level_required": str(
            data.get("seniority_level_required", "not specified") or "not specified"
        ).strip(),
        "target_titles": _ensure_str_list(data.get("target_titles", [])),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_section_text(doc: dict) -> str:
    """Build prompt-ready text from a preprocessed doc dict."""
    sections = doc.get("sections", {})
    cleaned = doc.get("cleaned", "")
    if sections:
        return "\n\n".join(f"[{name}]\n{body}" for name, body in sections.items())
    return cleaned


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from an LLM response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned malformed JSON.\n"
            f"Preview: {text[:300]}\nError: {exc}"
        ) from exc


def _ensure_str_list(value: object) -> list[str]:
    """Coerce value to a list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _ensure_int(value: object) -> int:
    """Coerce value to a non-negative integer."""
    try:
        return max(0, int(value))
    except (ValueError, TypeError):
        return 0
