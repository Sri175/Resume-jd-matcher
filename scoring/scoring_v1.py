"""Scoring module v1 — importance-weighted keyword/skill overlap.

VERSION:  1.1
DATE:     2026-08-22
CHANGELOG:
  v1.0  Weighted set-intersection across skill/tool/keyword categories
        (skills 50%, tools 30%, keywords 20%). Score: float 0.0-1.0.
  v1.1  Replaced category-weights with IMPORTANCE weights derived from
        JD tags. Must-have = 70% of score, nice-to-have = 30%.
        Score is now a 0-100 INTEGER (round to nearest int).
        `missing` is now list[dict] with {"name": str, "importance": str}.
        Added `extra` field (resume terms not required by JD).
        Flat-string JD items are treated as "must-have" for backward compat.

FORMULA (v1.1):
  Let M  = set of must-have terms from JD (skills + tools_tech with
            importance="must-have", plus all flat-string items)
  Let N  = set of nice-to-have terms from JD (importance="nice-to-have"
            plus all keywords -- domain signals, not hard requirements)
  Let R  = set of all terms from resume (skills + tools_tech + keywords)

  must_ratio  = |M & R| / |M|   (1.0 if M is empty)
  nice_ratio  = |N & R| / |N|   (1.0 if N is empty)
  score       = round(70 * must_ratio + 30 * nice_ratio)   -> 0-100 int

DESIGN NOTES:
  - Stateless and dependency-free (pure Python).
  - Imported by Agent 2 (active) and Agent 3 (disabled, parked for v2).
  - To create scoring_v2.py: copy, bump VERSION/DATE/CHANGELOG, update
    formula. Agents import by module path -- one-line swap to upgrade.
"""

from __future__ import annotations

import re as _re

__version__ = "1.1"
__date__ = "2026-08-22"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_match(resume_profile: dict, jd_profile: dict) -> dict:
    """Compute an importance-weighted match score between resume and JD.

    Parameters
    ----------
    resume_profile:
        Resume profile with flat string lists:
            {"skills": [str], "tools_tech": [str], "keywords": [str], ...}

    jd_profile:
        JD profile. skills/tools_tech may be either:
          - flat strings (treated as "must-have" -- backward compat for Agent 3)
          - dicts: [{"name": str, "importance": "must-have"|"nice-to-have"}]
        keywords: always flat strings (treated as "nice-to-have").

    Returns
    -------
    dict:
        ``score``        -- int 0-100
        ``matching``     -- sorted list[str] of terms in both profiles
        ``missing``      -- list[dict] {"name": str, "importance": str}
                           sorted: must-have first, then nice-to-have
        ``extra``        -- sorted list[str] in resume, not required by JD
        ``breakdown``    -- {must_have_score, nice_to_have_score,
                            must_have_matched, must_have_total,
                            nice_to_have_matched, nice_to_have_total}
    """
    # 1. Parse resume terms into a flat normalised set
    resume_terms: set[str] = set()
    for field in ("skills", "tools_tech", "keywords"):
        resume_terms |= _normalise_list(resume_profile.get(field, []))

    # 2. Split JD terms into must-have / nice-to-have buckets
    must_have: set[str] = set()
    nice_to_have: set[str] = set()

    for field in ("skills", "tools_tech"):
        for item in jd_profile.get(field, []):
            if isinstance(item, dict):
                name = _normalise_str(item.get("name", ""))
                imp = str(item.get("importance", "must-have")).lower().strip()
                if name:
                    if imp == "nice-to-have":
                        nice_to_have.add(name)
                    else:
                        must_have.add(name)  # default: must-have
            elif isinstance(item, str):
                name = _normalise_str(item)
                if name:
                    must_have.add(name)  # flat strings -> must-have

    # Keywords: domain signals -> nice-to-have (if not already must-have)
    for kw in _normalise_list(jd_profile.get("keywords", [])):
        if kw not in must_have:
            nice_to_have.add(kw)

    all_jd_terms = must_have | nice_to_have

    # 3. Compute overlaps
    must_matched = sorted(must_have & resume_terms)
    must_missing = sorted(must_have - resume_terms)
    nice_matched = sorted(nice_to_have & resume_terms)
    nice_missing = sorted(nice_to_have - resume_terms)

    all_matching = sorted(set(must_matched) | set(nice_matched))
    extra = sorted(resume_terms - all_jd_terms)

    # Missing -- must-have first for UI grouping
    missing_tagged = (
        [{"name": t, "importance": "must-have"} for t in must_missing]
        + [{"name": t, "importance": "nice-to-have"} for t in nice_missing]
    )

    # 4. Score formula (see module docstring)
    must_ratio = len(must_matched) / len(must_have) if must_have else 1.0
    nice_ratio = len(nice_matched) / len(nice_to_have) if nice_to_have else 1.0
    score = round(70 * must_ratio + 30 * nice_ratio)

    return {
        "score": score,
        "matching": all_matching,
        "missing": missing_tagged,
        "extra": extra,
        "breakdown": {
            "must_have_score": round(must_ratio * 100),
            "nice_to_have_score": round(nice_ratio * 100),
            "must_have_matched": len(must_matched),
            "must_have_total": len(must_have),
            "nice_to_have_matched": len(nice_matched),
            "nice_to_have_total": len(nice_to_have),
        },
    }


def build_job_profile(job: dict) -> dict:
    """Convert a normalised job dict (Agent 3) to a minimal flat profile.

    Returns flat string lists (all treated as must-have by score_match).
    Preserved for Agent 3 backward compatibility -- see agents/job_finder.py.
    """
    tags = _normalise_list(job.get("tags", []))
    description_text = (job.get("description") or "").lower()
    tech_words = set(_extract_tech_terms(description_text))

    return {
        "skills": list(tags),
        "tools_tech": list(tech_words),
        "keywords": list(tags),
        "experience_years": 0,
        "seniority_level": "",
        "target_titles": [job.get("title", "")],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_list(items: list) -> set[str]:
    """Lowercase, strip, and deduplicate a list of strings."""
    result: set[str] = set()
    for item in items:
        if isinstance(item, str) and item.strip():
            result.add(item.strip().lower())
    return result


def _normalise_str(s: str) -> str:
    """Lowercase and strip a single string."""
    return str(s).strip().lower()


_TECH_TERM_RE = _re.compile(
    r"\b([A-Z][a-z]+[A-Z]\w*"   # CamelCase: React, TypeScript
    r"|[A-Z]{2,}"               # ALL-CAPS: AWS, SQL
    r"|[a-z]+\.[a-z]+"          # dot-notation: node.js
    r"|[a-z]+-[a-z]+"           # hyphenated: machine-learning
    r")\b"
)


def _extract_tech_terms(text: str) -> list[str]:
    """Extract tech-shaped words from raw text (Agent 3 heuristic)."""

    matches = _TECH_TERM_RE.findall(text)
    return [m.lower() for m in matches if len(m) >= 3]
