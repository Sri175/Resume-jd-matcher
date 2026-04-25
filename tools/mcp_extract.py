"""MCP Tool: extract_structured_info

Preprocesses raw document text before it is passed to the LLM.
Responsibilities:
  - Clean whitespace and control characters
  - Split resume text into sections (EXPERIENCE, EDUCATION, SKILLS, etc.)
  - Extract text from PDF bytes (via pypdf)
  - Return a cleaned, structured string ready for LLM prompting

This is a Python function acting as an MCP tool. It performs purely local,
deterministic processing — no LLM calls here.
"""

from __future__ import annotations

import io
import re
import unicodedata


# ---------------------------------------------------------------------------
# Section heading patterns — covers common resume / JD conventions
# Note: inline (?i) flags are placed ONLY on the individual patterns here;
# the combined re.compile() call uses re.IGNORECASE | re.MULTILINE so all
# patterns share those flags. Python 3.14+ forbids inline global flags at
# positions other than the very start of the final compiled expression.
# ---------------------------------------------------------------------------

_SECTION_PATTERNS = [
    r"^(SUMMARY|OBJECTIVE|PROFILE|ABOUT)\b",
    r"^(EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|PROFESSIONAL EXPERIENCE)\b",
    r"^(EDUCATION|ACADEMIC|QUALIFICATIONS)\b",
    r"^(SKILLS|TECHNICAL SKILLS|CORE SKILLS|COMPETENCIES)\b",
    r"^(PROJECTS|PERSONAL PROJECTS|SIDE PROJECTS)\b",
    r"^(CERTIFICATIONS?|CERTIFICATES?|LICENSES?)\b",
    r"^(AWARDS?|ACHIEVEMENTS?|HONORS?)\b",
    r"^(LANGUAGES?)\b",
    r"^(PUBLICATIONS?|RESEARCH)\b",
    r"^(RESPONSIBILITIES|REQUIREMENTS|QUALIFICATIONS|DUTIES)\b",
    r"^(BENEFITS?|WHAT WE OFFER|WHY JOIN)\b",
]

_SECTION_RE = re.compile(
    "|".join(f"({p})" for p in _SECTION_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


def extract_structured_info(text: str, doc_type: str = "resume") -> dict:
    """Preprocess a resume or job description text.

    Parameters
    ----------
    text:
        Raw document text (already extracted from PDF if needed).
    doc_type:
        Either ``"resume"`` or ``"jd"`` (job description).

    Returns
    -------
    dict with keys:
        ``cleaned``  – whitespace-normalised full text
        ``sections`` – {section_name: section_body} dict
        ``doc_type`` – echoed back for downstream consumers
    """
    cleaned = _clean_text(text)
    sections = _split_sections(cleaned)
    return {
        "cleaned": cleaned,
        "sections": sections,
        "doc_type": doc_type,
    }


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pypdf.

    Returns empty string if no text layer is found (e.g. scanned images).
    Callers should check for empty return and show the user a warning.
    """
    try:
        from pypdf import PdfReader  # local import — only needed for PDF path
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        return "\n\n".join(pages)
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise unicode, strip control chars, collapse blank lines."""
    # Normalise unicode (NFC) — handles accented chars, fancy quotes etc.
    text = unicodedata.normalize("NFC", text)
    # Strip non-printable control characters (keep \n and \t)
    text = re.sub(r"[^\S\n\t]+", " ", text)          # collapse spaces
    text = re.sub(r"\t", " ", text)                    # tabs → single space
    text = re.sub(r" +", " ", text)                    # multiple spaces → one
    text = re.sub(r"\n{3,}", "\n\n", text)             # max 2 consecutive newlines
    # Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _split_sections(text: str) -> dict[str, str]:
    """Split text into named sections based on common headings.

    Returns a dict where keys are normalised section names (uppercased) and
    values are the body text that follows each heading.
    """
    lines = text.splitlines()
    sections: dict[str, str] = {}
    current_section = "HEADER"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if _is_section_heading(stripped):
            # Save the previous section
            body = "\n".join(current_lines).strip()
            if body:
                sections[current_section] = body
            current_section = _normalise_heading(stripped)
            current_lines = []
        else:
            current_lines.append(line)

    # Save the last section
    body = "\n".join(current_lines).strip()
    if body:
        sections[current_section] = body

    return sections


def _is_section_heading(line: str) -> bool:
    """Return True if *line* looks like a resume/JD section heading."""
    if not line or len(line) > 60:
        return False
    # All-caps line or matches known pattern
    if line.isupper() and len(line.split()) <= 5:
        return True
    return bool(_SECTION_RE.match(line))


def _normalise_heading(heading: str) -> str:
    """Normalise a heading string to a canonical uppercase key."""
    heading = heading.upper().strip().rstrip(":")
    # Map variants to canonical names
    _CANONICAL = {
        "WORK EXPERIENCE": "EXPERIENCE",
        "EMPLOYMENT": "EXPERIENCE",
        "PROFESSIONAL EXPERIENCE": "EXPERIENCE",
        "ACADEMIC": "EDUCATION",
        "QUALIFICATIONS": "EDUCATION",
        "TECHNICAL SKILLS": "SKILLS",
        "CORE SKILLS": "SKILLS",
        "COMPETENCIES": "SKILLS",
        "CERTIFICATES": "CERTIFICATIONS",
        "LICENSES": "CERTIFICATIONS",
        "ACHIEVEMENTS": "AWARDS",
        "HONORS": "AWARDS",
        "REQUIREMENTS": "RESPONSIBILITIES",
    }
    return _CANONICAL.get(heading, heading)
