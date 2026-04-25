# Job Finder (Agent 3) is implemented but disabled for v1 — see
# agents/job_finder.py. Re-enable by uncommenting the Job Matches tab
# and Agent 3 call below.

"""Resume-JD Match Analyzer — Streamlit App (v1)

Active pipeline (v1):
  Sidebar : provider selector, API key, test connection, recent runs
  Main    : resume input (paste/PDF) + JD input (required)
  Output  : match score (0-100), matching skills, missing skills,
            extra skills, experience match, suggestions, rewritten bullets

Disabled for v1 (implemented, not wired):
  - Job Matches tab (Agent 3 / live job search)
  - Job search filters (keyword, remote-only, location)

All LLM calls go through the LLMProvider abstraction (llm/base.py).
The API key is stored only in st.session_state — never written to disk.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime

import streamlit as st

from llm.factory import get_provider, list_providers
from tools.mcp_extract import extract_text_from_pdf
import agents.extractor as agent_extractor
import agents.matcher as agent_matcher

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Resume-JD Match Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

_LOG_PATH = pathlib.Path("logs/runs.jsonl")
_LOG_PATH.parent.mkdir(exist_ok=True)


def _log_run(entry: dict) -> None:
    """Append a run entry to logs/runs.jsonl. Never logs the API key."""
    entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
    entry.pop("api_key", None)  # paranoia — ensure key is never logged
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Logging failure must never crash the app


def _read_recent_runs(n: int = 5) -> list[dict]:
    """Read the last n entries from the run log."""
    if not _LOG_PATH.exists():
        return []
    try:
        lines = _LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines[-n:]][::-1]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Custom CSS — premium dark UI
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}

.metric-card {
    background: linear-gradient(135deg, #1e2433 0%, #252d3d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.score-ring {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    font-family: 'Space Grotesk', sans-serif;
    margin: 0 auto;
    box-shadow: 0 0 40px rgba(99,102,241,0.35);
}
.score-high { background: conic-gradient(#10b981 var(--pct), #1e2433 0); }
.score-mid  { background: conic-gradient(#f59e0b var(--pct), #1e2433 0); }
.score-low  { background: conic-gradient(#ef4444 var(--pct), #1e2433 0); }

.chip-container { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.5rem 0; }
.chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}
.chip-match   { background: rgba(16,185,129,0.15); border: 1px solid #10b981; color: #34d399; }
.chip-missing { background: rgba(239,68,68,0.15);  border: 1px solid #ef4444; color: #f87171; }
.chip-missing-nice { background: rgba(245,158,11,0.15); border: 1px solid #f59e0b; color: #fbbf24; }
.chip-extra   { background: rgba(148,163,184,0.1); border: 1px solid #475569; color: #94a3b8; }

.exp-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
}
.exp-Meets         { background: rgba(16,185,129,0.15); border: 1px solid #10b981; color: #34d399; }
.exp-Exceeds       { background: rgba(99,102,241,0.15); border: 1px solid #6366f1; color: #a5b4fc; }
.exp-Below         { background: rgba(239,68,68,0.15);  border: 1px solid #ef4444; color: #f87171; }
.exp-Unclear       { background: rgba(148,163,184,0.1); border: 1px solid #475569; color: #94a3b8; }

.coming-soon-box {
    border: 1px dashed #374151;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    background: rgba(17,24,39,0.5);
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.5);
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(30,36,51,0.8);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
}

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #6366f1);
    background-size: 200% auto;
    animation: shimmer 2s linear infinite;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for _key, _default in [
    ("api_key", ""),
    ("provider_name", "Gemini 2.5 Flash"),
    ("results", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎯 Resume Analyzer")
    st.caption("AI-powered resume × JD matching")
    st.divider()

    # Provider selector
    providers = list_providers()
    sel_idx = providers.index(st.session_state.provider_name) if st.session_state.provider_name in providers else 0
    selected_provider = st.selectbox(
        "🤖 LLM Provider",
        options=providers,
        index=sel_idx,
        help="Select which AI model powers the analysis.",
    )
    st.session_state.provider_name = selected_provider

    # API key input — stored in session state only, never on disk
    api_key_input = st.text_input(
        "🔑 API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="Paste your API key here...",
        help="Stored in session memory only — gone when you close the tab.",
    )
    st.session_state.api_key = api_key_input

    if not st.session_state.api_key:
        st.warning("⚠️ Enter your API key to enable analysis.")
    else:
        st.success("✅ API key set")

    # Test Connection button
    if st.button("🔌 Test Connection", key="btn_test_connection", disabled=not st.session_state.api_key):
        with st.spinner("Testing..."):
            try:
                provider = get_provider(st.session_state.provider_name, st.session_state.api_key)
                ok, msg = provider.test_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
            except ValueError as e:
                st.error(f"Provider error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    st.divider()

    # Recent Runs expander
    with st.expander("📊 Recent Runs", expanded=False):
        runs = _read_recent_runs(5)
        if not runs:
            st.caption("No runs yet — analyze a resume to see logs here.")
        else:
            for run in runs:
                ts = run.get("timestamp", "")[:16].replace("T", " ")
                provider_short = run.get("provider", "?").split("(")[0].strip()
                jd_score = run.get("jd_score")
                st.markdown(f"""
<div style="background:rgba(30,36,51,0.8);border:1px solid #30363d;border-radius:8px;
            padding:0.6rem 0.8rem;margin:0.3rem 0;">
<small style="color:#8b949e;">{ts} · {provider_short}</small><br>
{"🎯 Score: <b>" + str(jd_score) + "/100</b>" if jd_score is not None else "⚪ Score: —"}
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Job Search — coming soon note ────────────────────────────────────
    # Job Finder (Agent 3) is implemented in agents/job_finder.py but is
    # disabled in v1. Re-enable by wiring it back into the pipeline below.
    st.markdown("""
<div class="coming-soon-box">
  🔍 <b>Job Search</b> — coming in v2<br>
  <small>Live job matching via Arbeitnow + RemoteOK APIs is built and ready
  (see <code>agents/job_finder.py</code>) — will be enabled in the next release.</small>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.caption("🔒 Keys are session-only. No data saved between sessions.")

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.markdown("""
<h1 style="
  background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 2.4rem;
  margin-bottom: 0.1rem;
">🎯 Resume-JD Match Analyzer</h1>
<p style="color:#8b949e;font-size:1rem;margin-top:0;">
  Paste your resume and job description — get a detailed AI match analysis in seconds.
</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Input columns — BOTH are required in v1
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📄 Your Resume **(required)**")

    input_mode = st.radio(
        "Input method",
        ["Paste text", "Upload PDF / TXT"],
        horizontal=True,
        label_visibility="collapsed",
    )

    resume_text = ""
    if input_mode == "Paste text":
        resume_text = st.text_area(
            "Resume text",
            height=300,
            placeholder="Paste your full resume here…\n\nInclude all sections: Experience, Skills, Education, etc.",
            label_visibility="collapsed",
        )
    else:
        uploaded = st.file_uploader(
            "Upload resume",
            type=["pdf", "txt"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            if uploaded.name.lower().endswith(".pdf"):
                with st.spinner("Extracting text from PDF…"):
                    try:
                        resume_text = extract_text_from_pdf(uploaded.read())
                        if not resume_text.strip():
                            st.warning(
                                "⚠️ This PDF has no extractable text layer (may be a scanned image). "
                                "Please paste the text manually instead."
                            )
                        else:
                            st.success(f"✅ Extracted {len(resume_text):,} characters.")
                            with st.expander("Preview extracted text"):
                                st.text(resume_text[:1200] + ("…" if len(resume_text) > 1200 else ""))
                    except RuntimeError as e:
                        st.error(f"PDF extraction failed: {e}")
            else:
                resume_text = uploaded.read().decode("utf-8", errors="replace")
                st.success(f"✅ Loaded {len(resume_text):,} characters.")

with col_right:
    st.markdown("### 📋 Job Description **(required)**")
    jd_text = st.text_area(
        "JD text",
        height=300,
        placeholder="Paste the full job description here…\n\nInclude responsibilities, requirements, and nice-to-haves.",
        label_visibility="collapsed",
    )
    if jd_text.strip():
        st.caption("✅ JD provided — ready to analyze.")
    else:
        st.caption("⚠️ JD is required to run the analysis.")

# ---------------------------------------------------------------------------
# Analyze button
# ---------------------------------------------------------------------------

st.divider()

_missing_key = not st.session_state.api_key
_missing_resume = not resume_text.strip()
_missing_jd = not jd_text.strip()
_disabled = _missing_key or _missing_resume or _missing_jd

if _missing_key:
    st.info("🔑 Enter your API key in the sidebar to enable analysis.")
elif _missing_resume:
    st.info("📄 Paste or upload your resume above.")
elif _missing_jd:
    st.info("📋 Paste the job description on the right to enable analysis.")

analyze_clicked = st.button(
    "🚀 Analyze Match",
    key="btn_analyze",
    disabled=_disabled,
    use_container_width=True,
    type="primary",
)

# ---------------------------------------------------------------------------
# Pipeline execution — Agents 1 + 2 only (v1)
#
# Agent 3 (Job Finder) is implemented in agents/job_finder.py but disabled.
# To re-enable:
#   1. Uncomment the Agent 3 block below
#   2. Add the "Job Matches" tab to the results section
#   3. Add job search filter inputs (keyword, remote_only, location)
# ---------------------------------------------------------------------------

if analyze_clicked and not _disabled:
    st.session_state.results = None

    log_entry: dict = {
        "provider": st.session_state.provider_name,
        "jd_provided": True,  # v1: JD is always required
    }

    try:
        # Instantiate provider (validates key at runtime)
        with st.spinner("🔌 Connecting to LLM provider…"):
            try:
                llm = get_provider(st.session_state.provider_name, st.session_state.api_key)
            except (ValueError, RuntimeError) as e:
                st.error(f"❌ Failed to initialise LLM provider: {e}")
                st.stop()

        # ── Agent 1: Extract structured profiles ──────────────────────────
        with st.spinner("🤖 Agent 1: Extracting structured profiles from resume and JD…"):
            try:
                extraction = agent_extractor.run(
                    llm=llm,
                    resume_text=resume_text,
                    jd_text=jd_text,
                )
            except ValueError as e:
                st.error(f"❌ Input error: {e}")
                st.stop()
            except RuntimeError as e:
                st.error(f"❌ Agent 1 (Extractor) failed: {e}")
                st.stop()

        resume_profile = extraction["resume_profile"]
        jd_profile = extraction.get("jd_profile")

        if jd_profile is None:
            st.error("❌ Could not extract a JD profile — please check your JD text.")
            st.stop()

        # ── Agent 2: Score + advise ────────────────────────────────────────
        with st.spinner("🤖 Agent 2: Scoring match and generating advice…"):
            try:
                match_result = agent_matcher.run(
                    llm=llm,
                    resume_profile=resume_profile,
                    jd_profile=jd_profile,
                )
                log_entry["jd_score"] = match_result["score"]
            except RuntimeError as e:
                st.error(f"❌ Agent 2 (Matcher) failed: {e}")
                st.stop()

        # ── Agent 3: DISABLED for v1 ───────────────────────────────────────
        # To re-enable, uncomment this block and add filters + tab above:
        #
        # with st.spinner("🤖 Agent 3: Searching live jobs…"):
        #     try:
        #         from agents import job_finder as agent_job_finder
        #         job_results = agent_job_finder.run(
        #             llm=llm,
        #             resume_profile=resume_profile,
        #             keyword=search_keyword,
        #             remote_only=remote_only,
        #             location=search_location,
        #         )
        #         log_entry["jobs_found"] = len(job_results)
        #         if job_results:
        #             log_entry["top_job_score"] = job_results[0]["fit_score"]
        #     except RuntimeError as e:
        #         st.warning(f"⚠️ Agent 3 (Job Finder) error: {e}")
        #         job_results = []

        st.session_state.results = {
            "resume_profile": resume_profile,
            "match_result": match_result,
        }

    finally:
        _log_run(log_entry)

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if st.session_state.results:
    res = st.session_state.results
    match = res["match_result"]
    score = match["score"]        # already 0–100 int from matcher
    breakdown = match["breakdown"]  # already 0–100 per category
    exp = match.get("experience_match", {})

    st.divider()
    st.markdown("## 📊 Match Analysis")

    # ── Score header ──────────────────────────────────────────────────────
    score_class = "score-high" if score >= 70 else "score-mid" if score >= 40 else "score-low"
    score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
    score_label = "Strong Match 🟢" if score >= 70 else "Partial Match 🟡" if score >= 40 else "Low Match 🔴"

    hc1, hc2, hc3 = st.columns([1, 2, 1])
    with hc2:
        st.markdown(f"""
<div style="text-align:center; padding:1.5rem 0 1rem;">
  <div style="
    display:inline-flex; align-items:center; justify-content:center;
    width:120px; height:120px; border-radius:50%;
    border: 6px solid {score_color};
    box-shadow: 0 0 30px {score_color}55;
    flex-direction:column;
  ">
    <span style="font-size:2rem; font-weight:700; color:{score_color};
                 font-family:'Space Grotesk',sans-serif; line-height:1;">{score}</span>
    <span style="font-size:0.7rem; color:#8b949e; margin-top:2px;">/ 100</span>
  </div>
  <h3 style="margin:0.7rem 0 0.1rem; color:#e2e8f0;">{score_label}</h3>
  <p style="color:#8b949e; font-size:0.85rem; margin:0;">
    Based on skills, tools &amp; keyword alignment
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Breakdown bars ─────────────────────────────────────────────────────
    bc1, bc2 = st.columns(2)
    for col, (label, emoji, key) in zip(
        [bc1, bc2],
        [("Must-Have Match", "🎯", "must_have_score"), ("Nice-to-Have Match", "✨", "nice_to_have_score")],
    ):
        with col:
            pct = breakdown.get(key, 0)
            col_color = "#10b981" if pct >= 70 else "#f59e0b" if pct >= 40 else "#ef4444"
            st.markdown(f"""
<div class="metric-card" style="text-align:center;">
  <div style="font-size:1.4rem;">{emoji}</div>
  <div style="font-size:1.7rem; font-weight:700; color:{col_color};
              font-family:'Space Grotesk',sans-serif;">{pct}%</div>
  <div style="color:#8b949e; font-size:0.8rem;">{label}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Experience Match ───────────────────────────────────────────────────
    st.markdown("#### 🏆 Experience Match")
    verdict = match.get("experience_verdict", "Unclear from JD")
    justification = match.get("experience_justification", "No assessment provided.")
    
    _VERDICT_ICONS = {"Meets": "✅", "Exceeds": "🚀", "Below": "⚠️", "Unclear from JD": "❓"}
    icon = _VERDICT_ICONS.get(verdict, "❓")
    badge_class = f"exp-{verdict.split()[0]}" # e.g. exp-Meets, exp-Unclear
    
    st.markdown(f"""
<div class="metric-card" style="display:flex; align-items:center; gap:1.5rem;">
  <div>
    <span class="exp-badge {badge_class}" style="font-size:1.1rem; padding:8px 20px;">
      {icon} {verdict}
    </span>
  </div>
  <div style="color:#cbd5e1; font-size:1rem; line-height:1.4;">
    {justification}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Skills sections ────────────────────────────────────────────────────
    sk1, sk2, sk3 = st.columns(3)

    with sk1:
        st.markdown("#### ✅ Matching Skills")
        matching = match.get("matching_skills", [])
        if matching:
            chips = "".join(f'<span class="chip chip-match">{s}</span>' for s in matching)
            st.markdown(f'<div class="chip-container">{chips}</div>', unsafe_allow_html=True)
        else:
            st.caption("No direct skill matches found.")

    with sk2:
        st.markdown("#### ❌ Missing Skills")
        missing = match.get("missing_skills", [])
        if missing:
            chips = ""
            for m in missing:
                # Handle dicts vs flat strings gracefully (just in case)
                if isinstance(m, dict):
                    name = m.get("name", "")
                    imp = m.get("importance", "must-have")
                else:
                    name = str(m)
                    imp = "must-have"
                
                cclass = "chip-missing" if imp == "must-have" else "chip-missing-nice"
                chips += f'<span class="chip {cclass}" title="{imp}">{name}</span>'
                
            st.markdown(f'<div class="chip-container">{chips}</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 No skill gaps — great match!")

    with sk3:
        st.markdown("#### ➕ Extra Skills")
        st.caption("In your resume, not required by this JD")
        extra = match.get("extra_skills", [])
        if extra:
            chips = "".join(f'<span class="chip chip-extra">{s}</span>' for s in extra[:20])
            st.markdown(f'<div class="chip-container">{chips}</div>', unsafe_allow_html=True)
        else:
            st.caption("No extra skills detected.")

    st.markdown("---")

    # ── Actionable Suggestions ─────────────────────────────────────────────
    st.markdown("#### 💡 Actionable Suggestions")
    suggestions = match.get("suggestions", [])
    if suggestions:
        for i, s in enumerate(suggestions, 1):
            st.markdown(f"""
<div class="metric-card" style="margin:0.4rem 0; border-left:3px solid #6366f1;">
  <span style="color:#818cf8; font-weight:700;">{i}.</span>&nbsp; {s}
</div>
""", unsafe_allow_html=True)
    else:
        st.caption("No suggestions generated.")

    # ── Rewritten Bullets ──────────────────────────────────────────────────
    st.markdown("#### ✍️ Suggested Resume Bullets")
    bullets = match.get("rewritten_bullets", [])
    if bullets:
        for bullet in bullets:
            st.markdown(f"""
<div class="metric-card" style="margin:0.4rem 0; border-left:3px solid #10b981;">
  {bullet}
</div>
""", unsafe_allow_html=True)
    else:
        st.caption("No rewritten bullets generated.")
