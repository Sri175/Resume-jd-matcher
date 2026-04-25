# 🎯 Resume-JD Match Analyzer

An AI-powered tool that compares your resume against a specific job description to give you a precise match score, highlight missing skills, and provide actionable advice to improve your application.

## Key Features

*   **Multi-Agent Architecture**: 
    *   **Extractor**: Analyzes your resume and the job description to intelligently identify skills, tools, and keywords.
    *   **Matcher & Advisor**: Compares both profiles, reconciles synonyms (e.g. "Kubernetes" vs. "Container Orchestration"), determines experience alignment, and suggests resume improvements.
*   **Bring Your Own LLM**: Supports multiple providers through a unified abstraction layer. You can dynamically select between:
    *   Gemini 2.5 Flash (`google-genai` SDK)
    *   OpenAI (`openai` SDK)
    *   Anthropic Claude (`anthropic` SDK)
*   **Importance-Weighted Scoring**: Evaluates requirements not just as a flat list, but categorizes them intelligently into "Must-Have" vs. "Nice-to-Have" based on the JD's context.

## Architecture

The project leverages a robust module-level tool architecture inspired by the Model Context Protocol (MCP). Note that these are local Python modules providing structured outputs, not a standalone external MCP server.

```text
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────────┐
│                 │      │                     │      │                         │
│   User Input    ├─────►│  Agent 1: Extractor ├─────►│ Agent 2: Matcher/Advisor│
│ (Resume + JD)   │      │                     │      │                         │
└─────────────────┘      └─────────┬───────────┘      └────────────┬────────────┘
                                   │                               │             
                                   ▼                               ▼             
                         ┌──────────────────────────────────────────────────┐    
                         │             LLM Provider Abstraction             │    
                         │         (Gemini | OpenAI | Claude API)           │    
                         └──────────────────────────────────────────────────┘    
```

### Design Decisions
| Feature | Decision & Rationale |
| :--- | :--- |
| **Scoring Engine** | **Deterministic Local Module:** Skill intersections are scored in pure Python (`scoring_v1.py`) rather than by the LLM. This guarantees consistency and transparency (0-100 score). |
| **Experience & Synonyms** | **LLM-Reasoned:** The LLM is used to reconcile missing skills with synonyms and determine if the candidate meets the seniority level contextually, something traditional keyword-matching fails at. |
| **Must-Have Weighting** | **70/30 Split:** Not all keywords are created equal. The LLM extracts skills with an `importance` tag ("must-have" vs "nice-to-have") which factors into the 70/30 weighted score. |

## Tech Stack
*   **Language**: Python 3.11+
*   **UI Framework**: Streamlit
*   **Testing**: pytest
*   **LLM Providers**: Google Gemini, OpenAI, Anthropic Claude

## How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Sri175/Resume-jd-matcher.git
    cd Resume-jd-matcher
    ```

2.  **Set up a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application**
    ```bash
    streamlit run app.py
    ```

5.  **Use the app**
    *   Open `http://localhost:8501` in your browser.
    *   Enter your chosen LLM provider's API key in the sidebar.
    *   Paste your resume and job description.
    *   Click "Analyze Match".

## Deployment (Streamlit Community Cloud)

To deploy this app publicly for free:
1. Push this repository to your GitHub account.
2. Log in to [share.streamlit.io](https://share.streamlit.io/) and click "New app".
3. Select this repository and set the main file path to `app.py`.
4. Deploy! No secrets need to be configured in Streamlit Cloud, as the user brings their own API key at runtime.

## Testing

The project includes a robust suite of over 100 tests covering the scoring logic, normalization, and provider abstraction layer.

```bash
python -m pytest tests/ -v
```

## Roadmap

**Job Finder (Agent 3) — Coming Soon**
A third agent is currently parked in the codebase (`agents/job_finder.py`) for a future release. It will allow you to bypass pasting a JD and instead use your parsed Resume profile to search live job boards (Arbeitnow & RemoteOK), returning the top matches ranked by your specific skills.

## Limitations

*   **LLM Variability**: Since extraction and advice rely on generative AI, outputs can slightly vary between runs or between different providers.
*   **Data Persistence**: This is a stateless application. User data and API keys are not stored beyond the active session.
*   **Rate Limits**: You are responsible for your own API key usage and any rate limits imposed by OpenAI, Google, or Anthropic.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
