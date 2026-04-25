"""Gemini 2.5 Flash provider implementation.

Uses the `google-genai` SDK (google.genai — the current, supported package).
This file is the ONLY place in the codebase that imports google.genai.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from llm.base import LLMProvider

_MODEL_ID = "gemini-2.5-flash"


class GeminiProvider(LLMProvider):
    """Wraps Gemini 2.5 Flash behind the LLMProvider interface."""

    def __init__(self, api_key: str, model: str = _MODEL_ID) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key must not be empty.")
        self._model_id = model
        self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return f"Gemini 2.5 Flash ({self._model_id})"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Call Gemini and return the response text.

        The new google-genai SDK accepts system_instruction as part of
        GenerateContentConfig, so no model rebuild is needed per call.
        """
        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
            ) if system else None

            response = self._client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as exc:
            raise RuntimeError(
                f"Gemini API error: {exc}"
            ) from exc
