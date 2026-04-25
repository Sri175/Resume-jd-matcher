"""OpenAI GPT-4o-mini provider implementation.

Uses the `openai` SDK.  This file is the ONLY place in the codebase that
imports openai.
"""

from __future__ import annotations

from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

from llm.base import LLMProvider

_MODEL_ID = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    """Wraps OpenAI GPT-4o-mini behind the LLMProvider interface."""

    def __init__(self, api_key: str, model: str = _MODEL_ID) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API key must not be empty.")
        self._model_id = model
        self._client = OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return f"OpenAI GPT-4o-mini ({self._model_id})"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Call the OpenAI chat completions endpoint and return text."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except AuthenticationError as exc:
            raise RuntimeError(
                "OpenAI authentication failed — check your API key."
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError(
                "OpenAI rate limit reached — please wait and try again."
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                f"OpenAI connection error — check your network: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc
