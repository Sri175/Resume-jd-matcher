"""Anthropic Claude provider implementation.

Uses the `anthropic` SDK.  This file is the ONLY place in the codebase that
imports anthropic.
"""

from __future__ import annotations

import anthropic
from anthropic import AuthenticationError, RateLimitError, APIConnectionError

from llm.base import LLMProvider

_MODEL_ID = "claude-3-5-haiku-20241022"


class ClaudeProvider(LLMProvider):
    """Wraps Anthropic Claude behind the LLMProvider interface.

    Uses claude-3-5-haiku by default — fast and affordable, while still
    powerful enough for structured extraction tasks.
    """

    def __init__(self, api_key: str, model: str = _MODEL_ID) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Anthropic API key must not be empty.")
        self._model_id = model
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return f"Claude ({self._model_id})"

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Call the Anthropic messages endpoint and return text."""
        kwargs: dict = {
            "model": self._model_id,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)
            # response.content is a list of ContentBlock objects
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
        except AuthenticationError as exc:
            raise RuntimeError(
                "Anthropic authentication failed — check your API key."
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError(
                "Anthropic rate limit reached — please wait and try again."
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                f"Anthropic connection error — check your network: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc
