"""Abstract base class for all LLM providers.

Design contract
---------------
Every concrete provider must subclass LLMProvider and implement `generate`.
Agents import ONLY this class — never a specific SDK.  Swapping the underlying
model requires zero changes to any agent.

Usage (from agent code)::

    from llm.base import LLMProvider

    def run(llm: LLMProvider, prompt: str) -> str:
        return llm.generate(prompt, system="You are a helpful assistant.")
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Provider-agnostic interface for text generation.

    Parameters passed to the constructor are provider-specific (api_key,
    model name, etc.) and handled by each subclass.  The public surface is
    intentionally tiny: just `generate`.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Send *prompt* to the model and return the response text.

        Parameters
        ----------
        prompt:
            The user-facing instruction / question.
        system:
            Optional system-level instruction.  Providers that support a
            dedicated system role will use it; others prepend it to the prompt.

        Returns
        -------
        str
            The raw text content of the model's first response candidate.

        Raises
        ------
        RuntimeError
            Wraps any SDK-level error (auth failure, rate limit, etc.) with a
            human-readable message so callers can surface it in the UI without
            knowing which SDK is in use.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name shown in the UI / logs (e.g. 'Gemini 2.5 Flash')."""

    def test_connection(self) -> tuple[bool, str]:
        """Make a trivial call to verify the API key is valid.

        Returns
        -------
        (True, "Connected to <provider_name>")   on success
        (False, "<error message>")               on failure
        """
        try:
            response = self.generate("Reply with the single word: OK")
            return True, f"✅ Connected to {self.provider_name} — model replied: {response.strip()[:40]}"
        except Exception as exc:  # noqa: BLE001
            return False, f"❌ {self.provider_name} error: {exc}"
