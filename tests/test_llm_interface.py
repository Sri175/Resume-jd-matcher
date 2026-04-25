"""Tests for the LLM provider abstraction layer.

All network calls are mocked — no real API key or internet access needed.
Uses patch.object on the already-imported module-level names so mocks
correctly intercept calls regardless of import order.

Tests verify:
  1. The abstract interface contract (generate / provider_name / test_connection)
  2. That each provider correctly maps system + prompt to its SDK's call shape
  3. That the factory resolves provider names correctly (including fuzzy matching)
  4. That errors from each SDK are wrapped in RuntimeError
  5. That agents depend ONLY on LLMProvider — no SDK-specific imports in agent files
"""

from __future__ import annotations

import pathlib
import sys
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# 1. Abstract interface — cannot instantiate LLMProvider directly
# ---------------------------------------------------------------------------

class TestAbstractInterface:
    def test_cannot_instantiate_abstract_class(self):
        from llm.base import LLMProvider
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_concrete_must_implement_generate(self):
        from llm.base import LLMProvider

        class BrokenProvider(LLMProvider):
            @property
            def provider_name(self):
                return "broken"
            # Missing generate() — should raise

        with pytest.raises(TypeError):
            BrokenProvider()

    def test_concrete_must_implement_provider_name(self):
        from llm.base import LLMProvider

        class BrokenProvider(LLMProvider):
            def generate(self, prompt, system=None):
                return ""
            # Missing provider_name — should raise

        with pytest.raises(TypeError):
            BrokenProvider()


# ---------------------------------------------------------------------------
# 2. Minimal concrete provider for interface testing
# ---------------------------------------------------------------------------

from llm.base import LLMProvider as _LLMProvider


class _FakeProvider(_LLMProvider):
    def generate(self, prompt: str, system=None) -> str:
        return f"echo:{prompt}"

    @property
    def provider_name(self) -> str:
        return "FakeProvider"


class TestMinimalConcreteProvider:
    def setup_method(self):
        self.provider = _FakeProvider()

    def test_generate_returns_string(self):
        assert isinstance(self.provider.generate("hi"), str)

    def test_provider_name_returns_string(self):
        assert isinstance(self.provider.provider_name, str)

    def test_test_connection_success(self):
        ok, msg = self.provider.test_connection()
        assert ok is True
        assert "FakeProvider" in msg

    def test_test_connection_failure_on_exception(self):
        class FailingProvider(_LLMProvider):
            @property
            def provider_name(self):
                return "Failing"

            def generate(self, prompt, system=None):
                raise RuntimeError("simulated failure")

        fp = FailingProvider()
        ok, msg = fp.test_connection()
        assert ok is False
        assert "simulated failure" in msg


# ---------------------------------------------------------------------------
# 3. Gemini provider — SDK call shape (patch.object on imported module ref)
# ---------------------------------------------------------------------------

import llm.gemini_provider as _gp_mod


class TestGeminiProvider:
    def _make_mock_client(self, text="Hello from Gemini"):
        """Build a mock genai module + client that returns *text*."""
        mock_genai_mod = MagicMock()  # replaces `from google import genai`
        mock_client = MagicMock()    # replaces genai.Client(...)
        mock_response = MagicMock()
        mock_response.text = text
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_mod.Client.return_value = mock_client
        return mock_genai_mod, mock_client

    def test_rejects_empty_api_key(self):
        mock_genai_mod, _ = self._make_mock_client()
        with patch.object(_gp_mod, "genai", mock_genai_mod):
            with pytest.raises(ValueError, match="API key"):
                _gp_mod.GeminiProvider(api_key="")

    def test_generate_without_system(self):
        mock_genai_mod, mock_client = self._make_mock_client("Hello from Gemini")
        with patch.object(_gp_mod, "genai", mock_genai_mod):
            provider = _gp_mod.GeminiProvider(api_key="fake")
            result = provider.generate("Tell me something")
        assert result == "Hello from Gemini"
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["contents"] == "Tell me something"
        # No system → config should be None
        assert call_kwargs.get("config") is None

    def test_generate_with_system_passes_config(self):
        mock_genai_mod, mock_client = self._make_mock_client("Gemini with system")
        mock_types = MagicMock()
        with patch.object(_gp_mod, "genai", mock_genai_mod), \
             patch.object(_gp_mod, "types", mock_types):
            provider = _gp_mod.GeminiProvider(api_key="fake")
            result = provider.generate("Prompt", system="You are an expert")
        assert result == "Gemini with system"
        # types.GenerateContentConfig should have been called with system_instruction
        mock_types.GenerateContentConfig.assert_called_once_with(
            system_instruction="You are an expert"
        )

    def test_generate_without_system_does_not_create_config(self):
        mock_genai_mod, mock_client = self._make_mock_client()
        mock_types = MagicMock()
        with patch.object(_gp_mod, "genai", mock_genai_mod), \
             patch.object(_gp_mod, "types", mock_types):
            provider = _gp_mod.GeminiProvider(api_key="fake")
            provider.generate("No system")
        # types.GenerateContentConfig should NOT be called
        mock_types.GenerateContentConfig.assert_not_called()

    def test_sdk_exception_wrapped_as_runtime_error(self):
        mock_genai_mod, mock_client = self._make_mock_client()
        mock_client.models.generate_content.side_effect = Exception("quota exceeded")
        with patch.object(_gp_mod, "genai", mock_genai_mod):
            provider = _gp_mod.GeminiProvider(api_key="fake")
            with pytest.raises(RuntimeError, match="Gemini API error"):
                provider.generate("prompt")

    def test_provider_name_contains_gemini(self):
        mock_genai_mod, _ = self._make_mock_client()
        with patch.object(_gp_mod, "genai", mock_genai_mod):
            provider = _gp_mod.GeminiProvider(api_key="fake")
        assert "gemini" in provider.provider_name.lower()



# ---------------------------------------------------------------------------
# 4. OpenAI provider — SDK call shape
# ---------------------------------------------------------------------------

import llm.openai_provider as _op_mod


class TestOpenAIProvider:
    def _make_mock_client(self, text="OpenAI reply"):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = text
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_rejects_empty_api_key(self):
        mock_client = self._make_mock_client()
        with patch.object(_op_mod, "OpenAI", return_value=mock_client):
            with pytest.raises(ValueError, match="API key"):
                _op_mod.OpenAIProvider(api_key="")

    def test_generate_sends_user_message(self):
        mock_client = self._make_mock_client()
        with patch.object(_op_mod, "OpenAI", return_value=mock_client):
            provider = _op_mod.OpenAIProvider(api_key="sk-fake")
            result = provider.generate("Hello OpenAI")
        assert result == "OpenAI reply"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        assert any(m.get("role") == "user" for m in messages)

    def test_generate_with_system_sends_system_message(self):
        mock_client = self._make_mock_client()
        with patch.object(_op_mod, "OpenAI", return_value=mock_client):
            provider = _op_mod.OpenAIProvider(api_key="sk-fake")
            provider.generate("Prompt", system="Be helpful")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        assert any(m.get("role") == "system" for m in messages)

    def test_generate_without_system_omits_system_message(self):
        mock_client = self._make_mock_client()
        with patch.object(_op_mod, "OpenAI", return_value=mock_client):
            provider = _op_mod.OpenAIProvider(api_key="sk-fake")
            provider.generate("No system")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        assert not any(m.get("role") == "system" for m in messages)

    def test_sdk_exception_wrapped_as_runtime_error(self):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.create.side_effect = Exception("rate limit")
        with patch.object(_op_mod, "OpenAI", return_value=mock_client):
            provider = _op_mod.OpenAIProvider(api_key="sk-fake")
            with pytest.raises(RuntimeError):
                provider.generate("prompt")

    def test_provider_name_contains_openai_or_gpt(self):
        mock_client = self._make_mock_client()
        with patch.object(_op_mod, "OpenAI", return_value=mock_client):
            provider = _op_mod.OpenAIProvider(api_key="sk-fake")
        name = provider.provider_name.lower()
        assert "openai" in name or "gpt" in name


# ---------------------------------------------------------------------------
# 5. Claude provider — SDK call shape
# ---------------------------------------------------------------------------

import llm.claude_provider as _cp_mod


class TestClaudeProvider:
    def _make_mock_client(self, text="Claude reply"):
        mock_client = MagicMock()
        mock_response = MagicMock()
        content_block = MagicMock()
        content_block.text = text
        mock_response.content = [content_block]
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_rejects_empty_api_key(self):
        mock_client = self._make_mock_client()
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            with pytest.raises(ValueError, match="API key"):
                _cp_mod.ClaudeProvider(api_key="")

    def test_generate_returns_text_from_content_blocks(self):
        mock_client = self._make_mock_client("Claude reply")
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            provider = _cp_mod.ClaudeProvider(api_key="sk-ant-fake")
            result = provider.generate("Hello Claude")
        assert result == "Claude reply"

    def test_generate_concatenates_multiple_content_blocks(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        b1, b2 = MagicMock(), MagicMock()
        b1.text = "Hello "
        b2.text = "Claude"
        mock_response.content = [b1, b2]
        mock_client.messages.create.return_value = mock_response
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            provider = _cp_mod.ClaudeProvider(api_key="sk-ant-fake")
            result = provider.generate("Hello")
        assert result == "Hello Claude"

    def test_system_prompt_passed_as_kwarg(self):
        mock_client = self._make_mock_client()
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            provider = _cp_mod.ClaudeProvider(api_key="sk-ant-fake")
            provider.generate("Prompt", system="You are expert")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "You are expert"

    def test_generate_without_system_omits_system_kwarg(self):
        mock_client = self._make_mock_client()
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            provider = _cp_mod.ClaudeProvider(api_key="sk-ant-fake")
            provider.generate("No system here")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    def test_sdk_exception_wrapped_as_runtime_error(self):
        mock_client = self._make_mock_client()
        mock_client.messages.create.side_effect = Exception("auth failed")
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            provider = _cp_mod.ClaudeProvider(api_key="sk-ant-fake")
            with pytest.raises(RuntimeError):
                provider.generate("prompt")

    def test_provider_name_contains_claude(self):
        mock_client = self._make_mock_client()
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            provider = _cp_mod.ClaudeProvider(api_key="sk-ant-fake")
        assert "claude" in provider.provider_name.lower()


# ---------------------------------------------------------------------------
# 6. Factory — provider resolution
# ---------------------------------------------------------------------------

class TestFactory:
    def test_factory_resolves_gemini(self):
        mock_genai_mod, _ = _make_gemini_mock()
        with patch.object(_gp_mod, "genai", mock_genai_mod):
            from llm.factory import get_provider
            provider = get_provider("Gemini 2.5 Flash", api_key="fake")
        assert "gemini" in provider.provider_name.lower()

    def test_factory_resolves_openai(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        with patch.object(_op_mod, "OpenAI", return_value=mock_client):
            from llm.factory import get_provider
            provider = get_provider("OpenAI GPT-4o-mini", api_key="sk-fake")
        name = provider.provider_name.lower()
        assert "openai" in name or "gpt" in name

    def test_factory_resolves_claude(self):
        mock_client = MagicMock()
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            from llm.factory import get_provider
            provider = get_provider("Claude (Haiku)", api_key="sk-ant-fake")
        assert "claude" in provider.provider_name.lower()

    def test_factory_resolves_anthropic_alias(self):
        mock_client = MagicMock()
        with patch.object(_cp_mod, "anthropic") as mock_ant:
            mock_ant.Anthropic.return_value = mock_client
            from llm.factory import get_provider
            provider = get_provider("anthropic", api_key="sk-ant-fake")
        assert "claude" in provider.provider_name.lower()

    def test_factory_raises_on_unknown_provider(self):
        from llm.factory import get_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("SomeOtherLLM", api_key="key")

    def test_list_providers_returns_three_entries(self):
        from llm.factory import list_providers
        providers = list_providers()
        assert len(providers) == 3
        assert any("Gemini" in p for p in providers)
        assert any("OpenAI" in p for p in providers)
        assert any("Claude" in p for p in providers)


# ---------------------------------------------------------------------------
# 7. Provider isolation — agents must not import SDK modules directly
# ---------------------------------------------------------------------------

class TestProviderIsolation:
    """Verify that agent modules do not directly import any LLM SDK."""

    AGENT_MODULES = [
        "agents/extractor.py",
        "agents/matcher.py",
        "agents/job_finder.py",
    ]
    FORBIDDEN_IMPORTS = [
        "google.generativeai",
        "import openai",
        "import anthropic",
    ]

    @pytest.mark.parametrize("agent_file", AGENT_MODULES)
    @pytest.mark.parametrize("sdk_import", FORBIDDEN_IMPORTS)
    def test_agent_does_not_import_sdk(self, agent_file, sdk_import):
        root = pathlib.Path(__file__).parent.parent
        full_path = root / agent_file
        if not full_path.exists():
            pytest.skip(f"{agent_file} not yet created (Phase 3)")

        source = full_path.read_text(encoding="utf-8")
        assert sdk_import not in source, (
            f"{agent_file} must not import '{sdk_import}' directly. "
            "Use the LLMProvider interface instead."
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_gemini_mock(text="ok"):
    mock_genai_mod = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = text
    mock_client.models.generate_content.return_value = mock_response
    mock_genai_mod.Client.return_value = mock_client
    return mock_genai_mod, mock_client
