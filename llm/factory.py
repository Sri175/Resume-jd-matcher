"""Factory for instantiating LLM providers by name.

Usage::

    from llm.factory import get_provider

    llm = get_provider("gemini", api_key="AIza...")
    response = llm.generate("Hello!")

Provider name strings (case-insensitive):
    "gemini"  / "gemini 2.5 flash"  → GeminiProvider
    "openai"  / "gpt-4o-mini"       → OpenAIProvider
    "claude"  / "anthropic"         → ClaudeProvider
"""

from __future__ import annotations

from llm.base import LLMProvider

# Map of normalised name fragments → import path + class
# We import lazily to avoid loading all SDK dependencies at startup.
_PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "gemini": ("llm.gemini_provider", "GeminiProvider"),
    "openai": ("llm.openai_provider", "OpenAIProvider"),
    "claude": ("llm.claude_provider", "ClaudeProvider"),
    "anthropic": ("llm.claude_provider", "ClaudeProvider"),
    "gpt": ("llm.openai_provider", "OpenAIProvider"),
}


def get_provider(provider_name: str, api_key: str) -> LLMProvider:
    """Return a concrete LLMProvider for *provider_name*.

    Parameters
    ----------
    provider_name:
        A string identifying the desired provider.  Matching is
        case-insensitive; any string *containing* a known key works,
        e.g. "Gemini 2.5 Flash" matches via the "gemini" key.
    api_key:
        The user-supplied API key for this provider.

    Returns
    -------
    LLMProvider
        A ready-to-use provider instance.

    Raises
    ------
    ValueError
        If *provider_name* does not match any known provider.
    """
    normalised = provider_name.lower().strip()

    module_path: str | None = None
    class_name: str | None = None

    for key, (mod, cls) in _PROVIDER_MAP.items():
        if key in normalised:
            module_path, class_name = mod, cls
            break

    if module_path is None:
        supported = ", ".join(sorted({m for m, _ in _PROVIDER_MAP.values()}))
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Supported providers: {supported}"
        )

    # Lazy import so only the chosen provider's SDK is imported at runtime.
    import importlib
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)
    return provider_class(api_key=api_key)


def list_providers() -> list[str]:
    """Return human-readable provider labels for use in UI dropdowns."""
    return [
        "Gemini 2.5 Flash",
        "OpenAI GPT-4o-mini",
        "Claude (Haiku)",
    ]
