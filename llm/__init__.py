"""LLM provider abstraction package.

Public surface:
    LLMProvider  – abstract base class (llm.base)
    get_provider – factory function (llm.factory)

Agents should only import from this package, never from individual provider
modules, to keep the provider-swapping guarantee intact.
"""

from llm.base import LLMProvider
from llm.factory import get_provider

__all__ = ["LLMProvider", "get_provider"]
