"""
Base class for provider-specific response parsers.

Each provider module (anthropic.py, openai.py, google.py) implements
``ProviderParser`` and registers itself in the PROVIDERS registry so the
interceptor can dispatch to the correct parser at runtime.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProviderParser(ABC):
    """Extract token counts and model information from a raw API response."""

    #: Canonical provider name stored in TokenEvent.provider
    provider_name: str = ""

    @abstractmethod
    def parse_response(self, response: Any, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Return a dict with keys:
            model, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens
        """
        ...


# Registry populated by each provider module at import time.
PROVIDERS: dict[str, ProviderParser] = {}


def register(parser: ProviderParser) -> None:
    PROVIDERS[parser.provider_name] = parser
