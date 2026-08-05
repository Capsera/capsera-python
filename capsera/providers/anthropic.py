"""
Anthropic-specific response parser.

Used by the interceptor to extract token counts from Anthropic Message
objects in a single, testable place.
"""
from __future__ import annotations

from typing import Any

from .base import ProviderParser, register


class AnthropicParser(ProviderParser):
    provider_name = "anthropic"

    def parse_response(self, response: Any, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Extract usage fields from an ``anthropic.types.Message`` object.

        Anthropic usage fields:
            response.usage.input_tokens
            response.usage.output_tokens
            response.usage.cache_read_input_tokens      (may be absent)
            response.usage.cache_creation_input_tokens  (may be absent)
        """
        usage = response.usage
        return {
            "model": getattr(response, "model", request_kwargs.get("model", "unknown")),
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0),
            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        }


# Register singleton at module import
register(AnthropicParser())
