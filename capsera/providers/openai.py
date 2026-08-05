"""
OpenAI-specific response parser.

Used by the interceptor to extract token counts from OpenAI ChatCompletion
objects in a single, testable place.
"""
from __future__ import annotations

from typing import Any

from .base import ProviderParser, register


class OpenAIParser(ProviderParser):
    provider_name = "openai"

    def parse_response(self, response: Any, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Extract usage fields from an ``openai.types.chat.ChatCompletion`` object.

        OpenAI usage fields:
            response.usage.prompt_tokens
            response.usage.completion_tokens
        """
        usage = response.usage
        return {
            "model": getattr(response, "model", request_kwargs.get("model", "unknown")),
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }


register(OpenAIParser())
