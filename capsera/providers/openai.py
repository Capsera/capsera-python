"""
OpenAI-specific response parser.

Used by the interceptor to extract token counts from OpenAI ChatCompletion
objects in a single, testable place.
"""
from __future__ import annotations

from typing import Any

from .._usage import cached_is_included, openai_cached_tokens, split_cached_input
from .base import ProviderParser, register


class OpenAIParser(ProviderParser):
    provider_name = "openai"

    def parse_response(self, response: Any, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Extract usage fields from an ``openai.types.chat.ChatCompletion`` object.

        OpenAI usage fields:
            response.usage.prompt_tokens             (includes cached tokens)
            response.usage.completion_tokens
            response.usage.prompt_tokens_details.cached_tokens
        """
        usage = response.usage
        input_tokens, cache_read_tokens = split_cached_input(
            getattr(usage, "prompt_tokens", 0),
            openai_cached_tokens(usage),
            included=cached_is_included(self.provider_name),
        )
        return {
            "model": getattr(response, "model", request_kwargs.get("model", "unknown")),
            "input_tokens": input_tokens,
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "cache_read_tokens": cache_read_tokens,
            # No write premium: OpenAI caching is automatic and unbilled.
            "cache_write_tokens": 0,
        }


register(OpenAIParser())
