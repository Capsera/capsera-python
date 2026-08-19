"""
Google Gemini-specific response parser.

Handles responses from ``google-generativeai`` (``genai``) SDK.
"""
from __future__ import annotations

from typing import Any

from .._usage import cached_is_included, gemini_cached_tokens, split_cached_input
from .base import ProviderParser, register


class GoogleParser(ProviderParser):
    provider_name = "google"

    def parse_response(self, response: Any, request_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Extract usage from a Gemini ``GenerateContentResponse``.

        The field path for the Google SDK:
            response.usage_metadata.prompt_token_count
            response.usage_metadata.candidates_token_count
        """
        meta = getattr(response, "usage_metadata", None)
        input_tokens, cache_read_tokens = split_cached_input(
            getattr(meta, "prompt_token_count", 0) if meta else 0,
            gemini_cached_tokens(meta),
            included=cached_is_included(self.provider_name),
        )
        return {
            "model": request_kwargs.get("model", "gemini-pro"),
            "input_tokens": input_tokens,
            "output_tokens": getattr(meta, "candidates_token_count", 0) if meta else 0,
            "cache_read_tokens": cache_read_tokens,
            # Explicit context caching is billed by storage-hour, not per
            # written token, so there is no per-call write count.
            "cache_write_tokens": 0,
        }


register(GoogleParser())
