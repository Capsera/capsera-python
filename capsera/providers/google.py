"""
Google Gemini-specific response parser.

Handles responses from ``google-generativeai`` (``genai``) SDK.
"""
from __future__ import annotations

from typing import Any

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
        return {
            "model": request_kwargs.get("model", "gemini-pro"),
            "input_tokens": getattr(meta, "prompt_token_count", 0) if meta else 0,
            "output_tokens": getattr(meta, "candidates_token_count", 0) if meta else 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }


register(GoogleParser())
