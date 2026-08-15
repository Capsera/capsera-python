"""
Prompt analysis engine for the Capsera SDK.

Extracts structural metadata from LLM request kwargs — hashes, token
estimates, message counts, tool result tracking, context window utilization,
and conversation turn depth — without ever storing or transmitting prompt
content.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ._config import Config
from ._models import PromptMetadata

CONTEXT_WINDOW_SIZES: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
}

_DEFAULT_CONTEXT_WINDOW = 128_000


def _get_context_window(model: str) -> int:
    if model in CONTEXT_WINDOW_SIZES:
        return CONTEXT_WINDOW_SIZES[model]
    for prefix, size in CONTEXT_WINDOW_SIZES.items():
        if model.startswith(prefix):
            return size
    return _DEFAULT_CONTEXT_WINDOW


class PromptAnalyzer:
    """Extracts prompt metadata from request kwargs. All operations are fast (<1ms)."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._prompt_size_threshold = config.prompt_size_threshold

    def analyze(self, kwargs: dict[str, Any], provider: str) -> PromptMetadata:
        system_text = self._extract_system_prompt(kwargs, provider)
        messages = self._extract_messages(kwargs, provider)

        system_hash = self._hash_text(system_text) if system_text else None
        system_tokens = self._estimate_tokens(system_text) if system_text else 0
        conv_tokens = self._estimate_conversation_tokens(messages)
        message_count = len(messages)
        few_shot_pairs, has_few_shot = self._detect_few_shot(messages)
        tool_result_tokens = self._count_tool_result_tokens(messages, provider)
        conversation_turns = self._count_conversation_turns(messages)

        model = kwargs.get("model", "")
        context_window = _get_context_window(model)
        total_est = system_tokens + conv_tokens
        utilization = min(total_est / context_window, 1.0) if context_window > 0 else 0.0

        metadata = PromptMetadata(
            system_prompt_hash=system_hash,
            system_prompt_tokens=system_tokens,
            message_count=message_count,
            conversation_tokens=conv_tokens,
            has_few_shot=has_few_shot,
            few_shot_pairs=few_shot_pairs,
            tool_result_tokens=tool_result_tokens,
            conversation_turns=conversation_turns,
            context_window_utilization=round(utilization, 4),
        )

        if self._config.on_oversized_prompt and total_est > self._prompt_size_threshold:
            try:
                self._config.on_oversized_prompt(total_est, self._prompt_size_threshold)
            except Exception:
                pass

        return metadata

    # ------------------------------------------------------------------ #
    # Provider-specific extraction
    # ------------------------------------------------------------------ #

    def _extract_system_prompt(
        self, kwargs: dict[str, Any], provider: str
    ) -> str | None:
        if provider == "anthropic":
            system = kwargs.get("system")
            if system is None:
                return None
            if isinstance(system, list):
                parts = []
                for block in system:
                    if isinstance(block, dict):
                        parts.append(block.get("text", ""))
                    else:
                        parts.append(str(block))
                text = " ".join(parts)
                return text if text.strip() else None
            text = str(system)
            return text if text.strip() else None

        if provider == "openai":
            for msg in kwargs.get("messages", []):
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        text = " ".join(
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict)
                        )
                        return text if text.strip() else None
                    text = str(content)
                    return text if text.strip() else None

        return None

    def _extract_messages(
        self, kwargs: dict[str, Any], provider: str
    ) -> list[dict[str, Any]]:
        messages = kwargs.get("messages", [])
        if provider == "openai":
            return [m for m in messages if m.get("role") != "system"]
        return list(messages)

    # ------------------------------------------------------------------ #
    # Token estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_tokens(text: str | None) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _estimate_conversation_tokens(
        self, messages: list[dict[str, Any]]
    ) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += self._estimate_tokens(part.get("text", ""))
        return total

    # ------------------------------------------------------------------ #
    # Hashing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash_text(text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # Few-shot detection (improved: requires short assistant replies)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_few_shot(
        messages: list[dict[str, Any]],
    ) -> tuple[int, bool]:
        if len(messages) < 3:
            return 0, False
        roles = [m.get("role") for m in messages]

        pairs = 0
        short_assistant_count = 0
        for i in range(0, len(roles) - 1, 2):
            if roles[i] == "user" and i + 1 < len(roles) and roles[i + 1] == "assistant":
                pairs += 1
                content = messages[i + 1].get("content", "")
                if isinstance(content, str) and len(content) < 500:
                    short_assistant_count += 1
            else:
                break

        if pairs < 2 or roles[-1] != "user":
            return 0, False

        # Require most assistant replies to be short — real few-shot examples
        # have concise target outputs, while multi-turn conversations have
        # longer assistant replies.
        if short_assistant_count < pairs * 0.7:
            return 0, False

        return pairs, True

    # ------------------------------------------------------------------ #
    # Tool result token tracking
    # ------------------------------------------------------------------ #

    def _count_tool_result_tokens(
        self, messages: list[dict[str, Any]], provider: str
    ) -> int:
        total = 0
        for msg in messages:
            role = msg.get("role", "")
            if provider == "anthropic" and role == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            inner = block.get("content", "")
                            if isinstance(inner, str):
                                total += self._estimate_tokens(inner)
                            elif isinstance(inner, list):
                                for part in inner:
                                    if isinstance(part, dict):
                                        total += self._estimate_tokens(part.get("text", ""))
            elif provider == "openai" and role == "tool":
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += self._estimate_tokens(content)
        return total

    # ------------------------------------------------------------------ #
    # Conversation turn counting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_conversation_turns(messages: list[dict[str, Any]]) -> int:
        turns = 0
        prev_role = None
        for msg in messages:
            role = msg.get("role", "")
            if role in ("user", "assistant") and role != prev_role:
                turns += 1
                prev_role = role
            elif role == "tool":
                prev_role = role
        return turns
