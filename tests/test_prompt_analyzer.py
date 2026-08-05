"""Tests for the prompt analysis engine."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from capsera._config import Config
from capsera._models import PromptMetadata
from capsera._prompt_analyzer import PromptAnalyzer, _get_context_window


def _make_analyzer(**overrides) -> PromptAnalyzer:
    cfg = Config(enable_prompt_analysis=True, **overrides)
    return PromptAnalyzer(cfg)


# ------------------------------------------------------------------ #
# System prompt extraction
# ------------------------------------------------------------------ #


class TestSystemPromptExtraction:
    def test_anthropic_string_system(self):
        analyzer = _make_analyzer()
        text = analyzer._extract_system_prompt(
            {"system": "You are a helpful assistant."}, "anthropic"
        )
        assert text == "You are a helpful assistant."

    def test_anthropic_content_block_system(self):
        analyzer = _make_analyzer()
        text = analyzer._extract_system_prompt(
            {"system": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]},
            "anthropic",
        )
        assert text == "Hello World"

    def test_openai_system_message(self):
        analyzer = _make_analyzer()
        text = analyzer._extract_system_prompt(
            {
                "messages": [
                    {"role": "system", "content": "You are a bot."},
                    {"role": "user", "content": "Hi"},
                ]
            },
            "openai",
        )
        assert text == "You are a bot."

    def test_no_system_prompt(self):
        analyzer = _make_analyzer()
        assert analyzer._extract_system_prompt({"messages": []}, "anthropic") is None
        assert analyzer._extract_system_prompt(
            {"messages": [{"role": "user", "content": "Hi"}]}, "openai"
        ) is None

    def test_empty_system_prompt(self):
        analyzer = _make_analyzer()
        assert analyzer._extract_system_prompt({"system": ""}, "anthropic") is None
        assert analyzer._extract_system_prompt({"system": "   "}, "anthropic") is None

    def test_openai_content_block_system(self):
        analyzer = _make_analyzer()
        text = analyzer._extract_system_prompt(
            {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "Instructions here"}]},
                    {"role": "user", "content": "Hi"},
                ]
            },
            "openai",
        )
        assert text == "Instructions here"

    def test_unknown_provider(self):
        analyzer = _make_analyzer()
        assert analyzer._extract_system_prompt({"system": "test"}, "google") is None


# ------------------------------------------------------------------ #
# Message extraction
# ------------------------------------------------------------------ #


class TestMessageExtraction:
    def test_anthropic_messages(self):
        analyzer = _make_analyzer()
        msgs = analyzer._extract_messages(
            {"messages": [{"role": "user", "content": "Hi"}]}, "anthropic"
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_openai_filters_system(self):
        analyzer = _make_analyzer()
        msgs = analyzer._extract_messages(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            },
            "openai",
        )
        assert len(msgs) == 2
        assert all(m["role"] != "system" for m in msgs)

    def test_empty_messages(self):
        analyzer = _make_analyzer()
        assert analyzer._extract_messages({}, "anthropic") == []
        assert analyzer._extract_messages({}, "openai") == []


# ------------------------------------------------------------------ #
# Token estimation
# ------------------------------------------------------------------ #


class TestTokenEstimation:
    def test_known_length(self):
        assert PromptAnalyzer._estimate_tokens("a" * 400) == 100

    def test_short_string(self):
        assert PromptAnalyzer._estimate_tokens("hi") == 1

    def test_empty(self):
        assert PromptAnalyzer._estimate_tokens("") == 0
        assert PromptAnalyzer._estimate_tokens(None) == 0

    def test_unicode(self):
        result = PromptAnalyzer._estimate_tokens("hello world")
        assert result > 0


# ------------------------------------------------------------------ #
# Hashing
# ------------------------------------------------------------------ #


class TestPromptHashing:
    def test_deterministic(self):
        h1 = PromptAnalyzer._hash_text("You are a helpful assistant.")
        h2 = PromptAnalyzer._hash_text("You are a helpful assistant.")
        assert h1 == h2

    def test_whitespace_normalization(self):
        h1 = PromptAnalyzer._hash_text("You  are   a    helpful   assistant.")
        h2 = PromptAnalyzer._hash_text("You are a helpful assistant.")
        assert h1 == h2

    def test_case_normalization(self):
        h1 = PromptAnalyzer._hash_text("You Are A Helpful Assistant.")
        h2 = PromptAnalyzer._hash_text("you are a helpful assistant.")
        assert h1 == h2

    def test_different_prompts_different_hashes(self):
        h1 = PromptAnalyzer._hash_text("You are a helpful assistant.")
        h2 = PromptAnalyzer._hash_text("You are a coding assistant.")
        assert h1 != h2

    def test_hash_is_sha256(self):
        text = "test prompt"
        result = PromptAnalyzer._hash_text(text)
        expected = hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
        assert result == expected
        assert len(result) == 64


# ------------------------------------------------------------------ #
# Few-shot detection (improved: requires short assistant replies)
# ------------------------------------------------------------------ #


class TestFewShotDetection:
    def test_single_turn_no_few_shot(self):
        msgs = [{"role": "user", "content": "Hello"}]
        pairs, detected = PromptAnalyzer._detect_few_shot(msgs)
        assert pairs == 0
        assert detected is False

    def test_two_pairs_short_replies_detected(self):
        msgs = [
            {"role": "user", "content": "Example 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Example 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Real question"},
        ]
        pairs, detected = PromptAnalyzer._detect_few_shot(msgs)
        assert pairs == 2
        assert detected is True

    def test_two_pairs_long_replies_not_detected(self):
        long_reply = "x" * 600
        msgs = [
            {"role": "user", "content": "Example 1"},
            {"role": "assistant", "content": long_reply},
            {"role": "user", "content": "Example 2"},
            {"role": "assistant", "content": long_reply},
            {"role": "user", "content": "Real question"},
        ]
        pairs, detected = PromptAnalyzer._detect_few_shot(msgs)
        assert pairs == 0
        assert detected is False

    def test_one_pair_not_detected(self):
        msgs = [
            {"role": "user", "content": "Example 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Real question"},
        ]
        pairs, detected = PromptAnalyzer._detect_few_shot(msgs)
        assert pairs == 0
        assert detected is False

    def test_assistant_only_not_detected(self):
        msgs = [
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A2"},
        ]
        pairs, detected = PromptAnalyzer._detect_few_shot(msgs)
        assert pairs == 0
        assert detected is False

    def test_three_pairs_short_replies(self):
        msgs = [
            {"role": "user", "content": "E1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "E2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "E3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Real question"},
        ]
        pairs, detected = PromptAnalyzer._detect_few_shot(msgs)
        assert pairs == 3
        assert detected is True

    def test_empty_messages(self):
        pairs, detected = PromptAnalyzer._detect_few_shot([])
        assert pairs == 0
        assert detected is False


# ------------------------------------------------------------------ #
# Tool result token tracking
# ------------------------------------------------------------------ #


class TestToolResultTokens:
    def test_anthropic_tool_result(self):
        analyzer = _make_analyzer()
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "x" * 400},
            ]},
        ]
        tokens = analyzer._count_tool_result_tokens(msgs, "anthropic")
        assert tokens == 100

    def test_anthropic_tool_result_with_blocks(self):
        analyzer = _make_analyzer()
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": [
                    {"type": "text", "text": "x" * 200},
                    {"type": "text", "text": "y" * 200},
                ]},
            ]},
        ]
        tokens = analyzer._count_tool_result_tokens(msgs, "anthropic")
        assert tokens == 100

    def test_openai_tool_role(self):
        analyzer = _make_analyzer()
        msgs = [
            {"role": "tool", "content": "z" * 800},
        ]
        tokens = analyzer._count_tool_result_tokens(msgs, "openai")
        assert tokens == 200

    def test_no_tool_results(self):
        analyzer = _make_analyzer()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert analyzer._count_tool_result_tokens(msgs, "anthropic") == 0
        assert analyzer._count_tool_result_tokens(msgs, "openai") == 0


# ------------------------------------------------------------------ #
# Conversation turn counting
# ------------------------------------------------------------------ #


class TestConversationTurns:
    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "Hi"}]
        assert PromptAnalyzer._count_conversation_turns(msgs) == 1

    def test_user_assistant_pair(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        assert PromptAnalyzer._count_conversation_turns(msgs) == 2

    def test_multi_turn_with_tool(self):
        msgs = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": "Let me search"},
            {"role": "tool", "content": "results"},
            {"role": "assistant", "content": "Here are results"},
            {"role": "user", "content": "Thanks"},
        ]
        assert PromptAnalyzer._count_conversation_turns(msgs) == 4

    def test_empty_messages(self):
        assert PromptAnalyzer._count_conversation_turns([]) == 0


# ------------------------------------------------------------------ #
# Context window utilization
# ------------------------------------------------------------------ #


class TestContextWindowUtilization:
    def test_known_model(self):
        assert _get_context_window("gpt-4o") == 128_000
        assert _get_context_window("claude-sonnet-4-6") == 200_000

    def test_prefix_match(self):
        assert _get_context_window("gpt-4o-2024-05-13") == 128_000

    def test_unknown_model_default(self):
        assert _get_context_window("some-unknown-model") == 128_000

    def test_utilization_in_analyze(self):
        analyzer = _make_analyzer()
        meta = analyzer.analyze(
            {
                "model": "gpt-4",  # 8192 context window
                "system": "x" * 4000,  # ~1000 tokens
                "messages": [{"role": "user", "content": "y" * 4000}],  # ~1000 tokens
            },
            "anthropic",
        )
        assert meta.context_window_utilization > 0
        assert meta.context_window_utilization <= 1.0


# ------------------------------------------------------------------ #
# Oversized prompt callback
# ------------------------------------------------------------------ #


class TestOversizedCallback:
    def test_callback_fired_above_threshold(self):
        cb = MagicMock()
        analyzer = _make_analyzer(prompt_size_threshold=10, on_oversized_prompt=cb)
        analyzer.analyze(
            {"model": "test", "system": "a" * 100, "messages": [{"role": "user", "content": "hello"}]},
            "anthropic",
        )
        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] > 10
        assert args[1] == 10

    def test_callback_not_fired_below_threshold(self):
        cb = MagicMock()
        analyzer = _make_analyzer(prompt_size_threshold=10000, on_oversized_prompt=cb)
        analyzer.analyze(
            {"model": "test", "system": "short", "messages": [{"role": "user", "content": "hi"}]},
            "anthropic",
        )
        cb.assert_not_called()

    def test_callback_exception_swallowed(self):
        def bad_callback(actual, threshold):
            raise RuntimeError("boom")

        analyzer = _make_analyzer(prompt_size_threshold=1, on_oversized_prompt=bad_callback)
        result = analyzer.analyze(
            {"model": "test", "system": "some system prompt", "messages": [{"role": "user", "content": "hi"}]},
            "anthropic",
        )
        assert isinstance(result, PromptMetadata)


# ------------------------------------------------------------------ #
# Full analyze() integration
# ------------------------------------------------------------------ #


class TestAnalyzeIntegration:
    def test_anthropic_full(self):
        analyzer = _make_analyzer()
        meta = analyzer.analyze(
            {
                "model": "claude-sonnet-4-6",
                "system": "You are a helpful assistant.",
                "messages": [
                    {"role": "user", "content": "Hello world"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
            "anthropic",
        )
        assert meta.system_prompt_hash is not None
        assert meta.system_prompt_tokens > 0
        assert meta.message_count == 3
        assert meta.conversation_tokens > 0
        assert meta.has_few_shot is False
        assert meta.conversation_turns == 3
        assert meta.context_window_utilization > 0

    def test_openai_full(self):
        analyzer = _make_analyzer()
        meta = analyzer.analyze(
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            },
            "openai",
        )
        assert meta.system_prompt_hash is not None
        assert meta.system_prompt_tokens > 0
        assert meta.message_count == 1  # system filtered out
        assert meta.conversation_tokens > 0
        assert meta.has_few_shot is False
        assert meta.conversation_turns == 1

    def test_no_system_prompt(self):
        analyzer = _make_analyzer()
        meta = analyzer.analyze(
            {"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            "anthropic",
        )
        assert meta.system_prompt_hash is None
        assert meta.system_prompt_tokens == 0
        assert meta.message_count == 1

    def test_few_shot_detected(self):
        analyzer = _make_analyzer()
        meta = analyzer.analyze(
            {
                "model": "test",
                "messages": [
                    {"role": "user", "content": "classify: cat"},
                    {"role": "assistant", "content": "animal"},
                    {"role": "user", "content": "classify: car"},
                    {"role": "assistant", "content": "vehicle"},
                    {"role": "user", "content": "classify: dog"},
                ],
            },
            "anthropic",
        )
        assert meta.has_few_shot is True
        assert meta.few_shot_pairs == 2

    def test_tool_results_tracked(self):
        analyzer = _make_analyzer()
        meta = analyzer.analyze(
            {
                "model": "claude-sonnet-4-6",
                "messages": [
                    {"role": "user", "content": "search for X"},
                    {"role": "assistant", "content": "searching..."},
                    {"role": "user", "content": [
                        {"type": "tool_result", "content": "x" * 2000},
                    ]},
                    {"role": "assistant", "content": "found it"},
                ],
            },
            "anthropic",
        )
        assert meta.tool_result_tokens == 500
        assert meta.conversation_turns == 4


# ------------------------------------------------------------------ #
# Disabled analysis
# ------------------------------------------------------------------ #


class TestDisabledAnalysis:
    def test_disabled_by_default(self):
        cfg = Config()
        assert cfg.enable_prompt_analysis is False

    def test_no_analyzer_when_disabled(self):
        cfg = Config(enable_prompt_analysis=False)
        assert cfg.enable_prompt_analysis is False


# ------------------------------------------------------------------ #
# Fail-open
# ------------------------------------------------------------------ #


class TestFailOpen:
    def test_exception_returns_none(self):
        from capsera._interceptor import _maybe_analyze

        class BrokenAnalyzer:
            def analyze(self, kwargs, provider):
                raise RuntimeError("analysis crashed")

        result = _maybe_analyze({"model": "test"}, "anthropic", BrokenAnalyzer())
        assert result is None
