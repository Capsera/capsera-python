"""
Cached-token capture and the arithmetic around it.

Cached prompts are the single largest source of cost error this SDK can
produce, because the mistake is silent and multiplicative: a heavy-caching
workload bills the same tokens twice and the number only looks wrong next to
the provider's own invoice. The test plan ranks over-reporting as the
third-worst failure the product has, above silent under-counting, so these
tests pin the direction of every rounding decision as well as the values.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from capsera._pricing import compute_cost, has_cache_rates
from capsera._usage import (
    cached_is_included,
    gemini_cached_tokens,
    openai_cached_tokens,
    split_cached_input,
)

# --------------------------------------------------------------------------- #
# The inclusion rule — the thing that is easy to get backwards
# --------------------------------------------------------------------------- #


def test_openai_prompt_total_contains_its_cached_tokens():
    """prompt_tokens is the whole prompt; cached_tokens is a subset of it."""
    assert split_cached_input(1000, 400, included=True) == (600, 400)


def test_anthropic_buckets_are_additive():
    """input_tokens excludes both cache counts, so nothing is subtracted —
    doing so would drop 400 real uncached tokens off the bill."""
    assert split_cached_input(600, 400, included=False) == (600, 400)


def test_provider_inclusion_is_looked_up_not_assumed():
    assert cached_is_included("openai") is True
    assert cached_is_included("google") is True
    assert cached_is_included("anthropic") is False
    assert cached_is_included("bedrock") is False


def test_unknown_providers_default_to_the_openai_shape():
    """Most OpenAI-compatible APIs inherit prompt_tokens semantics with the
    wire format. The default also picks the safer error: under-counting the
    uncached remainder rather than double-billing the cached part."""
    assert cached_is_included("some-new-gateway") is True
    assert cached_is_included(None) is True


def test_no_cached_tokens_leaves_the_prompt_total_untouched():
    assert split_cached_input(1000, 0, included=True) == (1000, 0)


def test_incoherent_cached_count_keeps_cost_right_and_drops_the_split():
    """Seen from proxies merging usage across upstream calls. Emitting the
    negative would show up on the dashboard as a credit."""
    assert split_cached_input(100, 500, included=True) == (100, 0)


def test_negative_inputs_never_produce_negative_counts():
    assert split_cached_input(-5, -5, included=True) == (0, 0)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def test_openai_cached_tokens_read_from_prompt_tokens_details():
    usage = MagicMock()
    usage.prompt_tokens_details.cached_tokens = 256
    assert openai_cached_tokens(usage) == 256


def test_openai_cached_tokens_absent_details_is_zero():
    assert openai_cached_tokens(MagicMock(spec=[])) == 0


def test_a_bare_mock_does_not_invent_a_cached_token():
    """int(MagicMock()) is 1, and auto-attributes make every mocked usage look
    like it cached something. Since cached tokens are subtracted from the
    prompt total, a fabricated 1 removes a real input token from the bill."""
    usage = MagicMock()
    usage.prompt_tokens = 80
    assert openai_cached_tokens(usage) == 0


def test_cached_tokens_survive_dict_shaped_usage():
    assert openai_cached_tokens({"prompt_tokens_details": {"cached_tokens": 12}}) == 12


def test_stringified_counts_are_accepted():
    assert openai_cached_tokens({"prompt_tokens_details": {"cached_tokens": "12"}}) == 12


def test_gemini_cached_tokens_accept_both_spellings():
    snake = MagicMock(spec=["cached_content_token_count"])
    snake.cached_content_token_count = 300
    assert gemini_cached_tokens(snake) == 300
    assert gemini_cached_tokens({"cachedContentTokenCount": 300}) == 300


def test_gemini_cached_tokens_absent_is_zero():
    assert gemini_cached_tokens(MagicMock(spec=[])) == 0
    assert gemini_cached_tokens(None) == 0


# --------------------------------------------------------------------------- #
# Cost consequences
# --------------------------------------------------------------------------- #


def test_anthropic_cache_read_is_billed_at_the_cache_rate():
    """The bug this whole change exists to fix: 1000 cached tokens billed as
    ordinary input costs an order of magnitude more than reality."""
    as_input = compute_cost("claude-sonnet-4-6", "anthropic", 1000, 0)
    as_cached = compute_cost("claude-sonnet-4-6", "anthropic", 0, 0, cache_read_tokens=1000)
    assert as_cached < as_input
    # Decimal throughout: compare exactly rather than through a float lens.
    assert as_cached * 10 == as_input


def test_cache_write_carries_a_premium_over_input():
    as_input = compute_cost("claude-sonnet-4-6", "anthropic", 1000, 0)
    as_write = compute_cost("claude-sonnet-4-6", "anthropic", 0, 0, cache_write_tokens=1000)
    assert as_write > as_input


def test_splitting_is_cost_neutral_where_no_cached_rate_is_known():
    """A model with no cached rate in the table must cost the same split or
    unsplit — otherwise shipping the capture fix would silently zero out real
    spend, trading over-reporting for under-reporting. (Originally pinned on
    gpt-4o; its verified rate has since landed, so the pin moved to a model
    that genuinely still lacks one.)"""
    assert not has_cache_rates("mistral-large-latest")
    unsplit = compute_cost("mistral-large-latest", "mistral", 1000, 100)
    split = compute_cost("mistral-large-latest", "mistral", 600, 100, cache_read_tokens=400)
    assert split == unsplit


def test_a_missing_cached_rate_never_makes_cached_tokens_free():
    assert compute_cost("mistral-large-latest", "mistral", 0, 0, cache_read_tokens=1000) > 0


def test_has_cache_rates_distinguishes_a_gap_from_a_real_rate():
    assert has_cache_rates("claude-sonnet-4-6") is True
    # Verified rates landed for OpenAI/Gemini (see test_cost_golden.py).
    assert has_cache_rates("gpt-4o") is True
    assert has_cache_rates("mistral-large-latest") is False


# --------------------------------------------------------------------------- #
# End to end through the interceptor's emit paths
# --------------------------------------------------------------------------- #


def test_openai_response_records_the_cached_split(base_config, capturing_emitter):
    import importlib

    runtime = importlib.import_module("capsera._interceptor")
    response = MagicMock()
    response.model = "gpt-4o"
    response.usage.prompt_tokens = 1000
    response.usage.completion_tokens = 100
    response.usage.prompt_tokens_details.cached_tokens = 400

    runtime._emit_openai_response(
        response=response,
        request_kwargs={"model": "gpt-4o"},
        latency_ms=5,
        config=base_config,
        emitter=capturing_emitter,
        original_model=None,
        caller_file="app.py:1",
    )

    (event,) = capturing_emitter.drain()
    assert event.cache_read_tokens == 400
    assert event.input_tokens == 600, "cached tokens must not be counted twice"
    assert event.input_tokens + event.cache_read_tokens == 1000


def test_gemini_response_records_the_cached_split(base_config, capturing_emitter):
    import importlib

    runtime = importlib.import_module("capsera._interceptor")
    response = MagicMock()
    response.model = "gemini-2.0-flash"
    response.usage_metadata.prompt_token_count = 1000
    response.usage_metadata.candidates_token_count = 50
    response.usage_metadata.cached_content_token_count = 750

    runtime._emit_google_response(
        response=response,
        request_kwargs={"model": "gemini-2.0-flash"},
        latency_ms=5,
        config=base_config,
        emitter=capturing_emitter,
        original_model=None,
        caller_file="app.py:1",
    )

    (event,) = capturing_emitter.drain()
    assert event.cache_read_tokens == 750
    assert event.input_tokens == 250


def test_anthropic_response_keeps_its_buckets_additive(base_config, capturing_emitter):
    """Regression guard: applying the OpenAI subtraction to Anthropic would
    quietly delete real uncached input."""
    import importlib

    runtime = importlib.import_module("capsera._interceptor")
    response = MagicMock()
    response.model = "claude-sonnet-4-6"
    response.usage.input_tokens = 600
    response.usage.output_tokens = 100
    response.usage.cache_read_input_tokens = 400
    response.usage.cache_creation_input_tokens = 0

    runtime._emit_anthropic_response(
        response=response,
        request_kwargs={"model": "claude-sonnet-4-6"},
        latency_ms=5,
        config=base_config,
        emitter=capturing_emitter,
        original_model=None,
        caller_file="app.py:1",
    )

    (event,) = capturing_emitter.drain()
    assert event.input_tokens == 600
    assert event.cache_read_tokens == 400
