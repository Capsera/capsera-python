"""Cost computation precision and edge cases.

compute_cost() output is what customers reconcile against provider
invoices — it must be exact Decimal arithmetic at any scale, never float
drift, and unknown models must price to zero rather than guess.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from capsera._pricing import PRICING, compute_cost


def test_exact_decimal_for_known_model():
    cost = compute_cost("claude-sonnet-4-6", "anthropic", input_tokens=1000, output_tokens=1000)
    assert cost == Decimal("0.018")        # 0.003 + 0.015, exactly


def test_zero_tokens_zero_cost():
    assert compute_cost("claude-sonnet-4-6", "anthropic", input_tokens=0, output_tokens=0) == 0


def test_unknown_model_uses_the_documented_conservative_fallback():
    # DESIGN DECISION (reviewed 2026-08-03): unknown models price at the
    # documented fallback ($0.001/$0.003 per 1k) rather than $0 — invisible
    # spend is worse for governance than approximate spend. This test pins
    # the fallback so a silent change to it is caught.
    cost = compute_cost("totally-unknown-model-v99", "unknown", input_tokens=10_000, output_tokens=10_000)
    assert cost == Decimal("0.040")  # 10 × 0.001 + 10 × 0.003


def test_prefix_fallback_prices_dated_variants():
    exact = compute_cost("claude-sonnet-4-6", "anthropic", input_tokens=1000, output_tokens=0)
    dated = compute_cost("claude-sonnet-4-6-20991231", "anthropic", input_tokens=1000, output_tokens=0)
    assert dated == exact and dated > 0


def test_billion_token_precision_no_float_drift():
    # 1e9 tokens = 1e6 thousand-token units × $0.003 = $3,000 exactly.
    # Float math would show drift at this scale; Decimal must not.
    cost = compute_cost("claude-sonnet-4-6", "anthropic", input_tokens=1_000_000_000, output_tokens=0)
    assert cost == Decimal("3000")


def test_single_token_cost_is_exact():
    cost = compute_cost("claude-sonnet-4-6", "anthropic", input_tokens=1, output_tokens=0)
    assert cost == Decimal("0.000003")


def test_cache_tokens_priced_when_supported():
    # Cache reads are cheaper than fresh input; the math must use the cache
    # rates, not the input rate.
    base = compute_cost("claude-sonnet-4-6", "anthropic", input_tokens=0, output_tokens=0)
    with_cache = compute_cost(
        "claude-sonnet-4-6", "anthropic", input_tokens=0, output_tokens=0,
        cache_read_tokens=100_000,
    )
    fresh = compute_cost("claude-sonnet-4-6", "anthropic", input_tokens=100_000, output_tokens=0)
    assert with_cache > base
    assert with_cache < fresh, "cache reads must be cheaper than fresh input"


@pytest.mark.parametrize("model", sorted(PRICING))
def test_every_pricing_row_is_nonnegative_decimal(model):
    p = PRICING[model]
    for rate in (p.input_per_1k, p.output_per_1k, p.cache_read_per_1k, p.cache_write_per_1k):
        assert isinstance(rate, Decimal)
        assert rate >= 0


def test_verified_third_party_rates_price_correctly():
    # Spot-check the 2026-08-03 audit corrections landed in the math.
    assert compute_cost("mistral-large-latest", "mistral", 1000, 1000) == Decimal("0.002")
    assert compute_cost("deepseek-v4-flash", "deepseek", 1000, 1000) == Decimal("0.00042")
    assert compute_cost(
        "anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock", 1000, 1000
    ) == Decimal("0.036")
