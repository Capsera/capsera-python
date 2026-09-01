"""Golden cost tests — testing-strategy §5 C1 (per-provider golden pricing)
and §5 C3 (cached tokens priced at their VERIFIED discounted rates).

Complements test_cached_tokens.py, which owns the split *semantics* (which
provider includes cached tokens in its prompt total, extraction shapes,
cost-neutral fallback for rate-less models). This file owns the *numbers*:
every expected value is hand-calculated from the provider's published rates,
so the code under test computes nothing that grades itself.

Rate sources (verified 2026-08-20): OpenAI pricing page — cached input is
50% of the input rate across gpt-4o/-mini and the o-series; Gemini pricing
page (ai.google.dev) — context-cache reads are 10% of input across the 2.5
and 3.x families. Gemini's separate per-hour cache STORAGE fee is invisible
in per-call usage metadata and deliberately out of scope.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from capsera._pricing import compute_cost, get_pricing

# --------------------------------------------------------------------------- #
# C1 — per-provider goldens (1k in / 1k out, hand-calculated)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-sonnet-4-6", Decimal("0.018")),          # 0.003 + 0.015
        ("gpt-4o", Decimal("0.0125")),                    # 0.0025 + 0.010
        ("gemini-2.5-flash", Decimal("0.0028")),          # 0.0003 + 0.0025
        ("mistral-large-latest", Decimal("0.002")),       # 0.0005 + 0.0015
        ("amazon.nova-pro-v1:0", Decimal("0.004")),       # 0.0008 + 0.0032
    ],
)
def test_c1_golden_cost_per_provider(model: str, expected: Decimal) -> None:
    assert compute_cost(model, "any", 1000, 1000) == expected


# --------------------------------------------------------------------------- #
# C3 — cached tokens at the verified discounted rates
# --------------------------------------------------------------------------- #

def test_c3_openai_cached_golden() -> None:
    """gpt-4o: 200 uncached + 800 cached + 100 out.

    Hand-calc: 200/1k*0.0025 + 100/1k*0.010 + 800/1k*0.00125
             = 0.0005 + 0.001 + 0.001 = 0.0025 — materially below the
    0.0035 the same call cost while cached tokens billed at full input.
    """
    cached = compute_cost("gpt-4o", "openai", 200, 100, cache_read_tokens=800)
    assert cached == Decimal("0.0025")
    assert cached < compute_cost("gpt-4o", "openai", 1000, 100)


def test_c3_gemini_cached_golden() -> None:
    """gemini-2.5-flash: 400 uncached + 600 cached + 100 out.

    Hand-calc: 400/1k*0.0003 + 100/1k*0.0025 + 600/1k*0.00003
             = 0.00012 + 0.00025 + 0.000018 = 0.000388.
    """
    assert compute_cost(
        "gemini-2.5-flash", "google", 400, 100, cache_read_tokens=600
    ) == Decimal("0.000388")


def test_c3_anthropic_cached_golden() -> None:
    """claude-sonnet-4-6: 200 in + 100 out + 800 cache-read + 500 cache-write.

    Hand-calc: 200/1k*0.003 + 100/1k*0.015 + 800/1k*0.0003 + 500/1k*0.00375
             = 0.0006 + 0.0015 + 0.00024 + 0.001875 = 0.004215.
    Exercises the read discount and the write premium in one absolute number.
    """
    assert compute_cost(
        "claude-sonnet-4-6", "anthropic", 200, 100,
        cache_read_tokens=800, cache_write_tokens=500,
    ) == Decimal("0.004215")


# The published discount is a RATIO of the input rate, so pin the ratio —
# a future input-price update that forgets the cache column fails here.

@pytest.mark.parametrize(
    "model",
    ["gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
     "o1", "o1-mini", "o3-mini"],
)
def test_c3_openai_cached_rate_is_half_of_input(model: str) -> None:
    pricing = get_pricing(model)
    assert pricing.cache_read_per_1k * 2 == pricing.input_per_1k


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
     "gemini-3-pro", "gemini-3-flash", "gemini-3.1-pro",
     "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.5-flash-lite"],
)
def test_c3_gemini_cached_rate_is_tenth_of_input(model: str) -> None:
    pricing = get_pricing(model)
    assert pricing.cache_read_per_1k * 10 == pricing.input_per_1k
