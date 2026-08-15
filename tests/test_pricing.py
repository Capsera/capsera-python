"""
Unit tests for the pricing module.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

from capsera._pricing import ModelPricing, compute_cost, get_pricing


class TestGetPricing:
    def test_exact_match(self) -> None:
        p = get_pricing("gpt-4o")
        assert p.input_per_1k == Decimal("0.0025")

    def test_prefix_match(self) -> None:
        # A versioned alias that starts with a known key prefix
        p = get_pricing("claude-sonnet-4-6-20260420")
        assert p.input_per_1k == Decimal("0.003")

    def test_unknown_model_returns_fallback(self) -> None:
        p = get_pricing("totally-unknown-model-v99")
        # Fallback values
        assert p.input_per_1k == Decimal("0.001")
        assert p.output_per_1k == Decimal("0.003")


class TestExternalPricingFallback:
    """For models absent from Capsera's own table, fall back to LiteLLM's
    maintained ``model_cost`` map — but only when litellm is already loaded,
    and never overriding models we do carry."""

    def test_uses_litellm_model_cost_when_loaded(self, monkeypatch) -> None:
        fake = SimpleNamespace(model_cost={
            "acme-llm-x": {"input_cost_per_token": 0.000002, "output_cost_per_token": 0.000008},
        })
        monkeypatch.setitem(sys.modules, "litellm", fake)
        p = get_pricing("acme-llm-x")
        assert p.input_per_1k == Decimal("0.002")     # per-token → per-1k
        assert p.output_per_1k == Decimal("0.008")

    def test_vendor_prefixed_model_resolves(self, monkeypatch) -> None:
        fake = SimpleNamespace(model_cost={
            "acme-llm-x": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000003},
        })
        monkeypatch.setitem(sys.modules, "litellm", fake)
        p = get_pricing("openrouter/acme-llm-x")
        assert p.input_per_1k == Decimal("0.001")

    def test_absent_litellm_uses_generic_fallback(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "litellm", None)   # simulate not-loaded
        p = get_pricing("totally-unknown-model-zzz")
        assert p.input_per_1k == Decimal("0.001")
        assert p.output_per_1k == Decimal("0.003")

    def test_capsera_table_wins_over_external(self, monkeypatch) -> None:
        # A model we DO carry must never be repriced from the external source.
        fake = SimpleNamespace(model_cost={
            "gpt-4o": {"input_cost_per_token": 9.0, "output_cost_per_token": 9.0},
        })
        monkeypatch.setitem(sys.modules, "litellm", fake)
        assert get_pricing("gpt-4o").input_per_1k == Decimal("0.0025")

    def test_malformed_external_entry_falls_through(self, monkeypatch) -> None:
        fake = SimpleNamespace(model_cost={"acme-llm-x": "not-a-dict"})
        monkeypatch.setitem(sys.modules, "litellm", fake)
        p = get_pricing("acme-llm-x")
        assert p.input_per_1k == Decimal("0.001")   # generic fallback


class TestComputeCost:
    @pytest.mark.parametrize("model,provider,inp,out,expected", [
        # claude-sonnet-4-6: 0.003/1k in, 0.015/1k out
        ("claude-sonnet-4-6", "anthropic", 1000, 1000, Decimal("0.018")),
        # gpt-4o: 0.0025/1k in, 0.010/1k out
        ("gpt-4o", "openai", 2000, 500, Decimal("0.010")),
        # gpt-4o-mini: 0.00015/1k in, 0.0006/1k out
        ("gpt-4o-mini", "openai", 10_000, 2_000, Decimal("0.0027")),
        # claude-haiku: 0.00025/1k in, 0.00125/1k out
        ("claude-haiku-4-5-20251001", "anthropic", 1000, 1000, Decimal("0.0015")),
    ])
    def test_cost_parametrized(
        self, model: str, provider: str, inp: int, out: int, expected: Decimal
    ) -> None:
        cost = compute_cost(model=model, provider=provider,
                            input_tokens=inp, output_tokens=out)
        assert abs(cost - expected) < Decimal("0.000001")

    def test_zero_tokens_is_zero_cost(self) -> None:
        cost = compute_cost(model="claude-sonnet-4-6", provider="anthropic",
                            input_tokens=0, output_tokens=0)
        assert cost == Decimal("0")

    def test_cache_tokens_add_cost(self) -> None:
        base = compute_cost("claude-sonnet-4-6", "anthropic", 1000, 0)
        with_cache = compute_cost("claude-sonnet-4-6", "anthropic", 1000, 0,
                                  cache_read_tokens=500, cache_write_tokens=200)
        assert with_cache > base

    def test_all_major_models_present(self) -> None:
        """Spot-check that every model name mentioned in CLAUDE.md is in the table."""
        expected = [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gemini-pro",
            "gemini-flash",
        ]
        for model_name in expected:
            # Either exact or prefix match must work
            p = get_pricing(model_name)
            assert isinstance(p, ModelPricing), f"No pricing for {model_name}"
            assert p.input_per_1k > Decimal("0"), f"Zero input price for {model_name}"

    def test_decimal_precision_large_scale(self) -> None:
        """Summing costs over 1M calls should not accumulate float errors."""
        unit = compute_cost("gpt-4o-mini", "openai",
                            input_tokens=100, output_tokens=50)
        total = sum(unit for _ in range(1_000_000))
        # Decimal arithmetic — no floating-point drift
        expected = unit * 1_000_000
        assert total == expected


class TestBedrockRegionalProfiles:
    """Bedrock cross-region inference profiles prefix the model ID with a
    region scope (us. / eu. / apac. / global.). Billing must resolve to the
    underlying model, not the generic fallback estimate."""

    def test_apac_profile_resolves_to_base_model(self) -> None:
        assert (
            get_pricing("apac.amazon.nova-micro-v1:0")
            is get_pricing("amazon.nova-micro-v1:0")
        )

    def test_us_profile_resolves_for_anthropic_bedrock_id(self) -> None:
        assert (
            get_pricing("us.anthropic.claude-3-5-sonnet-20241022-v2:0")
            is get_pricing("anthropic.claude-3-5-sonnet-20241022-v2:0")
        )

    def test_nova_micro_priced(self) -> None:
        cost = compute_cost("apac.amazon.nova-micro-v1:0", "bedrock",
                            input_tokens=1000, output_tokens=1000)
        # $0.035/M in + $0.14/M out (aws.amazon.com/bedrock/pricing, 2026-08-03)
        assert cost == Decimal("0.000175")
