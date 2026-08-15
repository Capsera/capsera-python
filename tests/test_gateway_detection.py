"""Tests for OpenAI-compatible gateway detection, vendor-prefix model
normalization, and the gateway field's path through the event pipeline.

Bifrost/OpenRouter/Helicone/Cloudflare-AI-Gateway/self-hosted-LiteLLM need no
integration code: users point the OpenAI (or Anthropic) SDK ``base_url`` at
the gateway, the existing interceptor captures the call, and these tests pin
the two pieces that make that useful — host→gateway tagging and pricing that
survives "vendor/model" IDs.
"""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from capsera._catalog import gateway_for, provider_for, strip_vendor_prefix
from capsera._config import Config
from capsera._interceptor import Interceptor, _build_event, _detect_gateway
from capsera._models import TokenEvent
from capsera._pricing import PRICING, get_pricing

from .conftest import CapturingEmitter


@pytest.fixture
def config() -> Config:
    return Config(api_key="test-key", endpoint="http://localhost:9999",
                  env="test", flush_interval_ms=99_999, debug=True)


@pytest.fixture
def emitter(config: Config):
    e = CapturingEmitter(config)
    e.start()
    yield e
    e.shutdown()


# --------------------------------------------------------------------------- #
# Host → gateway mapping
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://api.portkey.ai/v1", "portkey"),
        ("https://oai.helicone.ai/v1", "helicone"),
        ("https://gateway.ai.cloudflare.com/v1/acct/gw/openai", "cloudflare"),
        ("http://localhost:4000", "litellm"),
        ("http://127.0.0.1:4000/v1", "litellm"),
        ("https://litellm.internal.acme.com", "litellm"),
        ("https://bifrost.acme.io/v1", "bifrost"),
        ("https://api.openai.com/v1", None),
        ("https://api.anthropic.com", None),
        (None, None),
    ],
)
def test_gateway_for(base_url, expected):
    assert gateway_for(base_url) == expected


def test_detect_gateway_reads_client_base_url():
    client = SimpleNamespace(_client=SimpleNamespace(base_url="https://openrouter.ai/api/v1"))
    assert _detect_gateway(client) == "openrouter"
    assert _detect_gateway(SimpleNamespace()) is None


# --------------------------------------------------------------------------- #
# Vendor-prefix stripping (gateway-style model IDs)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("openai/gpt-4o", "gpt-4o"),
        ("openrouter/anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
        # Not a vendor prefix — untouched:
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("accounts/fireworks/models/kimi-k3", "accounts/fireworks/models/kimi-k3"),
    ],
)
def test_strip_vendor_prefix(model, expected):
    assert strip_vendor_prefix(model) == expected


def test_pricing_resolves_prefixed_models():
    assert get_pricing("anthropic/claude-sonnet-4-6") is PRICING["claude-sonnet-4-6"]
    assert get_pricing("openrouter/openai/gpt-4o") is PRICING["gpt-4o"]
    # Real slash-keyed rows still win exactly:
    assert (get_pricing("meta-llama/Llama-3.3-70B-Instruct-Turbo")
            is PRICING["meta-llama/Llama-3.3-70B-Instruct-Turbo"])


def test_pricing_prefixed_cost_is_not_the_fallback():
    pricing = get_pricing("anthropic/claude-sonnet-4-6")
    assert pricing.input_per_1k == Decimal("0.003")
    assert pricing.output_per_1k == Decimal("0.015")


def test_provider_for_resolves_prefixed_models():
    assert provider_for("anthropic/claude-sonnet-4-6") == "anthropic"
    assert provider_for("openai/gpt-4o") == "openai"


# --------------------------------------------------------------------------- #
# Gateway field through the event pipeline
# --------------------------------------------------------------------------- #

def _fake_openai_namespace(base_url: str):
    class Completions:
        _client = SimpleNamespace(base_url=base_url)

        def create(self, **kwargs):
            return SimpleNamespace(
                model=kwargs.get("model", "gpt-4o"),
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    return SimpleNamespace(
        resources=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(Completions=Completions))
        )
    ), Completions


def test_openrouter_call_tagged_with_gateway(config, emitter):
    """OpenAI SDK pointed at OpenRouter: provider keeps its existing
    resolution ("openrouter") and the new gateway dimension is set too."""
    fake_openai, completions_cls = _fake_openai_namespace("https://openrouter.ai/api/v1")
    Interceptor(config, emitter)._patch_openai_sync(fake_openai)

    completions_cls().create(model="anthropic/claude-sonnet-4-6", messages=[])

    (ev,) = emitter.drain()
    assert ev.gateway == "openrouter"
    assert ev.provider == "openrouter"
    # Prefixed model ID still prices correctly (not the $0.001/$0.003 fallback):
    # 10 in × $0.003/1k + 5 out × $0.015/1k
    assert ev.cost_usd == pytest.approx(0.000105)


def test_direct_openai_call_has_no_gateway(config, emitter):
    fake_openai, completions_cls = _fake_openai_namespace("https://api.openai.com/v1")
    Interceptor(config, emitter)._patch_openai_sync(fake_openai)

    completions_cls().create(model="gpt-4o", messages=[])

    (ev,) = emitter.drain()
    assert ev.gateway is None
    assert ev.provider == "openai"


def test_default_gateway_applies_when_undetected(emitter):
    """init(gateway="bifrost") tags calls whose base_url is unrecognisable
    (self-hosted gateway on a vanity host)."""
    config = Config(api_key="test-key", endpoint="http://localhost:9999",
                    env="test", flush_interval_ms=99_999, default_gateway="bifrost")
    fake_openai, completions_cls = _fake_openai_namespace("https://llm.internal.acme.com/v1")
    Interceptor(config, emitter)._patch_openai_sync(fake_openai)

    completions_cls().create(model="gpt-4o", messages=[])

    (ev,) = emitter.drain()
    assert ev.gateway == "bifrost"


def test_detected_gateway_wins_over_default(emitter):
    config = Config(api_key="test-key", endpoint="http://localhost:9999",
                    env="test", flush_interval_ms=99_999, default_gateway="bifrost")
    fake_openai, completions_cls = _fake_openai_namespace("https://openrouter.ai/api/v1")
    Interceptor(config, emitter)._patch_openai_sync(fake_openai)

    completions_cls().create(model="gpt-4o", messages=[])

    (ev,) = emitter.drain()
    assert ev.gateway == "openrouter"


def test_build_event_gateway_contract_normalization(config):
    event = _build_event(
        model="gpt-4o", provider="openai",
        input_tokens=1, output_tokens=1,
        cache_read_tokens=0, cache_write_tokens=0,
        latency_ms=0, config=config,
        gateway="g" * 100,
    )
    assert len(event.gateway) == 64  # truncated to the ingest contract


def test_token_event_serialises_gateway():
    event = TokenEvent(
        id="i", timestamp="t", request_id="r", model="m", provider="p",
        input_tokens=1, output_tokens=1, gateway="litellm",
    )
    assert event.to_dict()["gateway"] == "litellm"
    # None gateway is dropped from the payload (backward-compatible ingest).
    event_no_gw = TokenEvent(
        id="i", timestamp="t", request_id="r", model="m", provider="p",
        input_tokens=1, output_tokens=1,
    )
    assert "gateway" not in event_no_gw.to_dict()
