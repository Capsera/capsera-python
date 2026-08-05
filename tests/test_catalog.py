"""Model catalog: single source of truth for provider + metadata.

The catalog holds everything EXCEPT price (price stays in _pricing.py, which
the backend already imports). Provider inference, tier/context data, and the
capability seam all read from here so that adding a model is one row in one
file instead of edits across the router, budget enforcer, model-optimization,
and backend registry.
"""
from capsera import _catalog as cat
from capsera._pricing import PRICING, get_pricing


def test_existing_models_have_catalog_metadata():
    for model in ("claude-sonnet-4-6", "gpt-4o", "gemini-1.5-pro"):
        meta = cat.model_meta(model)
        assert meta is not None, f"{model} missing from catalog"
        assert meta.provider in {"anthropic", "openai", "google"}
        assert meta.context_window and meta.context_window > 0
        assert meta.quality_tier in {"high", "medium", "low"}


def test_provider_inference_for_first_party():
    assert cat.provider_for("claude-sonnet-4-6") == "anthropic"
    assert cat.provider_for("gpt-4o") == "openai"
    assert cat.provider_for("o3-mini") == "openai"
    assert cat.provider_for("gemini-2.0-flash") == "google"


def test_openai_compatible_provider_resolved_by_base_url():
    # These ride the OpenAI SDK, so the model kwarg alone can't tell them
    # apart from real OpenAI — the base_url host does.
    assert cat.provider_for("deepseek-chat", base_url="https://api.deepseek.com/v1") == "deepseek"
    assert cat.provider_for("llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1") == "groq"
    assert cat.provider_for("grok-2", base_url="https://api.x.ai/v1") == "xai"
    # No base_url and an unknown model → the catalog's declared provider, or "unknown".
    assert cat.provider_for("deepseek-chat") == "deepseek"


def test_azure_host_maps_to_azure_even_for_openai_models():
    # Azure serves gpt-* under deployment names; the host disambiguates.
    assert cat.provider_for("gpt-4o", base_url="https://acme.openai.azure.com/") == "azure"


def test_native_providers_present_with_prices():
    for model in ("mistral-large-latest", "command-r-plus", "deepseek-chat"):
        meta = cat.model_meta(model)
        assert meta is not None
        # Price must resolve to a real (non-fallback) row.
        assert model in PRICING, f"{model} has no pricing row"
        pricing = get_pricing(model)
        assert pricing.input_per_1k >= 0


def test_prefix_fallback_matches_pricing_semantics():
    # A versioned suffix resolves to its base entry, mirroring get_pricing.
    meta = cat.model_meta("mistral-large-latest-2411")
    assert meta is not None and meta.provider == "mistral"


def test_capabilities_seam_present_for_routing_guardrails():
    # The future cross-provider routing guardrail keys on capabilities; the
    # data must exist even though nothing enforces it yet.
    meta = cat.model_meta("gpt-4o")
    assert meta is not None
    assert isinstance(meta.capabilities, frozenset)


def test_every_catalogued_model_has_a_price_row():
    # Drift guard: a model in the catalog with no price silently falls back to
    # the $0.001/$0.003 estimate — wrong spend, no warning at authoring time.
    missing = [m for m in cat.MODELS if m not in PRICING]
    assert missing == [], f"catalogued models with no pricing row: {missing}"


def test_every_new_price_row_has_catalog_metadata():
    # The inverse: a price row with no metadata can't be attributed to a
    # provider or tier, so optimization/routing can't see it.
    missing = [m for m in PRICING if cat.model_meta(m) is None]
    assert missing == [], f"priced models with no catalog metadata: {missing}"
