"""
Pricing tables for major LLM providers and cost computation logic.

Prices are in USD per 1,000 tokens and stored as Decimal for precision.
The table is embedded in the SDK as a fallback; the backend pricing_worker
can push updated tables, but the SDK always has a valid default.

Last updated: 2026-04-19
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPricing:
    """Per-1k token pricing for one model."""

    input_per_1k: Decimal
    output_per_1k: Decimal
    cache_read_per_1k: Decimal = Decimal("0")
    cache_write_per_1k: Decimal = Decimal("0")


# ---------------------------------------------------------------------------
# Pricing table
# Keys use the canonical model ID as returned by the provider API response.
# We also register common alias strings so lookups work even if callers use
# slightly different names.
# ---------------------------------------------------------------------------

PRICING: dict[str, ModelPricing] = {
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-opus-4-6": ModelPricing(
        input_per_1k=Decimal("0.015"),
        output_per_1k=Decimal("0.075"),
        cache_read_per_1k=Decimal("0.0015"),
        cache_write_per_1k=Decimal("0.01875"),
    ),
    # Aliases
    "claude-opus-4": ModelPricing(
        input_per_1k=Decimal("0.015"),
        output_per_1k=Decimal("0.075"),
        cache_read_per_1k=Decimal("0.0015"),
        cache_write_per_1k=Decimal("0.01875"),
    ),
    "claude-sonnet-4-6": ModelPricing(
        input_per_1k=Decimal("0.003"),
        output_per_1k=Decimal("0.015"),
        cache_read_per_1k=Decimal("0.0003"),
        cache_write_per_1k=Decimal("0.00375"),
    ),
    "claude-sonnet-4": ModelPricing(
        input_per_1k=Decimal("0.003"),
        output_per_1k=Decimal("0.015"),
        cache_read_per_1k=Decimal("0.0003"),
        cache_write_per_1k=Decimal("0.00375"),
    ),
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_per_1k=Decimal("0.003"),
        output_per_1k=Decimal("0.015"),
        cache_read_per_1k=Decimal("0.0003"),
        cache_write_per_1k=Decimal("0.00375"),
    ),
    "claude-3-5-sonnet-20240620": ModelPricing(
        input_per_1k=Decimal("0.003"),
        output_per_1k=Decimal("0.015"),
        cache_read_per_1k=Decimal("0.0003"),
        cache_write_per_1k=Decimal("0.00375"),
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_per_1k=Decimal("0.00025"),
        output_per_1k=Decimal("0.00125"),
        cache_read_per_1k=Decimal("0.000025"),
        cache_write_per_1k=Decimal("0.0003"),
    ),
    "claude-haiku-4-5": ModelPricing(
        input_per_1k=Decimal("0.00025"),
        output_per_1k=Decimal("0.00125"),
        cache_read_per_1k=Decimal("0.000025"),
        cache_write_per_1k=Decimal("0.0003"),
    ),
    "claude-3-haiku-20240307": ModelPricing(
        input_per_1k=Decimal("0.00025"),
        output_per_1k=Decimal("0.00125"),
        cache_read_per_1k=Decimal("0.000025"),
        cache_write_per_1k=Decimal("0.0003"),
    ),
    "claude-3-opus-20240229": ModelPricing(
        input_per_1k=Decimal("0.015"),
        output_per_1k=Decimal("0.075"),
        cache_read_per_1k=Decimal("0.0015"),
        cache_write_per_1k=Decimal("0.01875"),
    ),
    # ── OpenAI ───────────────────────────────────────────────────────────
    "gpt-4o": ModelPricing(
        input_per_1k=Decimal("0.0025"),
        output_per_1k=Decimal("0.010"),
    ),
    "gpt-4o-2024-11-20": ModelPricing(
        input_per_1k=Decimal("0.0025"),
        output_per_1k=Decimal("0.010"),
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_1k=Decimal("0.00015"),
        output_per_1k=Decimal("0.0006"),
    ),
    "gpt-4o-mini-2024-07-18": ModelPricing(
        input_per_1k=Decimal("0.00015"),
        output_per_1k=Decimal("0.0006"),
    ),
    "gpt-4-turbo": ModelPricing(
        input_per_1k=Decimal("0.010"),
        output_per_1k=Decimal("0.030"),
    ),
    "gpt-4": ModelPricing(
        input_per_1k=Decimal("0.030"),
        output_per_1k=Decimal("0.060"),
    ),
    "gpt-3.5-turbo": ModelPricing(
        input_per_1k=Decimal("0.0005"),
        output_per_1k=Decimal("0.0015"),
    ),
    "gpt-3.5-turbo-0125": ModelPricing(
        input_per_1k=Decimal("0.0005"),
        output_per_1k=Decimal("0.0015"),
    ),
    "o1": ModelPricing(
        input_per_1k=Decimal("0.015"),
        output_per_1k=Decimal("0.060"),
    ),
    "o1-mini": ModelPricing(
        input_per_1k=Decimal("0.003"),
        output_per_1k=Decimal("0.012"),
    ),
    "o3-mini": ModelPricing(
        input_per_1k=Decimal("0.0011"),
        output_per_1k=Decimal("0.0044"),
    ),
    # ── OpenAI Embeddings ────────────────────────────────────────────────
    "text-embedding-3-small": ModelPricing(
        input_per_1k=Decimal("0.00002"),
        output_per_1k=Decimal("0"),
    ),
    "text-embedding-3-large": ModelPricing(
        input_per_1k=Decimal("0.00013"),
        output_per_1k=Decimal("0"),
    ),
    "text-embedding-ada-002": ModelPricing(
        input_per_1k=Decimal("0.0001"),
        output_per_1k=Decimal("0"),
    ),
    # ── Google ───────────────────────────────────────────────────────────
    "gemini-1.5-pro": ModelPricing(
        input_per_1k=Decimal("0.00125"),
        output_per_1k=Decimal("0.005"),
    ),
    "gemini-pro": ModelPricing(
        input_per_1k=Decimal("0.00125"),
        output_per_1k=Decimal("0.005"),
    ),
    "gemini-1.5-flash": ModelPricing(
        input_per_1k=Decimal("0.000075"),
        output_per_1k=Decimal("0.0003"),
    ),
    "gemini-flash": ModelPricing(
        input_per_1k=Decimal("0.000075"),
        output_per_1k=Decimal("0.0003"),
    ),
    "gemini-2.0-flash": ModelPricing(
        input_per_1k=Decimal("0.0001"),
        output_per_1k=Decimal("0.0004"),
    ),
    # Gemini 2.5 family — Developer API rates (≤200k-token prompts), verified
    # live 2026-08-03; the 2.0 free tier is retired (limit: 0), so customers
    # on the Developer API are on 2.5 models.
    "gemini-2.5-pro": ModelPricing(
        input_per_1k=Decimal("0.00125"),
        output_per_1k=Decimal("0.01"),
    ),
    "gemini-2.5-flash": ModelPricing(
        input_per_1k=Decimal("0.0003"),
        output_per_1k=Decimal("0.0025"),
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        input_per_1k=Decimal("0.0001"),
        output_per_1k=Decimal("0.0004"),
    ),
    # Gemini 3.x family — Developer API paid-tier rates (≤200k-token prompts),
    # verified against ai.google.dev/gemini-api/docs/pricing on 2026-08-03.
    "gemini-3-pro": ModelPricing(
        input_per_1k=Decimal("0.002"),
        output_per_1k=Decimal("0.012"),
    ),
    "gemini-3-flash": ModelPricing(
        input_per_1k=Decimal("0.0005"),
        output_per_1k=Decimal("0.003"),
    ),
    "gemini-3.1-pro": ModelPricing(
        input_per_1k=Decimal("0.002"),
        output_per_1k=Decimal("0.012"),
    ),
    "gemini-3.1-flash-lite": ModelPricing(
        input_per_1k=Decimal("0.00025"),
        output_per_1k=Decimal("0.0015"),
    ),
    "gemini-3.5-flash": ModelPricing(
        input_per_1k=Decimal("0.0015"),
        output_per_1k=Decimal("0.009"),
    ),
    "gemini-3.5-flash-lite": ModelPricing(
        input_per_1k=Decimal("0.0003"),
        output_per_1k=Decimal("0.0025"),
    ),
    "gemini-3.6-flash": ModelPricing(
        input_per_1k=Decimal("0.0015"),
        output_per_1k=Decimal("0.0075"),
    ),
    "gemini-2.0-pro": ModelPricing(
        input_per_1k=Decimal("0.0025"),
        output_per_1k=Decimal("0.010"),
    ),

    # ── Third-party providers ────────────────────────────────────────────
    # Verified against official provider pricing pages 2026-08-03 (see
    # docs/provider-pricing-to-verify.md for per-row sources). Rows for
    # retired models keep their LAST published rate so historical events
    # still price; _catalog.py marks them retired.
    # Per-1k-token USD. Metadata (provider/tier/context) lives in _catalog.py.
    #
    # DeepSeek — v4 lineup (cache-miss rate); chat/reasoner aliases retired 2026-07-24
    "deepseek-v4-flash": ModelPricing(input_per_1k=Decimal("0.00014"), output_per_1k=Decimal("0.00028")),
    "deepseek-v4-pro": ModelPricing(input_per_1k=Decimal("0.000435"), output_per_1k=Decimal("0.00087")),
    "deepseek-chat": ModelPricing(input_per_1k=Decimal("0.00027"), output_per_1k=Decimal("0.0011")),
    "deepseek-reasoner": ModelPricing(input_per_1k=Decimal("0.00055"), output_per_1k=Decimal("0.00219")),
    # Groq (mixtral retired 2025-03)
    "llama-3.3-70b-versatile": ModelPricing(input_per_1k=Decimal("0.00059"), output_per_1k=Decimal("0.00079")),
    "llama-3.1-8b-instant": ModelPricing(input_per_1k=Decimal("0.00005"), output_per_1k=Decimal("0.00008")),
    "mixtral-8x7b-32768": ModelPricing(input_per_1k=Decimal("0.00024"), output_per_1k=Decimal("0.00024")),
    # xAI Grok — 4.x lineup. NOTE: xAI doubles these rates for prompts ≥200k
    # tokens; flat-rate compute_cost() underestimates such calls.
    "grok-4.5": ModelPricing(input_per_1k=Decimal("0.002"), output_per_1k=Decimal("0.006")),
    "grok-4.3": ModelPricing(input_per_1k=Decimal("0.00125"), output_per_1k=Decimal("0.0025")),
    "grok-4.20-0309-reasoning": ModelPricing(input_per_1k=Decimal("0.00125"), output_per_1k=Decimal("0.0025")),
    "grok-4.20-0309-non-reasoning": ModelPricing(input_per_1k=Decimal("0.00125"), output_per_1k=Decimal("0.0025")),
    "grok-4.20-multi-agent-0309": ModelPricing(input_per_1k=Decimal("0.00125"), output_per_1k=Decimal("0.0025")),
    "grok-build-0.1": ModelPricing(input_per_1k=Decimal("0.001"), output_per_1k=Decimal("0.002")),
    # grok-2 era (retired)
    "grok-2": ModelPricing(input_per_1k=Decimal("0.002"), output_per_1k=Decimal("0.010")),
    "grok-2-mini": ModelPricing(input_per_1k=Decimal("0.0003"), output_per_1k=Decimal("0.0005")),
    "grok-beta": ModelPricing(input_per_1k=Decimal("0.005"), output_per_1k=Decimal("0.015")),
    # Together (8B Turbo left serverless 2026-03; 70B repriced $0.88→$1.04/1M)
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPricing(input_per_1k=Decimal("0.00104"), output_per_1k=Decimal("0.00104")),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelPricing(input_per_1k=Decimal("0.00018"), output_per_1k=Decimal("0.00018")),
    # Fireworks (Standard-tier serverless rates; llama-v3p3 retired from serverless)
    "accounts/fireworks/models/deepseek-v4-pro": ModelPricing(input_per_1k=Decimal("0.00174"), output_per_1k=Decimal("0.00348")),
    "accounts/fireworks/models/kimi-k3": ModelPricing(input_per_1k=Decimal("0.003"), output_per_1k=Decimal("0.015")),
    "accounts/fireworks/models/llama-v3p3-70b-instruct": ModelPricing(input_per_1k=Decimal("0.0009"), output_per_1k=Decimal("0.0009")),
    # Perplexity
    "sonar": ModelPricing(input_per_1k=Decimal("0.001"), output_per_1k=Decimal("0.001")),
    "sonar-pro": ModelPricing(input_per_1k=Decimal("0.003"), output_per_1k=Decimal("0.015")),
    # Mistral (Large 3 repriced $2/$6 → $0.5/$1.5 per 1M; Small 4 input $0.15)
    "mistral-large-latest": ModelPricing(input_per_1k=Decimal("0.0005"), output_per_1k=Decimal("0.0015")),
    "mistral-small-latest": ModelPricing(input_per_1k=Decimal("0.00015"), output_per_1k=Decimal("0.0006")),
    "open-mistral-nemo": ModelPricing(input_per_1k=Decimal("0.00015"), output_per_1k=Decimal("0.00015")),
    "codestral-latest": ModelPricing(input_per_1k=Decimal("0.0003"), output_per_1k=Decimal("0.0009")),
    # Cohere — command-r-plus-08-2024 is the only current chat model with a
    # published token rate (verified 2026-08-03); the command-a lineup is
    # instance-hour priced and deliberately has no rows (see _catalog.py).
    # Undated aliases retired 2025-09-15; last published rates kept.
    "command-r-plus-08-2024": ModelPricing(input_per_1k=Decimal("0.0025"), output_per_1k=Decimal("0.010")),
    "command-r-plus": ModelPricing(input_per_1k=Decimal("0.0025"), output_per_1k=Decimal("0.010")),
    "command-r": ModelPricing(input_per_1k=Decimal("0.00015"), output_per_1k=Decimal("0.0006")),
    "command": ModelPricing(input_per_1k=Decimal("0.001"), output_per_1k=Decimal("0.002")),
    "embed-english-v3.0": ModelPricing(input_per_1k=Decimal("0.0001"), output_per_1k=Decimal("0")),
    # AWS Bedrock (gateway; keyed on Bedrock model IDs; on-demand US rates)
    # Sonnet v2 moved to "Public Extended Access" 2025-12-01 at 2x the old
    # rate — the $3/$15 row still shown in Bedrock's main table is stale.
    "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelPricing(input_per_1k=Decimal("0.006"), output_per_1k=Decimal("0.030")),
    "anthropic.claude-3-haiku-20240307-v1:0": ModelPricing(input_per_1k=Decimal("0.00025"), output_per_1k=Decimal("0.00125")),
    "meta.llama3-1-70b-instruct-v1:0": ModelPricing(input_per_1k=Decimal("0.00072"), output_per_1k=Decimal("0.00072")),
    "amazon.nova-pro-v1:0": ModelPricing(input_per_1k=Decimal("0.0008"), output_per_1k=Decimal("0.0032")),
    # Nova Micro/Lite on-demand rates verified against aws.amazon.com/bedrock/pricing 2026-08-03.
    "amazon.nova-lite-v1:0": ModelPricing(input_per_1k=Decimal("0.00006"), output_per_1k=Decimal("0.00024")),
    "amazon.nova-micro-v1:0": ModelPricing(input_per_1k=Decimal("0.000035"), output_per_1k=Decimal("0.00014")),
    "mistral.mistral-large-2407-v1:0": ModelPricing(input_per_1k=Decimal("0.002"), output_per_1k=Decimal("0.006")),
    # Google Vertex (gateway for Gemini). Vertex prices differ from the Gemini
    # Developer API: 2.0-flash is $0.15/$0.60 per 1M on Vertex (the cheaper
    # $0.10/$0.40 figure is the Developer API rate, keyed as "gemini-2.0-flash").
    "gemini-1.5-pro-002": ModelPricing(input_per_1k=Decimal("0.00125"), output_per_1k=Decimal("0.005")),
    "gemini-1.5-flash-002": ModelPricing(input_per_1k=Decimal("0.000075"), output_per_1k=Decimal("0.0003")),
    "gemini-2.0-flash-001": ModelPricing(input_per_1k=Decimal("0.00015"), output_per_1k=Decimal("0.0006")),
}

# Fallback used when the exact model string is not found
_UNKNOWN_PRICING = ModelPricing(
    input_per_1k=Decimal("0.001"),
    output_per_1k=Decimal("0.003"),
)


# Bedrock cross-region inference profiles prefix model IDs with a geo scope.
_BEDROCK_PROFILE_SCOPES = ("us.", "eu.", "apac.", "global.", "us-gov.", "jp.", "au.")

_THOUSAND = Decimal("1000")


def _external_pricing(model: str) -> ModelPricing | None:
    """Fallback pricing from a maintained public source, for models absent from
    Capsera's own table.

    Reads LiteLLM's ``model_cost`` map (loaded from its published
    ``model_prices_and_context_window.json``, refreshed every litellm release)
    — but **only when litellm is ALREADY imported** in this process. We never
    import it ourselves: litellm is heavy and optional, so this is free where
    the customer already uses it (SDK/proxy processes) and a silent no-op
    everywhere else (e.g. the backend). LiteLLM quotes per-token USD; we convert
    to Capsera's per-1k convention. Returns None when the source is unavailable
    or has no entry, so the caller proceeds to the generic estimate.
    """
    litellm = sys.modules.get("litellm")
    cost_map = getattr(litellm, "model_cost", None) if litellm is not None else None
    if not isinstance(cost_map, dict):
        return None

    # Local import avoids a cycle at module load (_catalog imports nothing from _pricing).
    from ._catalog import strip_vendor_prefix

    for key in (model, strip_vendor_prefix(model)):
        entry = cost_map.get(key)
        if not isinstance(entry, dict):
            continue
        in_c = entry.get("input_cost_per_token")
        out_c = entry.get("output_cost_per_token")
        if in_c is None and out_c is None:
            continue
        try:
            return ModelPricing(
                input_per_1k=Decimal(str(in_c or 0)) * _THOUSAND,
                output_per_1k=Decimal(str(out_c or 0)) * _THOUSAND,
                cache_read_per_1k=Decimal(str(entry.get("cache_read_input_token_cost") or 0)) * _THOUSAND,
                cache_write_per_1k=Decimal(str(entry.get("cache_creation_input_token_cost") or 0)) * _THOUSAND,
            )
        except (ValueError, ArithmeticError):
            return None
    return None


def get_pricing(model: str) -> ModelPricing:
    """
    Return pricing for the given model ID.

    Lookup strategy:
    1. Exact match
    2. Exact match with Bedrock regional inference-profile scope stripped
       (e.g. "apac.amazon.nova-micro-v1:0" → "amazon.nova-micro-v1:0")
    3. Exact match with gateway vendor prefixes stripped
       (e.g. "anthropic/claude-sonnet-4-6" via LiteLLM/OpenRouter/Portkey
       → "claude-sonnet-4-6") — after exact lookups so real slash-keyed rows
       (Together "meta-llama/...", Fireworks "accounts/...") always win first
    4. Prefix match (e.g. "claude-sonnet-4-6-..." → "claude-sonnet-4-6")
    5. Fallback to a generic $0.001 / $0.003 estimate and log a warning
    """
    # Local import: _catalog does not import _pricing, so this is cycle-free.
    from ._catalog import strip_vendor_prefix

    candidates = [model]
    for scope in _BEDROCK_PROFILE_SCOPES:
        if model.startswith(scope):
            candidates.append(model[len(scope):])
            break
    stripped = strip_vendor_prefix(model)
    if stripped != model:
        candidates.append(stripped)

    for candidate in candidates:
        if candidate in PRICING:
            return PRICING[candidate]

    # Prefix match — handles versioned suffixes like "claude-opus-4-6-20260101"
    for candidate in candidates:
        for key in PRICING:
            if candidate.startswith(key):
                return PRICING[key]

    # Maintained public source (litellm's table, if already loaded) beats a
    # blind $0.001/$0.003 guess for models we simply don't carry yet.
    external = _external_pricing(model)
    if external is not None:
        return external

    import logging
    logging.getLogger("capsera").warning(
        "capsera: no pricing found for model %r — using fallback estimate", model
    )
    return _UNKNOWN_PRICING


def compute_cost(
    model: str,
    provider: str,  # noqa: ARG001 — reserved for future provider-specific overrides
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal:
    """
    Compute the USD cost for one LLM API call.

    All arithmetic is done in Decimal to avoid floating-point drift when
    costs are summed across thousands of events.
    """
    pricing = get_pricing(model)

    input_cost = (Decimal(input_tokens) / Decimal("1000")) * pricing.input_per_1k
    output_cost = (Decimal(output_tokens) / Decimal("1000")) * pricing.output_per_1k
    cache_read_cost = (Decimal(cache_read_tokens) / Decimal("1000")) * pricing.cache_read_per_1k
    cache_write_cost = (
        (Decimal(cache_write_tokens) / Decimal("1000")) * pricing.cache_write_per_1k
    )

    return input_cost + output_cost + cache_read_cost + cache_write_cost
