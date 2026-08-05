"""Model catalog — the single source of truth for model METADATA.

Deliberately price-free: prices live in ``_pricing.py`` (which the backend
already imports), and everything else — which provider a model belongs to,
its context window, quality/latency tier, and capability flags — lives here.
Provider inference used to be duplicated across the SDK router and three
backend modules; they all now read ``provider_for``/``model_meta`` so adding a
model is one row in one file.

``api_style`` records HOW the SDK captures the call:
  - ``anthropic`` / ``openai`` / ``google`` — native first-party SDKs.
  - ``openai_compatible`` — rides the ``openai`` SDK with a different
    ``base_url``; the OpenAI interceptor patch already captures the call, so
    the host is what disambiguates the provider (see ``OPENAI_COMPAT_HOSTS``).
  - ``mistral`` / ``cohere`` / ``bedrock`` / ``vertex`` — native SDKs with
    their own interceptor patches.

PRICING AND MODEL IDS: third-party model IDs are the real provider API IDs,
but the prices are published figures as of authoring and MUST be verified
before production billing — they are all in one file (``_pricing.py``) for
exactly that audit. First-party model IDs follow this repo's own timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class ModelMeta:
    provider: str
    api_style: str
    context_window: int
    quality_tier: str      # "high" | "medium" | "low"
    latency_tier: str      # "slow" | "medium" | "fast"
    capabilities: frozenset[str] = field(default_factory=frozenset)
    # The provider no longer offers this model. The row stays so historical
    # events keep their provider/tier labels and pricing, but nothing may
    # RECOMMEND a retired model (optimizer registry and margin downgrades
    # filter on this).
    retired: bool = False


# Capability vocabulary (for the future cross-provider routing guardrail):
# "tools" (function calling), "vision", "json" (structured output),
# "reasoning" (extended-thinking / o-series), "embeddings".
_C_CHAT = frozenset({"tools", "json"})
_C_CHAT_VISION = frozenset({"tools", "json", "vision"})
_C_REASON = frozenset({"reasoning", "json"})
_C_EMBED = frozenset({"embeddings"})


def _m(provider, api_style, ctx, quality, latency, caps=_C_CHAT, retired=False):
    return ModelMeta(provider, api_style, ctx, quality, latency, caps, retired)


MODELS: dict[str, ModelMeta] = {
    # ── Anthropic (first-party) ──────────────────────────────────────────
    "claude-opus-4-6": _m("anthropic", "anthropic", 200_000, "high", "slow", _C_CHAT_VISION),
    "claude-opus-4": _m("anthropic", "anthropic", 200_000, "high", "slow", _C_CHAT_VISION),
    "claude-3-opus-20240229": _m("anthropic", "anthropic", 200_000, "high", "slow", _C_CHAT_VISION),
    "claude-sonnet-4-6": _m("anthropic", "anthropic", 200_000, "high", "medium", _C_CHAT_VISION),
    "claude-sonnet-4": _m("anthropic", "anthropic", 200_000, "high", "medium", _C_CHAT_VISION),
    "claude-3-5-sonnet-20241022": _m("anthropic", "anthropic", 200_000, "high", "medium", _C_CHAT_VISION),
    "claude-3-5-sonnet-20240620": _m("anthropic", "anthropic", 200_000, "high", "medium", _C_CHAT_VISION),
    "claude-haiku-4-5-20251001": _m("anthropic", "anthropic", 200_000, "low", "fast", _C_CHAT_VISION),
    "claude-haiku-4-5": _m("anthropic", "anthropic", 200_000, "low", "fast", _C_CHAT_VISION),
    "claude-3-haiku-20240307": _m("anthropic", "anthropic", 200_000, "low", "fast", _C_CHAT_VISION),
    # ── OpenAI (first-party) ─────────────────────────────────────────────
    "gpt-4o": _m("openai", "openai", 128_000, "high", "medium", _C_CHAT_VISION),
    "gpt-4o-2024-11-20": _m("openai", "openai", 128_000, "high", "medium", _C_CHAT_VISION),
    "gpt-4o-mini": _m("openai", "openai", 128_000, "medium", "fast", _C_CHAT_VISION),
    "gpt-4o-mini-2024-07-18": _m("openai", "openai", 128_000, "medium", "fast", _C_CHAT_VISION),
    "gpt-4-turbo": _m("openai", "openai", 128_000, "high", "medium", _C_CHAT_VISION),
    "gpt-4": _m("openai", "openai", 8_192, "high", "slow"),
    "gpt-3.5-turbo": _m("openai", "openai", 16_385, "low", "fast"),
    "gpt-3.5-turbo-0125": _m("openai", "openai", 16_385, "low", "fast"),
    "o1": _m("openai", "openai", 200_000, "high", "slow", _C_REASON),
    "o1-mini": _m("openai", "openai", 128_000, "medium", "medium", _C_REASON),
    "o3-mini": _m("openai", "openai", 200_000, "medium", "medium", _C_REASON),
    "text-embedding-3-small": _m("openai", "openai", 8_191, "low", "fast", _C_EMBED),
    "text-embedding-3-large": _m("openai", "openai", 8_191, "medium", "fast", _C_EMBED),
    "text-embedding-ada-002": _m("openai", "openai", 8_191, "low", "fast", _C_EMBED),
    # ── Google Gemini (first-party) ──────────────────────────────────────
    "gemini-1.5-pro": _m("google", "google", 2_000_000, "high", "medium", _C_CHAT_VISION),
    "gemini-pro": _m("google", "google", 2_000_000, "high", "medium", _C_CHAT_VISION),
    "gemini-1.5-flash": _m("google", "google", 1_000_000, "medium", "fast", _C_CHAT_VISION),
    "gemini-flash": _m("google", "google", 1_000_000, "medium", "fast", _C_CHAT_VISION),
    "gemini-2.0-flash": _m("google", "google", 1_000_000, "medium", "fast", _C_CHAT_VISION),
    "gemini-2.0-pro": _m("google", "google", 2_000_000, "high", "medium", _C_CHAT_VISION),
    "gemini-2.5-pro": _m("google", "google", 1_048_576, "high", "medium", _C_CHAT_VISION),
    "gemini-2.5-flash": _m("google", "google", 1_048_576, "medium", "fast", _C_CHAT_VISION),
    "gemini-2.5-flash-lite": _m("google", "google", 1_048_576, "low", "fast", _C_CHAT_VISION),
    "gemini-3-pro": _m("google", "google", 1_048_576, "high", "medium", _C_CHAT_VISION),
    "gemini-3-flash": _m("google", "google", 1_048_576, "medium", "fast", _C_CHAT_VISION),
    "gemini-3.1-pro": _m("google", "google", 1_048_576, "high", "medium", _C_CHAT_VISION),
    "gemini-3.1-flash-lite": _m("google", "google", 1_048_576, "low", "fast", _C_CHAT_VISION),
    "gemini-3.5-flash": _m("google", "google", 1_048_576, "medium", "fast", _C_CHAT_VISION),
    "gemini-3.5-flash-lite": _m("google", "google", 1_048_576, "low", "fast", _C_CHAT_VISION),
    "gemini-3.6-flash": _m("google", "google", 1_048_576, "medium", "fast", _C_CHAT_VISION),

    # ── DeepSeek (OpenAI-compatible) ─────────────────────────────────────
    # v4 lineup verified 2026-08-03 (api-docs.deepseek.com). The chat/reasoner
    # aliases were discontinued 2026-07-24; both mapped to v4-flash modes.
    "deepseek-v4-flash": _m("deepseek", "openai_compatible", 1_000_000, "high", "fast",
                            frozenset({"tools", "json", "reasoning"})),
    "deepseek-v4-pro": _m("deepseek", "openai_compatible", 1_000_000, "high", "medium"),
    "deepseek-chat": _m("deepseek", "openai_compatible", 64_000, "high", "medium", retired=True),
    "deepseek-reasoner": _m("deepseek", "openai_compatible", 64_000, "high", "slow", _C_REASON, retired=True),
    # ── Groq (OpenAI-compatible) ─────────────────────────────────────────
    # Both llama rows verified 2026-08-03; Groq schedules them off free/dev
    # tiers 2026-08-16 (successors: gpt-oss-20b/120b) — revisit after that.
    "llama-3.3-70b-versatile": _m("groq", "openai_compatible", 128_000, "high", "fast"),
    "llama-3.1-8b-instant": _m("groq", "openai_compatible", 128_000, "low", "fast"),
    "mixtral-8x7b-32768": _m("groq", "openai_compatible", 32_768, "medium", "fast", retired=True),
    # ── xAI Grok (OpenAI-compatible) ─────────────────────────────────────
    # grok-4.x lineup verified 2026-08-03 (docs.x.ai). Published rates double
    # for prompts ≥200k tokens — compute_cost() is flat-rate, so long-context
    # xAI costs are underestimated. grok-2/beta era removed from the API.
    "grok-4.5": _m("xai", "openai_compatible", 500_000, "high", "medium"),
    "grok-4.3": _m("xai", "openai_compatible", 1_000_000, "high", "medium"),
    "grok-4.20-0309-reasoning": _m("xai", "openai_compatible", 1_000_000, "high", "slow", _C_REASON),
    "grok-4.20-0309-non-reasoning": _m("xai", "openai_compatible", 1_000_000, "high", "medium"),
    "grok-4.20-multi-agent-0309": _m("xai", "openai_compatible", 1_000_000, "high", "medium"),
    "grok-build-0.1": _m("xai", "openai_compatible", 256_000, "medium", "fast"),
    "grok-2": _m("xai", "openai_compatible", 131_072, "high", "medium", _C_CHAT_VISION, retired=True),
    "grok-2-mini": _m("xai", "openai_compatible", 131_072, "medium", "fast", retired=True),
    "grok-beta": _m("xai", "openai_compatible", 131_072, "high", "medium", retired=True),
    # ── Together (OpenAI-compatible) ─────────────────────────────────────
    # 70B verified 2026-08-03 ($1.04/$1.04 per 1M). The 8B Turbo left the
    # serverless price list 2026-03-06 (dedicated endpoints only).
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": _m("together", "openai_compatible", 131_072, "high", "medium"),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": _m("together", "openai_compatible", 131_072, "low", "fast", retired=True),
    # ── Fireworks (OpenAI-compatible) ────────────────────────────────────
    # Serverless lineup verified 2026-08-03. llama-v3p3-70b dropped from
    # serverless (on-demand GPU only) — no per-token price applies.
    "accounts/fireworks/models/deepseek-v4-pro": _m("fireworks", "openai_compatible", 1_048_576, "high", "medium"),
    "accounts/fireworks/models/kimi-k3": _m("fireworks", "openai_compatible", 1_048_576, "high", "medium"),
    "accounts/fireworks/models/llama-v3p3-70b-instruct": _m("fireworks", "openai_compatible", 131_072, "high", "fast", retired=True),
    # ── Perplexity (OpenAI-compatible) ───────────────────────────────────
    "sonar": _m("perplexity", "openai_compatible", 127_072, "medium", "fast"),
    "sonar-pro": _m("perplexity", "openai_compatible", 200_000, "high", "medium"),

    # ── Mistral (native mistralai SDK) ───────────────────────────────────
    "mistral-large-latest": _m("mistral", "mistral", 128_000, "high", "medium", _C_CHAT_VISION),
    "mistral-small-latest": _m("mistral", "mistral", 128_000, "medium", "fast"),
    "open-mistral-nemo": _m("mistral", "mistral", 128_000, "low", "fast"),
    "codestral-latest": _m("mistral", "mistral", 256_000, "medium", "fast"),
    # ── Cohere (native cohere SDK) ───────────────────────────────────────
    # Cohere moved to instance-hour "Model Vault" pricing: the current
    # command-a lineup (command-a-plus-05-2026, command-a-03-2025,
    # command-a-reasoning-08-2025, command-r7b-12-2024, embed-v4.0) publishes
    # NO per-token rate, so those models are deliberately absent — a catalog
    # row without a price would silently report $0 spend. Capture still works
    # (the interceptor labels provider="cohere" itself); only tier metadata
    # and cost are unavailable. command-r-plus-08-2024 is the one Live chat
    # model with a published token rate. The undated aliases retired 2025-09-15.
    "command-r-plus-08-2024": _m("cohere", "cohere", 128_000, "high", "medium"),
    "command-r-plus": _m("cohere", "cohere", 128_000, "high", "medium", retired=True),
    "command-r": _m("cohere", "cohere", 128_000, "medium", "fast", retired=True),
    "command": _m("cohere", "cohere", 4_096, "low", "fast", retired=True),
    "embed-english-v3.0": _m("cohere", "cohere", 512, "medium", "fast", _C_EMBED),

    # ── AWS Bedrock (native boto3) — gateway, keyed on Bedrock model IDs ──
    "anthropic.claude-3-5-sonnet-20241022-v2:0": _m("bedrock", "bedrock", 200_000, "high", "medium", _C_CHAT_VISION),
    "anthropic.claude-3-haiku-20240307-v1:0": _m("bedrock", "bedrock", 200_000, "low", "fast", _C_CHAT_VISION),
    "meta.llama3-1-70b-instruct-v1:0": _m("bedrock", "bedrock", 128_000, "high", "medium"),
    "amazon.nova-pro-v1:0": _m("bedrock", "bedrock", 300_000, "high", "medium", _C_CHAT_VISION),
    "amazon.nova-lite-v1:0": _m("bedrock", "bedrock", 300_000, "medium", "fast", _C_CHAT_VISION),
    "amazon.nova-micro-v1:0": _m("bedrock", "bedrock", 128_000, "low", "fast", _C_CHAT),
    "mistral.mistral-large-2407-v1:0": _m("bedrock", "bedrock", 128_000, "high", "medium"),
    # ── Google Vertex (native vertexai) — gateway for Gemini ─────────────
    "gemini-1.5-pro-002": _m("vertex", "vertex", 2_000_000, "high", "medium", _C_CHAT_VISION),
    "gemini-1.5-flash-002": _m("vertex", "vertex", 1_000_000, "medium", "fast", _C_CHAT_VISION),
    "gemini-2.0-flash-001": _m("vertex", "vertex", 1_000_000, "medium", "fast", _C_CHAT_VISION),
}


# Host substrings → provider, for OpenAI-compatible + Azure disambiguation at
# capture time. Ordered longest/most-specific first is unnecessary because the
# substrings are disjoint. Local runtimes (Ollama default :11434) map to a
# provider so their $0 spend is still attributed.
OPENAI_COMPAT_HOSTS: tuple[tuple[str, str], ...] = (
    ("api.deepseek.com", "deepseek"),
    ("api.groq.com", "groq"),
    ("api.x.ai", "xai"),
    ("api.together.xyz", "together"),
    ("api.together.ai", "together"),
    ("api.fireworks.ai", "fireworks"),
    ("api.perplexity.ai", "perplexity"),
    ("openrouter.ai", "openrouter"),
    ("api.mistral.ai", "mistral"),          # Mistral also exposes an OpenAI-compat endpoint
    (".openai.azure.com", "azure"),
    ("localhost:11434", "ollama"),
    ("127.0.0.1:11434", "ollama"),
    (":11434", "ollama"),
)


def _host_provider(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url if "//" in base_url else "//" + base_url)
    hay = (parsed.netloc + parsed.path).lower()
    for needle, provider in OPENAI_COMPAT_HOSTS:
        if needle in hay:
            return provider
    return None


def model_meta(model: str) -> ModelMeta | None:
    """Metadata for a model ID, exact then prefix match (mirrors get_pricing)."""
    if model in MODELS:
        return MODELS[model]
    for key, meta in MODELS.items():
        if model.startswith(key):
            return meta
    return None


def provider_for(model: str, base_url: str | None = None) -> str:
    """Resolve the provider for a captured call.

    The base_url host wins when present — it is the only thing that
    distinguishes OpenAI-compatible providers (and Azure) from real OpenAI,
    since they all ride the ``openai`` SDK. Otherwise fall back to the model's
    catalogued provider, then ``"unknown"``.
    """
    host = _host_provider(base_url)
    if host:
        return host
    meta = model_meta(model)
    return meta.provider if meta else "unknown"
