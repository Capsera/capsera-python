"""
Cached-token accounting, and the one thing every provider disagrees about.

A cached prompt is billed in two parts: the tokens served from cache (cheap)
and the tokens that were not (full price). To record that split we need the
cached count *and* we need to know whether the provider's headline prompt
total already contains it. Providers differ, and the difference is invisible
in the payload:

    Anthropic   input_tokens EXCLUDES cache_read_input_tokens and
                cache_creation_input_tokens. The three are additive; the
                prompt is their sum.

    OpenAI      prompt_tokens INCLUDES prompt_tokens_details.cached_tokens.
                The cached count is a subset, not an addition.

    Gemini      prompt_token_count INCLUDES cached_content_token_count, same
                shape as OpenAI.

Get this backwards in one direction and every cached token is billed twice —
once at the input rate inside the prompt total, once at the cache rate beside
it. Get it backwards in the other and the uncached remainder goes unbilled.
Neither shows up as an error; both just quietly produce a number the partner
can disprove against their provider invoice, which §2 of the test plan ranks
as the third-worst failure this product has.

So the rule is stated per provider, in one place, and applied through one
function rather than open-coded at each of the eight emit sites.
"""
from __future__ import annotations

# Providers whose reported prompt/input token count already contains the
# tokens that were served from cache. For these the cached count must be
# subtracted out before it is recorded separately.
#
# Anthropic is deliberately absent: it reports the three counts as disjoint
# buckets, so subtracting would drop real uncached input on the floor.
CACHED_TOKENS_INCLUDED_IN_INPUT: frozenset[str] = frozenset({
    "openai",
    "azure_openai",
    "google",
    "vertex_ai",
    "mistral",
    "deepseek",
    "groq",
    "xai",
})


def cached_is_included(provider: str | None) -> bool:
    """Does this provider's prompt total already contain its cached tokens?

    Unknown providers default to True — the OpenAI shape. Almost every
    OpenAI-compatible API inherits ``prompt_tokens`` semantics along with the
    wire format, and the failure modes are not symmetric: assuming *included*
    for a provider that actually reports disjoint buckets under-counts the
    uncached remainder, while assuming *disjoint* for an OpenAI-shaped one
    double-bills every cached token. Under-counting is the safer wrong answer
    (test plan §2 ranks it P1; over-reporting is P0).
    """
    return (provider or "").lower() not in _DISJOINT_PROVIDERS


# The inverse of CACHED_TOKENS_INCLUDED_IN_INPUT, for the default-True rule
# above: providers known to report cache counts as separate buckets.
_DISJOINT_PROVIDERS: frozenset[str] = frozenset({"anthropic", "bedrock"})


def split_cached_input(
    prompt_tokens: int,
    cached_tokens: int,
    *,
    included: bool,
) -> tuple[int, int]:
    """Split a prompt total into (uncached_input, cache_read).

    ``included`` says whether ``prompt_tokens`` already contains
    ``cached_tokens`` — see ``cached_is_included``.

    A cached count larger than the prompt total is incoherent, and has been
    seen from proxies that merge usage from several upstream calls. Rather
    than emit a negative input count that would show up as a credit on the
    dashboard, the cached figure is discarded and the prompt total kept whole:
    the cost stays right even though the split is lost.
    """
    prompt = max(int(prompt_tokens or 0), 0)
    cached = max(int(cached_tokens or 0), 0)

    if not cached:
        return prompt, 0
    if not included:
        return prompt, cached
    if cached > prompt:
        return prompt, 0
    return prompt - cached, cached


def openai_cached_tokens(usage: object) -> int:
    """Read ``usage.prompt_tokens_details.cached_tokens``, 0 when absent.

    The details object is missing entirely on older API versions and on
    responses that used no cache, and some OpenAI-compatible servers return it
    as a plain dict.
    """
    details = _get(usage, "prompt_tokens_details")
    if details is None:
        return 0
    return _int(_get(details, "cached_tokens"))


def gemini_cached_tokens(usage_metadata: object) -> int:
    """Read Gemini's cached-content count across both SDK spellings.

    ``google-generativeai`` and ``google-genai`` expose snake_case attributes;
    the REST payload and some wrappers keep the camelCase JSON name.
    """
    for name in ("cached_content_token_count", "cachedContentTokenCount"):
        value = _get(usage_metadata, name)
        if value:
            return _int(value)
    return 0


def _get(obj: object, name: str) -> object | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(value: object) -> int:
    """Coerce a usage field to a count, accepting only real numbers.

    Deliberately not ``int(value)``: that coerces anything implementing
    ``__int__``, and a token count is not a thing worth guessing at. Test
    doubles are the loud case — a bare ``MagicMock`` auto-creates
    ``prompt_tokens_details.cached_tokens`` and ``int()`` on it yields 1, which
    would invent a cached token on every mocked call. The quiet case is the one
    that matters in production: an SDK that returns a sentinel or lazy proxy
    for an absent field would silently become a nonzero cached count, and
    cached tokens are subtracted from the prompt total — so a fabricated 1 does
    not merely add noise, it removes a real input token from the bill.

    JSON numbers arrive as int or float; some proxies stringify them. Anything
    else is treated as absent.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value.strip()), 0)
        except ValueError:
            return 0
    return 0
