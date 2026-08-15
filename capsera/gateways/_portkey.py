"""Portkey integration — capture for calls made through the ``portkey_ai`` SDK.

Portkey is reachable two ways, and they are deliberately NOT equal in how much
they couple Capsera to Portkey's code:

1. **OpenAI SDK + Portkey base_url** (RECOMMENDED, lowest coupling) — pointing
   the ``openai`` client at Portkey's OpenAI-compatible URL. Captured by the
   OpenAI interceptor patch; the gateway tag comes from base_url host detection
   (``_catalog.GATEWAY_HOSTS``). This path rides a *stable public HTTP
   contract*, so it survives ``portkey_ai`` version changes — prefer it when a
   choice exists (see docs/gateway-integrations.md). Nothing to do here.
2. **Native ``portkey_ai`` client** (``Portkey().chat.completions.create``)
   — the automatic FALLBACK, patched here for code that uses the portkey_ai
   client directly. The client rides Portkey's own HTTP layer (not the openai
   package), so without this patch those calls are invisible. This path reaches
   into ``portkey_ai`` internals (absorbed by the candidate class lists below),
   so it is inherently more exposed to that library's churn than path 1.

Portkey responses mirror the OpenAI schema (``.usage.prompt_tokens`` /
``.completion_tokens`` / ``.model``), so extraction reuses the interceptor's
OpenAI emit helpers. The provider is resolved from the model ID via the
catalog (Portkey routes to any provider); unresolvable models fall back to
``"portkey"`` so the spend is still attributed somewhere visible.

Streaming through the native client is passed through untouched (no success
capture) in this version — errors are still recorded.
"""
from __future__ import annotations

import functools
import importlib
import logging
import time
from typing import Any

from .._catalog import provider_for, strip_vendor_prefix

# See the note in _litellm.py: the runtime module must be resolved via
# importlib because the package attribute of the same name is shadowed by
# the Interceptor-instance singleton in capsera/__init__.
_runtime = importlib.import_module("capsera._interceptor")

logger = logging.getLogger("capsera")

GATEWAY_NAME = "portkey"

# (module path, [sync class candidates], [async class candidates], emit style)
# Candidate lists absorb portkey_ai's internal naming drift across versions;
# the first class that exists with a ``create`` method is patched.
_SURFACES = (
    (
        "portkey_ai.api_resources.apis.chat_complete",
        ("Completions", "ChatCompletions", "ChatCompletion"),
        ("AsyncCompletions", "AsyncChatCompletions", "AsyncChatCompletion"),
        "chat",
    ),
    (
        "portkey_ai.api_resources.apis.embeddings",
        ("Embeddings",),
        ("AsyncEmbeddings",),
        "embedding",
    ),
)


def _provider_for_model(model: Any) -> str:
    resolved = provider_for(strip_vendor_prefix(str(model or "")))
    return resolved if resolved != "unknown" else GATEWAY_NAME


def _emit_success(style: str, response: Any, kwargs: dict[str, Any],
                  latency_ms: int, caller: str) -> None:
    cfg = _runtime.config
    emitter = _runtime.emitter
    if cfg is None or emitter is None:
        return
    model = getattr(response, "model", None) or kwargs.get("model", "unknown")
    if style == "embedding":
        _runtime._emit_openai_embedding_response(
            response=response,
            request_kwargs=kwargs,
            latency_ms=latency_ms,
            config=cfg,
            emitter=emitter,
            caller_file=caller,
            provider=_provider_for_model(model),
            gateway=GATEWAY_NAME,
        )
    else:
        _runtime._emit_openai_response(
            response=response,
            request_kwargs=kwargs,
            latency_ms=latency_ms,
            config=cfg,
            emitter=emitter,
            original_model=None,
            caller_file=caller,
            provider=_provider_for_model(model),
            gateway=GATEWAY_NAME,
        )


def _wrap_sync(original: Any, style: str) -> Any:
    @functools.wraps(original)
    def patched(self_res: Any, *args: Any, **kwargs: Any) -> Any:
        caller = _runtime._caller_location()
        start = time.perf_counter()
        try:
            response = original(self_res, *args, **kwargs)
        except Exception as exc:
            _runtime._emit_simple_error(
                provider=_provider_for_model(kwargs.get("model")),
                model=str(kwargs.get("model") or "unknown"),
                start=start, error=exc, caller_file=caller,
            )
            raise
        if kwargs.get("stream"):
            return response
        _emit_success(style, response, kwargs,
                      int((time.perf_counter() - start) * 1000), caller)
        return response

    return patched


def _wrap_async(original: Any, style: str) -> Any:
    @functools.wraps(original)
    async def patched(self_res: Any, *args: Any, **kwargs: Any) -> Any:
        caller = _runtime._caller_location()
        start = time.perf_counter()
        try:
            response = await original(self_res, *args, **kwargs)
        except Exception as exc:
            _runtime._emit_simple_error(
                provider=_provider_for_model(kwargs.get("model")),
                model=str(kwargs.get("model") or "unknown"),
                start=start, error=exc, caller_file=caller,
            )
            raise
        if kwargs.get("stream"):
            return response
        _emit_success(style, response, kwargs,
                      int((time.perf_counter() - start) * 1000), caller)
        return response

    return patched


def _patch_first(module: Any, candidates: tuple[str, ...], wrap: Any, style: str) -> bool:
    for cls_name in candidates:
        cls = getattr(module, cls_name, None)
        if cls is None or not hasattr(cls, "create"):
            continue
        if getattr(cls, "_capsera_patched_create", False):
            return True
        cls.create = wrap(cls.create, style)
        cls._capsera_patched_create = True
        return True
    return False


def install_portkey() -> bool:
    """Patch the portkey_ai chat/embeddings surfaces. Idempotent, fail-open."""
    try:
        importlib.import_module("portkey_ai")
    except ImportError:
        return False
    except Exception:  # noqa: BLE001 — a broken install must not kill init()
        logger.warning("capsera: portkey_ai import failed — Portkey calls will not be tracked",
                       exc_info=True)
        return False

    patched_any = False
    for module_path, sync_candidates, async_candidates, style in _SURFACES:
        try:
            module = importlib.import_module(module_path)
        except Exception:  # noqa: BLE001
            continue
        try:
            patched_any = _patch_first(module, sync_candidates, _wrap_sync, style) or patched_any
            patched_any = _patch_first(module, async_candidates, _wrap_async, style) or patched_any
        except Exception:  # noqa: BLE001
            logger.debug("capsera: portkey patch failed for %s", module_path, exc_info=True)

    if not patched_any:
        logger.warning(
            "capsera: portkey_ai installed but no known chat surface found — "
            "Portkey native-SDK calls will not be tracked "
            "(OpenAI-SDK-with-Portkey-base_url capture is unaffected)"
        )
        return False

    # Defensive: if a portkey_ai version ever drives the openai package
    # internally, the inner patch must not double-count.
    _runtime.register_gateway_suppression("portkey_ai")
    if _runtime.config is not None and _runtime.config.debug:
        logger.debug("capsera: patched portkey_ai surfaces")
    return True
