"""
Monkey-patch interceptor for Anthropic and OpenAI LLM clients.

This module wraps the SDK methods of each provider at import time so that
every LLM call — regardless of the framework used (LangGraph, CrewAI, raw
API calls, etc.) — is transparently recorded without any changes to the
caller's code.

Patching strategy
-----------------
* Anthropic: ``anthropic.resources.messages.Messages.create`` (sync) and
  ``anthropic.resources.messages.AsyncMessages.create`` (async).
  Streaming variants are wrapped so the usage block from the final stream
  event is captured after iteration completes.
* OpenAI: ``openai.resources.chat.completions.Completions.create`` (sync)
  and ``openai.resources.chat.completions.AsyncCompletions.create`` (async).
  Streaming variants capture usage from the final ``ChatCompletionChunk``
  that carries a ``usage`` object (requires ``stream_options={"include_usage": True}``
  which we inject automatically).
* Google Gemini: ``google.generativeai.GenerativeModel.generate_content``
  and ``generate_content_async`` plus ``google.genai.models.Models.generate_content``
  when available.

Call-stack inspection
---------------------
We walk up the Python call stack looking for the first frame that does NOT
belong to the capsera package, httpx, anthropic, or openai internals.
That frame's ``filename:lineno`` is stored as ``caller_file``.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import sys
import logging
import threading
import time
import weakref
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Generator, Optional
from uuid import uuid4

import httpx

from . import _catalog
from ._config import Config
from ._context import _context_stack
from ._emitter import Emitter
from ._models import BudgetExceededError, PromptMetadata, TokenEvent
from ._pricing import compute_cost
from ._prompt_analyzer import PromptAnalyzer
from ._router import Router
from ._version import __version__

logger = logging.getLogger("capsera")

# --------------------------------------------------------------------------- #
# Active runtime
#
# The patched provider methods reference these module globals *at call time*
# (they intentionally do NOT capture locals). Re-running capsera.init() swaps
# them via Interceptor.install(), so already-patched classes immediately use
# the new emitter/config/router/analyzer. Binding these into the closures at
# patch time (the previous behaviour) stranded events in a retired emitter
# after a second init() — shutdown()/atexit flushed only the new emitter and
# the queued events were silently lost.
# --------------------------------------------------------------------------- #
config: Config = None  # type: ignore[assignment]  # set by Interceptor.install()
emitter: Emitter = None  # type: ignore[assignment]  # set by Interceptor.install()
router: Optional[Router] = None
analyzer: Optional[PromptAnalyzer] = None

# Frames belonging to these path fragments are skipped when identifying the
# true caller in the user's code.
_SKIP_FRAME_FRAGMENTS_BASE = [
    "capsera",
    "site-packages/anthropic",
    "site-packages/openai",
    "site-packages/google",
    "site-packages/google_generativeai",
    "site-packages/httpx",
    "site-packages/httpcore",
    # Windows path separators
    "site-packages\\anthropic",
    "site-packages\\openai",
    "site-packages\\google",
    "site-packages\\google_generativeai",
    "site-packages\\httpx",
    "site-packages\\httpcore",
]

# Frameworks to auto-detect: (import_name, path_fragment_posix, path_fragment_windows)
_FRAMEWORK_DETECT = [
    ("langchain", "site-packages/langchain", "site-packages\\langchain"),
    ("langgraph", "site-packages/langgraph", "site-packages\\langgraph"),
    ("crewai", "site-packages/crewai", "site-packages\\crewai"),
    ("llama_index", "site-packages/llama_index", "site-packages\\llama_index"),
    ("autogen", "site-packages/autogen", "site-packages\\autogen"),
    ("haystack", "site-packages/haystack", "site-packages\\haystack"),
    ("dspy", "site-packages/dspy", "site-packages\\dspy"),
]

_skip_fragments_initialized = False


def _detect_frameworks() -> tuple[str, ...]:
    """Auto-detect installed AI frameworks and build the skip-frame list."""
    global _skip_fragments_initialized
    if _skip_fragments_initialized:
        return _SKIP_FRAME_FRAGMENTS

    fragments = list(_SKIP_FRAME_FRAGMENTS_BASE)
    for import_name, posix_frag, win_frag in _FRAMEWORK_DETECT:
        try:
            __import__(import_name)
            fragments.append(posix_frag)
            fragments.append(win_frag)
            logger.debug("capsera: detected framework %s, added to skip list", import_name)
        except ImportError:
            pass

    _skip_fragments_initialized = True
    return tuple(fragments)


_SKIP_FRAME_FRAGMENTS: tuple[str, ...] = tuple(_SKIP_FRAME_FRAGMENTS_BASE)


def _caller_location(stack_start: int = 1) -> str:
    """
    Walk the call stack and return the first non-SDK frame as "filename:lineno".

    ``stack_start`` allows the caller to skip extra wrapper frames at the top
    of the stack.

    Walks raw frames via sys._getframe rather than inspect.stack(): this runs
    on the hot path of every LLM call, and inspect.stack() resolves source
    context for every frame (hundreds of ms on the first call while linecache
    warms up).
    """
    try:
        skip = _SKIP_FRAME_FRAGMENTS
        frame = sys._getframe(stack_start)
        while frame is not None:
            filename = frame.f_code.co_filename
            if not any(frag in filename for frag in skip):
                short = filename.replace("\\", "/").split("/")[-1]
                return f"{short}:{frame.f_lineno}"
            frame = frame.f_back
    except Exception:
        pass
    return "unknown"


def _build_event(
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    latency_ms: int,
    config: Config,
    original_model: Optional[str] = None,
    caller_file: str = "unknown",
    prompt_meta: Optional[PromptMetadata] = None,
    is_error: bool = False,
    error_type: Optional[str] = None,
) -> TokenEvent:
    """Construct a TokenEvent from extracted response data and current context."""
    ctx = _context_stack.current()
    agent_id = (ctx.agent if ctx else None) or config.default_agent or "unknown"
    team_id = (ctx.team if ctx else None) or config.default_team
    task_type = ctx.task_type if ctx else None
    session_id = (ctx.session_id if ctx else None) or _context_stack.get_session()
    customer_id = ctx.customer_id if ctx else None
    cost_center = ctx.cost_center if ctx else None

    cost = compute_cost(
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )

    event = TokenEvent(
        id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=str(uuid4()),
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost_usd=float(cost),
        agent_id=agent_id,
        team_id=team_id,
        task_type=task_type,
        session_id=session_id,
        customer_id=customer_id,
        cost_center=cost_center,
        env=config.env,
        latency_ms=latency_ms,
        caller_file=caller_file,
        was_routed=original_model is not None,
        original_model=original_model,
        is_error=is_error,
        error_type=error_type,
    )

    if prompt_meta:
        event.system_prompt_hash = prompt_meta.system_prompt_hash
        event.system_prompt_tokens = prompt_meta.system_prompt_tokens
        event.message_count = prompt_meta.message_count
        event.conversation_tokens = prompt_meta.conversation_tokens
        event.has_few_shot = prompt_meta.has_few_shot
        event.few_shot_pairs = prompt_meta.few_shot_pairs
        event.tool_result_tokens = prompt_meta.tool_result_tokens
        event.conversation_turns = prompt_meta.conversation_turns
        event.context_window_utilization = prompt_meta.context_window_utilization

    if config.debug:
        # The debug tap: one structured line per intercepted call, at the
        # single choke point every emit path flows through.
        logger.info(
            "capsera: intercepted provider=%s model=%s tokens=%d/%d "
            "cost_usd=%s agent=%s routed=%s error=%s caller=%s",
            provider, model, input_tokens, output_tokens, event.cost_usd,
            agent_id, event.was_routed, is_error, caller_file,
        )

    return event


def get_interception_report() -> dict[str, str]:
    """Which known LLM surfaces are actually hooked, right now.

    Values: ``"patched"`` (calls are captured), ``"unpatched"`` (library
    importable but not hooked — call ``capsera.init()``), ``"not-installed"``
    (library absent). Lets an app verify at startup that everything it uses
    is being tracked, instead of discovering silence on the dashboard.
    """
    report: dict[str, str] = {}

    def probe(surface: str, resolve: Any, guard: Any) -> None:
        try:
            target = resolve()
        except Exception:  # noqa: BLE001 — absence, not failure
            report[surface] = "not-installed"
            return
        try:
            report[surface] = "patched" if guard(target) else "unpatched"
        except Exception:  # noqa: BLE001
            report[surface] = "unpatched"

    def cls_flag(target: Any) -> bool:
        return bool(getattr(target, "_capsera_patched", False))

    def method_flag(name: str) -> Any:
        return lambda target: bool(getattr(target, f"_capsera_patched_{name}", False))

    def fn_flag(name: str) -> Any:
        return lambda target: bool(
            getattr(getattr(target, name), "_capsera_patched", False))

    def _p(path: str) -> Any:
        module_name, _, attr_path = path.partition(":")
        def resolve() -> Any:
            target: Any = __import__(module_name, fromlist=["_"])
            for part in attr_path.split("."):
                target = getattr(target, part)
            return target
        return resolve

    probe("anthropic.messages", _p("anthropic:resources.messages.Messages"), cls_flag)
    probe("anthropic.messages_async",
          _p("anthropic:resources.messages.AsyncMessages"), cls_flag)
    probe("openai.chat_completions",
          _p("openai:resources.chat.completions.Completions"), cls_flag)
    probe("openai.chat_completions_async",
          _p("openai:resources.chat.completions.AsyncCompletions"), cls_flag)
    probe("openai.embeddings", _p("openai:resources.embeddings.Embeddings"), cls_flag)
    probe("openai.embeddings_async",
          _p("openai:resources.embeddings.AsyncEmbeddings"), cls_flag)
    probe("google.generativeai",
          _p("google.generativeai:GenerativeModel"), fn_flag("generate_content"))
    probe("google.genai", _p("google.genai:models.Models"), fn_flag("generate_content"))
    probe("mistral.chat_complete", _p("mistralai.chat:Chat"), method_flag("complete"))
    probe("cohere.chat", _p("cohere:ClientV2"), method_flag("chat"))
    probe("vertex.generate_content",
          _p("vertexai.generative_models:GenerativeModel"),
          method_flag("generate_content"))
    probe("bedrock.converse", _p("botocore.client:BaseClient"),
          method_flag("_make_api_call"))
    return report


class Interceptor:
    """
    Installs monkey-patches on Anthropic and OpenAI SDK classes.

    The ``install()`` method is idempotent — re-patching an already-patched
    class is a no-op (the attribute ``_capsera_patched`` is set on the class to
    detect this).
    """

    def __init__(
        self,
        config: Config,
        emitter: Emitter,
        router: Optional[Router] = None,
        analyzer: Optional[PromptAnalyzer] = None,
    ) -> None:
        self._config = config
        self._emitter = emitter
        self._router = router
        self._analyzer = analyzer

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def _publish_runtime(self) -> None:
        """
        Expose this interceptor's config/emitter/router/analyzer as the
        module-level active runtime that patched closures resolve at call
        time. Published on every install/patch attempt — including when the
        target class is already patched — which is what makes repeated
        capsera.init() calls take effect instead of leaving closures bound to
        a retired emitter.
        """
        global config, emitter, router, analyzer
        config = self._config
        emitter = self._emitter
        router = self._router
        analyzer = self._analyzer

    def install(self) -> None:
        """Attempt to patch every known provider; silently skip missing ones."""
        global _SKIP_FRAME_FRAGMENTS
        _SKIP_FRAME_FRAGMENTS = _detect_frameworks()
        self._publish_runtime()
        self._patch_anthropic()
        self._patch_openai()
        self._patch_google()
        self._patch_openai_embeddings()
        # Native-SDK providers. Each is guarded: a missing package or a changed
        # internal path just skips (no capture, no crash), exactly like the
        # first-party patches above. Structurally tested against the shapes
        # patched here; validate each once against the live SDK.
        self._patch_mistral()
        self._patch_cohere()
        self._patch_vertex()
        self._patch_bedrock()

    # ================================================================== #
    # Native-SDK providers (Mistral, Cohere, Vertex, Bedrock)
    # ================================================================== #

    def _patch_method(self, cls: Any, method_name: str, make_wrapper: Any) -> None:
        """Idempotently monkeypatch ``cls.method_name`` with ``make_wrapper``.

        ``make_wrapper(original)`` returns the replacement. A per-class
        ``_capsera_patched_<method>`` guard makes re-init a no-op instead of
        double-wrapping. Fail-open on any AttributeError.
        """
        guard = f"_capsera_patched_{method_name}"
        try:
            original = getattr(cls, method_name)
        except AttributeError:
            return
        if getattr(cls, guard, False):
            return
        setattr(cls, method_name, make_wrapper(original))
        setattr(cls, guard, True)

    def _patch_mistral(self) -> None:
        self._publish_runtime()
        try:
            from mistralai.chat import Chat  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return

        def wrap(original: Any) -> Any:
            @functools.wraps(original)
            def patched(self_client: Any, **kwargs: Any) -> Any:
                caller = _caller_location()
                start = time.perf_counter()
                try:
                    resp = original(self_client, **kwargs)
                except Exception as exc:
                    _emit_simple_error(provider="mistral",
                                       model=kwargs.get("model", "unknown"),
                                       start=start, error=exc, caller_file=caller)
                    raise
                _emit_native_response(
                    resp, provider="mistral",
                    fallback_model=kwargs.get("model", "unknown"),
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    caller_file=caller, style="openai_usage",
                )
                return resp
            return patched

        self._patch_method(Chat, "complete", wrap)

    def _patch_cohere(self) -> None:
        self._publish_runtime()
        try:
            import cohere  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return

        def wrap(original: Any) -> Any:
            @functools.wraps(original)
            def patched(self_client: Any, *args: Any, **kwargs: Any) -> Any:
                caller = _caller_location()
                start = time.perf_counter()
                try:
                    resp = original(self_client, *args, **kwargs)
                except Exception as exc:
                    _emit_simple_error(provider="cohere",
                                       model=kwargs.get("model", "unknown"),
                                       start=start, error=exc, caller_file=caller)
                    raise
                _emit_native_response(
                    resp, provider="cohere",
                    fallback_model=kwargs.get("model", "unknown"),
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    caller_file=caller, style="cohere",
                )
                return resp
            return patched

        for cls_name in ("ClientV2", "Client"):
            cls = getattr(cohere, cls_name, None)
            if cls is not None:
                self._patch_method(cls, "chat", wrap)

    def _patch_vertex(self) -> None:
        self._publish_runtime()
        try:
            from vertexai.generative_models import GenerativeModel  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return

        def wrap(original: Any) -> Any:
            @functools.wraps(original)
            def patched(self_model: Any, *args: Any, **kwargs: Any) -> Any:
                caller = _caller_location()
                model_name = (getattr(self_model, "_model_name", None)
                              or getattr(self_model, "model_name", "unknown"))
                start = time.perf_counter()
                try:
                    resp = original(self_model, *args, **kwargs)
                except Exception as exc:
                    _emit_simple_error(provider="vertex", model=model_name,
                                       start=start, error=exc, caller_file=caller)
                    raise
                _emit_native_response(
                    resp, provider="vertex",
                    fallback_model=model_name,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    caller_file=caller, style="google_usage",
                )
                return resp
            return patched

        self._patch_method(GenerativeModel, "generate_content", wrap)

    def _patch_bedrock(self) -> None:
        self._publish_runtime()
        try:
            from botocore.client import BaseClient  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return

        def wrap(original: Any) -> Any:
            @functools.wraps(original)
            def patched(self_client: Any, operation_name: str, api_params: dict) -> Any:
                # boto3 clients are dynamically generated, so we patch the one
                # shared entry point and filter to bedrock-runtime's Converse.
                # InvokeModel bodies are provider-specific JSON (a follow-up).
                svc = getattr(getattr(self_client, "meta", None), "service_model", None)
                service = getattr(svc, "service_name", "")
                if service != "bedrock-runtime" or operation_name not in ("Converse", "ConverseStream"):
                    return original(self_client, operation_name, api_params)
                caller = _caller_location()
                model_id = (api_params.get("modelId", "unknown")
                            if isinstance(api_params, dict) else "unknown")
                start = time.perf_counter()
                try:
                    resp = original(self_client, operation_name, api_params)
                except Exception as exc:
                    _emit_simple_error(provider="bedrock", model=model_id,
                                       start=start, error=exc, caller_file=caller)
                    raise
                _emit_native_response(
                    resp, provider="bedrock",
                    fallback_model=model_id,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    caller_file=caller, style="bedrock_converse",
                )
                return resp
            return patched

        self._patch_method(BaseClient, "_make_api_call", wrap)

    # ================================================================== #
    # Anthropic
    # ================================================================== #

    def _patch_anthropic(self) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError:
            logger.warning("capsera: could not patch anthropic — LLM calls will not be tracked")
            return

        self._patch_anthropic_sync(anthropic)
        self._patch_anthropic_async(anthropic)

    def _patch_anthropic_sync(self, anthropic: Any) -> None:
        self._publish_runtime()
        try:
            messages_cls = anthropic.resources.messages.Messages
        except AttributeError:
            return

        if getattr(messages_cls, "_capsera_patched", False):
            return

        original_create = messages_cls.create

        @functools.wraps(original_create)
        def patched_create(self_client: Any, **kwargs: Any) -> Any:
            caller = _caller_location()
            kwargs, original_model = _maybe_route(kwargs, config, router)
            prompt_meta = _maybe_analyze(kwargs, "anthropic", analyzer)
            preflight_original = _enforce_budget(kwargs, "anthropic", config, prompt_meta)
            original_model = original_model or preflight_original

            start = time.perf_counter()
            try:
                response = original_create(self_client, **kwargs)
            except Exception as exc:
                _emit_error_event(
                    provider="anthropic",
                    request_kwargs=kwargs,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    error=exc,
                    config=config,
                    emitter=emitter,
                    original_model=original_model,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                )
                raise
            latency_ms = int((time.perf_counter() - start) * 1000)

            _emit_anthropic_response(
                response=response,
                request_kwargs=kwargs,
                latency_ms=latency_ms,
                config=config,
                emitter=emitter,
                original_model=original_model,
                caller_file=caller,
                prompt_meta=prompt_meta,
            )
            return response

        messages_cls.create = patched_create

        # --- streaming variant -------------------------------------------
        # anthropic.Messages.stream returns a context-manager / MessageStream.
        # We patch it to capture usage from the final_message after __exit__.
        if hasattr(messages_cls, "stream"):
            original_stream = messages_cls.stream

            @functools.wraps(original_stream)
            def patched_stream(self_client: Any, **kwargs: Any) -> Any:
                caller = _caller_location()
                kwargs, original_model = _maybe_route(kwargs, config, router)
                prompt_meta = _maybe_analyze(kwargs, "anthropic", analyzer)
                preflight_original = _enforce_budget(kwargs, "anthropic", config, prompt_meta)
                original_model = original_model or preflight_original
                start = time.perf_counter()
                try:
                    stream_mgr = original_stream(self_client, **kwargs)
                except Exception as exc:
                    _emit_error_event(
                        provider="anthropic",
                        request_kwargs=kwargs,
                        latency_ms=int((time.perf_counter() - start) * 1000),
                        error=exc,
                        config=config,
                        emitter=emitter,
                        original_model=original_model,
                        caller_file=caller,
                        prompt_meta=prompt_meta,
                    )
                    raise

                # Wrap in a thin proxy that records usage on context exit
                return _AnthropicStreamProxy(
                    stream_mgr=stream_mgr,
                    start_time=start,
                    request_kwargs=kwargs,
                    config=config,
                    emitter=emitter,
                    original_model=original_model,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                )

            messages_cls.stream = patched_stream

        messages_cls._capsera_patched = True
        if config.debug:
            logger.debug("capsera: patched anthropic.resources.messages.Messages")

    def _patch_anthropic_async(self, anthropic: Any) -> None:
        self._publish_runtime()
        try:
            async_messages_cls = anthropic.resources.messages.AsyncMessages
        except AttributeError:
            return

        if getattr(async_messages_cls, "_capsera_patched", False):
            return

        original_create = async_messages_cls.create

        @functools.wraps(original_create)
        async def async_patched_create(self_client: Any, **kwargs: Any) -> Any:
            caller = _caller_location()
            kwargs, original_model = _maybe_route(kwargs, config, router)
            prompt_meta = _maybe_analyze(kwargs, "anthropic", analyzer)
            preflight_original = await _enforce_budget_async(kwargs, "anthropic", config, prompt_meta)
            original_model = original_model or preflight_original

            start = time.perf_counter()
            try:
                response = await original_create(self_client, **kwargs)
            except Exception as exc:
                _emit_error_event(
                    provider="anthropic",
                    request_kwargs=kwargs,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    error=exc,
                    config=config,
                    emitter=emitter,
                    original_model=original_model,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                )
                raise
            latency_ms = int((time.perf_counter() - start) * 1000)

            _emit_anthropic_response(
                response=response,
                request_kwargs=kwargs,
                latency_ms=latency_ms,
                config=config,
                emitter=emitter,
                original_model=original_model,
                caller_file=caller,
                prompt_meta=prompt_meta,
            )
            return response

        async_messages_cls.create = async_patched_create

        # async stream
        if hasattr(async_messages_cls, "stream"):
            original_stream = async_messages_cls.stream

            @functools.wraps(original_stream)
            def async_patched_stream(self_client: Any, **kwargs: Any) -> Any:
                caller = _caller_location()
                kwargs, original_model = _maybe_route(kwargs, config, router)
                prompt_meta = _maybe_analyze(kwargs, "anthropic", analyzer)
                preflight_original = _enforce_budget(kwargs, "anthropic", config, prompt_meta)
                original_model = original_model or preflight_original
                start = time.perf_counter()
                try:
                    stream_mgr = original_stream(self_client, **kwargs)
                except Exception as exc:
                    _emit_error_event(
                        provider="anthropic",
                        request_kwargs=kwargs,
                        latency_ms=int((time.perf_counter() - start) * 1000),
                        error=exc,
                        config=config,
                        emitter=emitter,
                        original_model=original_model,
                        caller_file=caller,
                        prompt_meta=prompt_meta,
                    )
                    raise
                return _AnthropicAsyncStreamProxy(
                    stream_mgr=stream_mgr,
                    start_time=start,
                    request_kwargs=kwargs,
                    config=config,
                    emitter=emitter,
                    original_model=original_model,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                )

            async_messages_cls.stream = async_patched_stream

        async_messages_cls._capsera_patched = True
        if config.debug:
            logger.debug("capsera: patched anthropic.resources.messages.AsyncMessages")

    # ================================================================== #
    # OpenAI
    # ================================================================== #

    def _patch_openai(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            logger.warning("capsera: could not patch openai — LLM calls will not be tracked")
            return

        self._patch_openai_sync(openai)
        self._patch_openai_async(openai)

    def _patch_openai_sync(self, openai: Any) -> None:
        self._publish_runtime()
        try:
            completions_cls = openai.resources.chat.completions.Completions
        except AttributeError:
            return

        if getattr(completions_cls, "_capsera_patched", False):
            return

        original_create = completions_cls.create

        @functools.wraps(original_create)
        def patched_create(self_client: Any, **kwargs: Any) -> Any:
            caller = _caller_location()
            kwargs, original_model = _maybe_route(kwargs, config, router)
            provider = _openai_provider(self_client, kwargs.get("model", ""))
            prompt_meta = _maybe_analyze(kwargs, "openai", analyzer)
            preflight_original = _enforce_budget(kwargs, "openai", config, prompt_meta)
            original_model = original_model or preflight_original
            is_streaming = kwargs.get("stream", False)

            if is_streaming:
                # Inject include_usage so the final chunk carries token counts
                stream_opts = dict(kwargs.get("stream_options") or {})
                stream_opts["include_usage"] = True
                kwargs = {**kwargs, "stream_options": stream_opts}

            start = time.perf_counter()
            try:
                response = original_create(self_client, **kwargs)
            except Exception as exc:
                _emit_error_event(
                    provider=provider,
                    request_kwargs=kwargs,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    error=exc,
                    config=config,
                    emitter=emitter,
                    original_model=original_model,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                )
                raise

            if is_streaming:
                return _OpenAIStreamWrapper(
                    stream=response,
                    start_time=start,
                    request_kwargs=kwargs,
                    config=config,
                    emitter=emitter,
                    original_model=original_model,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                    provider=provider,
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            _emit_openai_response(
                response=response,
                request_kwargs=kwargs,
                latency_ms=latency_ms,
                config=config,
                emitter=emitter,
                original_model=original_model,
                caller_file=caller,
                prompt_meta=prompt_meta,
                provider=provider,
            )
            return response

        completions_cls.create = patched_create
        completions_cls._capsera_patched = True
        if config.debug:
            logger.debug(
                "capsera: patched openai.resources.chat.completions.Completions"
            )

    def _patch_openai_async(self, openai: Any) -> None:
        self._publish_runtime()
        try:
            async_cls = openai.resources.chat.completions.AsyncCompletions
        except AttributeError:
            return

        if getattr(async_cls, "_capsera_patched", False):
            return

        original_create = async_cls.create

        @functools.wraps(original_create)
        async def async_patched_create(self_client: Any, **kwargs: Any) -> Any:
            caller = _caller_location()
            kwargs, original_model = _maybe_route(kwargs, config, router)
            provider = _openai_provider(self_client, kwargs.get("model", ""))
            prompt_meta = _maybe_analyze(kwargs, "openai", analyzer)
            preflight_original = await _enforce_budget_async(kwargs, "openai", config, prompt_meta)
            original_model = original_model or preflight_original
            is_streaming = kwargs.get("stream", False)

            if is_streaming:
                stream_opts = dict(kwargs.get("stream_options") or {})
                stream_opts["include_usage"] = True
                kwargs = {**kwargs, "stream_options": stream_opts}

            start = time.perf_counter()
            try:
                response = await original_create(self_client, **kwargs)
            except Exception as exc:
                _emit_error_event(
                    provider=provider,
                    request_kwargs=kwargs,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    error=exc,
                    config=config,
                    emitter=emitter,
                    original_model=original_model,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                )
                raise

            if is_streaming:
                return _OpenAIAsyncStreamWrapper(
                    stream=response,
                    start_time=start,
                    request_kwargs=kwargs,
                    config=config,
                    emitter=emitter,
                    original_model=original_model,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                    provider=provider,
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            _emit_openai_response(
                response=response,
                request_kwargs=kwargs,
                latency_ms=latency_ms,
                config=config,
                emitter=emitter,
                original_model=original_model,
                caller_file=caller,
                prompt_meta=prompt_meta,
                provider=provider,
            )
            return response

        async_cls.create = async_patched_create
        async_cls._capsera_patched = True
        if config.debug:
            logger.debug(
                "capsera: patched openai.resources.chat.completions.AsyncCompletions"
            )

    # ================================================================== #
    # Google Gemini
    # ================================================================== #

    def _patch_google(self) -> None:
        patched = False
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError:
            genai = None
        if genai is not None:
            patched = self._patch_google_generativeai(genai) or patched

        try:
            import google.genai as google_genai  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError:
            google_genai = None
        if google_genai is not None:
            patched = self._patch_google_genai(google_genai) or patched

        if not patched:
            logger.warning("capsera: could not patch google gemini — Gemini calls will not be tracked")

    def _patch_google_generativeai(self, genai: Any) -> bool:
        self._publish_runtime()
        model_cls = getattr(genai, "GenerativeModel", None)
        if model_cls is None:
            return False

        patched = False
        if hasattr(model_cls, "generate_content") and not getattr(model_cls.generate_content, "_capsera_patched", False):
            original_generate = model_cls.generate_content

            @functools.wraps(original_generate)
            def patched_generate(self_model: Any, *args: Any, **kwargs: Any) -> Any:
                caller = _caller_location()
                model_name = _google_model_name(self_model, kwargs)
                request_kwargs = {**kwargs, "model": model_name}
                prompt_meta = _maybe_analyze(request_kwargs, "google", analyzer)
                _enforce_budget(dict(request_kwargs), "google", config, prompt_meta)

                start = time.perf_counter()
                try:
                    response = original_generate(self_model, *args, **kwargs)
                except Exception as exc:
                    _emit_error_event(
                        provider="google",
                        request_kwargs=request_kwargs,
                        latency_ms=int((time.perf_counter() - start) * 1000),
                        error=exc,
                        config=config,
                        emitter=emitter,
                        original_model=None,
                        caller_file=caller,
                        prompt_meta=prompt_meta,
                    )
                    raise
                latency_ms = int((time.perf_counter() - start) * 1000)
                _emit_google_response(
                    response=response,
                    request_kwargs=request_kwargs,
                    latency_ms=latency_ms,
                    config=config,
                    emitter=emitter,
                    original_model=None,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                )
                return response

            patched_generate._capsera_patched = True  # type: ignore[attr-defined]
            model_cls.generate_content = patched_generate
            patched = True

        if hasattr(model_cls, "generate_content_async") and not getattr(model_cls.generate_content_async, "_capsera_patched", False):
            original_async_generate = model_cls.generate_content_async

            @functools.wraps(original_async_generate)
            async def patched_async_generate(self_model: Any, *args: Any, **kwargs: Any) -> Any:
                caller = _caller_location()
                model_name = _google_model_name(self_model, kwargs)
                request_kwargs = {**kwargs, "model": model_name}
                prompt_meta = _maybe_analyze(request_kwargs, "google", analyzer)
                await _enforce_budget_async(dict(request_kwargs), "google", config, prompt_meta)

                start = time.perf_counter()
                try:
                    response = await original_async_generate(self_model, *args, **kwargs)
                except Exception as exc:
                    _emit_error_event(
                        provider="google",
                        request_kwargs=request_kwargs,
                        latency_ms=int((time.perf_counter() - start) * 1000),
                        error=exc,
                        config=config,
                        emitter=emitter,
                        original_model=None,
                        caller_file=caller,
                        prompt_meta=prompt_meta,
                    )
                    raise
                latency_ms = int((time.perf_counter() - start) * 1000)
                _emit_google_response(
                    response=response,
                    request_kwargs=request_kwargs,
                    latency_ms=latency_ms,
                    config=config,
                    emitter=emitter,
                    original_model=None,
                    caller_file=caller,
                    prompt_meta=prompt_meta,
                )
                return response

            patched_async_generate._capsera_patched = True  # type: ignore[attr-defined]
            model_cls.generate_content_async = patched_async_generate
            patched = True

        if patched and self._config.debug:
            logger.debug("capsera: patched google.generativeai.GenerativeModel")
        return patched

    def _patch_google_genai(self, google_genai: Any) -> bool:
        self._publish_runtime()
        models_mod = getattr(google_genai, "models", None)
        models_cls = getattr(models_mod, "Models", None)
        if models_cls is None or not hasattr(models_cls, "generate_content"):
            return False
        if getattr(models_cls.generate_content, "_capsera_patched", False):
            return True

        original_generate = models_cls.generate_content

        @functools.wraps(original_generate)
        def patched_generate(self_models: Any, *args: Any, **kwargs: Any) -> Any:
            caller = _caller_location()
            call_kwargs = dict(kwargs)
            prompt_meta = _maybe_analyze(call_kwargs, "google", analyzer)
            preflight_original = _enforce_budget(call_kwargs, "google", config, prompt_meta)
            original_model = preflight_original

            start = time.perf_counter()
            try:
                response = original_generate(self_models, *args, **call_kwargs)
            except Exception as exc:
                _emit_simple_error(provider="google",
                                   model=call_kwargs.get("model", "unknown"),
                                   start=start, error=exc, caller_file=caller)
                raise
            latency_ms = int((time.perf_counter() - start) * 1000)
            _emit_google_response(
                response=response,
                request_kwargs=call_kwargs,
                latency_ms=latency_ms,
                config=config,
                emitter=emitter,
                original_model=original_model,
                caller_file=caller,
                prompt_meta=prompt_meta,
            )
            return response

        patched_generate._capsera_patched = True  # type: ignore[attr-defined]
        models_cls.generate_content = patched_generate
        if self._config.debug:
            logger.debug("capsera: patched google.genai.models.Models.generate_content")
        return True

    # ================================================================== #
    # OpenAI Embeddings
    # ================================================================== #

    def _patch_openai_embeddings(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError:
            return

        self._patch_openai_embeddings_sync(openai)
        self._patch_openai_embeddings_async(openai)

    def _patch_openai_embeddings_sync(self, openai: Any) -> None:
        self._publish_runtime()
        try:
            embeddings_cls = openai.resources.embeddings.Embeddings
        except AttributeError:
            return

        if getattr(embeddings_cls, "_capsera_patched", False):
            return

        original_create = embeddings_cls.create

        @functools.wraps(original_create)
        def patched_create(self_client: Any, **kwargs: Any) -> Any:
            caller = _caller_location()
            provider = _openai_provider(self_client, kwargs.get("model", ""))
            start = time.perf_counter()
            try:
                response = original_create(self_client, **kwargs)
            except Exception as exc:
                _emit_simple_error(provider=provider,
                                   model=kwargs.get("model", "unknown"),
                                   start=start, error=exc, caller_file=caller)
                raise
            latency_ms = int((time.perf_counter() - start) * 1000)

            _emit_openai_embedding_response(
                response=response,
                request_kwargs=kwargs,
                latency_ms=latency_ms,
                config=config,
                emitter=emitter,
                caller_file=caller,
                provider=provider,
            )
            return response

        embeddings_cls.create = patched_create
        embeddings_cls._capsera_patched = True
        if config.debug:
            logger.debug("capsera: patched openai.resources.embeddings.Embeddings")

    def _patch_openai_embeddings_async(self, openai: Any) -> None:
        self._publish_runtime()
        try:
            async_cls = openai.resources.embeddings.AsyncEmbeddings
        except AttributeError:
            return

        if getattr(async_cls, "_capsera_patched", False):
            return

        original_create = async_cls.create

        @functools.wraps(original_create)
        async def async_patched_create(self_client: Any, **kwargs: Any) -> Any:
            caller = _caller_location()
            provider = _openai_provider(self_client, kwargs.get("model", ""))
            start = time.perf_counter()
            try:
                response = await original_create(self_client, **kwargs)
            except Exception as exc:
                _emit_simple_error(provider=provider,
                                   model=kwargs.get("model", "unknown"),
                                   start=start, error=exc, caller_file=caller)
                raise
            latency_ms = int((time.perf_counter() - start) * 1000)

            _emit_openai_embedding_response(
                response=response,
                request_kwargs=kwargs,
                latency_ms=latency_ms,
                config=config,
                emitter=emitter,
                caller_file=caller,
                provider=provider,
            )
            return response

        async_cls.create = async_patched_create
        async_cls._capsera_patched = True
        if config.debug:
            logger.debug(
                "capsera: patched openai.resources.embeddings.AsyncEmbeddings"
            )


# ======================================================================== #
# Helpers — provider-specific response parsing
# ======================================================================== #

def _emit_error_event(
    *,
    provider: str,
    request_kwargs: dict[str, Any],
    latency_ms: int,
    error: BaseException,
    config: Config,
    emitter: Emitter,
    original_model: Optional[str],
    caller_file: str,
    prompt_meta: Optional[PromptMetadata] = None,
) -> None:
    """Record a failed LLM call as a zero-cost error event.

    Routing attribution (``original_model`` / ``was_routed``) is preserved so a
    route's error-rate impact can be measured alongside its cost and latency.
    Never raises into the caller's code.
    """
    try:
        event = _build_event(
            model=request_kwargs.get("model", "unknown"),
            provider=provider,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=latency_ms,
            config=config,
            original_model=original_model,
            caller_file=caller_file,
            prompt_meta=prompt_meta,
            is_error=True,
            error_type=type(error).__name__,
        )
        emitter.enqueue(event)
    except Exception:
        if config.debug:
            logger.exception("capsera: failed to record error event")


def _emit_simple_error(
    *,
    provider: str,
    model: str,
    start: float,
    error: BaseException,
    caller_file: str,
) -> None:
    """Error capture for surfaces without routing/analysis context (native
    providers, embeddings, google.genai). Resolves config/emitter from the
    module runtime like every patched closure. Never raises."""
    _emit_error_event(
        provider=provider,
        request_kwargs={"model": model},
        latency_ms=int((time.perf_counter() - start) * 1000),
        error=error,
        config=config,
        emitter=emitter,
        original_model=None,
        caller_file=caller_file,
    )


def _emit_anthropic_response(
    *,
    response: Any,
    request_kwargs: dict[str, Any],
    latency_ms: int,
    config: Config,
    emitter: Emitter,
    original_model: Optional[str],
    caller_file: str,
    prompt_meta: Optional[PromptMetadata] = None,
) -> None:
    """Extract usage from an Anthropic Message response and enqueue an event."""
    try:
        usage = response.usage
        event = _build_event(
            model=getattr(response, "model", request_kwargs.get("model", "unknown")),
            provider="anthropic",
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            latency_ms=latency_ms,
            config=config,
            original_model=original_model,
            caller_file=caller_file,
            prompt_meta=prompt_meta,
        )
        emitter.enqueue(event)
    except Exception:
        if config.debug:
            logger.exception("capsera: failed to record Anthropic response")


def _emit_native_response(
    response: Any,
    *,
    provider: str,
    fallback_model: str,
    latency_ms: int,
    caller_file: str,
    style: str,
) -> None:
    """Extract usage from a native-provider response and enqueue an event.

    One helper, several shapes:
      - ``openai_usage`` (Mistral): ``resp.usage.prompt_tokens/.completion_tokens``.
      - ``google_usage`` (Vertex): ``resp.usage_metadata.prompt_token_count/…``.
      - ``cohere``: ``resp.usage.tokens.{input,output}_tokens`` or ``billed_units``.
      - ``bedrock_converse``: ``resp["usage"]["inputTokens"/"outputTokens"]`` (dict).
    Fail-open: any shape mismatch is logged (debug) and dropped, never raised
    into the customer's call.
    """
    try:
        model = fallback_model
        in_tok = out_tok = 0

        if style == "bedrock_converse" and isinstance(response, dict):
            usage = response.get("usage") or {}
            in_tok = int(usage.get("inputTokens", 0) or 0)
            out_tok = int(usage.get("outputTokens", 0) or 0)
        elif style == "google_usage":
            um = getattr(response, "usage_metadata", None)
            in_tok = int(getattr(um, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(um, "candidates_token_count", 0) or 0)
            model = getattr(response, "model", None) or model
        elif style == "cohere":
            usage = getattr(response, "usage", None)
            bucket = getattr(usage, "tokens", None) or getattr(usage, "billed_units", None)
            in_tok = int(getattr(bucket, "input_tokens", 0) or 0)
            out_tok = int(getattr(bucket, "output_tokens", 0) or 0)
            model = getattr(response, "model", None) or model
        else:  # openai_usage (Mistral and any OpenAI-shaped native response)
            if not hasattr(response, "usage") and hasattr(response, "parse"):
                response = response.parse()
            usage = getattr(response, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            model = getattr(response, "model", None) or model

        event = _build_event(
            model=model,
            provider=provider,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=latency_ms,
            config=config,
            caller_file=caller_file,
        )
        emitter.enqueue(event)
    except Exception:
        if config.debug:
            logger.exception("capsera: failed to record %s response", provider)


def _openai_provider(self_client: Any, model: str) -> str:
    """Resolve the provider for a call captured through the ``openai`` SDK.

    OpenAI-compatible providers (Groq, DeepSeek, xAI, Together, Fireworks,
    Perplexity, OpenRouter, Azure, local Ollama/vLLM) all ride this same SDK,
    so the client's ``base_url`` is the only capture-time signal that
    distinguishes them from real OpenAI. A real ``Completions`` resource
    carries the top-level client on ``self._client`` (which has ``base_url``);
    fall back to a direct ``base_url`` attribute, then to the catalog's
    declared provider for the model, then ``"openai"``.
    """
    base_url = None
    for obj in (getattr(self_client, "_client", None), self_client):
        bu = getattr(obj, "base_url", None)
        if bu:
            base_url = str(bu)
            break
    resolved = _catalog.provider_for(model or "", base_url=base_url)
    return resolved if resolved != "unknown" else "openai"


def _emit_openai_response(
    *,
    response: Any,
    request_kwargs: dict[str, Any],
    latency_ms: int,
    config: Config,
    emitter: Emitter,
    original_model: Optional[str],
    caller_file: str,
    prompt_meta: Optional[PromptMetadata] = None,
    provider: str = "openai",
) -> None:
    """Extract usage from an OpenAI ChatCompletion response and enqueue an event."""
    try:
        # Handle LegacyAPIResponse wrappers (e.g. when OpenAI SDK is used via LangChain)
        if not hasattr(response, "usage") and hasattr(response, "parse"):
            response = response.parse()
        usage = response.usage
        model = getattr(response, "model", request_kwargs.get("model", "unknown"))
        event = _build_event(
            model=model,
            provider=provider,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=latency_ms,
            config=config,
            original_model=original_model,
            caller_file=caller_file,
            prompt_meta=prompt_meta,
        )
        emitter.enqueue(event)
    except Exception:
        if config.debug:
            logger.exception("capsera: failed to record OpenAI response")


def _emit_openai_embedding_response(
    *,
    response: Any,
    request_kwargs: dict[str, Any],
    latency_ms: int,
    config: Config,
    emitter: Emitter,
    caller_file: str,
    provider: str = "openai",
) -> None:
    """Extract usage from an OpenAI Embedding response and enqueue an event."""
    try:
        if not hasattr(response, "usage") and hasattr(response, "parse"):
            response = response.parse()
        usage = response.usage
        model = getattr(response, "model", request_kwargs.get("model", "unknown"))
        input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0)
        event = _build_event(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=latency_ms,
            config=config,
            original_model=None,
            caller_file=caller_file,
        )
        event.task_type = event.task_type or "embedding"
        emitter.enqueue(event)
    except Exception:
        if config.debug:
            logger.exception("capsera: failed to record OpenAI embedding response")


def _google_model_name(model_obj: Any, kwargs: dict[str, Any]) -> str:
    model = kwargs.get("model")
    if model:
        return str(model)
    for attr in ("model_name", "_model_name", "model"):
        value = getattr(model_obj, attr, None)
        if value:
            return str(value).replace("models/", "")
    return "gemini-pro"


def _emit_google_response(
    *,
    response: Any,
    request_kwargs: dict[str, Any],
    latency_ms: int,
    config: Config,
    emitter: Emitter,
    original_model: Optional[str],
    caller_file: str,
    prompt_meta: Optional[PromptMetadata] = None,
) -> None:
    """Extract usage from a Google Gemini response and enqueue an event."""
    try:
        meta = getattr(response, "usage_metadata", None)
        input_tokens = getattr(meta, "prompt_token_count", 0) if meta else 0
        output_tokens = getattr(meta, "candidates_token_count", 0) if meta else 0
        # Prefer the concrete resolved model over the requested name: apps
        # calling rolling aliases ("gemini-flash-lite-latest") must be billed
        # against the model that actually served them (model_version).
        model = (
            getattr(response, "model", None)
            or getattr(response, "model_version", None)
            or request_kwargs.get("model")
            or "gemini-pro"
        )
        event = _build_event(
            model=str(model).replace("models/", ""),
            provider="google",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=latency_ms,
            config=config,
            original_model=original_model,
            caller_file=caller_file,
            prompt_meta=prompt_meta,
        )
        emitter.enqueue(event)
    except Exception:
        if config.debug:
            logger.exception("capsera: failed to record Google Gemini response")


def _maybe_analyze(
    kwargs: dict[str, Any],
    provider: str,
    analyzer: Optional[PromptAnalyzer] = None,
) -> Optional[PromptMetadata]:
    """Run prompt analysis if enabled. Fail-open: exceptions return None."""
    if analyzer is None:
        return None
    try:
        return analyzer.analyze(kwargs, provider)
    except Exception:
        return None


def _estimate_request_cost(kwargs: dict[str, Any], provider: str, prompt_meta: Optional[PromptMetadata]) -> float:
    model = kwargs.get("model", "")
    if not model:
        return 0.0

    input_tokens = 0
    if prompt_meta:
        input_tokens = prompt_meta.system_prompt_tokens + prompt_meta.conversation_tokens

    output_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 0
    try:
        output_tokens = int(output_tokens)
    except (TypeError, ValueError):
        output_tokens = 0

    try:
        return float(compute_cost(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))
    except Exception:
        return 0.0


def _budget_check_payload(
    kwargs: dict[str, Any],
    provider: str,
    config: Config,
    prompt_meta: Optional[PromptMetadata],
) -> dict[str, Any]:
    ctx = _context_stack.current()
    agent_id = (ctx.agent if ctx else None) or config.default_agent or "unknown"
    team_id = (ctx.team if ctx else None) or config.default_team
    task_type = ctx.task_type if ctx else None
    return {
        "agent_id": agent_id,
        "team_id": team_id,
        "task_type": task_type,
        "customer_id": ctx.customer_id if ctx else None,
        "cost_center": ctx.cost_center if ctx else None,
        "model": kwargs.get("model"),
        "provider": provider,
        "estimated_cost_usd": _estimate_request_cost(kwargs, provider, prompt_meta),
    }


def _raise_if_blocked(decision: dict[str, Any]) -> None:
    if decision.get("allowed", True):
        return
    message = decision.get("message") or "Capsera blocked this LLM call."
    raise BudgetExceededError(message)


def _apply_preflight_decision(
    kwargs: dict[str, Any],
    decision: dict[str, Any],
    config: Config,
) -> Optional[str]:
    _raise_if_blocked(decision)
    if decision.get("action") != "downgrade":
        return None

    target_model = decision.get("target_model")
    original_model = kwargs.get("model")
    if not target_model or not original_model or target_model == original_model:
        return None

    ctx = _context_stack.current()
    task_type = ctx.task_type if ctx else None
    if task_type and task_type in set(config.critical_task_types):
        logger.debug("capsera: margin downgrade skipped for critical task_type=%s", task_type)
        return None

    kwargs["model"] = target_model
    return original_model


# --------------------------------------------------------------------------- #
# Pooled HTTP clients for the pre-call budget check
#
# Budget enforcement runs on the hot path of every LLM call (it is on by
# default), so we reuse a single connection-pooled client instead of building
# and tearing one down per call. The sync client is thread-safe and shared
# process-wide; async clients are bound to an event loop, so we keep one per
# running loop. Clients are keyed by (endpoint, api_key, timeout) and rebuilt
# transparently if init() is called again with different settings.
# --------------------------------------------------------------------------- #

_BudgetClientKey = tuple[str, str, float]

_sync_budget_client: Optional[httpx.Client] = None
_sync_budget_client_key: Optional[_BudgetClientKey] = None
_sync_budget_client_lock = threading.Lock()

_async_budget_clients: "weakref.WeakKeyDictionary[Any, tuple[_BudgetClientKey, httpx.AsyncClient]]" = (
    weakref.WeakKeyDictionary()
)
_async_budget_clients_lock = threading.Lock()


def _budget_client_key(config: Config) -> _BudgetClientKey:
    return (config.endpoint, config.api_key, config.budget_check_timeout)


def _budget_client_headers(config: Config) -> dict[str, str]:
    return {
        "X-API-Key": config.api_key,
        "User-Agent": f"capsera-sdk/{__version__}",
    }


def _get_sync_budget_client(config: Config) -> httpx.Client:
    global _sync_budget_client, _sync_budget_client_key
    key = _budget_client_key(config)
    with _sync_budget_client_lock:
        if _sync_budget_client is None or _sync_budget_client_key != key:
            if _sync_budget_client is not None:
                try:
                    _sync_budget_client.close()
                except Exception:
                    pass
            _sync_budget_client = httpx.Client(
                base_url=config.endpoint,
                headers=_budget_client_headers(config),
                timeout=config.budget_check_timeout,
            )
            _sync_budget_client_key = key
        return _sync_budget_client


def _get_async_budget_client(config: Config) -> httpx.AsyncClient:
    key = _budget_client_key(config)
    loop = asyncio.get_running_loop()
    with _async_budget_clients_lock:
        existing = _async_budget_clients.get(loop)
        if existing is not None and existing[0] == key:
            return existing[1]
        client = httpx.AsyncClient(
            base_url=config.endpoint,
            headers=_budget_client_headers(config),
            timeout=config.budget_check_timeout,
        )
        _async_budget_clients[loop] = (key, client)
        return client


def _reset_budget_clients() -> None:
    """Close and drop pooled budget-check clients (called on re-init)."""
    global _sync_budget_client, _sync_budget_client_key
    with _sync_budget_client_lock:
        if _sync_budget_client is not None:
            try:
                _sync_budget_client.close()
            except Exception:
                pass
        _sync_budget_client = None
        _sync_budget_client_key = None
    with _async_budget_clients_lock:
        _async_budget_clients.clear()


def _enforce_budget(
    kwargs: dict[str, Any],
    provider: str,
    config: Config,
    prompt_meta: Optional[PromptMetadata],
) -> Optional[str]:
    if not config.enable_budget_enforcement:
        return None
    try:
        client = _get_sync_budget_client(config)
        resp = client.post(
            "/v1/budgets/check",
            json=_budget_check_payload(kwargs, provider, config, prompt_meta),
        )
        if resp.status_code == 200:
            return _apply_preflight_decision(kwargs, resp.json(), config)
    except BudgetExceededError:
        raise
    except Exception:
        logger.debug("capsera: budget preflight failed — passing through", exc_info=True)
    return None


async def _enforce_budget_async(
    kwargs: dict[str, Any],
    provider: str,
    config: Config,
    prompt_meta: Optional[PromptMetadata],
) -> Optional[str]:
    if not config.enable_budget_enforcement:
        return None
    try:
        client = _get_async_budget_client(config)
        resp = await client.post(
            "/v1/budgets/check",
            json=_budget_check_payload(kwargs, provider, config, prompt_meta),
        )
        if resp.status_code == 200:
            return _apply_preflight_decision(kwargs, resp.json(), config)
    except BudgetExceededError:
        raise
    except Exception:
        logger.debug("capsera: budget preflight failed — passing through", exc_info=True)
    return None


def _maybe_route(
    kwargs: dict[str, Any],
    config: Config,
    router: Optional[Router] = None,
) -> tuple[dict[str, Any], Optional[str]]:
    """
    Apply model routing if enabled.

    Delegates to the Router engine when ``enable_routing`` is True and a
    Router instance is available.  Otherwise returns kwargs unchanged.

    Returns ``(modified_kwargs, original_model_or_None)``.
    """
    if not config.enable_routing or router is None:
        return kwargs, None

    ctx = _context_stack.current()
    decision = router.evaluate(kwargs, ctx)
    if decision.routed and decision.target_model:
        new_kwargs = {**kwargs, "model": decision.target_model}
        return new_kwargs, decision.original_model
    return kwargs, None


# ======================================================================== #
# Stream proxy wrappers
# ======================================================================== #

class _AnthropicStreamProxy:
    """
    Thin proxy around an Anthropic MessageStream context manager.

    Captures usage from ``final_message`` after the ``__exit__`` call.
    All attribute access and method calls are forwarded transparently to
    the underlying stream manager.
    """

    def __init__(
        self,
        stream_mgr: Any,
        start_time: float,
        request_kwargs: dict[str, Any],
        config: Config,
        emitter: Emitter,
        original_model: Optional[str],
        caller_file: str,
        prompt_meta: Optional[PromptMetadata] = None,
    ) -> None:
        self._stream_mgr = stream_mgr
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._config = config
        self._emitter = emitter
        self._original_model = original_model
        self._caller_file = caller_file
        self._prompt_meta = prompt_meta

    def __enter__(self) -> Any:
        self._entered_stream = self._stream_mgr.__enter__()
        return self._entered_stream

    def __exit__(self, *args: Any) -> Any:
        # Record BEFORE closing: get_final_message() may need to drain the
        # remaining stream events, which requires a live response.
        self._record_from_stream()
        return self._stream_mgr.__exit__(*args)

    def __iter__(self) -> Any:
        return iter(self._stream_mgr)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream_mgr, name)

    def _record_from_stream(self) -> None:
        latency_ms = int((time.perf_counter() - self._start_time) * 1000)
        try:
            # `.get_final_message()` / `.final_message` live on the
            # MessageStream returned by __enter__, NOT on the
            # MessageStreamManager we wrap — asking the manager silently
            # yields nothing and the call would go untracked.
            stream = getattr(self, "_entered_stream", None) or self._stream_mgr
            msg = None
            if hasattr(stream, "get_final_message"):
                try:
                    msg = stream.get_final_message()
                except Exception:
                    pass
            if msg is None and hasattr(stream, "final_message"):
                msg = stream.final_message

            if msg is not None:
                _emit_anthropic_response(
                    response=msg,
                    request_kwargs=self._request_kwargs,
                    latency_ms=latency_ms,
                    config=self._config,
                    emitter=self._emitter,
                    original_model=self._original_model,
                    caller_file=self._caller_file,
                    prompt_meta=self._prompt_meta,
                )
        except Exception:
            if self._config.debug:
                logger.exception("capsera: failed to record Anthropic stream usage")


class _AnthropicAsyncStreamProxy:
    """Async variant of _AnthropicStreamProxy."""

    def __init__(
        self,
        stream_mgr: Any,
        start_time: float,
        request_kwargs: dict[str, Any],
        config: Config,
        emitter: Emitter,
        original_model: Optional[str],
        caller_file: str,
        prompt_meta: Optional[PromptMetadata] = None,
    ) -> None:
        self._stream_mgr = stream_mgr
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._config = config
        self._emitter = emitter
        self._original_model = original_model
        self._caller_file = caller_file
        self._prompt_meta = prompt_meta

    async def __aenter__(self) -> Any:
        self._entered_stream = await self._stream_mgr.__aenter__()
        return self._entered_stream

    async def __aexit__(self, *args: Any) -> Any:
        # Record BEFORE closing — see _AnthropicStreamProxy.__exit__.
        latency_ms = int((time.perf_counter() - self._start_time) * 1000)
        try:
            stream = getattr(self, "_entered_stream", None) or self._stream_mgr
            msg = None
            if hasattr(stream, "get_final_message"):
                try:
                    msg = await stream.get_final_message()
                except Exception:
                    pass
            if msg is None and hasattr(stream, "final_message"):
                msg = stream.final_message
            if msg is not None:
                _emit_anthropic_response(
                    response=msg,
                    request_kwargs=self._request_kwargs,
                    latency_ms=latency_ms,
                    config=self._config,
                    emitter=self._emitter,
                    original_model=self._original_model,
                    caller_file=self._caller_file,
                    prompt_meta=self._prompt_meta,
                )
        except Exception:
            if self._config.debug:
                logger.exception(
                    "capsera: failed to record Anthropic async stream usage"
                )
        return await self._stream_mgr.__aexit__(*args)

    def __aiter__(self) -> Any:
        return self._stream_mgr.__aiter__()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream_mgr, name)


class _OpenAIStreamWrapper:
    """
    Wraps an OpenAI streaming response (an iterable of ChatCompletionChunks).

    Captures usage from the final chunk (which carries a ``usage`` field when
    ``stream_options={"include_usage": True}`` is set).
    """

    def __init__(
        self,
        stream: Any,
        start_time: float,
        request_kwargs: dict[str, Any],
        config: Config,
        emitter: Emitter,
        original_model: Optional[str],
        caller_file: str,
        prompt_meta: Optional[PromptMetadata] = None,
        provider: str = "openai",
    ) -> None:
        self._provider = provider
        self._stream = stream
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._config = config
        self._emitter = emitter
        self._original_model = original_model
        self._caller_file = caller_file
        self._prompt_meta = prompt_meta
        self._last_usage: Any = None
        self._last_model: str = request_kwargs.get("model", "unknown")

    def __iter__(self) -> Generator[Any, None, None]:
        try:
            for chunk in self._stream:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    self._last_usage = chunk.usage
                if hasattr(chunk, "model") and chunk.model:
                    self._last_model = chunk.model
                yield chunk
        finally:
            self._record()

    def __enter__(self) -> "_OpenAIStreamWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _record(self) -> None:
        if self._last_usage is None:
            return
        latency_ms = int((time.perf_counter() - self._start_time) * 1000)
        try:
            event = _build_event(
                model=self._last_model,
                provider=self._provider,
                input_tokens=getattr(self._last_usage, "prompt_tokens", 0),
                output_tokens=getattr(self._last_usage, "completion_tokens", 0),
                cache_read_tokens=0,
                cache_write_tokens=0,
                latency_ms=latency_ms,
                config=self._config,
                original_model=self._original_model,
                caller_file=self._caller_file,
                prompt_meta=self._prompt_meta,
            )
            self._emitter.enqueue(event)
        except Exception:
            if self._config.debug:
                logger.exception("capsera: failed to record OpenAI stream usage")


class _OpenAIAsyncStreamWrapper:
    """Async variant of _OpenAIStreamWrapper."""

    def __init__(
        self,
        stream: Any,
        start_time: float,
        request_kwargs: dict[str, Any],
        config: Config,
        emitter: Emitter,
        original_model: Optional[str],
        caller_file: str,
        prompt_meta: Optional[PromptMetadata] = None,
        provider: str = "openai",
    ) -> None:
        self._provider = provider
        self._stream = stream
        self._start_time = start_time
        self._request_kwargs = request_kwargs
        self._config = config
        self._emitter = emitter
        self._original_model = original_model
        self._caller_file = caller_file
        self._prompt_meta = prompt_meta
        self._last_usage: Any = None
        self._last_model: str = request_kwargs.get("model", "unknown")

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        try:
            async for chunk in self._stream:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    self._last_usage = chunk.usage
                if hasattr(chunk, "model") and chunk.model:
                    self._last_model = chunk.model
                yield chunk
        finally:
            self._record()

    async def __aenter__(self) -> "_OpenAIAsyncStreamWrapper":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _record(self) -> None:
        if self._last_usage is None:
            return
        latency_ms = int((time.perf_counter() - self._start_time) * 1000)
        try:
            event = _build_event(
                model=self._last_model,
                provider=self._provider,
                input_tokens=getattr(self._last_usage, "prompt_tokens", 0),
                output_tokens=getattr(self._last_usage, "completion_tokens", 0),
                cache_read_tokens=0,
                cache_write_tokens=0,
                latency_ms=latency_ms,
                config=self._config,
                original_model=self._original_model,
                caller_file=self._caller_file,
                prompt_meta=self._prompt_meta,
            )
            self._emitter.enqueue(event)
        except Exception:
            if self._config.debug:
                logger.exception(
                    "capsera: failed to record OpenAI async stream usage"
                )
