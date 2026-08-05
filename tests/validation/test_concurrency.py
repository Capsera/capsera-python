"""Thread-safety of the pieces that customers hit concurrently.

Agentic apps run LLM calls from thread pools and async tasks; the context
stack, session propagation, queue, and patch installation must all hold
under that. Failures here are attribution corruption (agent A's spend on
agent B's line) rather than crashes — the worst kind for a spend product.
"""
from __future__ import annotations

import queue as queue_mod
import sys
import threading
from types import ModuleType, SimpleNamespace

from capsera._config import Config
from capsera._context import AgentContext, ContextStack
from capsera._interceptor import Interceptor
from capsera._models import TokenEvent

from ..conftest import CapturingEmitter


def _mk_event(i: int) -> TokenEvent:
    return TokenEvent(
        id=f"id-{i}", timestamp="2026-08-03T12:00:00+00:00", request_id=f"r-{i}",
        model="m", provider="anthropic", input_tokens=1, output_tokens=1,
        cost_usd=0.0, agent_id=f"agent-{i}",
    )


def test_context_stack_is_isolated_per_thread():
    stack = ContextStack()
    barrier = threading.Barrier(8)
    corruption: list[str] = []

    def worker(idx: int) -> None:
        stack.push(AgentContext(agent=f"agent-{idx}"))
        barrier.wait()                    # everyone has pushed
        current = stack.current()
        if current is None or current.agent != f"agent-{idx}":
            corruption.append(f"thread {idx} saw {current!r}")
        stack.pop()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert corruption == []


def test_session_is_isolated_per_thread():
    stack = ContextStack()
    barrier = threading.Barrier(6)
    corruption: list[str] = []

    def worker(idx: int) -> None:
        stack.set_session(f"sess-{idx}")
        barrier.wait()
        got = stack.get_session()
        if got != f"sess-{idx}":
            corruption.append(f"thread {idx} saw {got!r}")
        stack.clear_session()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert corruption == []


def test_concurrent_enqueue_loses_nothing_below_capacity():
    config = Config(api_key="k", endpoint="http://localhost:9999", env="test",
                    flush_interval_ms=99_999)
    emitter = CapturingEmitter(config)
    threads = [
        threading.Thread(target=lambda base: [emitter.enqueue(_mk_event(base + i)) for i in range(200)],
                         args=(t * 1000,))
        for t in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(emitter.drain()) == 8 * 200


def test_concurrent_enqueue_at_capacity_drops_without_blocking_or_raising():
    config = Config(api_key="k", endpoint="http://localhost:9999", env="test",
                    flush_interval_ms=99_999)
    emitter = CapturingEmitter(config)
    emitter._queue = queue_mod.Queue(maxsize=50)
    errors: list[BaseException] = []

    def worker(base: int) -> None:
        try:
            for i in range(100):
                emitter.enqueue(_mk_event(base + i))
        except BaseException as exc:  # noqa: BLE001 — the assertion IS "never raises"
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t * 1000,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads), "enqueue blocked under pressure"
    assert errors == []
    assert emitter._queue.qsize() <= 50


def test_patch_is_idempotent_under_concurrent_install():
    """Racing installs (framework + user both calling init paths) must end
    with EXACTLY one wrapper — double-wrapping double-counts every call."""
    config = Config(api_key="k", endpoint="http://localhost:9999", env="test",
                    flush_interval_ms=99_999)
    emitter = CapturingEmitter(config)
    emitter.start()

    mistralai = ModuleType("mistralai")
    chat_mod = ModuleType("mistralai.chat")
    mistralai.chat = chat_mod

    class Chat:
        def complete(self, **kwargs):
            return SimpleNamespace(
                model="mistral-large-latest",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    chat_mod.Chat = Chat
    saved = {k: sys.modules.get(k) for k in ("mistralai", "mistralai.chat")}
    sys.modules.update({"mistralai": mistralai, "mistralai.chat": chat_mod})
    try:
        interceptor = Interceptor(config, emitter)
        barrier = threading.Barrier(4)

        def install() -> None:
            barrier.wait()
            interceptor._patch_mistral()

        threads = [threading.Thread(target=install) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        Chat().complete(model="mistral-large-latest", messages=[])
        events = emitter.drain()
        assert len(events) == 1, f"double-wrapped: one call emitted {len(events)} events"
    finally:
        emitter.shutdown()
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
