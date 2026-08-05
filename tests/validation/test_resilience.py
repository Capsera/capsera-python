"""Emitter resilience against a hostile network and backend.

The emitter sits inline with customer LLM calls (enqueue) and owns the only
copy of undelivered events, so the bar is: never raise into the caller,
never lose events the queue could still hold, always tell the operator
(on_error) when data is dropped, and recover cleanly when the backend does.
"""
from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from capsera._config import Config
from capsera._emitter import Emitter
from capsera._models import TokenEvent


def _mk_event(i: int = 0) -> TokenEvent:
    return TokenEvent(
        id=f"id-{i}",
        timestamp="2026-08-03T12:00:00+00:00",
        request_id=f"req-{i}",
        model="claude-sonnet-4-6",
        provider="anthropic",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0001,
        agent_id="resilience-test",
    )


class ScriptedTransport(httpx.BaseTransport):
    """Serves each scripted item once (Response or raised exception); the
    final item repeats forever. Records request count."""

    def __init__(self, *script):
        self._script: Iterator = iter(script)
        self._last = script[-1]
        self.calls = 0

    def handle_request(self, request):
        self.calls += 1
        item = next(self._script, self._last)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body, request=request)


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    # NOTE: `import capsera._emitter as em` would bind the package-level
    # `_emitter` SINGLETON (None until init()) that shadows the submodule
    # attribute — resolve the real module via sys.modules.
    import sys

    import capsera._emitter  # noqa: F401 — ensure it's imported

    em = sys.modules["capsera._emitter"]
    fake = FakeClock()
    monkeypatch.setattr(em.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(em.time, "sleep", fake.sleep)
    return fake


def _emitter(transport: httpx.BaseTransport, *, on_error=None, max_retries: int = 3) -> Emitter:
    config = Config(
        api_key="k",
        endpoint="http://backend.test",
        env="test",
        flush_interval_ms=3_600_000,  # never auto-flush; tests drive flush()
        debug=True,
    )
    emitter = Emitter(config, max_retries=max_retries, on_error=on_error)
    emitter._client = httpx.Client(transport=transport, base_url=config.endpoint)
    return emitter


# ————— Retry & recovery —————

def test_transient_500s_recover_within_one_send(clock):
    transport = ScriptedTransport((500, {}), (500, {}), (200, {"stored": 1}))
    emitter = _emitter(transport)
    emitter.enqueue(_mk_event())
    emitter.flush()
    assert transport.calls == 3            # two failures + the success
    assert emitter._queue.qsize() == 0     # delivered, nothing re-queued
    assert emitter._consecutive_failures == 0


def test_connection_refused_requeues_events_for_the_next_flush(clock):
    # The queue still has room and the backend may come back — dropping the
    # batch after 3 attempts loses data the SDK could still deliver. Dedup by
    # request_id makes eventual double-delivery safe.
    transport = ScriptedTransport(
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
        httpx.ConnectError("refused"),
        (200, {"stored": 2}),
    )
    seen: list[str] = []
    emitter = _emitter(transport, on_error=seen.append)
    emitter.enqueue(_mk_event(1))
    emitter.enqueue(_mk_event(2))

    emitter.flush()                        # all 3 attempts fail
    assert emitter._queue.qsize() == 2, "undelivered events must return to the queue"
    assert any("failed to deliver" in m for m in seen)

    emitter.flush()                        # backend recovered
    assert emitter._queue.qsize() == 0
    assert transport.calls == 4


def test_remote_protocol_error_is_retried_like_any_transport_error(clock):
    # Server closing the connection mid-response (RemoteProtocolError) is as
    # transient as a refused connection — it must retry, not bail after one
    # attempt via the generic exception path.
    transport = ScriptedTransport(
        httpx.RemoteProtocolError("server disconnected"),
        (200, {"stored": 1}),
    )
    emitter = _emitter(transport)
    emitter.enqueue(_mk_event())
    emitter.flush()
    assert transport.calls == 2
    assert emitter._queue.qsize() == 0


# ————— Circuit breaker —————

def test_circuit_opens_after_threshold_and_recovers_after_window(clock):
    transport = ScriptedTransport(
        *([httpx.ConnectError("down")] * 6),
        (200, {"stored": 1}),
    )
    emitter = _emitter(transport)
    emitter.enqueue(_mk_event(1))
    emitter.flush()                        # 3 failures
    emitter.flush()                        # +2 → threshold at 5, breaker opens mid-send
    assert emitter._consecutive_failures >= emitter._FAILURE_THRESHOLD
    assert emitter._circuit_open_until > clock.monotonic()

    calls_when_open = transport.calls
    emitter.flush()                        # circuit open → no network traffic
    assert transport.calls == calls_when_open
    assert emitter._queue.qsize() > 0      # events held, not lost

    clock.now += emitter._CIRCUIT_OPEN_SECONDS + 1
    emitter.flush()                        # window elapsed → delivery resumes
    assert emitter._queue.qsize() == 0


# ————— Client errors —————

def test_422_drop_notifies_the_operator(clock):
    # A 4xx means the payload can never succeed — dropping is right, but
    # dropping SILENTLY is not: on_error exists precisely for this.
    transport = ScriptedTransport((422, {"detail": "validation error"}))
    seen: list[str] = []
    emitter = _emitter(transport, on_error=seen.append)
    emitter.enqueue(_mk_event())
    emitter.flush()
    assert transport.calls == 1            # no retry on 4xx
    assert emitter._queue.qsize() == 0     # not re-queued — it can never succeed
    assert any("422" in m for m in seen), "operator was never told events were dropped"


def test_429_enters_limit_mode_and_enqueue_drops_quietly(clock):
    transport = ScriptedTransport(
        (429, {"detail": {"resets_at": "2099-09-01T00:00:00Z"}}),
    )
    emitter = _emitter(transport)
    emitter.enqueue(_mk_event())
    emitter.flush()
    assert emitter._limit_reached is True

    emitter.enqueue(_mk_event(2))          # dropped in limit mode
    assert emitter._queue.qsize() == 0


# ————— The callback itself is untrusted —————

def test_raising_on_error_never_reaches_the_caller(clock):
    def bomb(_msg: str) -> None:
        raise RuntimeError("observer exploded")

    transport = ScriptedTransport((422, {"detail": "bad"}))
    emitter = _emitter(transport, on_error=bomb)
    emitter.enqueue(_mk_event())
    emitter.flush()                        # must not raise

    # enqueue path: queue-full triggers on_error inline with the customer's
    # LLM call — a raising observer must not turn a dropped event into a
    # crashed provider call.
    emitter._QUEUE_MAXSIZE = 1
    small = Emitter(
        Config(api_key="k", endpoint="http://backend.test", env="test",
               flush_interval_ms=3_600_000),
        on_error=bomb,
    )
    small._client = httpx.Client(transport=ScriptedTransport((200, {})),
                                 base_url="http://backend.test")
    import queue as queue_mod

    small._queue = queue_mod.Queue(maxsize=1)
    small.enqueue(_mk_event(1))
    small.enqueue(_mk_event(2))            # full → on_error fires → must not raise


# ————— Queue behavior under pressure —————

def test_full_queue_drops_newest_without_blocking(clock):
    import queue as queue_mod

    transport = ScriptedTransport((200, {}))
    dropped: list[str] = []
    emitter = _emitter(transport, on_error=dropped.append)
    emitter._queue = queue_mod.Queue(maxsize=3)

    for i in range(10):
        emitter.enqueue(_mk_event(i))      # must never block
    assert emitter._queue.qsize() == 3
    assert sum("queue full" in m for m in dropped) == 7

    emitter.flush()
    assert emitter._queue.qsize() == 0


def test_requeue_on_failure_respects_queue_capacity(clock):
    # Re-queueing a failed batch must not block when the queue refilled while
    # the send was in flight — overflow is dropped like any other enqueue.
    import queue as queue_mod

    transport = ScriptedTransport(
        httpx.ConnectError("down"),
        httpx.ConnectError("down"),
        httpx.ConnectError("down"),
        (200, {}),
    )
    emitter = _emitter(transport)
    emitter._queue = queue_mod.Queue(maxsize=2)
    emitter.enqueue(_mk_event(1))
    emitter.enqueue(_mk_event(2))
    emitter.flush()                        # drains 2, fails, re-queues into empty queue
    assert emitter._queue.qsize() == 2

    emitter.enqueue(_mk_event(3))          # full again — drop, don't block
    assert emitter._queue.qsize() == 2

