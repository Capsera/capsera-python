"""Lifecycle semantics (shutdown/flush ordering) and hostile init input.

shutdown() is the last chance to deliver a process's tail of events; the
flush interval drives a daemon loop that must never busy-spin; and init
kwargs come from customer code, so pathological values must degrade to
something safe rather than a hot loop or a crash at import time.
"""
from __future__ import annotations

import httpx
import pytest

from capsera._config import Config
from capsera._emitter import Emitter
from capsera._models import TokenEvent

from .test_resilience import ScriptedTransport, _mk_event


def _emitter(transport, **config_overrides) -> Emitter:
    cfg = dict(api_key="k", endpoint="http://backend.test", env="test",
               flush_interval_ms=3_600_000)
    cfg.update(config_overrides)
    emitter = Emitter(Config(**cfg))
    emitter._client = httpx.Client(transport=transport, base_url="http://backend.test")
    return emitter


def test_shutdown_flushes_pending_events():
    transport = ScriptedTransport((200, {"stored": 3}))
    emitter = _emitter(transport)
    for i in range(3):
        emitter.enqueue(_mk_event(i))
    emitter.shutdown()
    assert transport.calls == 1
    assert emitter._queue.qsize() == 0


def test_double_shutdown_is_safe():
    transport = ScriptedTransport((200, {}))
    emitter = _emitter(transport)
    emitter.enqueue(_mk_event(1))
    emitter.shutdown()
    emitter.shutdown()                     # second call must be a no-op, not a crash


def test_enqueue_after_shutdown_does_not_raise():
    transport = ScriptedTransport((200, {}))
    emitter = _emitter(transport)
    emitter.shutdown()
    emitter.enqueue(_mk_event(1))          # nothing listens anymore; must not raise


def test_flush_with_empty_queue_makes_no_network_call():
    transport = ScriptedTransport((200, {}))
    emitter = _emitter(transport)
    emitter.flush()
    assert transport.calls == 0


def test_worker_interval_never_busy_spins():
    """flush_interval_ms=0 (or negative) must floor to a real interval —
    Event.wait(0) returns immediately, which turns the daemon thread into a
    100%-CPU loop inside the customer's process."""
    import capsera._emitter  # noqa: F401

    waits: list[float] = []

    class OneShotStop:
        def __init__(self) -> None:
            self._fired = False

        def wait(self, timeout=None):
            waits.append(timeout)
            self._fired = True
            return True                    # stop immediately after recording

        def clear(self) -> None:
            pass

        def set(self) -> None:
            pass

        def is_set(self) -> bool:
            return self._fired

    for bad_interval in (0, -500):
        emitter = _emitter(ScriptedTransport((200, {})), flush_interval_ms=bad_interval)
        emitter._stop_event = OneShotStop()
        emitter._worker()                  # run the loop body directly
        assert waits[-1] is not None and waits[-1] >= 0.05, (
            f"flush_interval_ms={bad_interval} produced wait({waits[-1]}) — busy spin"
        )


def test_config_accepts_endpoint_with_trailing_slash():
    emitter = _emitter(ScriptedTransport((200, {})), endpoint="http://backend.test/")
    emitter.enqueue(_mk_event(1))
    emitter.flush()                        # must reach the transport, not 404 on '//'


@pytest.mark.parametrize("api_key", [None, ""])
def test_none_or_empty_api_key_degrades_to_auth_failure_not_crash(api_key):
    # A missing env var passed straight through must never crash init — the
    # emitter builds a client with an empty key and lets ingest 401.
    emitter = Emitter(Config(api_key=api_key, endpoint="http://backend.test",
                             env="test", flush_interval_ms=3_600_000))
    assert emitter._client.headers["X-API-Key"] == ""


def test_record_public_api_produces_contract_valid_event():
    """capsera.record() is the manual escape hatch — its output must meet
    the same contract as intercepted events."""
    import capsera

    from .conftest import backend_event_schema

    captured: list[TokenEvent] = []

    class _Sink:
        def enqueue(self, event: TokenEvent) -> None:
            captured.append(event)

    original = capsera._emitter
    capsera._emitter = _Sink()
    try:
        capsera.record(
            model="x" * 300,               # hostile on purpose
            provider="custom",
            input_tokens=100,
            output_tokens=-5,
            agent_id="manual-agent",
        )
    finally:
        capsera._emitter = original

    assert len(captured) == 1
    backend_event_schema().model_validate(captured[0].to_dict())
