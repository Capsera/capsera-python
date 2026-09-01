"""Faults on the enforcement hot path.

`test_resilience.py` covers the *ingest* path, which is off the hot path — a
failure there loses telemetry. This covers the path in front of the customer's
provider call, where a failure loses their request. The bar is different: ingest
may drop data, but enforcement must never take the call down with it, and must
never take longer than its own timeout to decide not to.

The one thing that may propagate is a genuine `block` verdict. Everything else
— every status code, every malformed body, every stall — has to end with the
provider call completing.
"""

from __future__ import annotations

import ast
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from ..conftest import make_anthropic_response

SDK_ROOT = Path(__file__).resolve().parents[2] / "capsera"


# --------------------------------------------------------------------------- #
# A fault-injecting stand-in for the backend
# --------------------------------------------------------------------------- #


class _FaultServer:
    """Answers the SDK's enforcement endpoints badly, on purpose.

    Modes are per-instance rather than per-path because the SDK is meant to be
    indifferent to *which* enforcement call failed — the requirement is that the
    customer's call proceeds regardless.
    """

    def __init__(self, mode: str, *, delay: float = 0.0) -> None:
        self.mode = mode
        self.delay = delay
        self.hits: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # noqa: ANN002 - silence the test log
                pass

            def _respond(self, status: int, body: bytes, ctype="application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve(self) -> None:
                outer.hits.append(self.path)
                if outer.delay:
                    time.sleep(outer.delay)
                mode = outer.mode
                if mode == "unauthorized":
                    return self._respond(401, b'{"detail":"revoked key"}')
                if mode == "forbidden":
                    return self._respond(403, b'{"detail":"insufficient scope"}')
                if mode == "server_error":
                    return self._respond(500, b'{"detail":"boom"}')
                if mode == "malformed":
                    return self._respond(200, b'{"allowed": tr')  # truncated JSON
                if mode == "html":
                    return self._respond(200, b"<html>proxy error</html>", "text/html")
                if mode == "unknown_action":
                    # A verdict from a newer backend than this SDK knows about.
                    return self._respond(200, json.dumps({
                        "allowed": True, "action": "quarantine-and-page-someone",
                    }).encode())
                return self._respond(200, b'{"allowed": true, "action": null}')

            def do_GET(self):
                self._serve()

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                self._serve()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()


@pytest.fixture
def fault_server():
    servers: list[_FaultServer] = []

    def make(mode: str, *, delay: float = 0.0) -> _FaultServer:
        server = _FaultServer(mode, delay=delay)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.stop()


def _call_through(config: Config, emitter, provider_response=None):
    """Make one patched provider call and return it, or raise what escaped."""
    from types import SimpleNamespace

    interceptor = Interceptor(config, emitter)
    response = provider_response or make_anthropic_response()
    captured = {}

    class Messages:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return response

    module = SimpleNamespace(
        Anthropic=type("Anthropic", (), {"messages": None}),
        resources=SimpleNamespace(messages=SimpleNamespace(Messages=Messages)),
    )
    interceptor._patch_anthropic_sync(module)
    result = Messages().create(model="claude-sonnet-4-6", max_tokens=8,
                               messages=[{"role": "user", "content": "hi"}])
    return result, captured


# --------------------------------------------------------------------------- #
# A1.3 / A1.5 / A1.6 — the enforcement call answers badly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", [
    "server_error",     # A1: backend 500
    "unauthorized",     # A1.6: revoked API key
    "forbidden",        # A1.6: scope refused
    "malformed",        # A1.4: truncated JSON
    "html",             # a proxy or WAF answering instead of the backend
    "unknown_action",   # A1.5: a verdict this SDK version has never heard of
])
def test_a_bad_enforcement_answer_never_takes_the_call_down(mode, fault_server, capturing_emitter):
    """Every one of these must end with the provider call completing.

    `unknown_action` is the subtle one: a newer backend returning a verdict this
    SDK does not implement must be treated as "allow", not as an error and
    certainly not as a block. Getting that wrong makes a backend deploy break
    every older SDK in the field.
    """
    server = fault_server(mode)
    config = Config(api_key="cap-test", endpoint=server.url, enforcement_mode="strict")

    result, _ = _call_through(config, capturing_emitter)

    assert result is not None, f"{mode}: the provider call did not complete"
    assert server.hits, f"{mode}: the SDK never attempted the enforcement call"


def test_a_stalled_enforcement_call_is_abandoned_at_the_timeout(fault_server, capturing_emitter):
    """Failing open on errors is not enough — it must not fail open *slowly*.

    This is the case the test plan calls the sneaky one: nobody gets paged for a
    hang, and a customer whose p99 doubled will not thank us for having
    technically allowed the call.
    """
    timeout = 0.4
    server = fault_server("ok", delay=5.0)
    config = Config(
        api_key="cap-test", endpoint=server.url,
        enforcement_mode="strict", budget_check_timeout=timeout,
    )

    started = time.perf_counter()
    result, _ = _call_through(config, capturing_emitter)
    elapsed = time.perf_counter() - started

    assert result is not None, "a stalled enforcement call blocked the provider call"
    # Generous headroom over the timeout, but far below the server's 5s stall:
    # the point is that the wait is bounded by our own setting, not by theirs.
    assert elapsed < timeout + 1.5, (
        f"waited {elapsed:.2f}s on a {timeout}s timeout — the deadline is not being honoured"
    )


def test_an_unreachable_backend_does_not_delay_the_call(capturing_emitter):
    """Connection refused resolves immediately, but assert it rather than assume:
    a retry loop here would multiply the cost of an outage by the retry count."""
    config = Config(
        api_key="cap-test", endpoint="http://127.0.0.1:1",   # nothing listens
        enforcement_mode="strict", budget_check_timeout=1.0,
    )
    started = time.perf_counter()
    result, _ = _call_through(config, capturing_emitter)
    elapsed = time.perf_counter() - started

    assert result is not None
    assert elapsed < 2.0, f"an unreachable backend cost {elapsed:.2f}s per call"


# --------------------------------------------------------------------------- #
# A2.1 — every outbound call carries a deadline
# --------------------------------------------------------------------------- #


def _httpx_calls_without_timeout(path: Path) -> list[str]:
    """Find httpx clients built, or requests made, with no timeout.

    A static audit rather than a runtime one on purpose: a missing timeout is
    only observable when the far end stalls, so a test that waits for that to
    happen would be slow, flaky, and would miss whichever call path the test did
    not happen to exercise. The parser sees all of them.
    """
    tree = ast.parse(path.read_text())
    offenders: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Attribute):
            name = func.attr
            root = func.value
            root_name = getattr(root, "id", "") or getattr(getattr(root, "value", None), "id", "")
        else:
            continue

        has_timeout = any(kw.arg == "timeout" for kw in node.keywords)

        # Constructing a client: the timeout may be set here for every request
        # it will make, which is the preferred form.
        if name in ("Client", "AsyncClient") and root_name == "httpx":
            if not has_timeout:
                offenders.append(f"{path.name}:{node.lineno} httpx.{name}() with no timeout")
            continue

        # A one-shot module-level request.
        if root_name == "httpx" and name in ("get", "post", "put", "patch", "delete", "request"):
            if not has_timeout:
                offenders.append(f"{path.name}:{node.lineno} httpx.{name}() with no timeout")

    return offenders


def test_every_httpx_client_and_request_sets_a_timeout():
    """No unbounded wait anywhere in the SDK.

    The SDK runs inside a customer's request path. One outbound call without a
    deadline turns a slow dependency of ours into a hang of theirs, and a hang
    is worse than a crash because nobody gets paged for it.
    """
    offenders: list[str] = []
    for path in sorted(SDK_ROOT.rglob("*.py")):
        offenders.extend(_httpx_calls_without_timeout(path))

    assert not offenders, "outbound calls with no deadline:\n  " + "\n  ".join(offenders)


# --------------------------------------------------------------------------- #
# A1.15 / A1.18 — lifecycle
# --------------------------------------------------------------------------- #


def test_re_init_does_not_double_patch(capturing_emitter):
    """`init()` is documented as safe to call more than once.

    Asserted by counting events, not by inspecting the guard flag: if a second
    patch layer stacked, one provider call would emit twice, and double-counted
    spend blocks a customer at half their real usage. An earlier version of this
    test checked the flag and could not fail.
    """
    config = Config(api_key="cap-test", endpoint="http://127.0.0.1:1", enforcement_mode="off")

    # Patch the same module three times through three interceptors, which is the
    # shape a repeated init() takes.
    from types import SimpleNamespace

    response = make_anthropic_response()
    seen = {"provider_calls": 0}

    class Messages:
        def create(self, **kwargs):
            seen["provider_calls"] += 1
            return response

    module = SimpleNamespace(
        Anthropic=type("Anthropic", (), {"messages": None}),
        resources=SimpleNamespace(messages=SimpleNamespace(Messages=Messages)),
    )
    for _ in range(3):
        Interceptor(config, capturing_emitter)._patch_anthropic_sync(module)

    module.resources.messages.Messages().create(
        model="claude-sonnet-4-6", max_tokens=8,
        messages=[{"role": "user", "content": "hi"}],
    )
    # drain(), not .captured: the emitter queues and drain() is the accessor.
    emitted = len(capturing_emitter.drain())

    assert seen["provider_calls"] == 1, (
        f"the provider was called {seen['provider_calls']} times for one request — "
        "patches are stacking"
    )
    assert emitted == 1, (
        f"one provider call produced {emitted} events — double counting would block "
        "a customer at half their real usage"
    )


def test_a_dead_emitter_thread_does_not_break_later_calls(capturing_emitter):
    """The emitter runs on a daemon thread. If that thread dies — an unhandled
    error, an interpreter quirk — the customer's calls must keep working, with
    telemetry degraded rather than the application broken."""
    config = Config(api_key="cap-test", endpoint="http://127.0.0.1:1", enforcement_mode="off")

    class DeadEmitter:
        def __init__(self):
            self.calls = 0

        def enqueue(self, event):
            self.calls += 1
            raise RuntimeError("emitter thread is gone")

    dead = DeadEmitter()
    result, _ = _call_through(config, dead)

    assert result is not None, "a broken emitter took the provider call down with it"
    assert dead.calls >= 1, "the SDK never tried to record the event"


# --------------------------------------------------------------------------- #
# A2.2 / A2.3 — the wait must be bounded by *our* deadline, not the far end's
# --------------------------------------------------------------------------- #


def test_a_hostname_that_does_not_resolve_does_not_stall_the_call(capturing_emitter):
    """DNS is an outbound call too.

    A resolver that cannot answer is a distinct failure from a server that will
    not: it happens before any socket exists, so a client-level timeout is not
    guaranteed to cover it. `.invalid` is reserved by RFC 2606 precisely so it
    can never resolve, which makes this deterministic rather than dependent on
    whatever the local resolver does with a typo.
    """
    config = Config(
        api_key="cap-test",
        endpoint="http://capsera-does-not-exist.invalid",
        enforcement_mode="strict",
        budget_check_timeout=1.0,
    )

    started = time.perf_counter()
    result, _ = _call_through(config, capturing_emitter)
    elapsed = time.perf_counter() - started

    assert result is not None, "an unresolvable backend host blocked the provider call"
    assert elapsed < 5.0, (
        f"DNS failure cost {elapsed:.2f}s on the hot path — a resolver timeout is "
        "not bounded by budget_check_timeout"
    )


class _BlackHoleTLS:
    """Accepts the TCP connection and then says nothing, forever.

    This is the shape of a stall that a connect timeout does not catch: the
    socket opens, so the client believes it is talking to something, and then
    waits for a TLS ServerHello that never comes. A load balancer draining, or a
    security appliance in the path, both look like this.
    """

    def __init__(self) -> None:
        import socket

        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._held: list = []
        self._stop = threading.Event()
        threading.Thread(target=self._accept_forever, daemon=True).start()

    def _accept_forever(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            # Hold the connection open without ever replying.
            self._held.append(conn)

    def stop(self) -> None:
        self._stop.set()
        for conn in self._held:
            try:
                conn.close()
            except OSError:
                pass
        self._sock.close()


def test_a_stalled_tls_handshake_is_bounded_by_the_timeout(capturing_emitter):
    """The socket opens and the handshake never completes.

    Worth its own case because it defeats the intuitive fix: a connect timeout
    is satisfied the moment the TCP connection is accepted, so only a total
    deadline saves the call.
    """
    black_hole = _BlackHoleTLS()
    try:
        config = Config(
            api_key="cap-test",
            endpoint=f"https://127.0.0.1:{black_hole.port}",
            enforcement_mode="strict",
            budget_check_timeout=0.5,
        )

        started = time.perf_counter()
        result, _ = _call_through(config, capturing_emitter)
        elapsed = time.perf_counter() - started

        assert result is not None, "a stalled TLS handshake blocked the provider call"
        assert elapsed < 3.0, (
            f"waited {elapsed:.2f}s on a 0.5s timeout — the handshake is outside the deadline"
        )
    finally:
        black_hole.stop()


# --------------------------------------------------------------------------- #
# A4.5 — nothing on the hot path accumulates
# --------------------------------------------------------------------------- #


def test_the_router_budget_cache_does_not_grow_with_traffic(capturing_emitter):
    """Caches sized by *data* are fine; caches sized by *traffic* are leaks.

    The router keeps a budget-utilisation map so it can route away from a
    nearly-exhausted budget. Its size must track the number of budgets, not the
    number of calls made — a soak would eventually surface the difference, but
    only after hours, and only if the leak is fast.
    """
    from capsera._router import Router

    config = Config(api_key="cap-test", endpoint="http://127.0.0.1:1", enable_routing=True)
    router = Router(config)

    router.set_budget_cache({"b1": 0.5, "b2": 0.9})
    assert len(router._budget_cache) == 2

    # Many decisions against the same two budgets.
    for i in range(2000):
        router.evaluate(
            {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": str(i)}]},
            None,
            refresh=False,   # no network: this is about accumulation, not fetching
        )

    assert len(router._budget_cache) == 2, (
        f"the budget cache grew to {len(router._budget_cache)} entries across 2000 calls — "
        "it is sized by traffic, not by the number of budgets"
    )

    # A replacement must replace, not merge: a budget that no longer exists
    # should stop being considered rather than linger forever.
    router.set_budget_cache({"b3": 0.1})
    assert set(router._budget_cache) == {"b3"}
