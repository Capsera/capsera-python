from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from ._config import Config
from ._models import TokenEvent
from ._version import __version__

logger = logging.getLogger("capsera")


class Emitter:
    _QUEUE_MAXSIZE = 10_000
    _FAILURE_THRESHOLD = 5
    _CIRCUIT_OPEN_SECONDS = 30

    def __init__(
        self,
        config: Config,
        max_retries: int = 3,
        on_error: Callable[[str], None] | None = None,
        budget_engine: Any | None = None,
    ) -> None:
        self._config = config
        self._max_retries = max_retries
        self._on_error = on_error
        # The local budget engine, when enforcement is on. The emitter drives
        # it because the emitter is where the two facts it needs already pass
        # through: every completed call (to count as unreported spend) and
        # every backend response (which carries a fresh snapshot).
        self._budget_engine = budget_engine
        self._queue: queue.Queue[TokenEvent] = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._consecutive_failures = 0
        self._circuit_open_until: float = 0.0

        self._limit_reached = False
        self._limit_resets_at: str | None = None

        self._client = httpx.Client(
            base_url=config.endpoint,
            headers={
                # `or ""`: a None api_key (unset env var passed straight
                # through by the caller) must degrade to auth failures on
                # ingest — never an AttributeError inside init() itself,
                # which crashes the host app at import time.
                "X-API-Key": config.api_key or "",
                "Content-Type": "application/json",
                "User-Agent": f"capsera-sdk/{__version__}",
            },
            timeout=5.0,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background flush thread.  Safe to call multiple times."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="capsera-emitter",
        )
        self._thread.start()

    def shutdown(self) -> None:
        """
        Signal the worker to stop, flush remaining events, then join the thread.

        Called automatically by capsera.shutdown().  Safe to call from any
        thread.  Blocks for up to 10 seconds waiting for the background thread.
        """
        self._stop_event.set()
        # One final flush on the calling thread so we don't lose the last batch
        # even if the worker thread has already exited.
        self.flush()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._client.close()

    # ------------------------------------------------------------------ #
    # Enqueueing
    # ------------------------------------------------------------------ #

    def enqueue(self, event: TokenEvent) -> None:
        # Before the drop checks below, deliberately: the money was spent
        # whether or not we manage to report it, and the local budget engine
        # is enforcing against reality, not against what we shipped.
        self._budget_hook("record_event", event)

        if self._limit_reached:
            now = datetime.now(UTC).strftime("%Y-%m")
            if self._limit_resets_at and now <= self._limit_resets_at:
                logger.debug("capsera: event dropped — monthly limit reached (resets %s)", self._limit_resets_at)
                return
            self._limit_reached = False
            self._limit_resets_at = None

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            msg = f"capsera: event queue full ({self._queue.qsize()}/{self._QUEUE_MAXSIZE}) — dropping event"
            logger.warning(msg)
            self._notify(msg)

    def _notify(self, msg: str) -> None:
        """Invoke the user's on_error observer, swallowing whatever it raises.

        enqueue() runs inline with the customer's LLM call — a raising
        observer must never turn a dropped event into a crashed provider
        call (and must never kill the flush path either).
        """
        if not self._on_error:
            return
        try:
            self._on_error(msg)
        except Exception:
            logger.warning("capsera: on_error callback raised", exc_info=self._config.debug)

    # ------------------------------------------------------------------ #
    # Flushing
    # ------------------------------------------------------------------ #

    def flush(self) -> None:
        now = time.monotonic()
        if now < self._circuit_open_until:
            qsize = self._queue.qsize()
            if qsize > self._QUEUE_MAXSIZE // 2:
                logger.warning("capsera: circuit open, queue at %d/%d", qsize, self._QUEUE_MAXSIZE)
            else:
                logger.debug("capsera: circuit open — skipping flush")
            return

        events: list[TokenEvent] = []
        try:
            while True:
                events.append(self._queue.get_nowait())
        except queue.Empty:
            pass

        if not events:
            return

        self._send(events)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _worker(self) -> None:
        """Background thread body: flush every flush_interval_ms milliseconds."""
        # Floor the interval: Event.wait(0) (a zero/negative flush_interval_ms
        # from customer config) returns immediately, turning this daemon into
        # a 100%-CPU busy loop inside the customer's process.
        interval = max(self._config.flush_interval_ms / 1000.0, 0.05)
        while not self._stop_event.wait(timeout=interval):
            try:
                self.flush()
                # Covers what the ingest response cannot: a process that has
                # not flushed yet, and one idle long enough that a sibling may
                # have moved a shared budget. Cheap — the engine decides
                # whether a fetch is actually due.
                self._budget_hook("maybe_refresh")
            except Exception:
                # Never let an exception kill the background thread.
                if self._config.debug:
                    logger.exception("capsera: unexpected error in emitter worker")

    def _send(self, events: list[TokenEvent]) -> None:
        payload: dict[str, Any] = {"events": [e.to_dict() for e in events]}
        blocks = self._drain_blocks()
        if blocks:
            payload["blocks"] = blocks
        delays = [1, 2, 4]
        was_circuit_open = self._consecutive_failures >= self._FAILURE_THRESHOLD

        for attempt in range(self._max_retries):
            try:
                resp = self._client.post("/v1/ingest/events", json=payload)

                if resp.status_code < 400:
                    if was_circuit_open:
                        logger.info("capsera: circuit closed, resuming delivery")
                    self._consecutive_failures = 0
                    self._limit_reached = False
                    # The backend has now recorded these events, and the
                    # snapshot it just returned includes them — so stop
                    # counting them locally or they are counted twice.
                    self._apply_snapshot(resp)
                    self._budget_hook("settle_events", events)
                    if self._config.debug:
                        logger.info("capsera: sent %d event(s) → status=%d", len(events), resp.status_code)
                    return

                if resp.status_code == 429:
                    self._consecutive_failures = 0
                    try:
                        body = resp.json()
                        detail = body.get("detail", body)
                        resets_at = detail.get("resets_at", "") if isinstance(detail, dict) else ""
                    except Exception:
                        resets_at = ""
                    self._limit_reached = True
                    self._limit_resets_at = resets_at[:7] if resets_at else None
                    msg = f"capsera: monthly event limit reached (resets {resets_at or 'next month'}). Events will not be tracked."
                    logger.warning(msg)
                    self._notify(msg)
                    # Never going to be recorded, so holding them in the ledger
                    # would tighten every local budget forever. Undercounting
                    # beats killing a healthy agent — see capsera._ledger.
                    self._budget_hook("settle_events", events)
                    return

                if 400 <= resp.status_code < 500:
                    # The payload can never succeed — dropping is correct,
                    # dropping SILENTLY is not: this is exactly the signal an
                    # operator needs (contract drift, bad api key, ...).
                    msg = (
                        f"capsera: backend rejected {len(events)} event(s) with "
                        f"{resp.status_code} — dropped (not retryable)"
                    )
                    logger.warning(msg)
                    self._notify(msg)
                    self._budget_hook("settle_events", events)
                    return

                self._record_failure()
                if attempt < self._max_retries - 1:
                    time.sleep(delays[min(attempt, len(delays) - 1)])

            except httpx.TransportError:
                # ConnectError, timeouts, ReadError, RemoteProtocolError
                # (server closed mid-response), … — all equally transient,
                # all equally worth the retry ladder.
                self._record_failure()
                if attempt < self._max_retries - 1:
                    time.sleep(delays[min(attempt, len(delays) - 1)])
            except Exception as exc:
                # Non-network failure (serialization bug, interpreter
                # teardown). Retrying cannot help; requeueing would poison
                # the next flush too.
                self._record_failure()
                msg = f"capsera: failed to send {len(events)} event(s): {exc}"
                logger.warning(msg)
                self._notify(msg)
                return

        # Retry ladder exhausted on transient failures. The backend may come
        # back and dedup-by-request_id makes a re-send safe, so the batch goes
        # back into the queue for the next flush instead of being dropped —
        # overflow (the queue refilled while we were sending) drops like any
        # other enqueue rather than blocking.
        requeued = 0
        for event in events:
            try:
                self._queue.put_nowait(event)
                requeued += 1
            except queue.Full:
                break
        msg = (
            f"capsera: failed to deliver {len(events)} event(s) after "
            f"{self._max_retries} attempts — {requeued} requeued for the next flush"
            + (f", {len(events) - requeued} dropped (queue full)" if requeued < len(events) else "")
        )
        logger.warning(msg)
        self._notify(msg)

    def _drain_blocks(self) -> list[dict[str, Any]]:
        """Locally-refused calls to report alongside this batch."""
        engine = self._budget_engine
        if engine is None:
            return []
        try:
            return engine.drain_blocks()
        except Exception:
            logger.debug("capsera: could not drain block reports", exc_info=True)
            return []

    def _budget_hook(self, method: str, *args: Any) -> None:
        """Call into the budget engine, swallowing anything it raises.

        Delivery is the emitter's job; enforcement bookkeeping is a passenger
        on it. A bug in the engine must not stop events reaching the backend,
        and must never surface on the caller's thread through enqueue().
        """
        engine = self._budget_engine
        if engine is None:
            return
        try:
            getattr(engine, method)(*args)
        except Exception:
            logger.debug("capsera: budget engine %s failed", method, exc_info=True)

    def _apply_snapshot(self, resp: httpx.Response) -> None:
        if self._budget_engine is None:
            return
        try:
            body = resp.json()
        except Exception:
            return
        if isinstance(body, dict):
            self._budget_hook("apply_snapshot", body.get("budget_snapshot"))

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._FAILURE_THRESHOLD:
            self._circuit_open_until = time.monotonic() + self._CIRCUIT_OPEN_SECONDS
            msg = f"capsera: circuit breaker open — pausing delivery for {self._CIRCUIT_OPEN_SECONDS}s"
            logger.warning(msg)
            if self._on_error:
                self._on_error(msg)
