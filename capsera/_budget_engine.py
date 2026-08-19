"""
The local budget engine: enforcement without a network hop on the hot path.

Every LLM call used to POST to ``/v1/budgets/check`` and wait for the verdict.
That put a round trip — and the backend's availability — in front of the
customer's provider call. This module answers the same question in-process,
in microseconds, from three things:

* a **snapshot** of budget state, refreshed in the background
  (``_budget_cache``);
* a **ledger** of what this process has spent since that snapshot, which it
  knows exactly because it just made the calls (``_ledger``);
* the **predicate**, shared byte-for-byte with the backend (``_budget_rules``).

    effective_spend = snapshot_baseline + local_pending

Refresh rides on the emitter
----------------------------
The snapshot is not polled on a timer of its own. The emitter is already
POSTing to ``/v1/ingest/events`` every 500 ms, and the backend returns the
current snapshot on that response — so refresh costs no extra request, and its
cadence tracks traffic automatically. A process making thousands of calls a
second refreshes constantly; an idle process does not refresh, and does not
need to, because it is not spending.

``maybe_refresh()`` covers the two cases the ingest response cannot: a process
that has not flushed anything yet, and one that has been idle long enough that
*another* process may have moved a shared budget. Its interval tightens as
utilisation climbs, because that is the only regime where a stale sibling
figure changes the verdict.

Failure is always "allow"
-------------------------
No snapshot, corrupt snapshot, unreachable backend, bug in this file: the call
proceeds. An enforcement layer that takes down a partner's application is a
worse product than one that occasionally lets a call through. The one
exception is a genuine ``block`` verdict, which raises — that is the feature.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

import httpx

from ._budget_cache import SnapshotCache, parse_snapshot
from ._budget_rules import PreflightRequest, allow, evaluate
from ._config import Config
from ._ledger import SpendLedger
from ._models import BudgetExceededError, TokenEvent
from ._version import __version__

logger = logging.getLogger("capsera")

# Refresh cadence for the idle path, in seconds. The floor applies near the
# cap, where another process's spend is what decides the verdict; the ceiling
# applies at low utilisation, where it cannot.
REFRESH_INTERVAL_MAX = 60.0
REFRESH_INTERVAL_MIN = 2.0

# Utilisation at which the interval starts contracting toward the floor.
ADAPTIVE_FLOOR_PCT = 0.5

_SNAPSHOT_OK = "ok"
_SNAPSHOT_FAILED = "failed"
# The endpoint is absent: a backend older than the local budget engine.
_SNAPSHOT_UNSUPPORTED = "unsupported"

MODE_LOCAL = "local"
MODE_STRICT = "strict"
MODE_OFF = "off"


class BudgetEngine:
    """Owns the snapshot, the ledger, and the refresh schedule."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self.ledger = SpendLedger()
        self.cache = SnapshotCache(
            endpoint=config.endpoint,
            api_key=config.api_key or "",
            persist=config.budget_cache_persist,
        )
        # Blocks refused locally, awaiting report on the next flush. Bounded:
        # an agent in a hard-blocked retry loop can generate these far faster
        # than they flush, and they all dedup to one alert anyway.
        self._blocks: deque[dict[str, Any]] = deque(maxlen=100)
        self._blocks_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._refreshing = False
        self._last_refresh = 0.0
        self._client: httpx.Client | None = None

        # A warm start from disk means a short-lived process (a Lambda, a CLI
        # run) enforces from its first call instead of never.
        try:
            self.cache.load()
        except Exception:
            logger.debug("capsera: budget cache warm start failed", exc_info=True)

    def prime(self) -> None:
        """Make sure a snapshot exists before the first LLM call.

        Without this, enforcement in a short-lived process is decorative. The
        background refresh rides on the emitter's flush, and a script that
        makes its calls and exits never reaches one — so a CLI run, a worker
        that processes one job, or a Lambda's first invocation would enforce
        nothing at all and never know it. That is a budget product not doing
        the one thing it claims.

        The cost is one request at startup, once per process. That is a
        different thing from the per-call round trip this engine exists to
        remove: it is paid on the caller's clock at init() time, where a
        hundred milliseconds is invisible, not in front of every LLM call. A
        usable snapshot on disk skips even that.

        Fail-open, and bounded by ``budget_check_timeout``: an unreachable
        backend must delay startup briefly and then be forgotten, never hang
        the host application's boot.
        """
        if self._config.enforcement_mode != MODE_LOCAL:
            return
        if self.cache.get() is not None:
            # Warm start from disk. Good enough to enforce against; the
            # emitter's first flush refreshes it anyway.
            return
        if self._fetch_snapshot() == _SNAPSHOT_UNSUPPORTED:
            self._fall_back_to_strict()
            return
        with self._refresh_lock:
            self._last_refresh = _monotonic()

    def _fall_back_to_strict(self) -> None:
        """Use the per-call check when the backend cannot serve snapshots.

        A backend older than the snapshot endpoint would leave local mode with
        nothing to enforce against — and it would fail *open*, silently, which
        is the worst possible shape for this: the dashboard still shows spend,
        budgets still look enabled, and nothing is actually being stopped.

        Falling back costs a round trip per call, which is exactly what local
        mode exists to avoid. It is still the right trade: enforcement that is
        slow beats enforcement that is absent and unannounced. It also means
        the SDK can be released independently of the backend rather than
        strictly after it.

        Only a definitive 404 triggers this. A timeout or connection error is a
        backend that is down, not a backend that is old, and strict mode would
        fare no better against it.

        404 is also what an older backend actually answers, though not for the
        reason it looks like: it has no /budgets/snapshot route, so the path is
        captured by ``GET /budgets/{budget_id}`` with the id "snapshot", which
        finds no such budget and 404s. Different mechanism, same correct
        conclusion — this backend cannot give us a snapshot. (Unauthenticated
        that shadowed route answers 401, which is why 401 must NOT be treated
        as "old": there it means the credential is wrong, and silently
        dropping to strict would hide a broken API key behind a slow path.)
        """
        logger.warning(
            "capsera: backend does not serve /v1/budgets/snapshot (needs the "
            "release that added it) — falling back to per-call budget checks. "
            "Enforcement still works; expect a round trip per LLM call."
        )
        self._config.enforcement_mode = MODE_STRICT

    # ------------------------------------------------------------------ #
    # Hot path
    # ------------------------------------------------------------------ #

    def decide(self, request: PreflightRequest) -> dict[str, Any]:
        """Evaluate the pending call against the cached snapshot.

        Pure CPU: a scan over the workspace's budgets and some float
        arithmetic. No I/O, no lock held across anything that can block.
        """
        snapshot = self.cache.get()
        if snapshot is None:
            # Cold start. Allow, and let the emitter tick fetch a snapshot —
            # blocking here to fetch one would reintroduce exactly the round
            # trip this engine exists to remove.
            return allow()
        try:
            return evaluate(snapshot, request, self.ledger)
        except Exception:
            logger.debug("capsera: budget evaluation failed — allowing", exc_info=True)
            return allow()

    def record_block(self, decision: dict[str, Any], request: PreflightRequest) -> None:
        """Remember that we refused a call, so the backend can alert on it.

        Local enforcement means no request reaches the backend at the moment
        of refusal, and "an agent just stopped" is precisely the event someone
        needs to be told about. Queued rather than sent: this runs inline with
        the caller's LLM call, and a notification is never worth a round trip
        on that path.
        """
        budget_id = decision.get("budget_id")
        if not budget_id:
            return
        with self._blocks_lock:
            self._blocks.append(
                {
                    "budget_id": str(budget_id),
                    "agent_id": request.agent_id,
                    "team_id": request.team_id,
                    "message": str(decision.get("message") or "")[:1024],
                    "pct_used": float(decision.get("pct_used") or 0.0),
                    "projected_spend_usd": float(
                        decision.get("projected_spend_usd") or 0.0
                    ),
                }
            )

    def drain_blocks(self) -> list[dict[str, Any]]:
        """Take the pending block reports, clearing the queue."""
        with self._blocks_lock:
            pending = list(self._blocks)
            self._blocks.clear()
        return pending

    # ------------------------------------------------------------------ #
    # Ledger hooks — driven by the emitter
    # ------------------------------------------------------------------ #

    def record_event(self, event: TokenEvent) -> None:
        """Count a completed call as spend the backend does not know about."""
        self.ledger.record(
            event.cost_usd, event.agent_id, event.team_id, event.customer_id
        )

    def settle_events(self, events: list[TokenEvent]) -> None:
        """Stop counting events the backend has now recorded — or never will.

        Called both on a successful flush (the next snapshot will contain them)
        and on a permanent rejection (it never will, and holding them would
        inflate the local total forever — see ``_ledger``).
        """
        for event in events:
            self.ledger.settle(
                event.cost_usd, event.agent_id, event.team_id, event.customer_id
            )

    def apply_snapshot(self, raw: dict[str, Any] | None) -> None:
        """Install a snapshot that arrived on an ingest response."""
        if not raw:
            return
        try:
            self.cache.set(parse_snapshot(raw))
        except Exception:
            logger.debug("capsera: could not apply budget snapshot", exc_info=True)

    # ------------------------------------------------------------------ #
    # Refresh
    # ------------------------------------------------------------------ #

    def maybe_refresh(self, *, now: float | None = None) -> None:
        """Fetch a snapshot if the cached one has gone stale.

        Called from the emitter's background thread, never from a call path.
        Single-flight: a refresh already in progress means this returns
        immediately rather than stacking a second request onto a backend that
        may simply be slow.
        """
        if self._config.enforcement_mode != MODE_LOCAL:
            return

        moment = now if now is not None else _monotonic()
        with self._refresh_lock:
            if self._refreshing:
                return
            if moment - self._last_refresh < self._refresh_interval():
                return
            self._refreshing = True

        try:
            self._fetch_snapshot()
        finally:
            with self._refresh_lock:
                self._refreshing = False
                self._last_refresh = _monotonic()

    def _refresh_interval(self) -> float:
        """Seconds between idle refreshes, tightening as the cap approaches.

        Below ``ADAPTIVE_FLOOR_PCT`` a sibling process's spend cannot flip a
        verdict, so there is nothing to buy by asking more often. Above it, the
        interval is what bounds overshoot across processes, so it contracts
        linearly to the floor at 100%.
        """
        pct = self._peak_utilisation()
        if pct <= ADAPTIVE_FLOOR_PCT:
            return REFRESH_INTERVAL_MAX
        if pct >= 1.0:
            return REFRESH_INTERVAL_MIN
        span = REFRESH_INTERVAL_MAX - REFRESH_INTERVAL_MIN
        travelled = (pct - ADAPTIVE_FLOOR_PCT) / (1.0 - ADAPTIVE_FLOOR_PCT)
        return REFRESH_INTERVAL_MAX - span * travelled

    def _peak_utilisation(self) -> float:
        snapshot = self.cache.get()
        if snapshot is None:
            return 1.0  # Nothing known yet — refresh eagerly.
        peak = 0.0
        for budget in snapshot.budgets:
            if budget.limit_usd <= 0:
                continue
            spend = budget.current_spend_usd + self.ledger.for_budget(budget)
            peak = max(peak, spend / budget.limit_usd)
        return peak

    def _fetch_snapshot(self) -> str:
        """Pull a fresh snapshot. Returns one of the _SNAPSHOT_* outcomes."""
        try:
            resp = self._http().get("/v1/budgets/snapshot")
            if resp.status_code == 200:
                self.cache.set(parse_snapshot(resp.json()))
                return _SNAPSHOT_OK
            if resp.status_code == 404:
                return _SNAPSHOT_UNSUPPORTED
            logger.debug("capsera: snapshot refresh → %d", resp.status_code)
        except Exception:
            # Unreachable backend is not an enforcement failure: the cached
            # snapshot plus this process's own ledger still enforces its own
            # spend correctly. It just stops seeing sibling processes.
            logger.debug("capsera: snapshot refresh failed", exc_info=True)
        return _SNAPSHOT_FAILED

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._config.endpoint,
                headers={
                    "X-API-Key": self._config.api_key or "",
                    "User-Agent": f"capsera-sdk/{__version__}",
                },
                timeout=self._config.budget_check_timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


def _monotonic() -> float:
    return time.monotonic()


def raise_if_blocked(decision: dict[str, Any]) -> None:
    """Turn a block verdict into the exception the caller can branch on."""
    if decision.get("allowed", True):
        return
    raise BudgetExceededError(
        decision.get("message") or "Capsera blocked this LLM call.",
        decision=decision,
    )
