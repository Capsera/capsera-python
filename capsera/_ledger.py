"""
In-process spend ledger: what this process has spent that the backend has not
been told about yet.

The point of the local budget engine is that a process already knows its own
spend. It just made the call and read the real token counts off the response;
no network round trip can tell it anything it does not already know. What it
cannot know is what *other* processes spent, and that is the only thing the
periodic snapshot is for. So the enforcement value is:

    effective_spend = snapshot_baseline + local_pending

``local_pending`` is exactly the events sitting in the emitter queue plus the
batch currently in flight — spend that has happened but that the backend's
``current_spend_usd`` does not include yet. Once a batch is acknowledged, the
backend's next snapshot will contain it, so it is subtracted here to avoid
counting it twice.

Buckets, not a list
-------------------
Every budget scope resolves to a counter this can answer in O(1): a global
budget wants the running total, an agent budget wants that agent's total, a
team budget wants that team's. Summing a list of pending events per call would
put an O(queue depth) loop on the hot path of every LLM request.

When counting has to be wrong, it undercounts
---------------------------------------------
Two deliberate choices, both in the same direction. Distinct attribution keys
are capped, and a batch that the backend permanently rejects is settled rather
than retained. Both lose a little spend from the local view. The alternative —
retaining spend the backend will never confirm — makes the local total drift
upward forever, and an upward drift on a ``block`` budget is a false-positive
kill of a healthy agent. Undercounting costs the partner money they were going
to spend anyway; overcounting costs them a production outage.
"""
from __future__ import annotations

import logging
import threading

from ._budget_rules import SCOPE_AGENT, SCOPE_GLOBAL, SCOPE_TEAM, BudgetRule, LocalSpend

logger = logging.getLogger("capsera")

# Distinct agents/teams/customers tracked before new keys stop being recorded
# individually. Generous for any real deployment; a bound at all is what stops
# a caller that mints an agent id per request from growing this without limit.
MAX_KEYS = 2_000


class SpendLedger(LocalSpend):
    """Thread-safe counters for spend the backend has not acknowledged.

    Implements ``LocalSpend`` so it can be handed straight to
    ``_budget_rules.evaluate()``.
    """

    def __init__(self, max_keys: int = MAX_KEYS) -> None:
        self._lock = threading.Lock()
        self._max_keys = max_keys
        self._total = 0.0
        self._by_agent: dict[str, float] = {}
        self._by_team: dict[str, float] = {}
        self._by_customer: dict[str, float] = {}
        self._warned_full = False

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record(
        self,
        cost_usd: float,
        agent_id: str | None,
        team_id: str | None,
        customer_id: str | None,
    ) -> None:
        """Add a completed call's cost to the pending total."""
        if not cost_usd:
            return
        with self._lock:
            self._total += cost_usd
            self._bump(self._by_agent, agent_id, cost_usd)
            self._bump(self._by_team, team_id, cost_usd)
            self._bump(self._by_customer, customer_id, cost_usd)

    def settle(
        self,
        cost_usd: float,
        agent_id: str | None,
        team_id: str | None,
        customer_id: str | None,
    ) -> None:
        """Remove a cost the backend has now recorded (or will never record)."""
        if not cost_usd:
            return
        with self._lock:
            self._total = _floor(self._total - cost_usd)
            self._drain(self._by_agent, agent_id, cost_usd)
            self._drain(self._by_team, team_id, cost_usd)
            self._drain(self._by_customer, customer_id, cost_usd)

    def _bump(self, bucket: dict[str, float], key: str | None, cost: float) -> None:
        if key is None:
            return
        if key not in bucket and len(bucket) >= self._max_keys:
            # Still counted in the running total, just not attributable to a
            # scoped budget. See the module docstring on undercounting.
            if not self._warned_full:
                logger.warning(
                    "capsera: local spend ledger hit %d distinct keys — "
                    "scoped budgets may undercount until the next snapshot",
                    self._max_keys,
                )
                self._warned_full = True
            return
        bucket[key] = bucket.get(key, 0.0) + cost

    def _drain(self, bucket: dict[str, float], key: str | None, cost: float) -> None:
        if key is None or key not in bucket:
            return
        remaining = _floor(bucket[key] - cost)
        if remaining <= 0.0:
            del bucket[key]
        else:
            bucket[key] = remaining

    # ------------------------------------------------------------------ #
    # LocalSpend
    # ------------------------------------------------------------------ #

    def for_budget(self, budget: BudgetRule) -> float:
        with self._lock:
            if budget.scope_type == SCOPE_GLOBAL:
                return self._total
            if budget.scope_type == SCOPE_AGENT:
                return self._by_agent.get(budget.scope_id or "", 0.0)
            if budget.scope_type == SCOPE_TEAM:
                return self._by_team.get(budget.scope_id or "", 0.0)
            return 0.0

    def for_customer(self, customer_id: str) -> float:
        with self._lock:
            return self._by_customer.get(customer_id, 0.0)

    # ------------------------------------------------------------------ #
    # Introspection / lifecycle
    # ------------------------------------------------------------------ #

    def pending_total(self) -> float:
        with self._lock:
            return self._total

    def reset(self) -> None:
        with self._lock:
            self._total = 0.0
            self._by_agent.clear()
            self._by_team.clear()
            self._by_customer.clear()
            self._warned_full = False


def _floor(value: float) -> float:
    """Clamp at zero.

    Record and settle race against each other across threads and accumulate
    float error over millions of calls; a total that drifts a hair below zero
    would then be added to the snapshot baseline as a negative, quietly
    inflating every budget's headroom.
    """
    return value if value > 0.0 else 0.0
