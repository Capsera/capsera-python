"""
The budget enforcement predicate — one decision rule, no I/O.

This module is the single definition of "should this LLM call be allowed,
downgraded, or blocked". It is pure: it takes a snapshot of budget state and
a description of the pending call, and returns a decision dict. It performs
no database access, no HTTP, and no clock reads it is not handed.

Why it is pure, and why it is duplicated
----------------------------------------
Enforcement runs in two places that cannot import each other:

* the SDK, in the customer's process, against a cached snapshot (the fast
  path — no network on the hot path of an LLM call);
* the backend, against the database, for ``POST /v1/budgets/check`` (strict
  mode) and for the authoritative recheck at ingest.

Two hand-written copies of a rule this fiddly WILL diverge, and a divergence
here is a false-positive kill — the second-worst failure this product has.
So there is exactly one authored copy (``sdk/capsera/_budget_rules.py``);
``backend/app/services/budget_rules.py`` is generated from it by
``scripts/sync_budget_rules.py``, and CI fails on drift. Do not edit the
backend copy.

The decision is returned as a plain dict, deliberately: it is also the wire
format of ``/v1/budgets/check``, and a dataclass in the middle would be one
more place for the shapes to drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Bumped when the decision semantics change. The snapshot carries it so a new
# SDK talking to an old backend (or the reverse) can tell.
RULES_VERSION = 1

SCOPE_GLOBAL = "global"
SCOPE_AGENT = "agent"
SCOPE_TEAM = "team"

ACTION_ALERT = "alert"
ACTION_THROTTLE = "throttle"
ACTION_BLOCK = "block"
ACTION_DOWNGRADE = "downgrade"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BudgetRule:
    """One enabled budget, as the predicate needs to see it."""

    id: str
    name: str
    scope_type: str
    scope_id: str | None
    action: str
    limit_usd: float
    current_spend_usd: float
    period_end: datetime
    per_call_ceiling_usd: float | None = None


@dataclass(frozen=True)
class CustomerRule:
    """One customer account, for the margin guardrail."""

    customer_id: str
    name: str
    monthly_revenue_usd: float
    margin_alert_threshold: float
    current_spend_usd: float


@dataclass(frozen=True)
class BudgetSnapshot:
    """Everything the predicate needs, as of one moment in time.

    ``margin_targets`` maps provider -> cheapest sensible downgrade model. It
    is resolved by whoever builds the snapshot (the backend, which owns the
    model catalog and knows what has been retired) rather than here, so this
    module stays free of a catalog dependency that would have to be duplicated
    too.
    """

    budgets: tuple[BudgetRule, ...] = ()
    customers: tuple[CustomerRule, ...] = ()
    margin_targets: dict[str, str] = field(default_factory=dict)
    fetched_at: datetime | None = None
    rules_version: int = RULES_VERSION


@dataclass(frozen=True)
class PreflightRequest:
    """The call about to be made."""

    agent_id: str
    team_id: str | None = None
    customer_id: str | None = None
    model: str | None = None
    provider: str | None = None
    estimated_cost_usd: float = 0.0


class LocalSpend:
    """Spend the caller knows about that the snapshot does not yet include.

    The SDK subclasses this with its in-process ledger: it has already made
    calls whose events are still queued, so the snapshot's ``current_spend_usd``
    is behind by exactly that much. The backend reads spend straight from the
    database and so passes ``NO_LOCAL_SPEND`` — the base class, which adds
    nothing, keeping both sides on one code path.
    """

    def for_budget(self, budget: BudgetRule) -> float:  # noqa: ARG002
        return 0.0

    def for_customer(self, customer_id: str) -> float:  # noqa: ARG002
        return 0.0


NO_LOCAL_SPEND = LocalSpend()


# --------------------------------------------------------------------------- #
# The predicate
# --------------------------------------------------------------------------- #


def scope_matches(budget: BudgetRule, request: PreflightRequest) -> bool:
    """Does this budget govern this call?"""
    if budget.scope_type == SCOPE_GLOBAL:
        return True
    if budget.scope_type == SCOPE_AGENT:
        return budget.scope_id == request.agent_id
    if budget.scope_type == SCOPE_TEAM:
        return budget.scope_id == request.team_id
    return False


def _aware(moment: datetime) -> datetime:
    """Treat a naive timestamp as UTC.

    Postgres hands back naive datetimes for ``timestamp without time zone``
    columns; comparing one to an aware ``now`` raises TypeError, which the
    fail-open wrapper would swallow into "allowed" — enforcement silently off.
    """
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


def allow() -> dict[str, Any]:
    """The decision when nothing objects."""
    return {"allowed": True, "action": None}


def evaluate(
    snapshot: BudgetSnapshot,
    request: PreflightRequest,
    local: LocalSpend = NO_LOCAL_SPEND,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Decide whether the pending call may proceed.

    Returns a decision dict. ``allowed`` is False only for a ``block`` budget;
    ``action`` carries what the budget asked for so the caller can downgrade,
    throttle, or merely record. A ``block`` short-circuits: it is the strongest
    outcome and no later budget can soften it.

    The scan is deliberately ordered by severity, not by budget: the first
    blocking budget wins, and otherwise the first non-null action is kept.
    Ceilings are checked before the running total because a single call
    exceeding the per-call ceiling is a different fault from the budget being
    exhausted, and it must be reported as such even when there is headroom.
    """
    moment = now or datetime.now(UTC)
    strongest: dict[str, Any] = allow()

    for budget in snapshot.budgets:
        if not scope_matches(budget, request):
            continue

        ceiling = budget.per_call_ceiling_usd
        if ceiling and request.estimated_cost_usd > ceiling:
            decision = _ceiling_decision(budget, request)
            if budget.action == ACTION_BLOCK:
                return decision
            if strongest["action"] is None:
                strongest = decision
            continue

        # Past the period end the backend has simply not rolled the budget over
        # yet. Reading the stale total would block every call in the new period
        # until a write happens to reset it.
        rolled_over = moment > _aware(budget.period_end)
        current_spend = 0.0 if rolled_over else budget.current_spend_usd
        if not rolled_over:
            current_spend += local.for_budget(budget)

        projected = current_spend + request.estimated_cost_usd
        if projected < budget.limit_usd:
            continue

        decision = _limit_decision(budget, current_spend, projected)
        if budget.action == ACTION_BLOCK:
            return decision
        if strongest["action"] is None:
            strongest = decision

    if strongest["action"] is not None:
        return strongest

    return _margin_decision(snapshot, request, local) or strongest


def _ceiling_decision(budget: BudgetRule, request: PreflightRequest) -> dict[str, Any]:
    return {
        "allowed": budget.action != ACTION_BLOCK,
        "action": budget.action,
        "budget_id": budget.id,
        "budget_name": budget.name,
        "current_spend_usd": round(budget.current_spend_usd, 6),
        "limit_usd": budget.limit_usd,
        "projected_spend_usd": round(
            budget.current_spend_usd + request.estimated_cost_usd, 6
        ),
        "pct_used": 0.0,
        "message": (
            f"Call cost ${request.estimated_cost_usd:.4f} exceeds the per-call "
            f"ceiling ${budget.per_call_ceiling_usd:.2f} of budget '{budget.name}'."
        ),
    }


def _limit_decision(
    budget: BudgetRule, current_spend: float, projected: float
) -> dict[str, Any]:
    pct_used = projected / budget.limit_usd if budget.limit_usd > 0 else 0.0
    return {
        "allowed": budget.action != ACTION_BLOCK,
        "action": budget.action,
        "budget_id": budget.id,
        "budget_name": budget.name,
        "current_spend_usd": round(current_spend, 6),
        "limit_usd": budget.limit_usd,
        "projected_spend_usd": round(projected, 6),
        "pct_used": round(pct_used, 4),
        "message": (
            f"Budget '{budget.name}' would exceed its limit "
            f"(${projected:.2f} / ${budget.limit_usd:.2f})."
        ),
    }


def _margin_decision(
    snapshot: BudgetSnapshot,
    request: PreflightRequest,
    local: LocalSpend,
) -> dict[str, Any] | None:
    """Downgrade when a customer's LLM spend is eating their revenue.

    Never blocks — a customer being unprofitable is a pricing problem, not a
    reason to stop serving them. Returns None when there is no customer, no
    revenue figure to compare against, or no cheaper model to move to.
    """
    if not request.customer_id:
        return None

    customer = next(
        (c for c in snapshot.customers if c.customer_id == request.customer_id), None
    )
    if customer is None or customer.monthly_revenue_usd <= 0:
        return None

    current_spend = customer.current_spend_usd + local.for_customer(customer.customer_id)
    projected = current_spend + request.estimated_cost_usd
    spend_ratio = projected / customer.monthly_revenue_usd
    if spend_ratio < customer.margin_alert_threshold:
        return None

    target_model = snapshot.margin_targets.get(request.provider or "")
    if not target_model or target_model == request.model:
        return None

    limit = customer.monthly_revenue_usd * customer.margin_alert_threshold
    return {
        "allowed": True,
        "action": ACTION_DOWNGRADE,
        "reason": "customer_margin_guardrail",
        "customer_id": customer.customer_id,
        "customer_name": customer.name,
        "current_spend_usd": round(current_spend, 6),
        "limit_usd": round(limit, 6),
        "projected_spend_usd": round(projected, 6),
        "pct_used": round(spend_ratio, 4),
        "target_model": target_model,
        "message": (
            f"Customer '{customer.name}' projected LLM spend is {spend_ratio:.0%} "
            f"of revenue; downgrade non-critical calls to {target_model}."
        ),
    }
