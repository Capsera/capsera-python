"""False-positive blocks: stopping a healthy agent.

Ranked second in the risk model, behind only breaking the customer's app, and
for the same reason — it kills production work and it destroys trust in the
thing we are selling. A missed block costs a customer some money; a wrong block
costs them an outage they did not cause.

These exercise the shared predicate directly. It is pure and has no I/O, so each
case is a precise statement about when we do and do not stop a call, which is
exactly the kind of thing that should be pinned rather than inferred.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from capsera._budget_rules import (
    ACTION_BLOCK,
    ACTION_THROTTLE,
    BudgetRule,
    BudgetSnapshot,
    PreflightRequest,
    evaluate,
)
from capsera._ledger import SpendLedger

FUTURE = datetime.now(UTC) + timedelta(days=20)


def budget(**overrides) -> BudgetRule:
    base = dict(
        id="b1", name="cap", scope_type="global", scope_id=None,
        action=ACTION_BLOCK, limit_usd=100.0, current_spend_usd=0.0,
        period_end=FUTURE, per_call_ceiling_usd=None,
    )
    base.update(overrides)
    return BudgetRule(**base)


def snapshot(*budgets: BudgetRule) -> BudgetSnapshot:
    return BudgetSnapshot(budgets=budgets, fetched_at=datetime.now(UTC))


def request(**overrides) -> PreflightRequest:
    base = dict(agent_id="agent-1", estimated_cost_usd=0.01)
    base.update(overrides)
    return PreflightRequest(**base)


def blocked(decision: dict) -> bool:
    return decision.get("action") == ACTION_BLOCK and not decision.get("allowed", True)


# --------------------------------------------------------------------------- #
# D1.1 — approaching the cap is not crossing it
# --------------------------------------------------------------------------- #


def test_an_agent_at_99_percent_keeps_running_cheap_calls():
    """The most damaging false positive: an agent that has been careful all
    month, stopped one call early because we rounded against it."""
    snap = snapshot(budget(limit_usd=100.0, current_spend_usd=99.0))
    local = SpendLedger()

    allowed = 0
    for _ in range(90):                 # 90 x $0.01 = $0.90, fits in the $1.00 left
        decision = evaluate(snap, request(estimated_cost_usd=0.01), local=local)
        if blocked(decision):
            break
        allowed += 1
        local.record(cost_usd=0.01, agent_id="agent-1", team_id=None, customer_id=None)

    assert allowed == 90, (
        f"blocked after {allowed} of 90 calls that all fit inside the remaining $1.00"
    )


def test_the_call_that_actually_crosses_the_cap_is_the_one_refused():
    """The boundary itself: not one call early, not one call late."""
    snap = snapshot(budget(limit_usd=100.0, current_spend_usd=99.99))
    local = SpendLedger()

    first = evaluate(snap, request(estimated_cost_usd=0.005), local=local)
    assert not blocked(first), "refused a call that fits in the remaining cent"
    local.record(cost_usd=0.005, agent_id="agent-1", team_id=None, customer_id=None)

    second = evaluate(snap, request(estimated_cost_usd=0.02), local=local)
    assert blocked(second), "allowed a call that crosses the cap"


# --------------------------------------------------------------------------- #
# D1.2 — an unpriced model must not be mistaken for a free one, or an expensive one
# --------------------------------------------------------------------------- #


def test_a_zero_cost_estimate_never_blocks_on_its_own():
    """A model missing from the pricing table estimates at zero. That must not
    trigger a block — and it must not be treated as free either, which is why
    the recorded cost is a separate concern from this decision."""
    snap = snapshot(budget(limit_usd=10.0, current_spend_usd=9.99))
    decision = evaluate(snap, request(estimated_cost_usd=0.0), local=SpendLedger())
    assert not blocked(decision), "a zero estimate blocked a call with headroom left"


# --------------------------------------------------------------------------- #
# D1.3 / D4.4 — time
# --------------------------------------------------------------------------- #


def test_a_period_that_has_already_ended_does_not_block_on_last_periods_spend():
    """Clock skew and a stale snapshot look identical from here. Either way, a
    budget whose period has closed must not hold last month's spend against
    this month's call."""
    snap = snapshot(budget(
        limit_usd=10.0, current_spend_usd=999.0,
        period_end=datetime.now(UTC) - timedelta(minutes=5),
    ))
    decision = evaluate(snap, request(estimated_cost_usd=1.0), local=SpendLedger())
    assert not blocked(decision), "an expired period blocked on the previous period's spend"


def test_a_snapshot_from_the_future_still_enforces():
    """The opposite skew: a host clock behind the backend's. Enforcement must
    not silently switch off because the snapshot looks like it has not happened
    yet."""
    snap = BudgetSnapshot(
        budgets=(budget(limit_usd=10.0, current_spend_usd=50.0),),
        fetched_at=datetime.now(UTC) + timedelta(hours=2),
    )
    decision = evaluate(snap, request(estimated_cost_usd=1.0), local=SpendLedger())
    assert blocked(decision), "a future-dated snapshot disabled enforcement"


# --------------------------------------------------------------------------- #
# D1.8 — scope matching must be exact
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scope_id,agent_id", [
    ("agent-1", "agent-1 "),      # trailing space
    ("agent-1", " agent-1"),      # leading space
    ("agent-1", "Agent-1"),       # different case
    ("agent-1", "agent-10"),      # prefix, not equal
    ("agent-1", "agent"),         # substring
])
def test_a_near_miss_agent_id_does_not_match_an_agent_budget(scope_id, agent_id):
    """A budget scoped to one agent must not catch a differently-named one.

    Every pair here is a plausible typo or a naming-convention drift, and each
    would stop an agent that has no budget at all — the hardest kind of block to
    diagnose, because the operator goes looking at the wrong agent's numbers.
    """
    snap = snapshot(budget(
        scope_type="agent", scope_id=scope_id,
        limit_usd=1.0, current_spend_usd=99.0,
    ))
    decision = evaluate(snap, request(agent_id=agent_id, estimated_cost_usd=1.0),
                        local=SpendLedger())
    assert not blocked(decision), (
        f"budget scoped to {scope_id!r} blocked agent {agent_id!r}"
    )


def test_the_exact_agent_id_does_match():
    """The control for the case above: exactness must not become deafness."""
    snap = snapshot(budget(
        scope_type="agent", scope_id="agent-1",
        limit_usd=1.0, current_spend_usd=99.0,
    ))
    decision = evaluate(snap, request(agent_id="agent-1", estimated_cost_usd=1.0),
                        local=SpendLedger())
    assert blocked(decision), "an exactly-matching agent budget did not engage"


# --------------------------------------------------------------------------- #
# D0.2 — what `throttle` actually does today
# --------------------------------------------------------------------------- #


def test_throttle_allows_the_call_and_only_reports():
    """Pins the current behaviour, which the test plan records as undefined.

    `throttle` returns allowed=True with the action named. Nothing in the SDK
    delays, queues or reshapes the call — so today it is an alert with a
    different label. That may be the right answer, but it should be a decision
    rather than a surprise, and a caller reading the action name could
    reasonably expect back-pressure.
    """
    snap = snapshot(budget(action=ACTION_THROTTLE, limit_usd=1.0, current_spend_usd=99.0))
    decision = evaluate(snap, request(estimated_cost_usd=1.0), local=SpendLedger())

    assert decision["allowed"] is True, "throttle should not refuse the call"
    assert decision["action"] == ACTION_THROTTLE, (
        "throttle must at least be reported, or a caller cannot implement back-pressure itself"
    )


def test_block_wins_when_a_throttle_budget_also_matches():
    """Two budgets, two verdicts. The strongest has to win, or a permissive
    budget could quietly cancel a blocking one."""
    snap = snapshot(
        budget(id="throttler", action=ACTION_THROTTLE, limit_usd=1.0, current_spend_usd=99.0),
        budget(id="blocker", action=ACTION_BLOCK, limit_usd=1.0, current_spend_usd=99.0),
    )
    decision = evaluate(snap, request(estimated_cost_usd=1.0), local=SpendLedger())
    assert blocked(decision), "a throttle budget masked a blocking one"
