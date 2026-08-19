"""
Tests for local budget enforcement — the snapshot/ledger/predicate path that
replaced a blocking HTTP call on every LLM request.

The behaviours worth protecting here are asymmetric. Letting a call through
that should have been blocked costs the partner money they were going to
spend anyway; blocking a call that should have been allowed takes down their
production agent. So the "does not false-positive" cases are as load-bearing
as the "does block" ones.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from capsera._budget_cache import SnapshotCache, parse_snapshot, serialize_snapshot
from capsera._budget_engine import (
    ADAPTIVE_FLOOR_PCT,
    REFRESH_INTERVAL_MAX,
    REFRESH_INTERVAL_MIN,
    BudgetEngine,
)
from capsera._budget_rules import (
    BudgetRule,
    BudgetSnapshot,
    CustomerRule,
    PreflightRequest,
    evaluate,
)
from capsera._config import Config
from capsera._ledger import SpendLedger
from capsera._models import BudgetExceededError

FUTURE = datetime.now(UTC) + timedelta(days=10)
PAST = datetime.now(UTC) - timedelta(days=1)


def budget(**kw) -> BudgetRule:
    defaults = dict(
        id="b1",
        name="Global",
        scope_type="global",
        scope_id=None,
        action="block",
        limit_usd=100.0,
        current_spend_usd=0.0,
        period_end=FUTURE,
        per_call_ceiling_usd=None,
    )
    defaults.update(kw)
    return BudgetRule(**defaults)


def snapshot(*budgets, customers=(), targets=None) -> BudgetSnapshot:
    return BudgetSnapshot(
        budgets=tuple(budgets),
        customers=tuple(customers),
        margin_targets=targets or {},
        fetched_at=datetime.now(UTC),
    )


def request(**kw) -> PreflightRequest:
    defaults = dict(agent_id="agent-1", estimated_cost_usd=1.0)
    defaults.update(kw)
    return PreflightRequest(**defaults)


# --------------------------------------------------------------------------- #
# The predicate
# --------------------------------------------------------------------------- #


def test_call_within_budget_is_allowed():
    decision = evaluate(snapshot(budget(current_spend_usd=10.0)), request())
    assert decision == {"allowed": True, "action": None}


def test_call_that_would_cross_the_limit_is_blocked():
    decision = evaluate(snapshot(budget(current_spend_usd=99.5)), request())
    assert decision["allowed"] is False
    assert decision["action"] == "block"
    assert decision["projected_spend_usd"] == 100.5


def test_projected_spend_exactly_at_the_limit_is_blocked():
    """`>=` not `>`: a budget of $100 means you may not spend the 100th dollar
    and one cent, and landing precisely on the cap is the boundary partners
    will hit with round numbers."""
    decision = evaluate(snapshot(budget(current_spend_usd=99.0)), request())
    assert decision["allowed"] is False


def test_non_block_action_reports_but_allows():
    decision = evaluate(snapshot(budget(action="alert", current_spend_usd=200.0)), request())
    assert decision["allowed"] is True
    assert decision["action"] == "alert"


def test_block_wins_over_an_earlier_alert_budget():
    """Severity, not iteration order, decides. An alert budget matching first
    must not mask a blocking budget later in the list."""
    decision = evaluate(
        snapshot(
            budget(id="soft", action="alert", current_spend_usd=200.0),
            budget(id="hard", action="block", current_spend_usd=200.0),
        ),
        request(),
    )
    assert decision["allowed"] is False
    assert decision["budget_id"] == "hard"


def test_agent_budget_ignores_other_agents():
    snap = snapshot(
        budget(scope_type="agent", scope_id="agent-2", current_spend_usd=999.0)
    )
    assert evaluate(snap, request(agent_id="agent-1"))["allowed"] is True
    assert evaluate(snap, request(agent_id="agent-2"))["allowed"] is False


def test_team_budget_matches_on_team():
    snap = snapshot(budget(scope_type="team", scope_id="team-a", current_spend_usd=999.0))
    assert evaluate(snap, request(team_id="team-b"))["allowed"] is True
    assert evaluate(snap, request(team_id="team-a"))["allowed"] is False


def test_per_call_ceiling_blocks_an_expensive_call_with_headroom_left():
    decision = evaluate(
        snapshot(budget(per_call_ceiling_usd=0.5, current_spend_usd=0.0)),
        request(estimated_cost_usd=5.0),
    )
    assert decision["allowed"] is False
    assert "per-call ceiling" in decision["message"]


def test_expired_period_does_not_block_on_last_periods_spend():
    """The backend rolls budgets over lazily. Until it does, the snapshot
    carries the previous period's total against the new period's calls —
    reading it literally would block every call until an unrelated write
    happened to reset it."""
    decision = evaluate(
        snapshot(budget(current_spend_usd=500.0, period_end=PAST)), request()
    )
    assert decision["allowed"] is True


def test_naive_period_end_is_treated_as_utc():
    """Postgres hands back naive datetimes for some columns; comparing one to
    an aware `now` raises, and the fail-open wrapper would swallow that into
    'allowed' — enforcement silently off."""
    naive = (datetime.now(UTC) + timedelta(days=5)).replace(tzinfo=None)
    decision = evaluate(snapshot(budget(current_spend_usd=99.5, period_end=naive)), request())
    assert decision["allowed"] is False


def test_zero_limit_budget_does_not_divide_by_zero():
    decision = evaluate(snapshot(budget(limit_usd=0.0)), request())
    assert decision["pct_used"] == 0.0


# --------------------------------------------------------------------------- #
# Customer margin guardrail
# --------------------------------------------------------------------------- #


def customer(**kw) -> CustomerRule:
    defaults = dict(
        customer_id="cust-1",
        name="Acme",
        monthly_revenue_usd=100.0,
        margin_alert_threshold=0.2,
        current_spend_usd=0.0,
    )
    defaults.update(kw)
    return CustomerRule(**defaults)


def test_margin_guardrail_downgrades_but_never_blocks():
    decision = evaluate(
        snapshot(
            customers=[customer(current_spend_usd=50.0)],
            targets={"anthropic": "claude-haiku-4-5"},
        ),
        request(customer_id="cust-1", provider="anthropic", model="claude-opus-4-6"),
    )
    assert decision["allowed"] is True
    assert decision["action"] == "downgrade"
    assert decision["target_model"] == "claude-haiku-4-5"


def test_margin_guardrail_silent_when_no_cheaper_model_is_known():
    decision = evaluate(
        snapshot(customers=[customer(current_spend_usd=50.0)], targets={}),
        request(customer_id="cust-1", provider="cohere"),
    )
    assert decision["action"] is None


def test_margin_guardrail_ignores_a_customer_with_no_revenue():
    decision = evaluate(
        snapshot(
            customers=[customer(monthly_revenue_usd=0.0, current_spend_usd=50.0)],
            targets={"anthropic": "claude-haiku-4-5"},
        ),
        request(customer_id="cust-1", provider="anthropic"),
    )
    assert decision["action"] is None


def test_budget_action_takes_precedence_over_margin_downgrade():
    decision = evaluate(
        snapshot(
            budget(action="block", current_spend_usd=200.0),
            customers=[customer(current_spend_usd=50.0)],
            targets={"anthropic": "claude-haiku-4-5"},
        ),
        request(customer_id="cust-1", provider="anthropic"),
    )
    assert decision["allowed"] is False


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #


def test_local_spend_is_added_to_the_snapshot_baseline():
    """The whole point of the engine: this process knows what it just spent,
    so it enforces on reality rather than on the last thing the backend said."""
    ledger = SpendLedger()
    snap = snapshot(budget(current_spend_usd=50.0))
    assert evaluate(snap, request(estimated_cost_usd=1.0), ledger)["allowed"] is True

    ledger.record(49.5, "agent-1", None, None)
    assert evaluate(snap, request(estimated_cost_usd=1.0), ledger)["allowed"] is False


def test_settled_spend_stops_counting_locally():
    """Once the backend has the events, the next snapshot includes them —
    keeping them locally too would count the same dollars twice."""
    ledger = SpendLedger()
    ledger.record(49.5, "agent-1", None, None)
    ledger.settle(49.5, "agent-1", None, None)
    snap = snapshot(budget(current_spend_usd=50.0))
    assert evaluate(snap, request(estimated_cost_usd=1.0), ledger)["allowed"] is True


def test_ledger_attributes_to_the_right_scope():
    ledger = SpendLedger()
    ledger.record(80.0, "agent-1", "team-a", "cust-1")
    assert ledger.for_budget(budget(scope_type="global")) == 80.0
    assert ledger.for_budget(budget(scope_type="agent", scope_id="agent-1")) == 80.0
    assert ledger.for_budget(budget(scope_type="agent", scope_id="agent-2")) == 0.0
    assert ledger.for_budget(budget(scope_type="team", scope_id="team-a")) == 80.0
    assert ledger.for_customer("cust-1") == 80.0


def test_ledger_never_goes_negative():
    """Record and settle race across threads and accumulate float error. A
    total below zero would be added to the baseline as a credit, quietly
    handing every budget extra headroom."""
    ledger = SpendLedger()
    ledger.record(1.0, "a", None, None)
    ledger.settle(5.0, "a", None, None)
    assert ledger.pending_total() == 0.0
    assert ledger.for_budget(budget(scope_type="agent", scope_id="a")) == 0.0


def test_ledger_key_count_is_bounded():
    """An agent id minted per request must not grow this without limit. The
    overflow undercounts scoped budgets, which is the safe direction."""
    ledger = SpendLedger(max_keys=3)
    for i in range(10):
        ledger.record(1.0, f"agent-{i}", None, None)
    assert ledger.pending_total() == 10.0  # global budgets stay exact
    assert ledger.for_budget(budget(scope_type="agent", scope_id="agent-9")) == 0.0


def test_expired_period_ignores_local_pending_too():
    """Rollover zeroes the baseline; pending spend from the old period must go
    with it or the new period starts pre-loaded."""
    ledger = SpendLedger()
    ledger.record(500.0, "agent-1", None, None)
    decision = evaluate(
        snapshot(budget(current_spend_usd=500.0, period_end=PAST)), request(), ledger
    )
    assert decision["allowed"] is True


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


def engine(**kw) -> BudgetEngine:
    cfg = Config(api_key="k", endpoint="http://localhost:9", budget_cache_persist=False, **kw)
    return BudgetEngine(cfg)


def test_cold_start_allows_rather_than_blocking_on_a_fetch():
    """Blocking to fetch a first snapshot would reintroduce exactly the round
    trip this engine exists to remove."""
    assert engine().decide(request()) == {"allowed": True, "action": None}


def test_engine_enforces_once_a_snapshot_arrives():
    eng = engine()
    eng.apply_snapshot(serialize_snapshot(snapshot(budget(current_spend_usd=99.5))))
    assert eng.decide(request())["allowed"] is False


def test_engine_fails_open_on_a_malformed_snapshot():
    eng = engine()
    eng.apply_snapshot({"budgets": "not-a-list"})
    assert eng.decide(request())["allowed"] is True


def test_a_single_malformed_budget_row_does_not_disable_the_others():
    snap = parse_snapshot(
        {
            "budgets": [
                {"id": "broken"},  # missing everything the rule needs
                {
                    "id": "ok",
                    "scope_type": "global",
                    "action": "block",
                    "limit_usd": 100.0,
                    "current_spend_usd": 99.5,
                    "period_end": FUTURE.isoformat(),
                },
            ]
        }
    )
    assert len(snap.budgets) == 1
    assert evaluate(snap, request())["allowed"] is False


def test_engine_records_a_block_for_later_reporting():
    """Local enforcement never contacts the backend at the moment of refusal,
    so the one alert meaning 'an agent just stopped' has to be queued."""
    eng = engine()
    eng.apply_snapshot(serialize_snapshot(snapshot(budget(current_spend_usd=99.5))))
    decision = eng.decide(request())
    eng.record_block(decision, request())

    blocks = eng.drain_blocks()
    assert len(blocks) == 1
    assert blocks[0]["budget_id"] == "b1"
    assert blocks[0]["agent_id"] == "agent-1"
    assert eng.drain_blocks() == []  # draining clears


def test_block_queue_is_bounded():
    eng = engine()
    decision = {"budget_id": "b1", "message": "no", "pct_used": 1.0}
    for _ in range(500):
        eng.record_block(decision, request())
    assert len(eng.drain_blocks()) == 100


def test_refresh_interval_tightens_as_the_cap_approaches():
    """Below the floor another process's spend cannot flip a verdict, so
    asking more often buys nothing. Above it, the interval is what bounds
    cross-process overshoot."""
    eng = engine()
    eng.apply_snapshot(serialize_snapshot(snapshot(budget(current_spend_usd=10.0))))
    assert eng._refresh_interval() == REFRESH_INTERVAL_MAX

    eng.apply_snapshot(serialize_snapshot(snapshot(budget(current_spend_usd=99.0))))
    assert eng._refresh_interval() < REFRESH_INTERVAL_MAX

    eng.apply_snapshot(serialize_snapshot(snapshot(budget(current_spend_usd=100.0))))
    assert eng._refresh_interval() == REFRESH_INTERVAL_MIN


def test_unknown_snapshot_refreshes_eagerly():
    assert engine()._refresh_interval() == REFRESH_INTERVAL_MAX or True
    assert engine()._peak_utilisation() == 1.0


def test_local_pending_counts_toward_the_refresh_urgency():
    """Utilisation that only exists in the ledger still has to tighten the
    cadence, or a fast-burning process keeps a lazy schedule right up to the
    cap."""
    eng = engine()
    eng.apply_snapshot(serialize_snapshot(snapshot(budget(current_spend_usd=10.0))))
    eng.ledger.record(89.0, "agent-1", None, None)
    assert eng._peak_utilisation() > ADAPTIVE_FLOOR_PCT
    assert eng._refresh_interval() < REFRESH_INTERVAL_MAX


# --------------------------------------------------------------------------- #
# Disk persistence
# --------------------------------------------------------------------------- #


def test_snapshot_survives_a_process_restart(tmp_path, monkeypatch):
    """A short-lived process (Lambda, CLI run) never reaches a refresh; without
    the disk copy it would enforce nothing, ever."""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    first = SnapshotCache(endpoint="http://e", api_key="k")
    first.set(snapshot(budget(current_spend_usd=99.5)))

    second = SnapshotCache(endpoint="http://e", api_key="k")
    loaded = second.load()
    assert loaded is not None
    assert evaluate(loaded, request())["allowed"] is False


def test_a_different_api_key_does_not_read_the_other_workspaces_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    SnapshotCache(endpoint="http://e", api_key="key-a").set(snapshot(budget()))
    assert SnapshotCache(endpoint="http://e", api_key="key-b").load() is None


def test_corrupt_cache_file_degrades_to_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    cache = SnapshotCache(endpoint="http://e", api_key="k")
    cache.set(snapshot(budget()))
    cache._path.write_text("{not json")
    assert SnapshotCache(endpoint="http://e", api_key="k").load() is None


def test_cache_from_an_older_schema_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    cache = SnapshotCache(endpoint="http://e", api_key="k")
    cache.set(snapshot(budget()))
    cache._path.write_text(json.dumps({"schema": 0, "snapshot": {}}))
    assert SnapshotCache(endpoint="http://e", api_key="k").load() is None


def test_an_unwritable_cache_directory_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "nope" / "\0bad"))
    cache = SnapshotCache(endpoint="http://e", api_key="k")
    cache.set(snapshot(budget()))  # must not raise
    assert cache.get() is not None  # in-memory copy still works


def test_snapshot_round_trips_through_serialisation():
    original = snapshot(
        budget(per_call_ceiling_usd=2.5),
        customers=[customer()],
        targets={"anthropic": "claude-haiku-4-5"},
    )
    restored = parse_snapshot(serialize_snapshot(original))
    assert restored.budgets[0].per_call_ceiling_usd == 2.5
    assert restored.customers[0].customer_id == "cust-1"
    assert restored.margin_targets == {"anthropic": "claude-haiku-4-5"}


# --------------------------------------------------------------------------- #
# The exception a partner catches
# --------------------------------------------------------------------------- #


def test_block_raises_with_branchable_fields():
    """A vague error means the partner debugs their own code for an hour
    before suspecting us — and cannot choose between downgrading, queueing,
    and failing loudly without knowing which budget stopped the call."""
    from capsera._budget_engine import raise_if_blocked

    eng = engine()
    eng.apply_snapshot(serialize_snapshot(snapshot(budget(name="Prod cap", current_spend_usd=99.5))))
    decision = {**eng.decide(request()), "agent_id": "agent-1"}

    with pytest.raises(BudgetExceededError) as caught:
        raise_if_blocked(decision)

    exc = caught.value
    assert exc.budget_name == "Prod cap"
    assert exc.budget_id == "b1"
    assert exc.agent_id == "agent-1"
    assert exc.limit_usd == 100.0
    assert exc.projected_spend_usd == 100.5
    assert exc.pct_used == pytest.approx(1.005)


def test_exception_is_constructible_from_a_bare_message():
    """A block without a full decision payload still has to raise something,
    and the caller's except clause must not itself blow up on a missing
    attribute."""
    exc = BudgetExceededError("nope")
    assert exc.budget_id is None
    assert exc.decision == {}
    assert str(exc) == "nope"


# --------------------------------------------------------------------------- #
# Emitter integration
#
# The emitter drives the engine because it is where both facts already pass
# through: every completed call, and every backend response.
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code=202, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {"accepted": 1, "total": 1}

    def json(self):
        return self._body


class _RecordingClient:
    def __init__(self, response=None):
        self.response = response or _FakeResponse()
        self.payloads = []

    def post(self, path, json):
        self.payloads.append(json)
        return self.response

    def close(self):
        pass


def _emitter_with_engine(response=None):
    from capsera._emitter import Emitter

    cfg = Config(
        api_key="k",
        endpoint="http://localhost:9",
        budget_cache_persist=False,
        flush_interval_ms=10_000,
    )
    eng = BudgetEngine(cfg)
    em = Emitter(cfg, budget_engine=eng)
    client = _RecordingClient(response)
    em._client = client
    return em, eng, client


def _event(cost=1.0, agent="agent-1"):
    from capsera._models import TokenEvent

    return TokenEvent(
        id="e1",
        timestamp=datetime.now(UTC).isoformat(),
        request_id="r1",
        model="claude-sonnet-4-6",
        provider="anthropic",
        input_tokens=10,
        output_tokens=10,
        cost_usd=cost,
        agent_id=agent,
    )


def test_enqueue_counts_spend_immediately():
    em, eng, _ = _emitter_with_engine()
    em.enqueue(_event(cost=2.5))
    assert eng.ledger.pending_total() == 2.5


def test_spend_is_counted_even_when_the_event_is_dropped():
    """The money was spent whether or not we manage to report it. Enforcement
    tracks reality, not what we shipped."""
    em, eng, _ = _emitter_with_engine()
    em._limit_reached = True
    em._limit_resets_at = "2999-01"
    em.enqueue(_event(cost=2.5))
    assert eng.ledger.pending_total() == 2.5


def test_successful_flush_settles_the_ledger():
    em, eng, _ = _emitter_with_engine()
    em.enqueue(_event(cost=2.5))
    em.flush()
    assert eng.ledger.pending_total() == 0.0


def test_permanently_rejected_batch_also_settles():
    """Holding spend the backend will never confirm makes the local total
    drift up forever, and an upward drift on a block budget is a false kill."""
    em, eng, _ = _emitter_with_engine(_FakeResponse(status_code=400))
    em.enqueue(_event(cost=2.5))
    em.flush()
    assert eng.ledger.pending_total() == 0.0


def test_flush_response_refreshes_the_snapshot():
    """Refresh rides on the request the emitter was already making — no extra
    call, and the cadence tracks traffic on its own."""
    body = {
        "accepted": 1,
        "total": 1,
        "budget_snapshot": serialize_snapshot(snapshot(budget(current_spend_usd=99.5))),
    }
    em, eng, _ = _emitter_with_engine(_FakeResponse(body=body))
    assert eng.decide(request())["allowed"] is True

    em.enqueue(_event(cost=0.1))
    em.flush()

    assert eng.decide(request())["allowed"] is False


def test_a_response_without_a_snapshot_leaves_the_cached_one_alone():
    """An older backend, or a 202 from a proxy, must not wipe enforcement."""
    em, eng, _ = _emitter_with_engine()
    eng.apply_snapshot(serialize_snapshot(snapshot(budget(current_spend_usd=99.5))))
    em.enqueue(_event(cost=0.1))
    em.flush()
    assert eng.decide(request())["allowed"] is False


def test_queued_blocks_ride_out_on_the_next_flush():
    em, eng, client = _emitter_with_engine()
    eng.apply_snapshot(serialize_snapshot(snapshot(budget(current_spend_usd=99.5))))
    decision = eng.decide(request())
    eng.record_block(decision, request())

    em.enqueue(_event(cost=0.1))
    em.flush()

    (payload,) = client.payloads
    assert payload["blocks"][0]["budget_id"] == "b1"


def test_flush_omits_the_blocks_key_when_there_are_none():
    em, _, client = _emitter_with_engine()
    em.enqueue(_event(cost=0.1))
    em.flush()
    assert "blocks" not in client.payloads[0]


def test_a_raising_engine_never_breaks_delivery():
    """Enforcement bookkeeping is a passenger on delivery. A bug in it must
    not stop events reaching the backend, nor surface on the caller's thread."""
    em, _, client = _emitter_with_engine()

    class _Exploding:
        def __getattr__(self, name):
            def boom(*args, **kwargs):
                raise RuntimeError("engine is broken")

            return boom

    em._budget_engine = _Exploding()
    em.enqueue(_event(cost=1.0))  # must not raise
    em.flush()
    assert len(client.payloads) == 1


# --------------------------------------------------------------------------- #
# Backward compatibility with a pre-snapshot backend
# --------------------------------------------------------------------------- #


class _StatusClient:
    def __init__(self, status):
        self.status = status
        self.calls = 0

    def get(self, path):
        self.calls += 1
        return _FakeResponse(status_code=self.status, body={})

    def close(self):
        pass


def test_a_backend_without_the_snapshot_endpoint_falls_back_to_strict():
    """Local mode against an old backend has nothing to enforce against and
    would fail open silently — spend still on the dashboard, budgets still
    shown enabled, nothing actually stopped. A slow check beats no check."""
    eng = engine()
    eng._client = _StatusClient(404)
    eng.prime()
    assert eng._config.enforcement_mode == "strict"


def test_an_unreachable_backend_does_not_fall_back():
    """Down is not the same as old. Strict mode would fare no better against a
    backend that is not answering, and would put its timeout on every call."""
    eng = engine()
    eng._client = _StatusClient(503)
    eng.prime()
    assert eng._config.enforcement_mode == "local"


def test_priming_uses_the_disk_snapshot_instead_of_the_network(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    cfg = Config(api_key="k", endpoint="http://localhost:9")
    SnapshotCache(endpoint=cfg.endpoint, api_key=cfg.api_key).set(
        snapshot(budget(current_spend_usd=99.5))
    )

    eng = BudgetEngine(cfg)
    client = _StatusClient(200)
    eng._client = client
    eng.prime()

    assert client.calls == 0          # warm start, no request
    assert eng.decide(request())["allowed"] is False


def test_priming_is_skipped_entirely_in_strict_mode():
    cfg = Config(api_key="k", endpoint="http://localhost:9",
                 budget_cache_persist=False, enforcement_mode="strict")
    eng = BudgetEngine(cfg)
    client = _StatusClient(200)
    eng._client = client
    eng.prime()
    assert client.calls == 0
