"""SDK → backend ingest contract.

Every event the SDK can construct must validate against the backend's
``EventIn`` schema. This matters more than it looks: the ingest route
parses the WHOLE batch as one Pydantic model, so a single out-of-contract
event 422s the entire flush — the emitter retries the identical payload
three times and then drops all of it. One oversized field = silent loss
of every event in that 500ms window.

The SDK therefore must NORMALIZE at event-construction time: clamp string
lengths to the contract's maxima, cap ratios into their valid range, and
floor negative token counts — provider responses and user tag() input are
not trustworthy.
"""
from __future__ import annotations

import pytest

from capsera._models import PromptMetadata, TokenEvent

from .conftest import backend_batch_schema, backend_event_schema


def _mk_event(**overrides) -> TokenEvent:
    base = dict(
        id="00000000-0000-0000-0000-000000000001",
        timestamp="2026-08-03T12:00:00+00:00",
        request_id="req-1",
        model="claude-sonnet-4-6",
        provider="anthropic",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        agent_id="test-agent",
        latency_ms=12,
    )
    base.update(overrides)
    return TokenEvent(**base)


def _validate(event: TokenEvent):
    EventIn = backend_event_schema()
    return EventIn.model_validate(event.to_dict())


# ————— Happy paths through every construction shape —————

def test_minimal_event_validates():
    _validate(_mk_event())


def test_full_attribution_event_validates():
    _validate(
        _mk_event(
            team_id="growth",
            task_type="summarize",
            session_id="sess-1",
            user_id="user-1",
            customer_id="cust-1",
            cost_center="cc-9",
            caller_file="app/main.py:42",
            env="production",
        )
    )


def test_routed_event_validates():
    _validate(_mk_event(was_routed=True, original_model="gpt-4o"))


def test_error_event_validates():
    _validate(
        _mk_event(
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            is_error=True,
            error_type="RateLimitError",
        )
    )


def test_prompt_analysis_event_validates():
    meta = PromptMetadata(
        system_prompt_hash="a" * 16,
        system_prompt_tokens=1200,
        message_count=14,
        conversation_tokens=9000,
        has_few_shot=True,
        few_shot_pairs=3,
        tool_result_tokens=400,
        conversation_turns=7,
        context_window_utilization=0.42,
    )
    ev = _mk_event(
        system_prompt_hash=meta.system_prompt_hash,
        system_prompt_tokens=meta.system_prompt_tokens,
        message_count=meta.message_count,
        conversation_tokens=meta.conversation_tokens,
        has_few_shot=meta.has_few_shot,
        few_shot_pairs=meta.few_shot_pairs,
        tool_result_tokens=meta.tool_result_tokens,
        conversation_turns=meta.conversation_turns,
        context_window_utilization=meta.context_window_utilization,
    )
    _validate(ev)


def test_every_catalog_model_id_fits_the_contract():
    # The contract caps model at 64 chars. If a catalogued model ID cannot
    # fit, every event for that model would poison its batch.
    from capsera._catalog import MODELS

    EventIn = backend_event_schema()
    too_long = [m for m in MODELS if len(m) > 64]
    assert too_long == [], f"catalog model IDs exceed the ingest contract: {too_long}"
    for model in MODELS:
        EventIn.model_validate(_mk_event(model=model).to_dict())


# ————— Hostile inputs must be normalized, not shipped —————
# Provider responses and tag()/record() arguments are untrusted. The SDK
# must clamp them into contract range at construction time.

def test_oversized_model_id_is_clamped():
    ev = _mk_event(model="ft:" + "x" * 200)
    assert len(ev.model) <= 64
    _validate(ev)


def test_oversized_agent_id_is_clamped():
    ev = _mk_event(agent_id="a" * 300)
    assert len(ev.agent_id) <= 128
    _validate(ev)


@pytest.mark.parametrize(
    "field,limit",
    [
        ("team_id", 128),
        ("task_type", 128),
        ("customer_id", 128),
        ("cost_center", 128),
        ("env", 128),
        ("provider", 64),
        ("original_model", 64),
        ("error_type", 128),
        ("system_prompt_hash", 64),
        ("caller_file", 512),
    ],
)
def test_oversized_string_fields_are_clamped(field, limit):
    ev = _mk_event(**{field: "x" * (limit + 200)})
    value = getattr(ev, field)
    assert value is not None and len(value) <= limit, (
        f"{field} shipped at {len(value or '')} chars; contract max is {limit} — "
        "one such event 422s the whole batch"
    )
    _validate(ev)


def test_context_window_utilization_above_one_is_capped():
    # An over-window prompt genuinely produces estimated_tokens > window;
    # the contract caps the ratio at 1.0.
    ev = _mk_event(context_window_utilization=1.73)
    assert 0.0 <= ev.context_window_utilization <= 1.0
    _validate(ev)


def test_negative_token_counts_are_floored():
    # Seen in the wild from partial/streaming usage objects.
    ev = _mk_event(input_tokens=-3, output_tokens=-1, cache_read_tokens=-2)
    assert ev.input_tokens >= 0 and ev.output_tokens >= 0
    assert ev.cache_read_tokens >= 0
    _validate(ev)


def test_negative_latency_is_floored():
    ev = _mk_event(latency_ms=-5)
    assert ev.latency_ms >= 0
    _validate(ev)


def test_negative_cost_is_floored():
    ev = _mk_event(cost_usd=-0.004)
    assert ev.cost_usd >= 0
    _validate(ev)


def test_unicode_attribution_survives():
    ev = _mk_event(agent_id="агент-поддержки", team_id="团队-α", task_type="résumé")
    _validate(ev)


def test_batch_of_hostile_events_validates_as_a_batch():
    # The batch-level parse is what actually gates ingest.
    EventBatch = backend_batch_schema()
    events = [
        _mk_event(request_id=f"r{i}", agent_id="a" * 300, model="m" * 100,
                  context_window_utilization=2.5, input_tokens=-1)
        for i in range(5)
    ]
    EventBatch.model_validate({"events": [e.to_dict() for e in events]})


def test_sdk_contract_tables_match_backend_schema():
    # The clamp tables in _models.py are a COPY of the backend's limits; this
    # keeps them in lockstep so a backend schema change can't silently strand
    # the SDK on stale limits.
    from annotated_types import Ge, Le, MaxLen

    from capsera._models import _CONTRACT_MAX_LEN, _CONTRACT_NON_NEGATIVE

    EventIn = backend_event_schema()

    for field_name, limit in _CONTRACT_MAX_LEN.items():
        meta = EventIn.model_fields[field_name].metadata
        backend_max = next((m.max_length for m in meta if isinstance(m, MaxLen)), None)
        assert backend_max == limit, (
            f"{field_name}: SDK clamps at {limit}, backend max_length is {backend_max}"
        )

    for field_name in _CONTRACT_NON_NEGATIVE:
        meta = EventIn.model_fields[field_name].metadata
        assert any(isinstance(m, Ge) and m.ge == 0 for m in meta), (
            f"{field_name}: SDK floors at 0 but backend declares no ge=0 bound"
        )

    util_meta = EventIn.model_fields["context_window_utilization"].metadata
    assert any(isinstance(m, Le) and m.le == 1.0 for m in util_meta)
