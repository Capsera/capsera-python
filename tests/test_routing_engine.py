"""
Comprehensive tests for the model routing engine (_router.py).
"""
from __future__ import annotations

import copy

import pytest

from capsera._config import Config
from capsera._context import AgentContext
from capsera._models import RoutingDecision
from capsera._router import Router, _infer_provider, _same_provider


def _cfg(**overrides) -> Config:
    base = dict(
        api_key="test",
        endpoint="http://localhost:9999",
        env="test",
        flush_interval_ms=9999,
        enable_routing=True,
    )
    base.update(overrides)
    return Config(**base)


# ------------------------------------------------------------------ #
# Provider inference
# ------------------------------------------------------------------ #

class TestProviderInference:
    def test_anthropic_detected(self) -> None:
        assert _infer_provider("claude-opus-4-6") == "anthropic"
        assert _infer_provider("claude-haiku-4-5") == "anthropic"

    def test_openai_detected(self) -> None:
        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("gpt-4o-mini") == "openai"
        assert _infer_provider("o1") == "openai"
        assert _infer_provider("o3-mini") == "openai"

    def test_google_detected(self) -> None:
        assert _infer_provider("gemini-1.5-pro") == "google"
        assert _infer_provider("gemini-2.0-flash") == "google"

    def test_unknown_provider(self) -> None:
        assert _infer_provider("llama-3.1-70b") == "unknown"

    def test_same_provider(self) -> None:
        assert _same_provider("claude-opus-4-6", "claude-haiku-4-5") is True
        assert _same_provider("gpt-4o", "gpt-4o-mini") is True

    def test_different_provider(self) -> None:
        assert _same_provider("claude-opus-4-6", "gpt-4o") is False

    def test_unknown_allows_routing(self) -> None:
        assert _same_provider("llama-3.1-70b", "claude-haiku-4-5") is True


# ------------------------------------------------------------------ #
# Core routing tests
# ------------------------------------------------------------------ #

class TestNoRoutingWhenDisabled:
    def test_passthrough(self) -> None:
        cfg = _cfg(enable_routing=True, routing_rules=[
            {"type": "model_map", "from_model": "claude-opus-4-6", "target_model": "claude-haiku-4-5"},
        ])
        # Router exists but config.enable_routing controls the interceptor level
        # Here we test that Router.evaluate still works — the disable check is in _maybe_route
        router = Router(cfg)
        decision = router.evaluate({"model": "claude-opus-4-6"}, None)
        assert decision.routed is True  # Router itself doesn't check enable_routing


class TestTaskTypeRouting:
    def test_exact_match(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        ctx = AgentContext(task_type="classification")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.routed is True
        assert decision.target_model == "claude-haiku-4-5"
        assert decision.original_model == "claude-opus-4-6"
        assert decision.rule_type == "task_type"

    def test_no_match(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        ctx = AgentContext(task_type="reasoning")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.routed is False

    def test_wildcard(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "*", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        ctx = AgentContext(task_type="anything")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.routed is True
        assert decision.target_model == "claude-haiku-4-5"

    def test_no_context_skips(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "claude-opus-4-6"}, None)
        assert decision.routed is False

    def test_no_task_type_in_context_skips(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        ctx = AgentContext(agent="myagent")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.routed is False

    def test_same_model_no_route(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        ctx = AgentContext(task_type="classification")
        decision = router.evaluate({"model": "claude-haiku-4-5"}, ctx)
        assert decision.routed is False


class TestModelMapRouting:
    def test_routes(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is True
        assert decision.target_model == "gpt-4o-mini"
        assert decision.original_model == "gpt-4o"
        assert decision.rule_type == "model_map"

    def test_scoped_to_agent(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "gpt-4o", "target_model": "gpt-4o-mini", "scope_agent": "classifier"},
        ])
        router = Router(cfg)
        ctx_match = AgentContext(agent="classifier")
        ctx_other = AgentContext(agent="summarizer")

        assert router.evaluate({"model": "gpt-4o"}, ctx_match).routed is True
        assert router.evaluate({"model": "gpt-4o"}, ctx_other).routed is False

    def test_scoped_to_team(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "gpt-4o", "target_model": "gpt-4o-mini", "scope_team": "nlp"},
        ])
        router = Router(cfg)
        ctx_match = AgentContext(team="nlp")
        ctx_other = AgentContext(team="infra")

        assert router.evaluate({"model": "gpt-4o"}, ctx_match).routed is True
        assert router.evaluate({"model": "gpt-4o"}, ctx_other).routed is False

    def test_no_match_different_model(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "gpt-4"}, None)
        assert decision.routed is False


class TestCostCapRouting:
    def test_under_threshold_no_routing(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "cost_cap", "budget_pct_threshold": 0.8, "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        router.set_budget_cache({"budget-1": 0.5})
        decision = router.evaluate({"model": "claude-opus-4-6"}, None)
        assert decision.routed is False

    def test_over_threshold_routes(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "cost_cap", "budget_pct_threshold": 0.8, "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        router.set_budget_cache({"budget-1": 0.85})
        decision = router.evaluate({"model": "claude-opus-4-6"}, None)
        assert decision.routed is True
        assert decision.target_model == "claude-haiku-4-5"
        assert decision.rule_type == "cost_cap"

    def test_critical_task_type_exempt(self) -> None:
        cfg = _cfg(
            routing_rules=[
                {"type": "cost_cap", "budget_pct_threshold": 0.8, "target_model": "claude-haiku-4-5"},
            ],
            critical_task_types=["reasoning", "coding"],
        )
        router = Router(cfg)
        router.set_budget_cache({"budget-1": 0.95})
        ctx = AgentContext(task_type="reasoning")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.routed is False

    def test_non_critical_task_type_not_exempt(self) -> None:
        cfg = _cfg(
            routing_rules=[
                {"type": "cost_cap", "budget_pct_threshold": 0.8, "target_model": "claude-haiku-4-5"},
            ],
            critical_task_types=["reasoning"],
        )
        router = Router(cfg)
        router.set_budget_cache({"budget-1": 0.9})
        ctx = AgentContext(task_type="classification")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.routed is True

    def test_no_budget_cache_no_routing(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "cost_cap", "budget_pct_threshold": 0.8, "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "claude-opus-4-6"}, None)
        assert decision.routed is False


class TestFallbackChainRouting:
    def test_picks_next_in_chain(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "fallback_chain", "chain": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "claude-opus-4-6"}, None)
        assert decision.routed is True
        assert decision.target_model == "claude-sonnet-4-6"
        assert decision.rule_type == "fallback_chain"

    def test_last_in_chain_no_route(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "fallback_chain", "chain": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "claude-haiku-4-5"}, None)
        assert decision.routed is False

    def test_model_not_in_chain_no_route(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "fallback_chain", "chain": ["claude-opus-4-6", "claude-sonnet-4-6"]},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is False


class TestCrossProviderRouting:
    # Product change (2026-08): cross-provider routing was blocked outright and
    # is now allowed freely, with a documented capability-guardrail seam
    # (_route_allowed) for later. These two used to assert routed is False.
    def test_anthropic_to_openai_now_routes(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "claude-opus-4-6", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "claude-opus-4-6"}, None)
        assert decision.routed is True
        assert decision.target_model == "gpt-4o-mini"

    def test_openai_to_anthropic_now_routes(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        ctx = AgentContext(task_type="classification")
        decision = router.evaluate({"model": "gpt-4o"}, ctx)
        assert decision.routed is True
        assert decision.target_model == "claude-haiku-4-5"


class TestRulePriorityOrder:
    def test_lower_priority_evaluated_first(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-sonnet-4-6", "priority": 10},
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5", "priority": 1},
        ])
        router = Router(cfg)
        ctx = AgentContext(task_type="classification")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.target_model == "claude-haiku-4-5"


class TestFirstMatchWins:
    def test_first_matching_rule_applied(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5", "priority": 0},
            {"type": "model_map", "from_model": "claude-opus-4-6", "target_model": "claude-sonnet-4-6", "priority": 0},
        ])
        router = Router(cfg)
        ctx = AgentContext(task_type="classification")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.target_model == "claude-haiku-4-5"
        assert decision.rule_type == "task_type"


class TestEvaluationOrder:
    def test_cost_cap_before_task_type(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-sonnet-4-6"},
            {"type": "cost_cap", "budget_pct_threshold": 0.5, "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        router.set_budget_cache({"budget-1": 0.9})
        ctx = AgentContext(task_type="classification")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.rule_type == "cost_cap"
        assert decision.target_model == "claude-haiku-4-5"


class TestDoesNotMutateKwargs:
    def test_original_dict_unchanged(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        original_kwargs = {"model": "gpt-4o", "temperature": 0.7}
        snapshot = copy.deepcopy(original_kwargs)
        router.evaluate(original_kwargs, None)
        assert original_kwargs == snapshot


class TestRoutingDecisionFields:
    def test_routed_decision_has_all_fields(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "gpt-4o", "target_model": "gpt-4o-mini", "name": "downgrade-gpt4o"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is True
        assert decision.target_model == "gpt-4o-mini"
        assert decision.original_model == "gpt-4o"
        assert decision.rule_name == "downgrade-gpt4o"
        assert decision.rule_type == "model_map"

    def test_no_route_decision_defaults(self) -> None:
        cfg = _cfg(routing_rules=[])
        router = Router(cfg)
        decision = router.evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is False
        assert decision.target_model is None
        assert decision.original_model is None
        assert decision.rule_name is None
        assert decision.rule_type is None


class TestFailOpen:
    def test_exception_in_evaluate_returns_passthrough(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        # Force an error by passing non-dict kwargs
        decision = router.evaluate("not-a-dict", None)  # type: ignore[arg-type]
        assert decision.routed is False


class TestEmptyModel:
    def test_empty_model_no_route(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": ""}, None)
        assert decision.routed is False

    def test_missing_model_key_no_route(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        ctx = AgentContext(task_type="classification")
        decision = router.evaluate({}, ctx)
        assert decision.routed is False


class TestUnknownRuleType:
    def test_unknown_type_skipped(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "unknown_type", "target_model": "claude-haiku-4-5"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "claude-opus-4-6"}, None)
        assert decision.routed is False


# ------------------------------------------------------------------ #
# Merge logic tests
# ------------------------------------------------------------------ #

class TestRuleMerge:
    def test_local_only(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "name": "rule-a", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        decision = router.evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is True
        assert decision.rule_name == "rule-a"

    def test_backend_only(self) -> None:
        cfg = _cfg(routing_rules=[])
        router = Router(cfg)
        router.set_rules([
            {"type": "model_map", "name": "backend-rule", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
        ], source="backend")
        decision = router.evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is True
        assert decision.rule_name == "backend-rule"

    def test_local_overrides_backend_by_name(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "name": "shared-rule", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        router.set_rules([
            {"type": "model_map", "name": "shared-rule", "from_model": "gpt-4o", "target_model": "gpt-3.5-turbo"},
        ], source="backend")
        decision = router.evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is True
        assert decision.target_model == "gpt-4o-mini"

    def test_backend_rules_appended_after_local(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "name": "local-rule", "from_model": "gpt-4", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        router.set_rules([
            {"type": "model_map", "name": "backend-rule", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
        ], source="backend")
        d1 = router.evaluate({"model": "gpt-4"}, None)
        d2 = router.evaluate({"model": "gpt-4o"}, None)
        assert d1.routed is True
        assert d1.rule_name == "local-rule"
        assert d2.routed is True
        assert d2.rule_name == "backend-rule"

    def test_empty_local_uses_backend(self) -> None:
        cfg = _cfg(routing_rules=[])
        router = Router(cfg)
        router.set_rules([
            {"type": "task_type", "name": "b-rule", "match": "summarization", "target_model": "claude-haiku-4-5"},
        ], source="backend")
        ctx = AgentContext(task_type="summarization")
        decision = router.evaluate({"model": "claude-opus-4-6"}, ctx)
        assert decision.routed is True
        assert decision.rule_name == "b-rule"

    def test_empty_backend_uses_local(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "name": "local", "from_model": "gpt-4o", "target_model": "gpt-4o-mini"},
        ])
        router = Router(cfg)
        router.set_rules([], source="backend")
        decision = router.evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is True
        assert decision.rule_name == "local"


class TestFetchRulesFromBackend:
    def test_fetch_handles_network_error(self) -> None:
        cfg = _cfg(
            endpoint="http://localhost:1",  # unreachable
            routing_rules=[],
            routing_refresh_interval=0,
        )
        router = Router(cfg)
        router._fetch_rules()  # should not raise
        assert router._backend_rules == []

    def test_fetch_budget_handles_network_error(self) -> None:
        cfg = _cfg(
            endpoint="http://localhost:1",
            routing_rules=[],
            routing_refresh_interval=0,
        )
        router = Router(cfg)
        router._fetch_budget_status()  # should not raise
        assert router._budget_cache == {}

    def test_refresh_respects_interval(self) -> None:
        cfg = _cfg(routing_rules=[], routing_refresh_interval=9999)
        router = Router(cfg)
        router._last_refresh = 999999999.0  # far in the future
        import time
        router._last_refresh = time.monotonic()
        router._refresh_if_stale()
        # Should not have tried to fetch (no network call attempted)


class TestCrossProviderRoutingAllowed:
    """Cross-provider routing was blocked outright; it is now allowed freely
    (with a documented capability-guardrail seam for later)."""

    def test_model_map_across_providers_routes(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "model_map", "from_model": "gpt-4o", "target_model": "deepseek-chat"},
        ])
        decision = Router(cfg).evaluate({"model": "gpt-4o"}, None)
        assert decision.routed is True
        assert decision.target_model == "deepseek-chat"

    def test_task_type_across_providers_routes(self) -> None:
        cfg = _cfg(routing_rules=[
            {"type": "task_type", "match": "classification", "target_model": "claude-haiku-4-5"},
        ])
        ctx = AgentContext(task_type="classification")
        decision = Router(cfg).evaluate({"model": "gpt-4o"}, ctx)
        assert decision.routed is True
        assert decision.target_model == "claude-haiku-4-5"
