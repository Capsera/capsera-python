"""
Intelligent model routing engine.

Evaluates routing rules against LLM call parameters and context to
automatically reroute requests to cheaper models.  Rules are evaluated
in priority order: cost-cap -> task-type -> model-map -> fallback-chain.
First match wins.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import _catalog
from ._config import Config
from ._context import AgentContext
from ._models import RoutingDecision
from ._pricing import get_pricing
from ._version import __version__

logger = logging.getLogger("capsera")


def _infer_provider(model: str) -> str:
    # Single source of truth (the shared catalog), so a model added there is
    # known to routing without editing this file.
    return _catalog.provider_for(model)


def _same_provider(model_a: str, model_b: str) -> bool:
    pa = _infer_provider(model_a)
    pb = _infer_provider(model_b)
    if pa == "unknown" or pb == "unknown":
        return True
    return pa == pb


def _route_allowed(current_model: str, target_model: str) -> bool:
    """Whether a routing rule may send ``current_model`` to ``target_model``.

    Cross-provider routing was blocked outright (``_same_provider``). Per the
    product decision it is now ALLOWED freely — a rule targeting any provider
    routes — so this returns True. It is the deliberate seam for the future
    capability guardrail: when that lands, this becomes e.g.

        req = _catalog.model_meta(current_model)
        tgt = _catalog.model_meta(target_model)
        return not (req and tgt) or req.capabilities <= tgt.capabilities

    so a rule can't silently downgrade a tool-using / vision call to a model
    that lacks the capability. Nothing enforces that yet — the catalog already
    carries the capability flags for when it does.
    """
    return True


def _model_cost_per_1k(model: str) -> float:
    pricing = get_pricing(model)
    return float(pricing.input_per_1k + pricing.output_per_1k)


# --------------------------------------------------------------------------- #
# Backend payload conversion
#
# Shared by the sync in-process fetch (Router._fetch_*) and the LiteLLM proxy's
# async refresh, so the wire shape of /v1/routing/rules and
# /v1/routing/budget-status is parsed in exactly one place.
# --------------------------------------------------------------------------- #

def convert_backend_rules(raw_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten backend RoutingRule rows into the engine's rule dicts.

    Drops disabled rules and merges each row's ``config_json`` up to the top
    level beside ``type``/``name``/``priority``.
    """
    converted: list[dict[str, Any]] = []
    for r in raw_rules or []:
        if not r.get("enabled", True):
            continue
        cfg = r.get("config_json", {}) or {}
        converted.append({
            "type": r.get("rule_type", ""),
            "name": r.get("name", ""),
            "priority": r.get("priority", 0),
            **cfg,
        })
    return converted


def convert_budget_status(budgets: list[dict[str, Any]]) -> dict[str, float]:
    """Map /v1/routing/budget-status rows to ``{budget_id: pct_used}``."""
    cache: dict[str, float] = {}
    for b in budgets or []:
        cache[b.get("budget_id", "")] = b.get("pct_used", 0.0)
    return cache


# ------------------------------------------------------------------ #
# Parsed rule types
# ------------------------------------------------------------------ #

@dataclass
class _TaskTypeRule:
    name: str
    match: str
    target_model: str
    priority: int = 0


@dataclass
class _ModelMapRule:
    name: str
    from_model: str
    target_model: str
    scope_agent: str | None = None
    scope_team: str | None = None
    priority: int = 0


@dataclass
class _CostCapRule:
    name: str
    budget_pct_threshold: float
    target_model: str
    priority: int = 0


@dataclass
class _FallbackChainRule:
    name: str
    chain: list[str] = field(default_factory=list)
    priority: int = 0


class Router:
    """
    Evaluates routing rules and returns a RoutingDecision.

    Thread-safe: evaluate() only reads from immutable rule lists and the
    budget cache (which is replaced atomically, not mutated in place).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._local_rules: list[dict[str, Any]] = list(config.routing_rules)
        self._backend_rules: list[dict[str, Any]] = []

        self._cost_cap_rules: list[_CostCapRule] = []
        self._task_type_rules: list[_TaskTypeRule] = []
        self._model_map_rules: list[_ModelMapRule] = []
        self._fallback_chain_rules: list[_FallbackChainRule] = []

        self._budget_cache: dict[str, float] = {}
        self._last_refresh: float = 0.0

        self._client = httpx.Client(
            base_url=config.endpoint,
            headers={
                "X-API-Key": config.api_key,
                "User-Agent": f"capsera-sdk/{__version__}",
            },
            timeout=5.0,
        )

        if config.routing_rules:
            self._rebuild_effective_rules()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        kwargs: dict[str, Any],
        context: AgentContext | None,
        *,
        refresh: bool = True,
    ) -> RoutingDecision:
        """Evaluate routing rules against a call.

        ``refresh=True`` (default) pulls fresh rules/budget-status from the
        backend via the Router's own **sync** client when stale — correct for
        the in-process SDK path. Callers that drive refresh themselves (the
        LiteLLM proxy refreshes asynchronously so it never blocks the event
        loop) pass ``refresh=False`` and this becomes pure, non-blocking rule
        evaluation against whatever rules/budget are currently loaded.
        """
        try:
            if refresh:
                self._refresh_if_stale()
            return self._evaluate_rules(kwargs, context)
        except Exception:
            logger.debug("capsera: routing evaluation failed — passing through", exc_info=True)
            return RoutingDecision()

    def set_budget_cache(self, cache: dict[str, float]) -> None:
        self._budget_cache = cache

    def set_rules(self, rules: list[dict[str, Any]], source: str = "backend") -> None:
        if source == "backend":
            self._backend_rules = rules
            self._rebuild_effective_rules()
        else:
            self._local_rules = rules
            self._rebuild_effective_rules()

    # ------------------------------------------------------------------ #
    # Backend sync
    # ------------------------------------------------------------------ #

    def _refresh_if_stale(self) -> None:
        now = time.monotonic()
        if now - self._last_refresh < self._config.routing_refresh_interval:
            return
        self._last_refresh = now
        self._fetch_rules()
        self._fetch_budget_status()

    def _fetch_rules(self) -> None:
        try:
            resp = self._client.get("/v1/routing/rules")
            if resp.status_code == 200:
                self._backend_rules = convert_backend_rules(resp.json())
                self._rebuild_effective_rules()
                if self._config.debug:
                    logger.debug(
                        "capsera: fetched %d routing rules from backend",
                        len(self._backend_rules),
                    )
        except Exception:
            logger.debug("capsera: failed to fetch routing rules from backend", exc_info=True)

    def _fetch_budget_status(self) -> None:
        try:
            resp = self._client.get("/v1/routing/budget-status")
            if resp.status_code == 200:
                self._budget_cache = convert_budget_status(resp.json())
                if self._config.debug:
                    logger.debug("capsera: refreshed budget cache (%d budgets)", len(self._budget_cache))
        except Exception:
            logger.debug("capsera: failed to fetch budget status from backend", exc_info=True)

    def _rebuild_effective_rules(self) -> None:
        local_names = {r.get("name") for r in self._local_rules if r.get("name")}
        merged = list(self._local_rules)
        for rule in self._backend_rules:
            if rule.get("name") and rule["name"] in local_names:
                continue
            merged.append(rule)
        self._parse_rules(merged, source="merged")

    # ------------------------------------------------------------------ #
    # Rule evaluation
    # ------------------------------------------------------------------ #

    def _evaluate_rules(
        self,
        kwargs: dict[str, Any],
        context: AgentContext | None,
    ) -> RoutingDecision:
        current_model = kwargs.get("model", "")
        if not current_model:
            return RoutingDecision()

        task_type = context.task_type if context else None
        agent_id = context.agent if context else None
        team_id = context.team if context else None
        critical_types = set(self._config.critical_task_types)

        decision = self._check_cost_cap(current_model, task_type, critical_types)
        if decision.routed:
            logger.debug("capsera: routing %s -> %s (cost-cap rule '%s')", current_model, decision.target_model, decision.rule_name)
            return decision

        decision = self._check_task_type(current_model, task_type)
        if decision.routed:
            logger.debug("capsera: routing %s -> %s (task-type rule '%s')", current_model, decision.target_model, decision.rule_name)
            return decision

        decision = self._check_model_map(current_model, agent_id, team_id)
        if decision.routed:
            logger.debug("capsera: routing %s -> %s (model-map rule '%s')", current_model, decision.target_model, decision.rule_name)
            return decision

        decision = self._check_fallback_chain(current_model)
        if decision.routed:
            logger.debug("capsera: routing %s -> %s (fallback-chain rule '%s')", current_model, decision.target_model, decision.rule_name)
            return decision

        logger.debug("capsera: no routing rule matched for model=%s task_type=%s", current_model, task_type)
        return RoutingDecision()

    def _check_cost_cap(
        self,
        current_model: str,
        task_type: str | None,
        critical_types: set[str],
    ) -> RoutingDecision:
        if task_type and task_type in critical_types:
            return RoutingDecision()

        if not self._budget_cache:
            return RoutingDecision()

        for rule in self._cost_cap_rules:
            max_pct = max(self._budget_cache.values()) if self._budget_cache else 0.0
            if max_pct >= rule.budget_pct_threshold:
                if not _route_allowed(current_model, rule.target_model):
                    continue
                if rule.target_model == current_model:
                    continue
                return RoutingDecision(
                    routed=True,
                    target_model=rule.target_model,
                    original_model=current_model,
                    rule_name=rule.name,
                    rule_type="cost_cap",
                )
        return RoutingDecision()

    def _check_task_type(
        self,
        current_model: str,
        task_type: str | None,
    ) -> RoutingDecision:
        if not task_type:
            return RoutingDecision()

        for rule in self._task_type_rules:
            if rule.match == task_type or rule.match == "*":
                if not _route_allowed(current_model, rule.target_model):
                    continue
                if rule.target_model == current_model:
                    continue
                return RoutingDecision(
                    routed=True,
                    target_model=rule.target_model,
                    original_model=current_model,
                    rule_name=rule.name,
                    rule_type="task_type",
                )
        return RoutingDecision()

    def _check_model_map(
        self,
        current_model: str,
        agent_id: str | None,
        team_id: str | None,
    ) -> RoutingDecision:
        for rule in self._model_map_rules:
            if rule.from_model != current_model:
                continue
            if rule.scope_agent and rule.scope_agent != agent_id:
                continue
            if rule.scope_team and rule.scope_team != team_id:
                continue
            if not _route_allowed(current_model, rule.target_model):
                continue
            return RoutingDecision(
                routed=True,
                target_model=rule.target_model,
                original_model=current_model,
                rule_name=rule.name,
                rule_type="model_map",
            )
        return RoutingDecision()

    def _check_fallback_chain(self, current_model: str) -> RoutingDecision:
        for rule in self._fallback_chain_rules:
            if current_model not in rule.chain:
                continue
            current_idx = rule.chain.index(current_model)
            candidates = rule.chain[current_idx + 1:]
            for candidate in candidates:
                if _route_allowed(current_model, candidate):
                    return RoutingDecision(
                        routed=True,
                        target_model=candidate,
                        original_model=current_model,
                        rule_name=rule.name,
                        rule_type="fallback_chain",
                    )
        return RoutingDecision()

    # ------------------------------------------------------------------ #
    # Rule parsing
    # ------------------------------------------------------------------ #

    def _parse_rules(self, rules: list[dict[str, Any]], source: str = "local") -> None:
        cost_cap: list[_CostCapRule] = []
        task_type: list[_TaskTypeRule] = []
        model_map: list[_ModelMapRule] = []
        fallback_chain: list[_FallbackChainRule] = []

        for i, raw in enumerate(rules):
            rtype = raw.get("type", "")
            name = raw.get("name", f"{source}-rule-{i}")
            priority = raw.get("priority", i)

            if rtype == "task_type":
                task_type.append(_TaskTypeRule(
                    name=name,
                    match=raw.get("match", ""),
                    target_model=raw.get("target_model", ""),
                    priority=priority,
                ))
            elif rtype == "model_map":
                model_map.append(_ModelMapRule(
                    name=name,
                    from_model=raw.get("from_model", ""),
                    target_model=raw.get("target_model", ""),
                    scope_agent=raw.get("scope_agent"),
                    scope_team=raw.get("scope_team"),
                    priority=priority,
                ))
            elif rtype == "cost_cap":
                cost_cap.append(_CostCapRule(
                    name=name,
                    budget_pct_threshold=float(raw.get("budget_pct_threshold", 0.8)),
                    target_model=raw.get("target_model", ""),
                    priority=priority,
                ))
            elif rtype == "fallback_chain":
                fallback_chain.append(_FallbackChainRule(
                    name=name,
                    chain=list(raw.get("chain", [])),
                    priority=priority,
                ))
            else:
                logger.warning("capsera: unknown routing rule type %r — skipping", rtype)

        self._cost_cap_rules = sorted(cost_cap, key=lambda r: r.priority)
        self._task_type_rules = sorted(task_type, key=lambda r: r.priority)
        self._model_map_rules = sorted(model_map, key=lambda r: r.priority)
        self._fallback_chain_rules = sorted(fallback_chain, key=lambda r: r.priority)

        if self._config.debug:
            total = len(cost_cap) + len(task_type) + len(model_map) + len(fallback_chain)
            logger.debug("capsera: parsed %d routing rules from %s", total, source)
