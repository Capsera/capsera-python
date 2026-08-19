"""
Configuration dataclass and global config holder for the Capsera SDK.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """Immutable configuration created once at init() time."""

    api_key: str = "dev"
    endpoint: str = "http://localhost:8000"
    default_agent: str | None = None
    default_team: str | None = None
    # Gateway tag applied to every event that has no auto-detected gateway.
    # Set via init(gateway="bifrost") for self-hosted gateways whose base_url
    # the host list in _catalog.GATEWAY_HOSTS cannot recognise.
    default_gateway: str | None = None
    env: str = "development"
    flush_interval_ms: int = 500
    enable_routing: bool = False
    routing_rules: list[dict[str, Any]] = field(default_factory=list)
    routing_refresh_interval: int = 60
    critical_task_types: list[str] = field(default_factory=list)
    enable_budget_enforcement: bool = True
    # "local"  — evaluate against a background-refreshed snapshot, no network
    #            on the hot path. The default: a blocking check per LLM call
    #            put our availability in front of the customer's.
    # "strict" — POST /v1/budgets/check and wait. Zero overshoot, at the cost
    #            of a round trip per call. For workloads where a single call
    #            over the cap is genuinely unacceptable.
    # "off"    — record spend, enforce nothing.
    enforcement_mode: str = "local"
    # Persist the snapshot to the system temp dir so a short-lived process
    # (Lambda, CLI, worker that exits) enforces from its first call rather
    # than never reaching a refresh.
    budget_cache_persist: bool = True
    budget_check_timeout: float = 1.0
    enable_prompt_analysis: bool = False
    prompt_size_threshold: int = 4000
    on_oversized_prompt: Callable[[int, int], None] | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        # Normalise endpoint — strip trailing slash so we can always append paths cleanly
        self.endpoint = self.endpoint.rstrip("/")
        # enable_budget_enforcement=False is the documented opt-out and predates
        # enforcement_mode. Honour it whatever the mode says, so existing code
        # that disables enforcement does not silently regain it on upgrade.
        if not self.enable_budget_enforcement:
            self.enforcement_mode = "off"
        elif self.enforcement_mode not in ("local", "strict", "off"):
            self.enforcement_mode = "local"


class _GlobalConfig:
    """Thread-safe singleton holder for the active Config instance."""

    def __init__(self) -> None:
        self._config: Config | None = None
        self._lock = threading.Lock()

    def set(self, config: Config) -> None:
        with self._lock:
            self._config = config

    def get(self) -> Config | None:
        # Read is atomic for CPython references; lock not strictly needed but
        # avoids any theoretical visibility issue on alternative runtimes.
        with self._lock:
            return self._config


# Module-level singleton — imported by other submodules
_global_config = _GlobalConfig()
