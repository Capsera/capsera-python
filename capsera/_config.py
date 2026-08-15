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
    budget_check_timeout: float = 1.0
    enable_prompt_analysis: bool = False
    prompt_size_threshold: int = 4000
    on_oversized_prompt: Callable[[int, int], None] | None = None
    debug: bool = False

    def __post_init__(self) -> None:
        # Normalise endpoint — strip trailing slash so we can always append paths cleanly
        self.endpoint = self.endpoint.rstrip("/")


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
