"""
Live smoke suite — real provider APIs, real interception, captured emitter.

Opt-in only: every module here skips unless CAPSERA_TEST_LIVE=1. Each test
additionally skips when its provider's key env var is unset. Spend is capped:
one call per surface, max_tokens<=16.

Usage:
    set -a; source ../.env.test; set +a          # from sdk/
    python3 -m pytest tests/live -v
"""
from __future__ import annotations

import os

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor

from tests.conftest import CapturingEmitter


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CAPSERA_TEST_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live tests disabled (set CAPSERA_TEST_LIVE=1)")
    for item in items:
        if "/tests/live/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(skip)


def require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"missing env: {', '.join(missing)}")


@pytest.fixture
def live_events():
    """Install the real interceptor with an in-memory capturing emitter.

    Budget enforcement is off (no backend in play); routing/analysis off —
    this suite tests capture fidelity against real provider responses only.
    """
    config = Config(
        api_key="cap-live-smoke",
        endpoint="http://127.0.0.1:9",  # unused: CapturingEmitter never sends
        enable_budget_enforcement=False,
    )
    emitter = CapturingEmitter(config)
    emitter.start()
    interceptor = Interceptor(config, emitter, None, None)
    interceptor.install()
    yield emitter
    emitter.shutdown()
