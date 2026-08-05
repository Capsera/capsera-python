"""Fixtures shared by the SDK validation suite.

The suite validates the SDK's OUTWARD behavior: the events it emits must
satisfy the backend's ingest contract, it must survive hostile
environments (dead backends, malformed provider responses, concurrency),
and its public API must reject or normalize bad input rather than
poisoning the event stream.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The backend's ingest schema IS the contract — import it from the sibling
# tree so the two sides are validated against each other in one process.
_BACKEND = Path(__file__).resolve().parents[3] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from capsera._config import Config  # noqa: E402


@pytest.fixture
def config() -> Config:
    return Config(
        api_key="test-key",
        endpoint="http://localhost:9",  # reserved port — always refused
        env="test",
        flush_interval_ms=10_000,
        debug=True,
    )


def backend_event_schema():
    from app.schemas.event import EventIn

    return EventIn


def backend_batch_schema():
    from app.schemas.event import EventBatch

    return EventBatch
