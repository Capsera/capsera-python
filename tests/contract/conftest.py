"""Contract layer: REAL provider client libraries → local mock LLM server.

The fake-based unit tests verify the interceptor against OUR assumptions
about provider internals; these tests verify the assumptions themselves.
When a provider release moves a patch target (e.g. the openai 1.x → 2.x major
bump), this layer fails loudly — the fakes cannot.

The server is evals/mock_provider.py verbatim (stdlib-only, deterministic
usage per call, proven against real anthropic/openai clients by the eval
corpus). Ground truth per test: the server's ledger vs the emitter's events.

No API keys and no network beyond 127.0.0.1. Tests importorskip their client
library, so an environment without a provider installed skips cleanly.
"""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest

from capsera._config import Config
from capsera._interceptor import Interceptor
from tests.conftest import CapturingEmitter

_REPO = Path(__file__).resolve().parents[3]


def _load_mock_provider():
    spec = importlib.util.spec_from_file_location(
        "capsera_contract_mock_provider", _REPO / "evals" / "mock_provider.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockLLMServer:
    def __init__(self, state, server) -> None:
        self.state = state
        self.server = server
        self.port = server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"

    @property
    def ledger(self) -> list[dict]:
        return self.state.snapshot()

    def reset(self) -> None:
        self.state.reset()

    def inject_error(self, call_id: str, status: int = 400) -> None:
        """Calls whose prompt contains [CAP-CALL:<call_id>] get a provider-
        shaped error response with the given status."""
        self.state.faults = {"error_call_ids": [call_id], "status": status}


@pytest.fixture(scope="session")
def mock_llm_server():
    mp = _load_mock_provider()
    state = mp.ProviderState()
    server = mp.make_provider_server(state, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield MockLLMServer(state, server)
    server.shutdown()


@pytest.fixture
def capsera_capture(mock_llm_server):
    """Fresh ledger + real interceptor install with an in-memory emitter."""
    mock_llm_server.reset()
    config = Config(api_key="cap-contract", endpoint="http://127.0.0.1:9",
                    env="test", flush_interval_ms=99_999,
                    enable_budget_enforcement=False)
    emitter = CapturingEmitter(config)
    emitter.start()
    Interceptor(config, emitter).install()
    yield emitter
    emitter.shutdown()
