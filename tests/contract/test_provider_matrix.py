"""Provider × surface contract matrix — real clients, local mock server.

Every cell asserts the full capture contract:
  1 request in the server ledger == 1 event emitted, exact token equality,
  and the response object reaches the app intact (non-interference).

Bedrock has no clean local base-url override for its mock — it stays covered
by fake-based unit tests plus the live smoke suite. Cohere cells activate
once the `cohere` package is installed (`pip install -e "sdk[test-providers]"`).
"""
from __future__ import annotations

import pytest

anthropic_mod = pytest.importorskip("anthropic")
openai_mod = pytest.importorskip("openai")

PROMPT = "Say OK"


def sole_capture(emitter, server):
    events = emitter.drain()
    ledger = server.ledger
    assert len(ledger) == 1, f"server saw {len(ledger)} calls: {ledger!r}"
    assert len(events) == 1, f"SDK emitted {len(events)} events"
    return events[0], ledger[0]


def assert_tokens_match(event, row) -> None:
    assert event.input_tokens == row["usage"]["input_tokens"]
    assert event.output_tokens == row["usage"]["output_tokens"]


# =========================================================================== #
# Anthropic — real anthropic client
# =========================================================================== #

class TestAnthropicContract:
    def _client(self, server):
        return anthropic_mod.Anthropic(base_url=server.base_url,
                                       api_key="test", max_retries=0)

    def _aclient(self, server):
        return anthropic_mod.AsyncAnthropic(base_url=server.base_url,
                                            api_key="test", max_retries=0)

    def test_sync_create(self, mock_llm_server, capsera_capture):
        resp = self._client(mock_llm_server).messages.create(
            model="claude-haiku-4-5", max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}])
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert_tokens_match(event, row)
        assert event.provider == "anthropic"
        assert event.cache_read_tokens == row["usage"]["cache_read_tokens"]
        assert event.cache_write_tokens == row["usage"]["cache_write_tokens"]
        assert resp.usage.input_tokens == row["usage"]["input_tokens"]  # NI
        assert resp.content[0].text.startswith("echo:")

    async def test_async_create(self, mock_llm_server, capsera_capture):
        resp = await self._aclient(mock_llm_server).messages.create(
            model="claude-haiku-4-5", max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}])
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert_tokens_match(event, row)
        assert resp.usage.output_tokens == row["usage"]["output_tokens"]

    def test_sync_stream(self, mock_llm_server, capsera_capture):
        with self._client(mock_llm_server).messages.stream(
                model="claude-haiku-4-5", max_tokens=16,
                messages=[{"role": "user", "content": PROMPT}]) as stream:
            text = "".join(stream.text_stream)
        assert text.startswith("echo:")
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert row["stream"] is True
        assert_tokens_match(event, row)

    async def test_async_stream(self, mock_llm_server, capsera_capture):
        chunks = []
        async with self._aclient(mock_llm_server).messages.stream(
                model="claude-haiku-4-5", max_tokens=16,
                messages=[{"role": "user", "content": PROMPT}]) as stream:
            async for part in stream.text_stream:
                chunks.append(part)
        assert "".join(chunks).startswith("echo:")
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert row["stream"] is True
        assert_tokens_match(event, row)

    def test_sync_error_passthrough(self, mock_llm_server, capsera_capture):
        mock_llm_server.inject_error("e1", status=400)
        with pytest.raises(anthropic_mod.BadRequestError):
            self._client(mock_llm_server).messages.create(
                model="claude-haiku-4-5", max_tokens=16,
                messages=[{"role": "user", "content": f"[CAP-CALL:e1] {PROMPT}"}])
        events = capsera_capture.drain()
        assert len(events) == 1
        assert events[0].is_error is True
        assert events[0].error_type == "BadRequestError"
        assert float(events[0].cost_usd) == 0.0


# =========================================================================== #
# OpenAI — real openai client (survived the 1.x → 2.x major bump; keep it so)
# =========================================================================== #

class TestOpenAIContract:
    def _client(self, server):
        return openai_mod.OpenAI(base_url=f"{server.base_url}/v1",
                                 api_key="test", max_retries=0)

    def _aclient(self, server):
        return openai_mod.AsyncOpenAI(base_url=f"{server.base_url}/v1",
                                      api_key="test", max_retries=0)

    def test_sync_create(self, mock_llm_server, capsera_capture):
        resp = self._client(mock_llm_server).chat.completions.create(
            model="gpt-4o-mini", max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}])
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert_tokens_match(event, row)
        assert event.provider == "openai"
        assert resp.usage.prompt_tokens == row["usage"]["input_tokens"]  # NI
        assert resp.choices[0].message.content.startswith("echo:")

    async def test_async_create(self, mock_llm_server, capsera_capture):
        resp = await self._aclient(mock_llm_server).chat.completions.create(
            model="gpt-4o-mini", max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}])
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert_tokens_match(event, row)
        assert resp.usage.completion_tokens == row["usage"]["output_tokens"]

    def test_sync_stream(self, mock_llm_server, capsera_capture):
        stream = self._client(mock_llm_server).chat.completions.create(
            model="gpt-4o-mini", max_tokens=16, stream=True,
            messages=[{"role": "user", "content": PROMPT}])
        chunks = list(stream)
        assert chunks, "app must still receive stream chunks"
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert row["stream"] is True
        assert_tokens_match(event, row)

    async def test_async_stream(self, mock_llm_server, capsera_capture):
        stream = await self._aclient(mock_llm_server).chat.completions.create(
            model="gpt-4o-mini", max_tokens=16, stream=True,
            messages=[{"role": "user", "content": PROMPT}])
        chunks = [c async for c in stream]
        assert chunks
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert row["stream"] is True
        assert_tokens_match(event, row)

    def test_sync_error_passthrough(self, mock_llm_server, capsera_capture):
        mock_llm_server.inject_error("e2", status=400)
        with pytest.raises(openai_mod.BadRequestError):
            self._client(mock_llm_server).chat.completions.create(
                model="gpt-4o-mini", max_tokens=16,
                messages=[{"role": "user", "content": f"[CAP-CALL:e2] {PROMPT}"}])
        events = capsera_capture.drain()
        assert len(events) == 1
        assert events[0].is_error is True
        assert events[0].error_type == "BadRequestError"

    def test_embeddings_sync(self, mock_llm_server, capsera_capture):
        resp = self._client(mock_llm_server).embeddings.create(
            model="text-embedding-3-small", input=PROMPT)
        assert resp.data[0].embedding  # NI
        events = capsera_capture.drain()
        ledger = mock_llm_server.ledger
        assert len(ledger) == 1 and len(events) == 1
        assert events[0].input_tokens == ledger[0]["usage"]["input_tokens"]

    async def test_embeddings_async(self, mock_llm_server, capsera_capture):
        resp = await self._aclient(mock_llm_server).embeddings.create(
            model="text-embedding-3-small", input=PROMPT)
        assert resp.data[0].embedding
        events = capsera_capture.drain()
        ledger = mock_llm_server.ledger
        assert len(ledger) == 1 and len(events) == 1
        assert events[0].input_tokens == ledger[0]["usage"]["input_tokens"]


# =========================================================================== #
# Google — real google.genai client
# =========================================================================== #

class TestGoogleGenAIContract:
    def _client(self, server):
        genai = pytest.importorskip("google.genai")
        return genai.Client(api_key="test",
                            http_options={"base_url": server.base_url})

    def test_sync_generate_content(self, mock_llm_server, capsera_capture):
        client = self._client(mock_llm_server)  # keep alive: GC closes httpx
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=PROMPT)
        assert resp.text.startswith("echo:")  # NI
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert event.provider == "google"
        assert event.model == "gemini-2.5-flash"
        assert_tokens_match(event, row)
        assert event.cost_usd > 0

    def test_sync_error_passthrough(self, mock_llm_server, capsera_capture):
        genai = pytest.importorskip("google.genai")
        mock_llm_server.inject_error("e3", status=400)
        client = self._client(mock_llm_server)
        with pytest.raises(genai.errors.ClientError):
            client.models.generate_content(
                model="gemini-2.5-flash", contents=f"[CAP-CALL:e3] {PROMPT}")
        events = capsera_capture.drain()
        assert len(events) == 1
        assert events[0].is_error is True
        assert float(events[0].cost_usd) == 0.0


# =========================================================================== #
# Mistral — real mistralai client (OpenAI-shaped API, ledger-tagged by UA)
# =========================================================================== #

class TestMistralContract:
    def _client(self, server):
        mistralai = pytest.importorskip("mistralai")
        return mistralai.Mistral(api_key="test", server_url=server.base_url)

    def test_chat_complete(self, mock_llm_server, capsera_capture):
        with self._client(mock_llm_server) as client:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": PROMPT}])
        assert resp.choices[0].message.content.startswith("echo:")  # NI
        event, row = sole_capture(capsera_capture, mock_llm_server)
        assert row["provider"] == "mistral", \
            "ledger must tag mistral traffic distinctly from openai"
        assert event.provider == "mistral"
        assert_tokens_match(event, row)
        assert event.cost_usd > 0

    def test_chat_error_passthrough(self, mock_llm_server, capsera_capture):
        mock_llm_server.inject_error("e4", status=400)
        with self._client(mock_llm_server) as client, \
                pytest.raises(Exception) as excinfo:
            client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user",
                           "content": f"[CAP-CALL:e4] {PROMPT}"}])
        assert "capsera" not in str(type(excinfo.value)).lower(), \
            "the app must see the provider's error, never a capsera one"
        events = capsera_capture.drain()
        assert len(events) == 1
        assert events[0].is_error is True
