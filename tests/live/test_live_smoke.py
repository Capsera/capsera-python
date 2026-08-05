"""One real API call per patched surface; assert the interceptor saw it.

Hard assertions on: exactly one emitted event, positive token counts, and
(for catalog-listed models) positive cost. A failure here is a genuine
finding — either an interception bug or a stale pricing catalog.
"""
from __future__ import annotations

import os

import pytest

from tests.live.conftest import require_env

PROMPT = "Say OK"

ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
# Rolling alias — this key's free tier serves it (resolves to a concrete 3.x
# version; the interceptor records response.model_version, not the alias).
GEMINI_MODEL = "gemini-flash-lite-latest"
# Bedrock in ap-south-1 serves most models via APAC cross-region inference
# profiles; plain IDs kept as fallbacks.
BEDROCK_CANDIDATES = [
    "apac.amazon.nova-micro-v1:0",
    "amazon.nova-micro-v1:0",
    "apac.anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]


def sole_event(emitter):
    events = emitter.drain()
    assert len(events) == 1, f"expected exactly 1 event, got {len(events)}: {events!r}"
    return events[0]


def assert_tokens(event):
    assert event.input_tokens > 0, f"input_tokens not captured: {event!r}"
    assert event.output_tokens > 0, f"output_tokens not captured: {event!r}"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

anthropic_mod = pytest.importorskip("anthropic")


class TestAnthropicLive:
    @pytest.fixture(autouse=True)
    def _key(self):
        require_env("ANTHROPIC_API_KEY")

    def test_sync_create(self, live_events):
        client = anthropic_mod.Anthropic()
        resp = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}],
        )
        assert resp.usage.input_tokens > 0  # NI: response object untouched
        event = sole_event(live_events)
        assert_tokens(event)
        assert event.provider == "anthropic"
        assert event.cost_usd > 0, f"model {event.model!r} missing from pricing catalog"

    async def test_async_create(self, live_events):
        client = anthropic_mod.AsyncAnthropic()
        resp = await client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}],
        )
        assert resp.usage.output_tokens > 0
        event = sole_event(live_events)
        assert_tokens(event)
        assert event.cost_usd > 0

    def test_sync_stream(self, live_events):
        client = anthropic_mod.Anthropic()
        with client.messages.stream(
            model=ANTHROPIC_MODEL, max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}],
        ) as stream:
            text = "".join(stream.text_stream)
        assert text  # the app really streamed something
        event = sole_event(live_events)
        assert_tokens(event)
        assert event.cost_usd > 0

    async def test_async_stream(self, live_events):
        client = anthropic_mod.AsyncAnthropic()
        chunks = []
        async with client.messages.stream(
            model=ANTHROPIC_MODEL, max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}],
        ) as stream:
            async for part in stream.text_stream:
                chunks.append(part)
        assert chunks
        event = sole_event(live_events)
        assert_tokens(event)


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #

openai_mod = pytest.importorskip("openai")


class TestOpenAILive:
    @pytest.fixture(autouse=True)
    def _key(self):
        require_env("OPENAI_API_KEY")

    def test_sync_create(self, live_events):
        client = openai_mod.OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}],
        )
        assert resp.usage.prompt_tokens > 0
        event = sole_event(live_events)
        assert_tokens(event)
        assert event.provider == "openai"
        assert event.cost_usd > 0

    async def test_async_create(self, live_events):
        client = openai_mod.AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL, max_tokens=16,
            messages=[{"role": "user", "content": PROMPT}],
        )
        assert resp.usage.completion_tokens > 0
        event = sole_event(live_events)
        assert_tokens(event)

    def test_sync_stream(self, live_events):
        client = openai_mod.OpenAI()
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, max_tokens=16, stream=True,
            messages=[{"role": "user", "content": PROMPT}],
        )
        chunks = list(stream)
        assert chunks
        event = sole_event(live_events)
        assert_tokens(event)  # usage must come from the injected include_usage final chunk

    async def test_async_stream(self, live_events):
        client = openai_mod.AsyncOpenAI()
        stream = await client.chat.completions.create(
            model=OPENAI_MODEL, max_tokens=16, stream=True,
            messages=[{"role": "user", "content": PROMPT}],
        )
        chunks = [c async for c in stream]
        assert chunks
        event = sole_event(live_events)
        assert_tokens(event)

    def test_embeddings_sync(self, live_events):
        client = openai_mod.OpenAI()
        resp = client.embeddings.create(model=EMBED_MODEL, input=PROMPT)
        assert resp.data[0].embedding
        event = sole_event(live_events)
        assert event.input_tokens > 0
        assert event.cost_usd > 0

    async def test_embeddings_async(self, live_events):
        client = openai_mod.AsyncOpenAI()
        resp = await client.embeddings.create(model=EMBED_MODEL, input=PROMPT)
        assert resp.data[0].embedding
        event = sole_event(live_events)
        assert event.input_tokens > 0


# --------------------------------------------------------------------------- #
# Google — modern google.genai and legacy google.generativeai
# --------------------------------------------------------------------------- #


class TestGoogleGenAILive:
    @pytest.fixture(autouse=True)
    def _key(self):
        require_env("GOOGLE_API_KEY")

    def test_genai_sync(self, live_events):
        genai = pytest.importorskip("google.genai")
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=PROMPT)
        assert resp.text
        event = sole_event(live_events)
        assert_tokens(event)
        assert event.cost_usd > 0

    def test_generativeai_legacy_sync(self, live_events):
        ggenai = pytest.importorskip("google.generativeai")
        ggenai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = ggenai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(PROMPT)
        assert resp.text
        event = sole_event(live_events)
        assert_tokens(event)


# --------------------------------------------------------------------------- #
# AWS Bedrock — Converse via bearer token (AWS_BEARER_TOKEN_BEDROCK)
# --------------------------------------------------------------------------- #


class TestBedrockLive:
    @pytest.fixture(autouse=True)
    def _key(self):
        require_env("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION")

    def test_converse(self, live_events):
        boto3 = pytest.importorskip("boto3")
        client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])
        resp = None
        errors = []
        for model_id in BEDROCK_CANDIDATES:
            try:
                resp = client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": PROMPT}]}],
                    inferenceConfig={"maxTokens": 16},
                )
                break
            except Exception as exc:  # model not enabled in this region — try next
                errors.append(f"{model_id}: {exc}")
        if resp is None:
            pytest.fail("no Bedrock candidate model callable:\n" + "\n".join(errors))
        assert resp["output"]["message"]["content"]
        event = sole_event(live_events)
        assert_tokens(event)
