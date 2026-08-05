# Capsera SDK

Per-agent token attribution and budget enforcement for multi-agent LLM systems.

Install once, and every LLM call your app makes is attributed to the agent that
made it — with budget envelopes that can block or downgrade a call *before* it
reaches the provider.

```bash
pip install capsera
```

## Quickstart

```python
import capsera

capsera.init(api_key="cap-...")          # from your Capsera dashboard

@capsera.agent("researcher", team="core")
def research(question):
    return client.messages.create(       # captured automatically
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
```

That's the whole integration. `init()` patches the provider clients already
installed in your environment; the decorator tells Capsera *which agent* the
spend belongs to. Calls made outside any decorated scope are still captured and
attributed to `unknown`.

## What gets captured

| Provider | Coverage |
| --- | --- |
| Anthropic | sync, async, streaming, errors |
| OpenAI chat | sync, async, streaming, errors |
| OpenAI embeddings | sync, async, errors |
| Google Gemini (`google-genai`, `google-generativeai`) | sync (plus async on the legacy client), errors |
| Mistral, Cohere, Vertex AI, AWS Bedrock | sync, errors |

Frameworks need no separate integration: they call the provider clients above,
so their calls are captured by construction. Where Capsera does extra work is
caller attribution — resolving the recorded file and line past the framework's
internals to *your* code. That skip-list covers LangChain, LangGraph, CrewAI,
LlamaIndex, AutoGen, Haystack, DSPy and the LiteLLM gateway.

Each captured call records token counts, cost from a maintained pricing catalog,
latency, the agent/team/session it belongs to, and the file and line that made
it.

## Design guarantees

**It stays out of the way.** Events are queued and shipped by a background
daemon thread, so your call path never blocks on Capsera. The queue is bounded
and drops rather than growing without limit. If the backend is unreachable, the
emitter retries with backoff, trips a circuit breaker, and gives up — your app
keeps working.

**It fails open.** Every interception path is wrapped: a bug or a shape change
in Capsera surfaces as missing telemetry, never as an exception in your
application. The one intentional exception is `BudgetExceededError`, raised only
when a blocking budget stops a call you configured it to stop.

**It never sends your prompts.** Optional prompt analysis records structure —
token estimates, message counts, a hash of the system prompt — and never
content. This is verified by a test that plants canary strings in prompts and
scans every emitted payload for them.

## Attribution

```python
from capsera import agent, tag, set_session

@agent("planner", team="core", task_type="planning")     # decorator
def plan(): ...

with tag("summarizer", customer_id="acme"):               # context manager
    ...

set_session("conv-42")     # thread-level, flows to every call without wrapping
```

## Configuration

```python
capsera.init(
    api_key="cap-...",
    enable_routing=True,              # route eligible calls to cheaper models
    enable_budget_enforcement=True,   # pre-call budget check (on by default)
    enable_prompt_analysis=True,      # structural metadata, never content
    debug=True,                       # one log line per intercepted call
)
```

`capsera.get_interception_report()` returns which provider surfaces are actually
hooked in the current process, so you can assert at startup that everything you
use is being tracked.

Requires Python 3.11+. The only runtime dependency is `httpx`.

## Links

- [Capsera](https://capsera.ai)
- [Dashboard](https://app.capsera.ai)

Licensed under the Apache License, Version 2.0.
