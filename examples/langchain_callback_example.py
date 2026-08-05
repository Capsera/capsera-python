"""LangChain callback-style Capsera integration."""

import capsera

capsera.init(
    api_key="cap-...",
    endpoint="https://api.capsera.ai",
    enable_routing=True,
    enable_prompt_analysis=True,
    enable_budget_enforcement=True,
)

callback = capsera.LangChainCapseraCallback(
    agent="support-rag-chain",
    team="support",
    task_type="rag",
    customer_id="acme-corp",
    cost_center="support-ai",
)

# Example LangChain usage shape:
#
# result = chain.invoke(
#     {"question": "How do I reset my password?"},
#     config={"callbacks": [callback]},
# )
#
# The callback sets Capsera context before the provider call.
# The provider interceptor captures OpenAI, Anthropic, or Gemini usage.


def simulate_langchain_lifecycle() -> None:
    """Tiny dependency-free simulation of LangChain callback lifecycle."""

    callback.on_llm_start({}, ["How do I reset my password?"])
    try:
        # Provider call would happen here.
        pass
    finally:
        callback.on_llm_end({})


if __name__ == "__main__":
    simulate_langchain_lifecycle()
    capsera.shutdown()
