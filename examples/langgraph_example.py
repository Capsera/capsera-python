"""LangGraph-style Capsera integration.

This example intentionally avoids importing LangGraph so it can be read as a
framework recipe. Apply the decorators to your actual LangGraph node functions.
"""

import capsera

capsera.init(
    api_key="cap-...",
    endpoint="https://api.capsera.ai",
    enable_routing=True,
    enable_prompt_analysis=True,
    enable_budget_enforcement=True,
    critical_task_types=["reasoning", "final_answer"],
)


@capsera.langgraph_node(
    "classify-intent",
    team="support",
    task_type="classification",
    customer_id="acme-corp",
    cost_center="support-ai",
)
def classify_intent(state: dict) -> dict:
    # Call OpenAI, Anthropic, or Gemini here.
    # The provider interceptor records the call under this node context.
    return state


@capsera.langgraph_node(
    "reason-about-ticket",
    team="support",
    task_type="reasoning",
    customer_id="acme-corp",
    cost_center="support-ai",
)
def reason_about_ticket(state: dict) -> dict:
    # Because "reasoning" is critical, margin-driven downgrades can be skipped.
    return state


if __name__ == "__main__":
    classify_intent({"message": "refund request"})
    reason_about_ticket({"message": "urgent production issue"})
    capsera.shutdown()
