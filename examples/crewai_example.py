"""CrewAI-style Capsera integration."""

import capsera

capsera.init(
    api_key="cap-...",
    endpoint="https://api.capsera.ai",
    enable_routing=True,
    enable_prompt_analysis=True,
    enable_budget_enforcement=True,
)


@capsera.crewai_agent(
    "market-research-agent",
    team="growth",
    task_type="research",
    customer_id="acme-corp",
    cost_center="growth-ai",
)
def run_market_research(topic: str) -> str:
    # CrewAI agent logic and provider calls go here.
    return f"Research summary for {topic}"


@capsera.crewai_task(
    "summarize-research",
    team="growth",
    customer_id="acme-corp",
    cost_center="growth-ai",
)
def summarize_research(notes: list[str]) -> str:
    # Provider calls here are attributed as task_type="crewai_task".
    return "\n".join(notes)


if __name__ == "__main__":
    run_market_research("AI FinOps")
    summarize_research(["LLM spend is growing", "Routing reduces simple-task cost"])
    capsera.shutdown()
