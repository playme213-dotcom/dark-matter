"""Triage multi-agent example.

A coordinator agent routes user requests to specialist agents:
- Sales Agent for pricing/features questions
- Support Agent for technical/troubleshooting questions
- General questions answered by the coordinator itself
"""

from dark_matter import Agent, Runner, function_tool


# Specialist agents
sales_agent = Agent(
    name="Sales Agent",
    instructions=(
        "You handle sales questions. Be persuasive, focus on pricing, features, "
        "and benefits. Do not answer technical support questions."
    ),
)

support_agent = Agent(
    name="Support Agent",
    instructions=(
        "You handle technical support and troubleshooting. Be patient, ask "
        "clarifying questions, and provide step-by-step solutions. Do not "
        "answer sales questions."
    ),
)


def transfer_to_sales():
    """Transfer to the sales specialist."""
    return sales_agent


def transfer_to_support():
    """Transfer to the support specialist."""
    return support_agent


# Coordinator / triage agent
triage_agent = Agent(
    name="Triage Coordinator",
    instructions=(
        "You are a triage coordinator. Route user requests to the right specialist:\n"
        "- Sales questions (pricing, features, buying) → transfer to sales agent\n"
        "- Support questions (technical, troubleshooting, bugs) → transfer to support agent\n"
        "- General questions → answer yourself\n\n"
        "Use the transfer functions to hand off to specialists."
    ),
    functions=[transfer_to_sales, transfer_to_support],
)


def main():
    print("=== Dark Matter Triage Agent ===")
    print("Specialists: Sales Agent, Support Agent")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        result = Runner.run_sync(triage_agent, user_input)
        print(f"Agent ({result.agent.name}): {result.final_output}\n")


if __name__ == "__main__":
    main()
