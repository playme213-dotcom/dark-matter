"""Parallel swarm example.

A coordinator fans out independent tasks to multiple specialist agents,
then synthesizes their results into a single answer.
"""

from dark_matter import Agent, Runner, function_tool


research_agent = Agent(
    name="Research Agent",
    instructions="Research the topic thoroughly and provide key facts and sources.",
)

writing_agent = Agent(
    name="Writing Agent",
    instructions="Write clear, engaging content based on research findings.",
)

review_agent = Agent(
    name="Review Agent",
    instructions="Critique content for accuracy, clarity, and completeness. Suggest improvements.",
)


def research_topic(topic: str) -> str:
    """Delegate research to the research agent."""
    result = Runner.run_sync(research_agent, f"Research: {topic}")
    return result.final_output


def write_content(research: str) -> str:
    """Delegate writing to the writing agent."""
    result = Runner.run_sync(writing_agent, f"Write content based on: {research}")
    return result.final_output


def review_content(content: str) -> str:
    """Delegate review to the review agent."""
    result = Runner.run_sync(review_agent, f"Review: {content}")
    return result.final_output


swarm_coordinator = Agent(
    name="Swarm Coordinator",
    instructions=(
        "You coordinate a parallel swarm of agents. For a given topic:\n"
        "1. Use research_topic to gather facts\n"
        "2. Use write_content to draft based on research\n"
        "3. Use review_content to critique the draft\n"
        "4. Synthesize all results into a final answer\n\n"
        "Run all steps and present the combined output."
    ),
    functions=[research_topic, write_content, review_content],
)


def main():
    print("=== Dark Matter Parallel Swarm ===")
    print("Agents: Research → Writing → Review")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("Topic: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        result = Runner.run_sync(swarm_coordinator, user_input)
        print(f"\nFinal Output:\n{result.final_output}\n")


if __name__ == "__main__":
    main()
