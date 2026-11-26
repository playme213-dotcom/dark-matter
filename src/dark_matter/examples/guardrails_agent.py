"""Guardrails example.

Demonstrates input and output validation to prevent unsafe content
from reaching the agent or being returned to the user.
"""

from dark_matter import Agent, Runner, function_tool
from dark_matter.guardrail import InputGuardrail, OutputGuardrail, GuardrailTripwireTriggered


def no_secrets_input(ctx, agent, input_text):
    """Block inputs containing API keys or secrets."""
    import re
    patterns = [r"sk-[A-Za-z0-9]{20,}", r"api[_-]?key[=:]\s*\S+"]
    for p in patterns:
        if re.search(p, input_text, re.IGNORECASE):
            return False, "Input contains potential API key or secret. Please remove it."
    return True, None


def no_harmful_output(ctx, agent, output_text):
    """Block outputs containing harmful instructions."""
    harmful = ["how to hack", "how to make a bomb", "how to steal"]
    text = output_text.lower()
    for h in harmful:
        if h in text:
            return False, "Output contains harmful content."
    return True, None


safe_agent = Agent(
    name="Safe Agent",
    instructions="You are a helpful assistant. Answer questions safely and accurately.",
    input_guardrails=[InputGuardrail(trigger=no_secrets_input)],
    output_guardrails=[OutputGuardrail(trigger=no_harmful_output)],
)


def main():
    print("=== Dark Matter Guardrails Agent ===")
    print("Input guardrail: blocks API keys/secrets")
    print("Output guardrail: blocks harmful content")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        try:
            result = Runner.run_sync(safe_agent, user_input)
            print(f"Agent: {result.final_output}\n")
        except GuardrailTripwireTriggered as e:
            print(f"Guardrail triggered: {e}\n")


if __name__ == "__main__":
    main()
