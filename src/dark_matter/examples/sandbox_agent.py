"""Sandbox agent example with human approval gate.

This agent has file read/write and shell command tools that require
human approval before execution, demonstrating safe multi-agent tool use.
"""

from dark_matter import Agent, Runner, function_tool
import os
import subprocess
from pathlib import Path

WORKSPACE = Path("./sandbox_workspace")
WORKSPACE.mkdir(exist_ok=True)


@function_tool
def read_file(path: str) -> str:
    """Read a file from the sandbox workspace."""
    try:
        full_path = WORKSPACE / path
        if not full_path.exists():
            return f"File not found: {path}"
        return full_path.read_text()
    except Exception as e:
        return f"Error reading {path}: {e}"


@function_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the sandbox workspace (requires approval)."""
    try:
        full_path = WORKSPACE / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


@function_tool
def run_command(command: str) -> str:
    """Run a shell command in the sandbox (requires approval)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(WORKSPACE),
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


sandbox_agent = Agent(
    name="Sandbox Agent",
    instructions=(
        "You are a sandbox agent with file read/write and shell command tools. "
        "You can read files, write files, and run shell commands in the sandbox workspace. "
        "Be careful with destructive operations. Always explain what you're doing before "
        "calling tools."
    ),
    tools=[read_file, write_file, run_command],
)


def main():
    print("=== Dark Matter Sandbox Agent ===")
    print(f"Workspace: {WORKSPACE.absolute()}")
    print("Tools: read_file, write_file, run_command")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        result = Runner.run_sync(sandbox_agent, user_input)
        print(f"Agent: {result.final_output}\n")


if __name__ == "__main__":
    main()
