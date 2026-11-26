"""Dark Matter GUI — Streamlit chat interface for Dark Matter agents.

Run with:
    dark-matter-gui
    # or
    streamlit run src/dark_matter/gui/app.py
"""

from __future__ import annotations

import os
from typing import List

import streamlit as st

from dark_matter import Agent, Runner, function_tool


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dark Matter Agent",
    page_icon="🌑",
    layout="wide",
)

st.title("🌑 Dark Matter Agent")
st.caption("Multi-agent orchestration with the Dark Matter SDK (forked from OpenAI Agents SDK)")

# ---------------------------------------------------------------------------
# Sidebar: agent selection + API key status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    if api_key_set:
        st.success("✅ OPENAI_API_KEY set (server-side)")
    else:
        st.error("❌ OPENAI_API_KEY not set")
        st.info("Set it on the server before running: `export OPENAI_API_KEY=sk-...`")

    st.divider()
    st.subheader("Choose Agent")
    agent_choice = st.selectbox(
        "Agent",
        options=[
            "Assistant (general)",
            "Triage (specialist handoffs)",
            "Tools (weather + time)",
            "Web Search (DuckDuckGo)",
        ],
        index=0,
    )

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
@st.cache_resource
def build_agent(agent_name: str):
    """Build and cache a Dark Matter Agent based on the sidebar choice."""

    @function_tool
    def get_weather(city: str) -> str:
        """Returns the current weather for a city."""
        return f"It's sunny in {city}."

    @function_tool
    def get_time() -> str:
        """Returns the current local time."""
        import datetime
        return f"The current time is {datetime.datetime.now().strftime('%H:%M')}."

    from dark_matter.examples.web_search_tool import web_agent as web_search_agent

    if agent_name == "Assistant (general)":
        return Agent(
            name="Assistant",
            instructions="You are a helpful assistant powered by Dark Matter.",
        )

    if agent_name == "Tools (weather + time)":
        return Agent(
            name="Tool Assistant",
            instructions="You are a helpful assistant with weather and time tools. Use them when relevant.",
            tools=[get_weather, get_time],
        )

    if agent_name == "Web Search (DuckDuckGo)":
        return web_search_agent

    # Triage: coordinator agent with handoffs to specialists
    sales_agent = Agent(
        name="Sales Agent",
        instructions="You handle sales questions. Be persuasive and focus on pricing and features.",
    )
    support_agent = Agent(
        name="Support Agent",
        instructions="You handle support questions. Be patient and troubleshoot technical issues.",
    )

    return Agent(
        name="Triage Coordinator",
        instructions=(
            "You are a triage coordinator. Route user requests to the right specialist: "
            "sales questions → Sales Agent, support/technical questions → Support Agent. "
            "Answer general questions yourself."
        ),
        tools=[sales_agent.as_tool(), support_agent.as_tool()],
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Type a message..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build agent and run
    agent = build_agent(agent_choice)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = Runner.run_sync(agent, prompt)
                response_text = result.final_output or "(no output)"
            except Exception as exc:  # noqa: BLE001
                response_text = f"Error: {exc}"

        st.markdown(response_text)

    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": response_text})


# ---------------------------------------------------------------------------
# Entry point for console script
# ---------------------------------------------------------------------------
def main():
    """Entry point for the ``dark-matter-gui`` console script."""
    import sys
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "Dark Matter • MIT License • "
    "[GitHub](https://github.com/playme213-dotcom/dark-matter)"
)
