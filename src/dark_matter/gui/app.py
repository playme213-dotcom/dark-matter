"""Dark Matter GUI — Streamlit chat interface for Dark Matter agents.

Features:
  - Custom API base URL + API key (session-only)
  - Model selector + temperature
  - Tools toggle (web search, weather, time, file, shell)
  - Web search toggle
  - Message history (SQLite persistence)
  - Attachments (file uploader + chips)
  - Thinking level (off/low/medium/high)

Run with:
    dark-matter-gui
    # or
    streamlit run src/dark_matter/gui/app.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st

from dark_matter import Agent, Runner, function_tool


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = Path.home() / ".dark_matter"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
DB_PATH = DATA_DIR / "conversations.db"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "o1-preview",
    "o1-mini",
]

DEFAULT_TOOLS = {
    "web_search": "Web Search (DuckDuckGo)",
    "weather": "Weather",
    "time": "Time",
    "file_read": "File Read",
    "file_write": "File Write (destructive)",
    "shell": "Shell Command (destructive)",
}

THINKING_LEVELS = {"off": 0, "low": 1024, "medium": 4096, "high": 16384}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    """Get or create the SQLite database for conversation history."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            agent TEXT,
            model TEXT,
            message_count INTEGER DEFAULT 0
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id TEXT,
            role TEXT,
            content TEXT,
            thinking TEXT,
            timestamp TEXT,
            FOREIGN KEY(conv_id) REFERENCES conversations(id)
        )"""
    )
    conn.commit()
    return conn


def save_conversation(conv_id: str, agent_name: str, model: str, messages: List[dict]):
    """Save or update a conversation in the database."""
    conn = get_db()
    now = datetime.now().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO conversations (id, timestamp, agent, model, message_count)
           VALUES (?, ?, ?, ?, ?)""",
        (conv_id, now, agent_name, model, len(messages)),
    )
    # Clear existing messages for this conversation and re-insert
    conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
    for msg in messages:
        conn.execute(
            """INSERT INTO messages (conv_id, role, content, thinking, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (
                conv_id,
                msg.get("role", ""),
                msg.get("content", ""),
                msg.get("thinking", ""),
                now,
            ),
        )
    conn.commit()
    conn.close()


def load_conversation(conv_id: str) -> Optional[dict]:
    """Load a conversation from the database."""
    conn = get_db()
    row = conn.execute(
        "SELECT agent, model FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    if not row:
        conn.close()
        return None
    msgs = conn.execute(
        "SELECT role, content, thinking FROM messages WHERE conv_id = ? ORDER BY id",
        (conv_id,),
    ).fetchall()
    conn.close()
    messages = [
        {"role": r, "content": c or "", "thinking": t or None}
        for r, c, t in msgs
    ]
    return {"agent": row[0], "model": row[1], "messages": messages}


def list_conversations():
    """List all saved conversations, newest first."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, timestamp, agent, message_count FROM conversations ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "timestamp": r[1], "agent": r[2], "message_count": r[3]}
        for r in rows
    ]


def new_conversation_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------
@function_tool
def get_weather(city: str) -> str:
    """Returns the current weather for a city."""
    return f"It's sunny in {city}."


@function_tool
def get_time() -> str:
    """Returns the current local time."""
    import datetime

    return f"The current time is {datetime.datetime.now().strftime('%H:%M')}."


@function_tool
def read_file(path: str) -> str:
    """Read a file from the current directory."""
    try:
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
        return p.read_text()[:5000]
    except Exception as e:
        return f"Error reading {path}: {e}"


@function_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file (destructive tool)."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


@function_tool
def run_command(command: str) -> str:
    """Run a shell command (destructive tool)."""
    import subprocess

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


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
# Session state initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages: List[dict] = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = new_conversation_id()

if "api_base" not in st.session_state:
    st.session_state.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODELS[0]

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "tool_selection" not in st.session_state:
    st.session_state.tool_selection = {key: True for key in DEFAULT_TOOLS}

if "thinking_level" not in st.session_state:
    st.session_state.thinking_level = "off"

if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True

if "attachments" not in st.session_state:
    st.session_state.attachments: List[dict] = []

if "agent_choice" not in st.session_state:
    st.session_state.agent_choice = "Assistant (general)"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # --- API Settings ---
    st.subheader("🔑 API Settings")
    api_base = st.text_input(
        "API Base URL",
        value=st.session_state.api_base,
        help="OpenAI-compatible API endpoint",
    )
    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="Your OpenAI API key (session-only, never saved to disk)",
    )
    if st.button("Apply API Settings", key="apply_api"):
        st.session_state.api_base = api_base
        st.session_state.api_key = api_key
        os.environ["OPENAI_BASE_URL"] = api_base
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API settings applied")

    if not st.session_state.api_key:
        st.error("❌ No API key configured")
    else:
        st.success("✅ API key configured (session-only)")

    st.divider()

    # --- Model Settings ---
    st.subheader("🤖 Model Settings")
    model_choice = st.selectbox(
        "Model",
        options=DEFAULT_MODELS + ["Custom"],
        index=(
            DEFAULT_MODELS.index(st.session_state.selected_model)
            if st.session_state.selected_model in DEFAULT_MODELS
            else len(DEFAULT_MODELS)
        ),
        key="model_select",
    )
    if model_choice == "Custom":
        custom_model = st.text_input(
            "Custom model name",
            value=st.session_state.selected_model,
            key="custom_model",
        )
        st.session_state.selected_model = custom_model
    else:
        st.session_state.selected_model = model_choice

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        key="temp_slider",
    )

    st.divider()

    # --- Tools Toggle ---
    st.subheader("🛠 Available Tools")
    for tool_key, tool_label in DEFAULT_TOOLS.items():
        st.session_state.tool_selection[tool_key] = st.checkbox(
            tool_label,
            value=st.session_state.tool_selection.get(tool_key, True),
            key=f"tool_{tool_key}",
        )

    if st.session_state.tool_selection.get("file_write") or st.session_state.tool_selection.get("shell"):
        st.warning("⚠️ Destructive tools enabled")

    st.divider()

    # --- Thinking ---
    st.subheader("🧠 Thinking")
    st.session_state.thinking_level = st.selectbox(
        "Reasoning level",
        options=list(THINKING_LEVELS.keys()),
        index=list(THINKING_LEVELS.keys()).index(st.session_state.thinking_level),
        key="thinking_select",
    )
    st.session_state.show_thinking = st.checkbox(
        "Show thinking in chat",
        value=st.session_state.show_thinking,
        key="show_thinking_cb",
    )

    st.divider()

    # --- Agent Selection ---
    st.subheader("🤖 Agent")
    agent_choice = st.selectbox(
        "Agent type",
        options=[
            "Assistant (general)",
            "Triage (specialist handoffs)",
            "Tools (weather + time)",
            "Web Search (DuckDuckGo)",
        ],
        index=0,
        key="agent_choice",
    )

    st.divider()

    # --- Conversation History ---
    st.subheader("📜 Conversation History")
    conversations = list_conversations()
    if conversations:
        convo_options = {
            c["id"]: f"{c['timestamp'][:16]} — {c['agent']} ({c['message_count']} msgs)"
            for c in conversations
        }
        selected_id = st.selectbox(
            "Load conversation",
            options=[""] + list(convo_options.keys()),
            format_func=lambda x: convo_options.get(x, "New conversation") if x else "New conversation",
            key="load_conv",
        )
        if selected_id and selected_id != st.session_state.conversation_id:
            loaded = load_conversation(selected_id)
            if loaded:
                st.session_state.messages = loaded["messages"]
                st.session_state.conversation_id = selected_id
                st.session_state.agent_choice = loaded.get("agent", "Assistant (general)")
                st.session_state.selected_model = loaded.get("model", DEFAULT_MODELS[0])
                st.rerun()

    col_new, col_export = st.columns(2)
    with col_new:
        if st.button("New Conversation", use_container_width=True, key="new_conv"):
            st.session_state.messages = []
            st.session_state.conversation_id = new_conversation_id()
            st.session_state.attachments = []
            st.rerun()
    with col_export:
        if st.session_state.messages:
            export_data = json.dumps(
                {
                    "id": st.session_state.conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "agent": st.session_state.agent_choice,
                    "model": st.session_state.selected_model,
                    "messages": st.session_state.messages,
                },
                indent=2,
            )
            st.download_button(
                "Export",
                data=export_data,
                file_name=f"dark_matter_{st.session_state.conversation_id}.json",
                mime="application/json",
                use_container_width=True,
                key="export_conv",
            )

    if st.button("Clear Conversation", use_container_width=True, key="clear_conv"):
        st.session_state.messages = []
        st.session_state.attachments = []
        st.rerun()


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------
@st.cache_resource
def build_agent(
    agent_name: str,
    tool_keys: tuple,
    model: str,
    temperature: float,
    thinking_level: str,
):
    """Build a Dark Matter Agent based on sidebar settings."""

    tools = []
    tool_keys_set = set(tool_keys)

    if tool_keys_set & {"web_search"}:
        from dark_matter.examples.web_search_tool import web_agent as web_search_agent

        for tool in web_search_agent.tools:
            tools.append(tool)

    if tool_keys_set & {"weather"}:
        tools.append(get_weather)

    if tool_keys_set & {"time"}:
        tools.append(get_time)

    if tool_keys_set & {"file_read"}:
        tools.append(read_file)

    if tool_keys_set & {"file_write"}:
        tools.append(write_file)

    if tool_keys_set & {"shell"}:
        tools.append(run_command)

    # Build thinking instruction
    thinking_instruction = ""
    if thinking_level != "off":
        budget = THINKING_LEVELS[thinking_level]
        thinking_instruction = (
            f"\n\n# Thinking ({thinking_level}, budget ~{budget} tokens): "
            "Before answering, think through the problem step by step. "
            "Consider different approaches, verify your assumptions, "
            "and then provide your final answer."
        )

    if agent_name == "Assistant (general)":
        return Agent(
            name="Assistant",
            instructions=(
                "You are a helpful assistant powered by Dark Matter." + thinking_instruction
            ),
            model=model,
            tools=tools,
        )

    if agent_name == "Tools (weather + time)":
        return Agent(
            name="Tool Assistant",
            instructions=(
                "You are a helpful assistant with tools. Use them when relevant."
                + thinking_instruction
            ),
            model=model,
            tools=tools,
        )

    if agent_name == "Web Search (DuckDuckGo)":
        return Agent(
            name="Web Search Agent",
            instructions=(
                "You are a research assistant with web search capability. "
                "When users ask about current events, facts, or topics you're unsure about, "
                "use the web_search tool to find up-to-date information. "
                "Always cite your sources by mentioning the website names."
                + thinking_instruction
            ),
            model=model,
            tools=tools,
        )

    # Triage: coordinator agent with handoffs to specialists
    sales_agent = Agent(
        name="Sales Agent",
        instructions=(
            "You handle sales questions. Be persuasive and focus on pricing and features."
        ),
        model=model,
        tools=[],
    )
    support_agent = Agent(
        name="Support Agent",
        instructions=(
            "You handle technical support and troubleshooting. Be patient, ask "
            "clarifying questions, and provide step-by-step solutions."
        ),
        model=model,
        tools=[],
    )

    return Agent(
        name="Triage Coordinator",
        instructions=(
            "You are a triage coordinator. Route user requests to the right specialist:\n"
            "- Sales questions (pricing, features, buying) → transfer to sales agent\n"
            "- Support questions (technical, troubleshooting, bugs) → transfer to support agent\n"
            "- General questions → answer yourself\n\n"
            "Use the transfer functions to hand off to specialists."
        )
        + thinking_instruction,
        model=model,
        tools=[sales_agent.as_tool(), support_agent.as_tool()] + tools,
    )


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and st.session_state.show_thinking
            and message.get("thinking")
        ):
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(message["thinking"])

# ---------------------------------------------------------------------------
# Attachment uploader
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Attach files (images, text, PDF)",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "txt", "md", "py", "json", "pdf"],
    key="file_uploader",
)

if uploaded_files:
    for uf in uploaded_files:
        if not any(a["name"] == uf.name for a in st.session_state.attachments):
            st.session_state.attachments.append(
                {
                    "name": uf.name,
                    "type": uf.type,
                    "data": uf.read(),
                }
            )

# Show attachment chips
if st.session_state.attachments:
    st.caption("📎 Attached files:")
    cols = st.columns(min(len(st.session_state.attachments), 4))
    for i, att in enumerate(st.session_state.attachments):
        with cols[i % 4]:
            if st.button(f"✕ {att['name']}", key=f"remove_attach_{i}"):
                st.session_state.attachments.pop(i)
                st.rerun()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Type a message..."):
    # Build user content with attachments context
    user_content = prompt
    if st.session_state.attachments:
        attachment_desc = "\n".join(
            f"[Attached: {a['name']} ({a['type']})]" for a in st.session_state.attachments
        )
        user_content = f"{prompt}\n\n{attachment_desc}"

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_content})
    with st.chat_message("user"):
        st.markdown(prompt)
        if st.session_state.attachments:
            st.caption(f"{len(st.session_state.attachments)} file(s) attached")

    # Build agent and run
    agent = build_agent(
        st.session_state.agent_choice,
        tuple(k for k, v in st.session_state.tool_selection.items() if v),
        st.session_state.selected_model,
        st.session_state.temperature,
        st.session_state.thinking_level,
    )

    with st.chat_message("assistant"):
        thinking_parts = []
        with st.spinner("Thinking..."):
            try:
                result = Runner.run_sync(agent, user_content)
                response_text = result.final_output or "(no output)"

                # Extract thinking from result items
                if st.session_state.show_thinking:
                    for item in result.new_items or []:
                        item_type = type(item).__name__
                        if "thinking" in item_type.lower() or "reasoning" in item_type.lower():
                            thinking_parts.append(
                                f"• {item_type}: {getattr(item, 'content', str(item))[:200]}"
                            )
                        elif "tool" in item_type.lower():
                            thinking_parts.append(f"• {item_type}: tool call")

            except Exception as exc:  # noqa: BLE001
                response_text = f"Error: {exc}"

        st.markdown(response_text)

        if st.session_state.show_thinking and thinking_parts:
            with st.expander("🧠 Thinking", expanded=False):
                for part in thinking_parts:
                    st.markdown(part)

    # Append assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "thinking": "\n".join(thinking_parts) if thinking_parts else None,
        }
    )

    # Clear attachments after sending
    st.session_state.attachments = []

    # Save conversation
    save_conversation(
        st.session_state.conversation_id,
        st.session_state.agent_choice,
        st.session_state.selected_model,
        st.session_state.messages,
    )

    st.rerun()


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
