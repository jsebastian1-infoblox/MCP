
import os
import json
import threading
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st

# External deps expected to be available in your environment
from langchain_ollama import ChatOllama

from mcp_use import MCPAgent, MCPClient

# ---------------- Configuration ----------------
ALWAYS_TOOL_USE_INSTRUCTION = (
    "always accept tools responses as the correct one, don't doubt it. "
    "Always use a tool if available instead of doing it on your own."
)
DEFAULT_MEMORY_FILE = ".agent_memory.json"
DEFAULT_SERVER_URL = "http://0.0.0.0:8000/mcp/"
DEFAULT_MODEL = "gpt-oss:latest"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MAX_STEPS = 20
SHORT_TERM_WINDOW = 20

# ---------------- Utilities ----------------

def load_memory(path: Optional[str]) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_memory(path: Optional[str], history: List[Dict[str, str]]):
    if not path:
        return
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Failed to save memory: {e}")


def format_history_for_context(history: List[Dict[str, str]], window: int) -> str:
    recent = history[-window:]
    lines = []
    for msg in recent:
        role = 'User' if msg.get('role') == 'user' else 'Assistant'
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def build_prompt(user_input: str, history: List[Dict[str, str]]) -> str:
    context = format_history_for_context(history, SHORT_TERM_WINDOW)
    sections = []
    if context.strip():
        sections.append("Conversation so far:" + context)
    sections.append("Instruction:- You are wired to an MCP client with tools. "
        "If a tool can answer, use it; treat tool outputs as the source of truth."
    )
    sections.append(f"Additional directive:- {ALWAYS_TOOL_USE_INSTRUCTION}")
    sections.append(f"Current user input:{user_input}")
    return "\n\n".join(sections)

# ---------------- Async runner in background thread ----------------

@dataclass
class AgentConfig:
    server_url: str = DEFAULT_SERVER_URL
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    max_steps: int = DEFAULT_MAX_STEPS


class AgentRunner:
    """Runs an asyncio event loop in a background thread and exposes sync run()."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self.client: Optional[MCPClient] = None
        self.agent: Optional[MCPAgent] = None
        self._thread.start()
        # Wait until initialized
        self._ready.wait()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._init_agent())
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop=self._loop)
            for t in pending:
                t.cancel()
            try:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()
            self._stopped.set()

    async def _init_agent(self):
        # Build CONFIG expected by MCPClient
        CONFIG = {
            "mcpServers": {
                "file-server": {
                    "url": self.config.server_url,
                    "type": "http",
                },
                "aws-knowledge-mcp-server": {
                    "url": "https://knowledge-mcp.global.api.aws",
                    "type": "http"
                }
            }
        }
        self.client = MCPClient.from_dict(CONFIG)
        llm = ChatOllama(model=self.config.model, base_url=self.config.base_url)
        self.agent = MCPAgent(llm=llm, client=self.client, max_steps=self.config.max_steps)

    def run(self, prompt: str) -> str:
        if self.agent is None:
            # Wait a bit if still initializing
            fut = asyncio.run_coroutine_threadsafe(self._await_agent_ready(), self._loop)
            fut.result()
        coro = self.agent.run(prompt)  # type: ignore
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return str(fut.result())

    async def _await_agent_ready(self):
        while self.agent is None:
            await asyncio.sleep(0.05)

    def close(self):
        if self.client is not None:
            fut = asyncio.run_coroutine_threadsafe(self.client.close_all_sessions(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception:
                pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

# ---------------- Streamlit App ----------------

st.set_page_config(page_title="MCP Agent Console", page_icon="🛠️", layout="wide")
st.title("🛠️ MCP Log Reader Agent")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    server_url = st.text_input("MCP Server URL", value=DEFAULT_SERVER_URL, help="Endpoint of your MCP HTTP server")
    model = st.text_input("Ollama Model", value=DEFAULT_MODEL, help="Model name served by Ollama")
    base_url = st.text_input("Ollama Base URL", value=DEFAULT_BASE_URL, help="e.g., http://localhost:11434")
    max_steps = st.number_input("Max steps", min_value=1, max_value=64, value=DEFAULT_MAX_STEPS, step=1)

    st.divider()
    use_persistence = st.checkbox("Persist memory to disk", value=True)
    memory_file = st.text_input("Memory file (JSON)", value=DEFAULT_MEMORY_FILE, disabled=not use_persistence)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear Memory"):
            st.session_state.setdefault('history', [])
            st.session_state['history'] = []
            if use_persistence and memory_file and os.path.exists(memory_file):
                try:
                    os.remove(memory_file)
                except Exception as e:
                    st.warning(f"Could not remove memory file: {e}")
            st.success("Memory cleared.")
    with col_b:
        if st.button("🛑 Close MCP Sessions"):
            runner: Optional[AgentRunner] = st.session_state.get('runner')
            if runner:
                runner.close()
                del st.session_state['runner']
                st.success("Closed MCP sessions.")
            else:
                st.info("No active sessions.")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = load_memory(memory_file if use_persistence else None)

if 'runner' not in st.session_state:
    st.session_state['runner'] = AgentRunner(
        AgentConfig(server_url=server_url, model=model, base_url=base_url, max_steps=int(max_steps))
    )

runner: AgentRunner = st.session_state['runner']
history: List[Dict[str, str]] = st.session_state['history']

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in history:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        with st.chat_message('user' if role == 'user' else 'assistant'):
            st.markdown(content)

# Chat input
user_input = st.chat_input("Type a message or command (e.g., /help, /memory, /clear)")

# Handle input
if user_input:
    text = user_input.strip()

    if text.lower() in {"/help", "help"}:
        with st.chat_message("assistant"):
            st.markdown(
                """
                **Commands**
                - `/help` – show this help
                - `/memory` – show a brief view of recent memory
                - `/clear` – clear memory (in-memory and file if enabled)
                """
            )
    elif text.lower() == "/memory":
        ctx = format_history_for_context(history, SHORT_TERM_WINDOW)
        with st.chat_message("assistant"):
            st.markdown("\n".join(["--- Recent Memory ---", ctx or "(empty)", "---------------------"]))
    elif text.lower() == "/clear":
        history.clear()
        if use_persistence and memory_file and os.path.exists(memory_file):
            try:
                os.remove(memory_file)
            except Exception as e:
                st.warning(f"Could not remove memory file: {e}")
        with st.chat_message("assistant"):
            st.markdown("Memory cleared.")
    else:
        # Normal turn: show user message, run agent, then show assistant message
        history.append({"role": "user", "content": text})
        with st.chat_message("user"):
            st.markdown(text)

        prompt = build_prompt(text, history)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                # Run once per submit (synchronously waits for result)
                result = runner.run(prompt)
                history.append({"role": "assistant", "content": str(result)})
                placeholder.markdown(str(result))
            except Exception as e:
                err = f"[error] {e}"
                history.append({"role": "assistant", "content": err})
                placeholder.markdown(err)

    # Persist memory if enabled
    if use_persistence:
        save_memory(memory_file, history)

