"""
MemoryOS — Domain-Adaptive AI Assistant with Persistent Memory
Streamlit UI

Run with:
    streamlit run app.py
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from chatbot.graph import build_chat_graph, run_chat_turn
from chatbot.personas import list_personas, get_persona

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MemoryOS",
    page_icon="🧠",
    layout="wide",
)

# ── Session initialisation ────────────────────────────────────────────────────

def init_session():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]
    if "persona_id" not in st.session_state:
        st.session_state.persona_id = "finance"
    if "chat_graph" not in st.session_state:
        _rebuild_graph()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "security_log" not in st.session_state:
        st.session_state.security_log = []  # list of blocked attempt strings


def _rebuild_graph():
    """Rebuild the LangGraph whenever user or persona changes."""
    with st.spinner("Loading memory system..."):
        graph, _, mem_manager = build_chat_graph(
            user_id=st.session_state.user_id,
            persona_id=st.session_state.persona_id,
        )
    st.session_state.chat_graph = graph
    st.session_state.mem_manager = mem_manager


init_session()

# ── Header ────────────────────────────────────────────────────────────────────

persona = get_persona(st.session_state.persona_id)
st.title(f"🧠 MemoryOS  ·  {persona.icon} {persona.display_name}")
st.caption("LLaMA3 · GROQ · Mem0 · ChromaDB · LangGraph  |  Fully free to run")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    # ── Persona selector ──────────────────────────────────────────────────────
    st.header("Choose a persona")
    personas = list_personas()
    persona_options = {p.id: f"{p.icon}  {p.display_name}" for p in personas}

    selected_persona_id = st.radio(
        "Domain",
        options=list(persona_options.keys()),
        format_func=lambda x: persona_options[x],
        index=list(persona_options.keys()).index(st.session_state.persona_id),
        label_visibility="collapsed",
    )

    if selected_persona_id != st.session_state.persona_id:
        st.session_state.persona_id = selected_persona_id
        st.session_state.chat_history = []
        _rebuild_graph()
        st.rerun()

    selected_persona = get_persona(selected_persona_id)
    st.caption(selected_persona.tagline)

    st.divider()

    # ── User profile card (long-term memory visualised) ───────────────────────
    st.header("Your profile")
    st.caption(f"User ID: `{st.session_state.user_id}`")

    memories = st.session_state.mem_manager.get_all_memories(
        user_id=st.session_state.user_id
    )

    if memories:
        st.success(f"{len(memories)} facts remembered")
        for m in memories:
            st.markdown(f"- {m.get('memory', '')}")
    else:
        st.info("No memories yet — start chatting!")

    st.divider()

    # ── Switch user ───────────────────────────────────────────────────────────
    st.subheader("Switch user")
    new_user = st.text_input(
        "User ID", placeholder="e.g. arjun_01", label_visibility="collapsed"
    )
    if st.button("Load user", use_container_width=True) and new_user.strip():
        st.session_state.user_id = new_user.strip()
        st.session_state.chat_history = []
        st.session_state.security_log = []
        _rebuild_graph()
        st.rerun()

    if st.button("Clear all memories", type="secondary", use_container_width=True):
        st.session_state.mem_manager.delete_all_memories(
            user_id=st.session_state.user_id
        )
        st.session_state.chat_history = []
        st.success("Memories cleared!")
        st.rerun()

    st.divider()

    # ── Security log ──────────────────────────────────────────────────────────
    st.subheader("Security log")
    if st.session_state.security_log:
        st.error(f"{len(st.session_state.security_log)} attempt(s) blocked")
        for i, entry in enumerate(st.session_state.security_log, 1):
            with st.expander(f"Attempt {i}"):
                st.code(entry, language=None)
    else:
        st.success("No injection attempts detected")

    st.divider()
    st.caption(
        "🟣 Short-term: last 5 turns, in-session only\n\n"
        "🟢 Long-term: Mem0 + ChromaDB, persists across sessions\n\n"
        "🔴 Security: two-layer injection detector on every message"
    )

# ── Starter questions (shown when chat is empty) ──────────────────────────────

if not st.session_state.chat_history:
    st.markdown(f"**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(selected_persona.starter_questions):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state._starter_prompt = q
            st.rerun()

# Handle starter question click
if hasattr(st.session_state, "_starter_prompt"):
    starter = st.session_state._starter_prompt
    del st.session_state._starter_prompt
    # Inject it as if the user typed it
    st.session_state.chat_history.append({"role": "user", "content": starter})
    with st.spinner("Thinking..."):
        response, blocked = run_chat_turn(
            graph=st.session_state.chat_graph,
            user_message=starter,
            user_id=st.session_state.user_id,
            persona_id=st.session_state.persona_id,
        )
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input(f"Ask your {persona.display_name}..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, blocked = run_chat_turn(
                graph=st.session_state.chat_graph,
                user_message=prompt,
                user_id=st.session_state.user_id,
                persona_id=st.session_state.persona_id,
            )
        st.markdown(response)

    # Log injection attempts
    if blocked:
        st.session_state.security_log.append(prompt)
        st.toast("Injection attempt blocked and logged", icon="🔴")

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()
