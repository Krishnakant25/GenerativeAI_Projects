"""
LangGraph conversation graph — MemoryOS edition.

Five nodes run in sequence each turn:
  0. security_check     — two-layer prompt injection detector (NEW)
  1. retrieve_long_term — fetch relevant memories from Mem0/ChromaDB
  2. build_prompt       — assemble persona system prompt + memories + history
  3. call_llm           — call GROQ / LLaMA3
  4. save_to_memory     — extract and persist new facts for future sessions

Persona is passed in at graph-build time, controlling the system prompt
and what Mem0 is told to extract.
"""

import os
from typing import TypedDict, Annotated
from operator import add

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from chatbot.memory_manager import LongTermMemoryManager
from chatbot.prompts import build_system_prompt
from chatbot.personas import Persona, get_persona
from chatbot.security import check_for_injection


# ── State schema ──────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    user_id: str
    persona_id: str
    user_message: str
    long_term_memories: list[dict]
    short_term_history: list[dict]
    system_prompt: str
    assistant_response: str
    is_injection: bool           # set by security node
    injection_reason: str        # human-readable reason if blocked
    messages: Annotated[list, add]


# ── Node 0: security check ────────────────────────────────────────────────────

def security_check(state: ChatState) -> dict:
    """
    Node 0: Run two-layer injection detection before anything else.
    If injection detected, sets is_injection=True and short-circuits the graph.
    """
    is_safe, reason = check_for_injection(state["user_message"])
    if not is_safe:
        return {
            "is_injection": True,
            "injection_reason": reason,
            "assistant_response": (
                "⚠️ I can't respond to that message — it appears to contain "
                "an attempt to override my instructions or access protected data. "
                "If this was a mistake, please rephrase your question."
            ),
        }
    return {"is_injection": False, "injection_reason": ""}


def should_continue(state: ChatState) -> str:
    """Conditional edge: skip to END if injection was detected."""
    return "blocked" if state.get("is_injection") else "continue"


# ── Node 1: retrieve long-term memory ────────────────────────────────────────

def make_retrieve_long_term(memory_manager: LongTermMemoryManager):
    def retrieve_long_term(state: ChatState) -> dict:
        memories = memory_manager.get_relevant_memories(
            query=state["user_message"],
            user_id=state["user_id"],
            limit=5,
        )
        return {"long_term_memories": memories}
    return retrieve_long_term


# ── Node 2: build prompt ──────────────────────────────────────────────────────

def make_build_prompt(short_term_memory: ConversationBufferWindowMemory):
    def build_prompt(state: ChatState) -> dict:
        persona = get_persona(state["persona_id"])

        # Inject memories into the persona's system prompt template
        system_prompt = build_system_prompt(
            persona_system_prompt=persona.system_prompt,
            memories=state["long_term_memories"],
        )

        # Pull short-term history from ConversationBufferWindowMemory
        history = short_term_memory.load_memory_variables({})
        chat_history = history.get("chat_history", [])
        short_term = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in chat_history
        ]

        return {"system_prompt": system_prompt, "short_term_history": short_term}
    return build_prompt


# ── Node 3: call LLM ──────────────────────────────────────────────────────────

def make_call_llm(llm: ChatGroq):
    def call_llm(state: ChatState) -> dict:
        messages = [SystemMessage(content=state["system_prompt"])]

        for turn in state["short_term_history"]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=state["user_message"]))

        response = llm.invoke(messages)
        return {
            "assistant_response": response.content,
            "messages": [AIMessage(content=response.content)],
        }
    return call_llm


# ── Node 4: save to memory ────────────────────────────────────────────────────

def make_save_to_memory(
    memory_manager: LongTermMemoryManager,
    short_term_memory: ConversationBufferWindowMemory,
):
    def save_to_memory(state: ChatState) -> dict:
        memory_manager.add_interaction(
            user_message=state["user_message"],
            assistant_message=state["assistant_response"],
            user_id=state["user_id"],
        )
        short_term_memory.save_context(
            {"input": state["user_message"]},
            {"output": state["assistant_response"]},
        )
        return {"messages": []}
    return save_to_memory


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_chat_graph(user_id: str = "default_user", persona_id: str = "finance"):
    """
    Build and compile the LangGraph StateGraph for a given user and persona.
    Returns (compiled_graph, short_term_memory, memory_manager).
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=1024,
    )

    memory_manager = LongTermMemoryManager()

    short_term_memory = ConversationBufferWindowMemory(
        k=5,
        return_messages=True,
        memory_key="chat_history",
    )

    retrieve_node = make_retrieve_long_term(memory_manager)
    build_node = make_build_prompt(short_term_memory)
    llm_node = make_call_llm(llm)
    save_node = make_save_to_memory(memory_manager, short_term_memory)

    graph = StateGraph(ChatState)

    # Register all nodes
    graph.add_node("security_check", security_check)
    graph.add_node("retrieve_long_term", retrieve_node)
    graph.add_node("build_prompt", build_node)
    graph.add_node("call_llm", llm_node)
    graph.add_node("save_to_memory", save_node)

    # Entry point
    graph.set_entry_point("security_check")

    # Conditional edge after security check
    graph.add_conditional_edges(
        "security_check",
        should_continue,
        {"continue": "retrieve_long_term", "blocked": END},
    )

    # Linear flow for safe messages
    graph.add_edge("retrieve_long_term", "build_prompt")
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "save_to_memory")
    graph.add_edge("save_to_memory", END)

    compiled = graph.compile()
    return compiled, short_term_memory, memory_manager


# ── Turn runner ───────────────────────────────────────────────────────────────

def run_chat_turn(
    graph,
    user_message: str,
    user_id: str,
    persona_id: str = "finance",
) -> tuple[str, bool]:
    """
    Run one conversational turn through the graph.
    Returns (assistant_response, was_blocked).
    """
    initial_state: ChatState = {
        "user_id": user_id,
        "persona_id": persona_id,
        "user_message": user_message,
        "long_term_memories": [],
        "short_term_history": [],
        "system_prompt": "",
        "assistant_response": "",
        "is_injection": False,
        "injection_reason": "",
        "messages": [HumanMessage(content=user_message)],
    }

    final_state = graph.invoke(initial_state)
    return final_state["assistant_response"], final_state.get("is_injection", False)
