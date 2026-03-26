"""
Prompt building — now persona-aware.

Each persona has its own system prompt template.
Long-term memories are injected into the {long_term_context} slot.
"""

LONG_TERM_CONTEXT_WITH_MEMORIES = """--- What you remember about this user ---
{memories}
-----------------------------------------"""

NO_MEMORIES_YET = """--- No prior memories for this user yet ---
This is an early interaction. Be attentive to any personal details they share."""


def build_system_prompt(persona_system_prompt: str, memories: list[dict]) -> str:
    """
    Inject long-term memories into the persona's system prompt template.

    Args:
        persona_system_prompt: The raw system prompt string from the Persona object,
                               containing a {long_term_context} placeholder.
        memories: List of memory dicts from Mem0, each with a 'memory' key.

    Returns:
        Fully formatted system prompt string ready to send to the LLM.
    """
    if memories:
        memory_lines = "\n".join(
            f"- {m['memory']}" for m in memories if m.get("memory")
        )
        long_term_context = LONG_TERM_CONTEXT_WITH_MEMORIES.format(
            memories=memory_lines
        )
    else:
        long_term_context = NO_MEMORIES_YET

    return persona_system_prompt.format(long_term_context=long_term_context)
