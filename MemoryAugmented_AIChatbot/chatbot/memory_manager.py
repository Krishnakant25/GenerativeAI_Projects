"""
Long-term memory manager using Mem0 with ChromaDB as the vector store.

Mem0 automatically:
- Extracts facts and preferences from conversations
- Embeds and stores them in ChromaDB
- Retrieves semantically relevant memories given a query
"""

import os
from mem0 import Memory


def get_memory_config() -> dict:
    """
    Returns Mem0 configuration using ChromaDB as the local vector store
    and GROQ as the LLM for memory extraction.
    
    ChromaDB persists to disk at ./chroma_db so memories survive restarts.
    """
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "chatbot_memories",
                "path": "./chroma_db",  # local persistence
            },
        },
        "llm": {
            "provider": "groq",
            "config": {
                "model": "llama-3.1-8b-instant",
                "api_key": os.environ.get("GROQ_API_KEY"),
                "temperature": 0.1,
                "max_tokens": 1000,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                # Free, runs locally — no API key needed
                "model": "multi-qa-MiniLM-L6-cos-v1",
            },
        },
    }


class LongTermMemoryManager:
    """
    Wraps Mem0 to provide a clean interface for storing and retrieving
    long-term memories per user.
    """

    def __init__(self):
        config = get_memory_config()
        self.mem0 = Memory.from_config(config)

    def add_interaction(self, user_message: str, assistant_message: str, user_id: str):
        """
        After each exchange, Mem0 analyses the conversation and decides
        what (if anything) is worth remembering long-term.
        
        It extracts facts like:
        - "User prefers Python over Java"
        - "User is building a Flask API for their startup"
        - "User's name is Arjun"
        """
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        self.mem0.add(messages, user_id=user_id)

    def get_relevant_memories(self, query: str, user_id: str, limit: int = 5) -> list[dict]:
        """
        Semantically search stored memories for ones relevant to the current query.
        Returns a list of memory dicts, each with a 'memory' key.
        """
        results = self.mem0.search(query=query, user_id=user_id, limit=limit)
        # Mem0 returns {"results": [...]} in newer versions
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []

    def get_all_memories(self, user_id: str) -> list[dict]:
        """
        Retrieve all stored memories for a user (useful for a 'memory panel' in the UI).
        """
        results = self.mem0.get_all(user_id=user_id)
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []

    def delete_all_memories(self, user_id: str):
        """Clear all memories for a user (useful for demo resets)."""
        self.mem0.delete_all(user_id=user_id)
