"""llama3.1:8b suggestion generator — turns rolling context into actionable tips."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import httpx

from screen_copilot.models import Suggestion
from screen_copilot.context import ContextManager

logger = logging.getLogger("copilot.suggester")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REASONING_MODEL = os.getenv("REASONING_MODEL", "llama3.1:8b-instruct-q4_0")
REASONING_TIMEOUT = 45.0
MAX_RETRIES = 2

_VALID_CATEGORIES = {"tip", "warning", "info", "general"}


async def generate_suggestion(context: ContextManager) -> Suggestion:
    """Generate a suggestion from the current rolling context.

    Args:
        context: Live ContextManager holding recent observations.

    Returns:
        A populated Suggestion object.
    """
    ctx = context.get_context_string()
    summary = context.get_summary()

    if summary["observations_count"] == 0:
        return Suggestion(
            suggestion_text="Keep working — I'll start helping once I understand your context.",
            category="info",
            confidence="low",
            based_on_observations=0,
        )

    time_on_task = summary["time_on_task_seconds"]
    urgency_note = ""
    if time_on_task > 120:
        urgency_note = "The user has been on this task a while — consider whether they may be stuck."

    latest_obs = context.window.observations[-1] if context.window.observations else None
    latest_topic = latest_obs.topic if latest_obs else ""
    latest_vision_suggestion = latest_obs.suggested_action if latest_obs else ""

    prompt = (
        "You are a helpful AI copilot watching someone's screen to assist them.\n"
        "MOST RECENT SCREEN ANALYSIS:\n"
        f"Topic detected: {latest_topic}\n"
        f"Vision system's suggestion: {latest_vision_suggestion}\n\n"
        "Use the above as your PRIMARY basis for your suggestion. You may\n"
        "rephrase or refine it, but it should be clearly related to\n"
        f"\"{latest_topic}\" — do not suggest something unrelated to this topic.\n\n"
        "CURRENT CONTEXT:\n\n"
        f"Active app: {summary['current_app']}\n\n"
        f"Time on current task: {summary['time_on_task_seconds']} seconds\n\n"
        f"App switches this session: {summary['switch_count']}\n"
        "RECENT ACTIVITY (last "
        f"{summary['observations_count']} observations):\n\n"
        f"{ctx}\n"
        "IMPORTANT: Your suggestion must be SPECIFIC to the actual topic/content\n"
        "the person is engaging with — reference the TOPIC or KEY_DETAIL from\n"
        "their recent activity if available. Avoid generic app-level tips\n"
        "(like \"use Ctrl+T for new tab\") unless no specific topic was detected.\n"
        "SUGGESTIONS ALREADY GIVEN (do NOT repeat these — give something NEW):\n"
        f"{context.get_recent_suggestions_string()}\n\n"
        "IMPORTANT: If the user has been stuck on the same task for over 60 seconds,\n"
        "prioritize a more urgent or specific suggestion. If everything looks fine,\n"
        "it is OK to give a CATEGORY of \"info\" with a low-key observation.\n"
        "Based on what this person is doing, provide ONE specific, helpful suggestion.\n\n"
        "Focus on: shortcuts they might not know, next logical steps, potential issues,\n\n"
        "or relevant resources.\n"
        f"{urgency_note}\n"
        "Respond in this EXACT format — no other text:\n"
        "SUGGESTION: [one specific actionable suggestion, max 25 words]\n\n"
        "CATEGORY: [tip | warning | info | general]\n\n"
        "CONFIDENCE: [high | medium | low]\n\n"
        "REASON: [one phrase explaining why this is relevant right now]"
    )

    raw = await _call_llama(prompt)
    suggestion = _parse_suggestion(raw)
    suggestion.based_on_observations = summary["observations_count"]

    logger.info(
        "suggester: generated [%s] suggestion confidence=%s",
        suggestion.category,
        suggestion.confidence,
    )
    return suggestion


async def generate_app_switch_message(old_app: str, new_app: str) -> Suggestion:
    """Generate a context-switch acknowledgment when the user changes apps.

    Args:
        old_app: Application the user was previously in.
        new_app: Application the user just switched to.

    Returns:
        A Suggestion acknowledging the switch with a tip for the new app.
    """
    prompt = (
        f"The user just switched from {old_app} to {new_app}.\n\n"
        f"Generate a brief, friendly acknowledgment and one helpful tip for {new_app}.\n"
        "Respond in this EXACT format:\n\n"
        "SUGGESTION: [brief acknowledgment + one tip for the new app, max 25 words]\n\n"
        "CATEGORY: info\n\n"
        "CONFIDENCE: high\n\n"
        "REASON: app switch detected"
    )

    raw = await _call_llama(prompt)
    return _parse_suggestion(raw)


async def generate_detailed_explanation(context: ContextManager,
                                          suggestion: Suggestion) -> str:
    """
    Generates a longer, more substantive explanation building on
    the short suggestion already shown. Called only on user request
    (click), not automatically.
    """
    latest_obs = context.window.observations[-1] if context.window.observations else None
    topic = latest_obs.topic if latest_obs else "the current screen"
    key_detail = latest_obs.key_detail if latest_obs else ""

    prompt = f"""You previously gave this brief suggestion: "{suggestion.suggestion_text}"

This relates to: {topic}
Additional context: {key_detail}

Now provide a more detailed, helpful explanation (3-5 sentences).
Include specific, actionable advice, relevant terms or next steps,
and anything else genuinely useful about "{topic}".

Respond with ONLY the explanation text, no headers or formatting."""

    raw = await _call_llama_detailed(prompt)
    return raw or "Could not generate additional detail right now."


async def _call_llama_detailed(prompt: str) -> Optional[str]:
    """Like _call_llama but allows longer output for detailed explanations."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=REASONING_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": REASONING_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 400}
                }
            )
            if resp.status_code != 200:
                return None
            return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("detailed explanation call failed: %s", e)
        return None


async def analyze_selection(selected_text: str) -> Suggestion:
    """
    Analyzes a specific text selection the user highlighted and
    provides a focused, relevant explanation or insight about it.
    """
    if not selected_text or len(selected_text.strip()) < 3:
        return Suggestion(
            suggestion_text="No text selected. Highlight something and press the hotkey again.",
            category="info", confidence="low"
        )

    prompt = f"""The user highlighted this specific text and wants help understanding it:

\"\"\"
{selected_text[:1200]}
\"\"\"

Provide a focused, genuinely useful response about THIS specific text.
Depending on what it is, you might: explain a concept, define terms,
summarize, point out something important, or suggest a next step.
Be specific to the actual content — 2-4 sentences.

Respond in this EXACT format:
SUGGESTION: [your focused, specific response]
CATEGORY: tip
CONFIDENCE: high
REASON: user selection"""

    raw = await _call_llama_detailed(prompt)  # reuse the longer-output caller
    if not raw:
        return Suggestion(
            suggestion_text="Could not analyze the selection right now.",
            category="info", confidence="low"
        )
    # parse the SUGGESTION line, fallback to raw text
    suggestion_text = raw
    for line in raw.splitlines():
        if line.strip().startswith("SUGGESTION:"):
            suggestion_text = line.split("SUGGESTION:", 1)[1].strip()
            break
    return Suggestion(
        suggestion_text=suggestion_text[:600],
        category="tip", confidence="high",
        based_on_observations=0
    )


async def _call_llama(prompt: str) -> Optional[str]:
    """POST a text-only prompt to llama3.1:8b via Ollama and return the response."""
    payload = {
        "model": REASONING_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 128},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=REASONING_TIMEOUT) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        except Exception as exc:
            logger.warning("suggester: llama attempt %d failed: %s", attempt, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1.5)

    return None


def _parse_suggestion(raw: Optional[str]) -> Suggestion:
    """Parse llama's structured KEY: value response into a Suggestion."""
    if not raw or not raw.strip():
        return Suggestion(
            suggestion_text="Analyzing your screen...",
            category="info",
            confidence="low",
            timestamp=datetime.utcnow(),
        )

    try:
        suggestion_text = "Analyzing your screen..."
        category = "general"
        confidence = "low"
        found_suggestion = False

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("SUGGESTION:"):
                text = line[len("SUGGESTION:"):].strip()[:200]
                if text:
                    suggestion_text = text
                    found_suggestion = True
            elif line.startswith("CATEGORY:"):
                val = line[len("CATEGORY:"):].strip().lower()
                category = val if val in _VALID_CATEGORIES else "general"
            elif line.startswith("CONFIDENCE:"):
                val = line[len("CONFIDENCE:"):].strip().lower()
                confidence = val if val in {"high", "medium", "low"} else "low"
            # REASON is parsed but intentionally not surfaced in suggestion_text

        if not found_suggestion:
            return Suggestion(
                suggestion_text="Analyzing your screen...",
                category="info",
                confidence="low",
                timestamp=datetime.utcnow(),
            )

        return Suggestion(
            suggestion_text=suggestion_text,
            category=category,
            confidence=confidence,
            timestamp=datetime.utcnow(),
        )

    except Exception as exc:
        logger.warning("_parse_suggestion failed: %s", exc)
        return Suggestion(
            suggestion_text="Analyzing your screen...",
            category="info",
            confidence="low",
            timestamp=datetime.utcnow(),
        )
