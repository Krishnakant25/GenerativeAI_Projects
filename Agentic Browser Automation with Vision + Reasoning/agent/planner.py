"""Planner module — LLaMA3.1 reasoning/action planning only (never receives image data)."""

import asyncio
import logging
import os
import json
import re
from typing import Optional

import httpx

from agent.models import VisionOutput, Action, Plan

logger = logging.getLogger("agent.planner")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
REASONING_MODEL = os.getenv("REASONING_MODEL", "llama3.1:8b-instruct-q4_0")
REASONING_TIMEOUT = 60.0
MAX_RETRIES = 3

_ALLOWED_ACTIONS = {
    "navigate",
    "click",
    "click_text",
    "type",
    "scroll",
    "extract",
    "finish",
}

_PLAN_PROMPT_TEMPLATE = """You are an AI agent controlling a web browser to complete a research task.
GOAL: {goal}

STEP: {step_number}
CURRENT PAGE ANALYSIS:

Page type: {page_type}
Visible text: {visible_text}
Suggested action from vision: {suggested_action}
Suggested target: {suggested_target}
Vision confidence: {confidence}
Vision reasoning: {reasoning}

RECENT ACTIONS:

{context_string}

DISCOVERED FACTS (use these to avoid re-researching):
{memory_context}

INSTRUCTIONS:

Decide the single best next action to make progress toward the goal
If vision confidence is low, prefer click_text or navigate over coordinate clicks
If you see the same action repeated 3 times in recent history, choose a different approach
If goal appears complete, use action: finish

Respond in this EXACT format — no other text:
ACTION: [navigate | click | click_text | type | scroll | extract | finish]

TARGET: [url, or (x,y) coordinates, or text string, or "page" for scroll/extract]

VALUE: [text to type if action is type, scroll direction if scroll (up/down), otherwise empty]

REASONING: [one sentence explaining this decision]

CONFIDENCE: [high | medium | low]

GOAL_COMPLETE: [yes | no]"""

_FINDINGS_PROMPT_TEMPLATE = """You are producing a structured research report.

GOAL: {goal}

RAW CONTENT:
{joined_texts}

Write a structured report with exactly these sections.
Use "N/A" for sections with no relevant content.

## Summary
2-3 sentence overview of what was found.

## Key Findings
Bullet points of the most important discoveries.

## Companies and Entities
Names of companies, organizations, or people mentioned.

## Funding Rounds
Any funding rounds found with amounts, dates, investors.

## Timeline
Key dates and events in chronological order if found.

## Sources
URLs or publication names referenced."""

_FACTS_PROMPT_TEMPLATE = """You are extracting key facts from web content to remember for a research task.

GOAL: {goal}

CONTENT:
{content}

List up to 5 key facts relevant to the goal. Each fact on one line.
Start each line with "FACT: ".
Only include facts, no commentary."""


async def plan_next_action(
    vision_output: VisionOutput,
    goal: str,
    step_number: int,
    action_history: list[dict],
    task_memory: list[str] = [],  # read-only — never mutated here
) -> Plan:
    """Produce the next action from the latest vision analysis and recent history."""
    try:
        if action_history:
            recent = action_history[-5:]
            context_string = "\n".join(
                f"Step {entry.get('step_number', '?')}: "
                f"{entry.get('action_taken', '')} → {entry.get('action_result', '')}"
                for entry in recent
            )
        else:
            context_string = "No actions taken yet."

        memory_context = "\n".join(task_memory) if task_memory else "None yet."

        prompt = _PLAN_PROMPT_TEMPLATE.format(
            goal=goal,
            step_number=step_number,
            page_type=vision_output.page_type,
            visible_text=", ".join(vision_output.visible_text),
            suggested_action=vision_output.suggested_action,
            suggested_target=vision_output.suggested_target,
            confidence=vision_output.confidence,
            reasoning=vision_output.reasoning,
            context_string=context_string,
            memory_context=memory_context,
        )

        raw = await _call_llama(prompt)
        return _parse_plan_response(raw)
    except Exception as exc:  # noqa: BLE001 — public function must never raise
        logger.error("plan_next_action failed: %s", exc)
        return Plan(success=False, action="navigate", confidence="low",
                    raw_response=f"plan_next_action error: {exc}")


async def check_loop_detection(
    action_history: list[dict], current_action: str, current_target: str
) -> bool:
    """Return True if the last 3 history entries share the same action and target."""
    if len(action_history) < 3:
        return False

    last_three = action_history[-3:]
    actions = {entry.get("action_taken") for entry in last_three}
    targets = {entry.get("target") for entry in last_three}

    if len(actions) == 1 and len(targets) == 1:
        return True
    return False


async def extract_key_facts(extracted_text: str, goal: str) -> list[str]:
    """Pull goal-relevant facts from extracted page text for the rolling task memory.

    Returns an empty list on any failure — never raises.
    """
    try:
        prompt = _FACTS_PROMPT_TEMPLATE.format(goal=goal, content=extracted_text[:1500])
        raw = await _call_llama(prompt)
        if raw is None:
            return []
        facts = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("FACT:"):
                facts.append(stripped[len("FACT:"):].strip())
        return facts
    except Exception as exc:  # noqa: BLE001 — never raise
        logger.error("extract_key_facts failed: %s", exc)
        return []


async def summarize_findings(extracted_texts: list[str], goal: str) -> str:
    """Compile extracted text into a structured findings summary via the reasoning model."""
    try:
        joined_texts = "\n---\n".join(extracted_texts)
        prompt = _FINDINGS_PROMPT_TEMPLATE.format(goal=goal, joined_texts=joined_texts)
        raw = await _call_llama(prompt)
        if raw is None:
            return "Summary generation failed — raw data available in agent trace."
        return raw
    except Exception as exc:  # noqa: BLE001 — never raise
        logger.error("summarize_findings failed: %s", exc)
        return "Summary generation failed — raw data available in agent trace."


async def _call_llama(prompt: str) -> Optional[str]:
    """POST to Ollama /api/generate (text-only) with retries. No image data is ever sent."""
    payload = {
        "model": REASONING_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }
    url = f"{OLLAMA_BASE_URL}/api/generate"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=REASONING_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                return response.json()["response"]
        except Exception as exc:  # noqa: BLE001 — retry on any failure
            logger.warning("LLaMA call attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2.0)

    logger.error("LLaMA call exhausted all %d retries", MAX_RETRIES)
    return None


def _parse_plan_response(raw: Optional[str]) -> Plan:
    """Parse llama's structured response into a Plan, line by line."""
    if not raw:
        return Plan(success=False, action="navigate", confidence="low", raw_response="")

    try:
        plan = Plan(raw_response=raw)
        action_recognized = False

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or ":" not in stripped:
                continue
            prefix, _, value = stripped.partition(":")
            key = prefix.strip().upper()
            value = value.strip()

            if key == "ACTION":
                candidate = value.lower()
                if candidate in _ALLOWED_ACTIONS:
                    plan.action = candidate
                    action_recognized = True
                else:
                    plan.action = "navigate"
            elif key == "TARGET":
                plan.target = value
            elif key == "VALUE":
                plan.value = value
            elif key == "REASONING":
                plan.reasoning = value
            elif key == "CONFIDENCE":
                plan.confidence = (
                    value.lower()
                    if value.lower() in ("high", "medium", "low")
                    else "low"
                )
            elif key == "GOAL_COMPLETE":
                plan.goal_complete = value.lower() == "yes"

        plan.success = action_recognized
        return plan
    except Exception as exc:  # noqa: BLE001 — never raise out of parser
        logger.error("_parse_plan_response failed: %s", exc)
        return Plan(success=False, action="navigate", confidence="low", raw_response=raw)
