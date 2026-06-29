"""LangGraph nodes — OBSERVE, THINK, ACT, VERIFY. Each takes and returns AgentState."""

import asyncio
import logging
import os
from typing import Any

from agent.models import AgentState, VisionOutput, Plan
from agent.vision import analyze_screenshot, analyze_region, analyze_screenshot_som
from agent.planner import (
    plan_next_action,
    check_loop_detection,
    summarize_findings,
    extract_key_facts,
)
from agent.actions import execute_action
from browser.screenshot import capture

logger = logging.getLogger("agent.nodes")

# Global controller registry — keyed by task_id
# Avoids LangGraph dict serialization destroying live Playwright objects
_browser_registry: dict = {}


def register_controller(task_id: str, controller) -> None:
    _browser_registry[task_id] = controller


def get_controller(task_id: str):
    return _browser_registry.get(task_id)


def unregister_controller(task_id: str) -> None:
    _browser_registry.pop(task_id, None)


async def observe_node(state: AgentState) -> AgentState:
    """OBSERVE — capture a screenshot of the current browser state."""
    if isinstance(state, dict):
        valid_keys = set(AgentState.model_fields.keys())
        filtered = {k: v for k, v in state.items() if k in valid_keys}
        # fill missing required fields with defaults
        filtered.setdefault("task_id", "")
        filtered.setdefault("goal", "")
        filtered.setdefault("step_number", 0)
        filtered.setdefault("max_steps", 25)
        filtered.setdefault("status", "running")
        filtered.setdefault("action_history", [])
        filtered.setdefault("extracted_texts", [])
        filtered.setdefault("screenshot_paths", [])
        filtered.setdefault("task_memory", [])
        filtered.setdefault("last_som_regions", [])
        try:
            state = AgentState(**filtered)
        except Exception as e:
            logger.error("state normalization failed: %s | keys: %s", e, list(state.keys()))
            # last resort — construct minimal valid state
            state = AgentState(
                task_id=filtered.get("task_id", ""),
                goal=filtered.get("goal", ""),
                status="failed",
                error_message=f"state normalization failed: {e}"
            )
    try:
        controller = get_controller(state.task_id)
        screenshot_path = await capture(controller, state.task_id, state.step_number)
        if screenshot_path is None:
            logger.warning(
                "observe_node: screenshot capture failed at step %s", state.step_number
            )
            state.error_message = "screenshot capture failed"
            state.status = "failed"
            return state

        state.current_screenshot = screenshot_path
        logger.info("observe_node: screenshot captured → %s", screenshot_path)
        return state
    except Exception as exc:  # noqa: BLE001 — nodes must never raise
        logger.error("observe_node raised: %s", exc)
        state.error_message = f"observe_node error: {exc}"
        state.status = "failed"
        return state


async def think_node(state: AgentState) -> AgentState:
    """THINK — run vision analysis, then plan the next action (with loop breaking)."""
    if isinstance(state, dict):
        valid_keys = set(AgentState.model_fields.keys())
        filtered = {k: v for k, v in state.items() if k in valid_keys}
        # fill missing required fields with defaults
        filtered.setdefault("task_id", "")
        filtered.setdefault("goal", "")
        filtered.setdefault("step_number", 0)
        filtered.setdefault("max_steps", 25)
        filtered.setdefault("status", "running")
        filtered.setdefault("action_history", [])
        filtered.setdefault("extracted_texts", [])
        filtered.setdefault("screenshot_paths", [])
        filtered.setdefault("task_memory", [])
        filtered.setdefault("last_som_regions", [])
        try:
            state = AgentState(**filtered)
        except Exception as e:
            logger.error("state normalization failed: %s | keys: %s", e, list(state.keys()))
            # last resort — construct minimal valid state
            state = AgentState(
                task_id=filtered.get("task_id", ""),
                goal=filtered.get("goal", ""),
                status="failed",
                error_message=f"state normalization failed: {e}"
            )
    try:
        if state.current_screenshot is None:
            logger.error("think_node: no current screenshot at step %s", state.step_number)
            state.error_message = "no screenshot to analyze"
            state.status = "failed"
            return state

        controller = get_controller(state.task_id)
        vision_output, som_regions = await analyze_screenshot_som(
            state.current_screenshot,
            state.goal,
            state.step_number,
            controller=controller,
            task_id=state.task_id,
        )
        state.last_vision = vision_output
        state.last_som_regions = som_regions
        logger.info(
            "think_node: SoM analysis complete — %d regions, confidence=%s",
            len(som_regions),
            vision_output.confidence,
        )

        if not vision_output.success:
            logger.warning(
                "think_node: vision analysis low confidence at step %s", state.step_number
            )
            if vision_output.clickable_elements:
                first = vision_output.clickable_elements[0]
                region_vision = await analyze_region(
                    controller,
                    int(first.get("x", 0)),
                    int(first.get("y", 0)),
                    200,
                    100,
                    state.task_id,
                    state.step_number,
                    state.goal,
                )
                if region_vision.success:
                    state.last_vision = region_vision

        looping = await check_loop_detection(
            state.action_history,
            state.last_vision.suggested_action,
            state.last_vision.suggested_target,
        )
        if looping:
            logger.warning(
                "think_node: loop detected at step %s — injecting scroll to break",
                state.step_number,
            )
            forced_plan = Plan(
                success=True,
                action="scroll",
                target="page",
                value="down",
                reasoning="loop detected — forced scroll",
                confidence="low",
            )
            state.last_plan = forced_plan
            return state

        plan = await plan_next_action(
            state.last_vision,
            state.goal,
            state.step_number,
            state.action_history,
            task_memory=state.task_memory,
        )
        state.last_plan = plan
        logger.info(
            "think_node: planned action=%s target=%s confidence=%s",
            plan.action,
            plan.target,
            plan.confidence,
        )
        return state
    except Exception as exc:  # noqa: BLE001 — nodes must never raise
        logger.error("think_node raised: %s", exc)
        state.error_message = f"think_node error: {exc}"
        state.status = "failed"
        return state


async def act_node(state: AgentState) -> AgentState:
    """ACT — execute the planned action and, on finish, build the final report."""
    if isinstance(state, dict):
        valid_keys = set(AgentState.model_fields.keys())
        filtered = {k: v for k, v in state.items() if k in valid_keys}
        # fill missing required fields with defaults
        filtered.setdefault("task_id", "")
        filtered.setdefault("goal", "")
        filtered.setdefault("step_number", 0)
        filtered.setdefault("max_steps", 25)
        filtered.setdefault("status", "running")
        filtered.setdefault("action_history", [])
        filtered.setdefault("extracted_texts", [])
        filtered.setdefault("screenshot_paths", [])
        filtered.setdefault("task_memory", [])
        filtered.setdefault("last_som_regions", [])
        try:
            state = AgentState(**filtered)
        except Exception as e:
            logger.error("state normalization failed: %s | keys: %s", e, list(state.keys()))
            # last resort — construct minimal valid state
            state = AgentState(
                task_id=filtered.get("task_id", ""),
                goal=filtered.get("goal", ""),
                status="failed",
                error_message=f"state normalization failed: {e}"
            )
    try:
        if state.last_plan is None:
            logger.error("act_node: no plan to execute at step %s", state.step_number)
            state.error_message = "no plan to execute"
            state.status = "failed"
            return state

        if state.last_plan.goal_complete:
            logger.info("act_node: goal_complete flagged — proceeding to finish")
            state.last_plan.action = "finish"

        if state.current_screenshot and state.last_plan.action in ["click", "click_text", "type"]:
            from agent.validator import validate_action
            validation = await validate_action(
                state.current_screenshot,
                state.last_plan.action,
                state.last_plan.target,
            )
            if not validation["valid"]:
                logger.warning(
                    "act_node: moondream rejected action %s → %s: %s",
                    state.last_plan.action,
                    state.last_plan.target,
                    validation["reason"],
                )
                state.last_plan.action = "scroll"
                state.last_plan.target = "page"
                state.last_plan.value = "down"
                state.last_plan.reasoning = f"moondream validation failed: {validation['reason']}"

        old_step = state.step_number
        action = state.last_plan.action

        result = await execute_action(
            controller=get_controller(state.task_id),
            action=action,
            target=state.last_plan.target,
            value=state.last_plan.value,
            task_id=state.task_id,
            step_number=state.step_number,
        )
        state.action_history.append(result)

        if action == "extract" and not str(result.get("action_result", "")).startswith(
            "failed"
        ):
            state.extracted_texts.append(result["action_result"])
            new_facts = await extract_key_facts(result["action_result"], state.goal)
            state.task_memory.extend(new_facts)
            state.task_memory = state.task_memory[-10:]
            logger.info(
                "act_node: extracted %d facts — memory now %d entries",
                len(new_facts),
                len(state.task_memory),
            )

        if action == "finish":
            summary_text = await summarize_findings(state.extracted_texts, state.goal)
            final_report = {
                "goal": state.goal,
                "status": "complete",
                "steps_taken": state.step_number,
                "findings": {
                    "summary": summary_text,
                    "extracted_raw": state.extracted_texts,
                },
                "agent_reasoning_trace": state.action_history,
                "screenshots": [
                    entry
                    for entry in state.action_history
                    if "screenshot" in str(entry)
                ],
            }
            state.final_report = final_report
            state.status = "complete"

        state.step_number += 1
        logger.info(
            "act_node: step %s executed %s → %s",
            old_step,
            action,
            str(result.get("action_result", ""))[:80],
        )
        return state
    except Exception as exc:  # noqa: BLE001 — nodes must never raise
        logger.error("act_node raised: %s", exc)
        state.error_message = f"act_node error: {exc}"
        state.status = "failed"
        return state


async def verify_node(state: AgentState) -> AgentState:
    """VERIFY — decide whether to continue, terminate with partial, or fail."""
    if isinstance(state, dict):
        valid_keys = set(AgentState.model_fields.keys())
        filtered = {k: v for k, v in state.items() if k in valid_keys}
        # fill missing required fields with defaults
        filtered.setdefault("task_id", "")
        filtered.setdefault("goal", "")
        filtered.setdefault("step_number", 0)
        filtered.setdefault("max_steps", 25)
        filtered.setdefault("status", "running")
        filtered.setdefault("action_history", [])
        filtered.setdefault("extracted_texts", [])
        filtered.setdefault("screenshot_paths", [])
        filtered.setdefault("task_memory", [])
        filtered.setdefault("last_som_regions", [])
        try:
            state = AgentState(**filtered)
        except Exception as e:
            logger.error("state normalization failed: %s | keys: %s", e, list(state.keys()))
            # last resort — construct minimal valid state
            state = AgentState(
                task_id=filtered.get("task_id", ""),
                goal=filtered.get("goal", ""),
                status="failed",
                error_message=f"state normalization failed: {e}"
            )
    try:
        if state.status in ("complete", "failed"):
            return state

        if state.step_number >= state.max_steps:
            logger.warning(
                "verify_node: max steps %s reached — terminating with partial report",
                state.max_steps,
            )
            summary_text = await summarize_findings(state.extracted_texts, state.goal)
            final_report = {
                "goal": state.goal,
                "status": "partial",
                "steps_taken": state.step_number,
                "findings": {
                    "summary": summary_text,
                    "extracted_raw": state.extracted_texts,
                },
                "agent_reasoning_trace": state.action_history,
                "screenshots": [
                    entry
                    for entry in state.action_history
                    if "screenshot" in str(entry)
                ],
            }
            state.final_report = final_report
            state.status = "partial"
            return state

        if state.error_message is not None:
            if state.step_number < 3:
                state.error_message = None
            else:
                state.status = "failed"
                return state

        logger.info(
            "verify_node: step %s/%s — continuing", state.step_number, state.max_steps
        )
        return state
    except Exception as exc:  # noqa: BLE001 — nodes must never raise
        logger.error("verify_node raised: %s", exc)
        state.error_message = f"verify_node error: {exc}"
        state.status = "failed"
        return state


def should_continue(state: AgentState) -> str:
    """Routing function (sync) — LangGraph uses this to pick the next node."""
    if isinstance(state, dict):
        valid_keys = set(AgentState.model_fields.keys())
        filtered = {k: v for k, v in state.items() if k in valid_keys}
        # fill missing required fields with defaults
        filtered.setdefault("task_id", "")
        filtered.setdefault("goal", "")
        filtered.setdefault("step_number", 0)
        filtered.setdefault("max_steps", 25)
        filtered.setdefault("status", "running")
        filtered.setdefault("action_history", [])
        filtered.setdefault("extracted_texts", [])
        filtered.setdefault("screenshot_paths", [])
        filtered.setdefault("task_memory", [])
        filtered.setdefault("last_som_regions", [])
        try:
            state = AgentState(**filtered)
        except Exception as e:
            logger.error("state normalization failed: %s | keys: %s", e, list(state.keys()))
            # last resort — construct minimal valid state
            state = AgentState(
                task_id=filtered.get("task_id", ""),
                goal=filtered.get("goal", ""),
                status="failed",
                error_message=f"state normalization failed: {e}"
            )
    if state.status in ("complete", "partial", "failed"):
        return "end"
    return "observe"
