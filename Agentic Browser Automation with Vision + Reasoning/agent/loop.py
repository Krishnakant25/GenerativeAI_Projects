"""Agent loop — assembles the LangGraph graph and exposes run_task() as the entry point."""

import asyncio
import glob
import logging
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from langgraph.graph import StateGraph, END

from agent.models import AgentState
from agent.nodes import (
    observe_node, think_node, act_node, verify_node, should_continue,
    register_controller, unregister_controller,
)
from agent.tracer import StepTracer
from browser.controller import BrowserController
from db.storage import (
    init_db,
    save_task,
    save_step,
    save_report,
    update_task_status,
)

logger = logging.getLogger("agent.loop")

SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "screenshots")


def build_graph() -> StateGraph:
    """Assemble and compile the OBSERVE → THINK → ACT → VERIFY agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("observe", observe_node)
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("observe")
    graph.add_edge("observe", "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "verify")
    graph.add_conditional_edges(
        "verify", should_continue, {"observe": "observe", "end": END}
    )

    return graph.compile()


async def run_task(
    goal: str, max_steps: int = 25, headless: bool = True
) -> AsyncGenerator[dict, None]:
    """Run the full agent loop for a goal, yielding progress events as dicts."""
    task_id = str(uuid.uuid4())
    logger.info("run_task started: task_id=%s goal=%s", task_id, goal[:60])

    await init_db()
    await save_task(task_id, goal, max_steps)

    # block heavy resources only in headless mode — headed runs keep normal rendering
    controller = BrowserController(headless=headless, block_resources=headless)
    tracer: Optional[StepTracer] = None

    try:
        await controller.start()
        register_controller(task_id, controller)

        initial_state = AgentState(
            task_id=task_id,
            goal=goal,
            step_number=0,
            max_steps=max_steps,
            status="running",
            action_history=[],
            extracted_texts=[],
            screenshot_paths=[],
        )
        tracer = StepTracer(task_id)

        yield {"event": "started", "task_id": task_id, "goal": goal}

        # Manual dispatch loop — bypasses LangGraph astream entirely.
        # Nodes receive a real AgentState object so Playwright objects are
        # never lost to dict serialization.
        state = initial_state
        last_history_len = 0

        while state.status == "running":
            # OBSERVE
            state = await observe_node(state)
            if state.status in ("complete", "partial", "failed"):
                break

            # THINK
            state = await think_node(state)
            if state.status in ("complete", "partial", "failed"):
                break

            # ACT
            state = await act_node(state)

            # emit step event for any new history entry
            if len(state.action_history) > last_history_len:
                last_step = state.action_history[-1]
                last_history_len = len(state.action_history)
                await save_step(task_id, last_step)
                yield {
                    "event": "step",
                    "task_id": task_id,
                    "step_number": state.step_number,
                    "action": last_step.get("action_taken"),
                    "result": str(last_step.get("action_result", ""))[:120],
                    "status": state.status,
                }
                tracer.log_step(
                    state.step_number,
                    "step",
                    {
                        "action": last_step.get("action_taken"),
                        "target": last_step.get("target"),
                        "result": str(last_step.get("action_result", ""))[:200],
                    },
                )

            if state.status in ("complete", "partial", "failed"):
                break

            # VERIFY
            state = await verify_node(state)

            if should_continue(state) == "end":
                break

        # --- finalization ---
        screenshot_paths = sorted(
            glob.glob(os.path.join("screenshots", task_id, "step_*.png"))
        )

        step_screenshots = {}
        for path in screenshot_paths:
            stem = Path(path).stem
            try:
                step_num = int(stem.split("_")[1])
                step_screenshots.setdefault(step_num, path)
            except (IndexError, ValueError):
                pass

        report = state.final_report or {
            "goal": goal,
            "status": state.status,
            "steps_taken": state.step_number,
            "findings": {"summary": "No findings — task did not complete."},
            "agent_reasoning_trace": state.action_history,
            "screenshot_paths": screenshot_paths,
            "step_screenshots": step_screenshots,
            "discovered_facts": state.task_memory,
        }
        if state.final_report:
            report["screenshot_paths"] = screenshot_paths
            report["step_screenshots"] = step_screenshots
            report["discovered_facts"] = state.task_memory

        await save_report(task_id, report)
        await update_task_status(task_id, state.status, state.step_number)
        await controller.stop()
        unregister_controller(task_id)

        yield {
            "event": "finished",
            "task_id": task_id,
            "status": state.status,
            "steps_taken": state.step_number,
            "report": report,
        }
    except Exception as exc:  # noqa: BLE001 — surface as an error event, never raise
        logger.error("run_task failed for task %s: %s", task_id, exc)
        if controller.is_running:
            await controller.stop()
        unregister_controller(task_id)
        await update_task_status(task_id, "failed", 0)
        yield {"event": "error", "task_id": task_id, "error": str(exc)}
    finally:
        if tracer is not None:
            tracer.close()
