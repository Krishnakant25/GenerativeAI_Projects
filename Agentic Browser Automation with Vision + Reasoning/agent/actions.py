"""Browser action dispatcher — the only module that directly drives BrowserController."""

import asyncio
import logging
import os
import re
from typing import Optional

from playwright.async_api import Page

from agent.models import Action, StepRecord
from browser.controller import BrowserController
from browser.screenshot import capture, encode_to_base64

logger = logging.getLogger("agent.actions")

SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "screenshots")

_COORD_PATTERN = re.compile(r"\(?(\d+),\s*(\d+)\)?")


async def execute_action(
    controller: BrowserController,
    action: str,
    target: str,
    value: str,
    task_id: str,
    step_number: int,
) -> dict:
    """Route an action string to its handler and return a history dict.

    Never raises — returns a failed history dict on any exception.
    """
    try:
        logger.info(
            "Executing action: %s | target: %s | value: %s", action, target, value
        )
        if action == "navigate":
            return await _do_navigate(controller, target, task_id, step_number)
        if action == "click":
            return await _do_click(controller, target, task_id, step_number)
        if action == "click_text":
            return await _do_click_text(controller, target, task_id, step_number)
        if action == "type":
            return await _do_type(controller, target, value, task_id, step_number)
        if action == "scroll":
            return await _do_scroll(controller, value, task_id, step_number)
        if action == "extract":
            return await _do_extract(controller, task_id, step_number)
        if action == "finish":
            return await _do_finish(task_id, step_number)

        return {
            "step_number": step_number,
            "action_taken": action,
            "target": target,
            "action_result": f"unknown action: {action}",
        }
    except Exception as exc:  # noqa: BLE001 — dispatcher must never raise
        logger.error("execute_action(%s) raised: %s", action, exc)
        return {
            "step_number": step_number,
            "action_taken": action,
            "target": target,
            "action_result": f"exception: {str(exc)}",
        }


async def _do_navigate(
    controller: BrowserController, url: str, task_id: str, step_number: int
) -> dict:
    """Navigate to a URL, wait for load, and capture a screenshot."""
    ok = await controller.navigate(url)
    await controller.wait_for_load(timeout=5000)
    await capture(controller, task_id, step_number)
    return {
        "step_number": step_number,
        "action_taken": "navigate",
        "target": url,
        "action_result": "success" if ok else "failed: navigation returned False",
    }


async def _do_click(
    controller: BrowserController, target: str, task_id: str, step_number: int
) -> dict:
    """Parse (x, y) from target and click. Never raises on bad coordinates."""
    match = _COORD_PATTERN.search(target or "")
    if not match:
        return {
            "step_number": step_number,
            "action_taken": "click",
            "target": target,
            "action_result": f"failed: could not parse coordinates from target: {target}",
        }

    x, y = int(match.group(1)), int(match.group(2))
    ok = await controller.click(x, y)
    await asyncio.sleep(0.5)
    await capture(controller, task_id, step_number)
    return {
        "step_number": step_number,
        "action_taken": "click",
        "target": target,
        "action_result": "success" if ok else "failed: click returned False",
    }


async def _do_click_text(
    controller: BrowserController, text: str, task_id: str, step_number: int
) -> dict:
    """Click the first element matching `text`."""
    ok = await controller.click_text(text)
    await asyncio.sleep(0.5)
    await capture(controller, task_id, step_number)
    return {
        "step_number": step_number,
        "action_taken": "click_text",
        "target": text,
        "action_result": "success"
        if ok
        else "failed: element not found or not clickable",
    }


async def _do_type(
    controller: BrowserController,
    selector: str,
    text: str,
    task_id: str,
    step_number: int,
) -> dict:
    """Type `text` into `selector`."""
    ok = await controller.type_text(selector, text)
    await asyncio.sleep(0.3)
    await capture(controller, task_id, step_number)
    return {
        "step_number": step_number,
        "action_taken": "type",
        "target": selector,
        "action_result": f"success: typed '{text}'"
        if ok
        else f"failed: could not type into {selector}",
    }


async def _do_scroll(
    controller: BrowserController, direction: str, task_id: str, step_number: int
) -> dict:
    """Scroll up or down. Defaults to 'down' for empty/unrecognized direction."""
    if direction not in ("up", "down"):
        direction = "down"
    ok = await controller.scroll(direction, amount=300)
    await asyncio.sleep(0.3)
    await capture(controller, task_id, step_number)
    return {
        "step_number": step_number,
        "action_taken": "scroll",
        "target": direction,
        "action_result": "success" if ok else "failed",
    }


async def _do_extract(
    controller: BrowserController, task_id: str, step_number: int
) -> dict:
    """Extract visible body text, hard-capped at 3000 chars, and capture a screenshot."""
    try:
        text = await controller.page.inner_text("body")
        text = text.strip()
        if len(text) > 3000:
            text = text[:3000] + "... [truncated]"
        await capture(controller, task_id, step_number)
        return {
            "step_number": step_number,
            "action_taken": "extract",
            "target": "page",
            "action_result": text,
        }
    except Exception as exc:  # noqa: BLE001 — extract must never raise out
        logger.error("_do_extract failed: %s", exc)
        return {
            "step_number": step_number,
            "action_taken": "extract",
            "target": "page",
            "action_result": f"failed: {exc}",
        }


async def _do_finish(task_id: str, step_number: int) -> dict:
    """Signal task completion. No browser interaction."""
    return {
        "step_number": step_number,
        "action_taken": "finish",
        "target": "goal_complete",
        "action_result": "agent signalled task complete",
    }
